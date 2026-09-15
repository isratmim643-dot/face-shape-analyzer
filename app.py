import cv2, numpy as np, os, base64, urllib.request, bz2, joblib
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from tensorflow.keras.models import load_model
import uvicorn
from PIL import Image
import io, dlib

app = FastAPI()

# Load models
model       = load_model("face_shape_model_final.h5", compile=False)
class_names = ['Oval', 'Round', 'Square']
IMG_SIZE    = (224, 224)

anthro_model = joblib.load("anthropometric_model.pkl") if os.path.exists("anthropometric_model.pkl") else None

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

DAT_PATH = "shape_predictor_68_face_landmarks.dat"
if not os.path.exists(DAT_PATH):
    urllib.request.urlretrieve("http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2", DAT_PATH+".bz2")
    with bz2.open(DAT_PATH+".bz2","rb") as f_in, open(DAT_PATH,"wb") as f_out: f_out.write(f_in.read())
    os.remove(DAT_PATH+".bz2")

detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(DAT_PATH)

LM_IDX = {"chin":8,"forehead_proxy":27,"left_cheek":1,"right_cheek":15,
           "left_jaw":3,"right_jaw":13,"left_temple":0,"right_temple":16,
           "left_eye_out":36,"right_eye_out":45,"nose_tip":33,"left_mouth":48,"right_mouth":54}

RECS = {
    "Oval":   {"Haircuts":["Long layers","Shoulder-length lob","Curtain bangs","Loose waves"],
               "Eyeglasses":["Square frames","Rectangle frames","Cat-eye frames","Geometric frames"],
               "Earrings":["Hoops","Teardrop earrings","Stud earrings","Geometric drops"]},
    "Round":  {"Haircuts":["Blunt bob below chin","Asymmetrical bob","Long layered cut","Side-swept bangs"],
               "Eyeglasses":["Rectangle frames","Square frames","Angular geometric","Browline frames"],
               "Earrings":["Long drop earrings","Dangle earrings","Slim teardrop","Angular/geometric"]},
    "Square": {"Haircuts":["Soft layered lob","Long waves","Side-swept bangs","Textured shoulder cut"],
               "Eyeglasses":["Round frames","Oval frames","Cat-eye frames","Thin curved frames"],
               "Earrings":["Round hoops","Oval earrings","Teardrop earrings","Curved drop earrings"]},
}

def dist(p1,p2): return float(np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2))

def get_landmarks(img_rgb):
    h,w = img_rgb.shape[:2]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    dets = detector(gray,1)
    if not dets: return None
    det   = max(dets, key=lambda d: d.width()*d.height())
    shape = predictor(gray, det)
    pts   = {n: (shape.part(i).x, shape.part(i).y) for n,i in LM_IDX.items()}
    ey    = (pts["left_eye_out"][1]+pts["right_eye_out"][1])/2
    ny    = pts["forehead_proxy"][1]
    pts["forehead"] = ((pts["left_eye_out"][0]+pts["right_eye_out"][0])//2, max(0,int(ny-2.2*(ny-ey))))
    return pts

def compute_ratios(pts):
    fh=dist(pts["forehead"],pts["chin"]); fw=dist(pts["left_cheek"],pts["right_cheek"])
    jw=dist(pts["left_jaw"],pts["right_jaw"]); wf=dist(pts["left_temple"],pts["right_temple"])*0.85
    lf=abs(pts["forehead"][1]-pts["chin"][1]); wc=fw
    return {"face_index":round(fh/fw if fw>0 else 0,3),
            "forehead_to_cheek":round(wf/wc if wc>0 else 0,3),
            "jaw_to_cheek":round(jw/wc if wc>0 else 0,3),
            "forehead_to_jaw":round(wf/jw if jw>0 else 0,3),
            "R_LC":round(lf/wc if wc>0 else 0,3),"R_FC":round(wf/wc if wc>0 else 0,3),
            "R_JC":round(jw/wc if wc>0 else 0,3),"R_FJ":round(wf/jw if jw>0 else 0,3),
            "face_height_px":round(fh,1),"face_width_px":round(fw,1),"jaw_width_px":round(jw,1)}

def annotate(img_bgr, pts, ratios, geo, cnn_cls, cnn_conf, val):
    vis = img_bgr.copy()
    def line(p1,p2,color,label,side="top"):
        p1,p2=tuple(map(int,p1)),tuple(map(int,p2))
        cv2.line(vis,p1,p2,color,2)
        dx,dy=p2[0]-p1[0],p2[1]-p1[1]; ln=max(np.sqrt(dx**2+dy**2),1)
        nx,ny=int(-dy/ln*8),int(dx/ln*8)
        for pt in [p1,p2]: cv2.line(vis,(pt[0]-nx,pt[1]-ny),(pt[0]+nx,pt[1]+ny),color,1)
        mx,my=(p1[0]+p2[0])//2,(p1[1]+p2[1])//2
        cv2.putText(vis,label,(mx-len(label)*4,my+(-22 if side=="top" else 18)),cv2.FONT_HERSHEY_SIMPLEX,0.38,color,1,cv2.LINE_AA)
    line(pts["forehead"],pts["chin"],(0,230,230),f"FH={ratios['face_height_px']:.0f}px")
    line(pts["left_cheek"],pts["right_cheek"],(0,220,60),f"CW={ratios['face_width_px']:.0f}px")
    line(pts["left_jaw"],pts["right_jaw"],(0,160,255),f"JW={ratios['jaw_width_px']:.0f}px","bottom")
    for nm in ["forehead","chin","left_cheek","right_cheek","left_jaw","right_jaw","nose_tip"]:
        pt=tuple(map(int,pts[nm])); cv2.circle(vis,pt,4,(255,255,0),-1); cv2.circle(vis,pt,4,(0,0,0),1)
    for i,l in enumerate([f"FI:{ratios['face_index']:.3f}",f"J/C:{ratios['jaw_to_cheek']:.3f}",
                           f"Geo:{geo}",f"CNN:{cnn_cls}({cnn_conf*100:.0f}%)",f"Val:{val}"]):
        y=20+i*18; cv2.putText(vis,l,(8,y),cv2.FONT_HERSHEY_SIMPLEX,0.42,(0,0,0),3,cv2.LINE_AA)
        cv2.putText(vis,l,(8,y),cv2.FONT_HERSHEY_SIMPLEX,0.42,(0,255,180) if "Geo" in l else (0,200,255) if "CNN" in l else (255,255,255),1,cv2.LINE_AA)
    return cv2.cvtColor(vis,cv2.COLOR_BGR2RGB)

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse(open("index.html").read())

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    data    = await file.read()
    img_rgb = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.1,5,minSize=(60,60))
    crop  = img_bgr[max(0,faces[0][1]-20):faces[0][1]+faces[0][3]+20,
                    max(0,faces[0][0]-20):faces[0][0]+faces[0][2]+20] if len(faces)>0 else img_bgr

    inp   = cv2.resize(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB),IMG_SIZE)/255.0
    probs = model.predict(np.expand_dims(inp,0),verbose=0)[0]
    idx   = int(np.argmax(probs)); cnn_cls=class_names[idx]; cnn_conf=float(probs[idx])

    pts = get_landmarks(img_rgb)
    if pts:
        ratios = compute_ratios(pts)
        fi,jc  = ratios["face_index"],ratios["jaw_to_cheek"]
        if fi>1.55: geo="Oblong"
        elif fi>=1.28 and jc<=0.85: geo="Oval"
        elif fi>=1.20 and jc<0.72:  geo="Heart"
        elif fi<=1.15 and jc>=0.85: geo="Square"
        elif fi<=1.20: geo="Round"
        else: geo="Oval"
        if anthro_model:
            raw=anthro_model.predict(np.array([[ratios["R_LC"],ratios["R_FC"],ratios["R_JC"],ratios["R_FJ"]]]))[0]
            try: ap=class_names[int(raw)]
            except: ap=str(raw)
        else: ap=geo
        val = "Supported" if ap==cnn_cls else "Not Supported"
        ann = annotate(img_bgr,pts,ratios,geo,cnn_cls,cnn_conf,val)
        buf = io.BytesIO(); Image.fromarray(ann).save(buf,format="JPEG"); buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode()
    else:
        ratios={}; geo="N/A"; val="Landmarks not detected"; img_b64=""

    return JSONResponse({"shape":cnn_cls,"confidence":round(cnn_conf*100,1),
        "geo":geo,"validation":val,
        "ratios":ratios,
        "probs":{c:round(float(probs[i])*100,1) for i,c in enumerate(class_names)},
        "recommendations":RECS.get(cnn_cls,{}),
        "annotated_image":img_b64})

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
