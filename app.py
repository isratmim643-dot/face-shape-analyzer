import gradio as gr
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import urllib.request
import os
import dlib
import bz2
from PIL import Image as PILImage
import joblib

# ── Load CNN model ────────────────────────────────────────────
MODEL_PATH  = "face_shape_model_final.h5"
model       = load_model(MODEL_PATH, compile=False)
class_names = ['Oval', 'Round', 'Square']
IMG_SIZE    = (224, 224)

# ── Load dlib landmark model ──────────────────────────────────
DAT_PATH = "shape_predictor_68_face_landmarks.dat"
DAT_GZ   = DAT_PATH + ".bz2"

if not os.path.exists(DAT_PATH):
    print("Downloading dlib model...")
    urllib.request.urlretrieve(
        "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
        DAT_GZ
    )
    with bz2.open(DAT_GZ, "rb") as f_in, open(DAT_PATH, "wb") as f_out:
        f_out.write(f_in.read())
    os.remove(DAT_GZ)
    print("dlib model ready.")

detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(DAT_PATH)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ── Anthropometric model ──────────────────────────────────────
ANTHRO_PATH  = "anthropometric_model.pkl"
anthro_model = joblib.load(ANTHRO_PATH) if os.path.exists(ANTHRO_PATH) else None

# ── Landmark indices (dlib 68-point) ─────────────────────────
LM_IDX = {
    "chin": 8,           "forehead_proxy": 27,
    "left_cheek": 1,     "right_cheek": 15,
    "left_jaw": 3,       "right_jaw": 13,
    "left_temple": 0,    "right_temple": 16,
    "left_eye_out": 36,  "right_eye_out": 45,
    "nose_tip": 33,
    "left_mouth": 48,    "right_mouth": 54,
}

def dist(p1, p2):
    return float(np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2))

def get_landmarks(img_rgb):
    h, w  = img_rgb.shape[:2]
    gray  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    dets  = detector(gray, 1)
    if len(dets) == 0:
        return None, h, w
    det   = max(dets, key=lambda d: d.width()*d.height())
    shape = predictor(gray, det)
    all_pts = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
    pts     = {name: all_pts[idx] for name, idx in LM_IDX.items()}
    # Estimate forehead by projecting above nose bridge
    eye_y  = (pts["left_eye_out"][1] + pts["right_eye_out"][1]) / 2
    nose_y = pts["forehead_proxy"][1]
    fh_y   = int(nose_y - 2.2 * (nose_y - eye_y))
    fh_x   = (pts["left_eye_out"][0] + pts["right_eye_out"][0]) // 2
    pts["forehead"] = (fh_x, max(0, fh_y))
    return pts, h, w

def compute_ratios(pts):
    fh  = dist(pts["forehead"],    pts["chin"])
    fw  = dist(pts["left_cheek"],  pts["right_cheek"])
    jw  = dist(pts["left_jaw"],    pts["right_jaw"])
    wf  = dist(pts["left_temple"], pts["right_temple"]) * 0.85
    eye_mid_y = (pts["left_eye_out"][1] + pts["right_eye_out"][1]) / 2
    lf  = abs(pts["forehead"][1] - pts["chin"][1])
    wc  = fw
    r_lc = lf / wc if wc > 0 else 0
    r_fc = wf / wc if wc > 0 else 0
    r_jc = jw / wc if wc > 0 else 0
    r_fj = wf / jw if jw > 0 else 0
    return {
        "face_index":        round(fh/fw if fw>0 else 0, 3),
        "forehead_to_cheek": round(r_fc, 3),
        "jaw_to_cheek":      round(r_jc, 3),
        "forehead_to_jaw":   round(r_fj, 3),
        "R_LC": round(r_lc, 3),
        "R_FC": round(r_fc, 3),
        "R_JC": round(r_jc, 3),
        "R_FJ": round(r_fj, 3),
        "face_height_px": round(fh, 1),
        "face_width_px":  round(fw, 1),
        "jaw_width_px":   round(jw, 1),
    }

def geo_classify(ratios):
    fi, jc = ratios["face_index"], ratios["jaw_to_cheek"]
    if fi > 1.55:                    return "Oblong"
    elif fi >= 1.28 and jc <= 0.85: return "Oval"
    elif fi >= 1.20 and jc < 0.72:  return "Heart"
    elif fi <= 1.15 and jc >= 0.85: return "Square"
    elif fi <= 1.20:                 return "Round"
    else:                            return "Oval"

def annotate(img_bgr, pts, ratios, geo, cnn_cls, cnn_conf, val_status):
    vis = img_bgr.copy()

    def line(p1, p2, color, label, side="top"):
        p1, p2 = tuple(map(int, p1)), tuple(map(int, p2))
        cv2.line(vis, p1, p2, color, 2)
        dx, dy = p2[0]-p1[0], p2[1]-p1[1]
        ln = max(np.sqrt(dx**2+dy**2), 1)
        nx, ny = int(-dy/ln*8), int(dx/ln*8)
        for pt in [p1, p2]:
            cv2.line(vis, (pt[0]-nx, pt[1]-ny), (pt[0]+nx, pt[1]+ny), color, 1)
        mx, my = (p1[0]+p2[0])//2, (p1[1]+p2[1])//2
        off = -22 if side == "top" else 18
        cv2.putText(vis, label, (mx-len(label)*4, my+off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

    line(pts["forehead"],   pts["chin"],        (0,230,230), f"FH={ratios['face_height_px']:.0f}px")
    line(pts["left_cheek"], pts["right_cheek"], (0,220,60),  f"CW={ratios['face_width_px']:.0f}px")
    line(pts["left_jaw"],   pts["right_jaw"],   (0,160,255), f"JW={ratios['jaw_width_px']:.0f}px", "bottom")

    for name in ["forehead","chin","left_cheek","right_cheek","left_jaw","right_jaw","nose_tip"]:
        pt = tuple(map(int, pts[name]))
        cv2.circle(vis, pt, 4, (255,255,0), -1)
        cv2.circle(vis, pt, 4, (0,0,0), 1)

    info = [
        f"FI  : {ratios['face_index']:.3f}",
        f"J/C : {ratios['jaw_to_cheek']:.3f}",
        f"Geo : {geo}",
        f"CNN : {cnn_cls} ({cnn_conf*100:.1f}%)",
        f"Val : {val_status}",
    ]
    for i, l in enumerate(info):
        y = 22 + i*18
        cv2.putText(vis, l, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 3, cv2.LINE_AA)
        col = (0,255,180) if "Geo" in l else \
              (0,200,255) if "CNN" in l else \
              (0,255,100) if "Supported" in l and "Not" not in l else \
              (0,120,255) if "Val" in l else (255,255,255)
        cv2.putText(vis, l, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

# ── Recommendations ───────────────────────────────────────────
RECS = {
    "Oval": {
        "Haircuts":   ["Long layers", "Shoulder-length lob", "Curtain bangs", "Loose waves"],
        "Eyeglasses": ["Square frames", "Rectangle frames", "Cat-eye frames", "Geometric frames"],
        "Earrings":   ["Hoops", "Teardrop earrings", "Stud earrings", "Geometric drops"],
    },
    "Round": {
        "Haircuts":   ["Blunt bob below chin", "Asymmetrical bob", "Long layered cut", "Side-swept bangs"],
        "Eyeglasses": ["Rectangle frames", "Square frames", "Angular geometric", "Browline frames"],
        "Earrings":   ["Long drop earrings", "Dangle earrings", "Slim teardrop", "Angular/geometric"],
    },
    "Square": {
        "Haircuts":   ["Soft layered lob", "Long waves", "Side-swept bangs", "Textured shoulder cut"],
        "Eyeglasses": ["Round frames", "Oval frames", "Cat-eye frames", "Thin curved frames"],
        "Earrings":   ["Round hoops", "Oval earrings", "Teardrop earrings", "Curved drop earrings"],
    },
}

# ── Main inference function ───────────────────────────────────
def analyze(pil_img):
    if pil_img is None:
        return None, "❌ No image uploaded.", ""

    img_rgb = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Face detection + crop
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
        pad = 20
        x1, y1 = max(0, x-pad), max(0, y-pad)
        x2, y2 = min(img_bgr.shape[1], x+w+pad), min(img_bgr.shape[0], y+h+pad)
        crop = img_bgr[y1:y2, x1:x2]
    else:
        crop = img_bgr

    # CNN prediction
    face_inp = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), IMG_SIZE) / 255.0
    probs    = model.predict(np.expand_dims(face_inp, 0), verbose=0)[0]
    cnn_idx  = int(np.argmax(probs))
    cnn_cls  = class_names[cnn_idx]
    cnn_conf = float(probs[cnn_idx])

    # Landmarks + anthropometric ratios
    pts, h, w = get_landmarks(img_rgb)

    if pts is None:
        out_img = pil_img
        ratios  = {}
        geo     = "N/A"
        val     = "Landmarks not detected"
    else:
        ratios = compute_ratios(pts)
        geo    = geo_classify(ratios)

        if anthro_model is not None:
            feat        = np.array([[ratios["R_LC"], ratios["R_FC"],
                                     ratios["R_JC"], ratios["R_FJ"]]])
            anthro_pred = class_names[int(anthro_model.predict(feat)[0])]
        else:
            anthro_pred = geo

        val     = "Supported" if anthro_pred == cnn_cls else "Not Supported"
        ann_rgb = annotate(img_bgr, pts, ratios, geo, cnn_cls, cnn_conf, val)
        out_img = PILImage.fromarray(ann_rgb)

    # Summary text
    def fmt(v): return f"{v:.3f}" if isinstance(v, float) else str(v)

    summary = (
        f"{'='*48}\n"
        f"  FACE SHAPE ANALYSIS RESULT\n"
        f"{'='*48}\n\n"
        f"🎯  Predicted Shape   : {cnn_cls}\n"
        f"📊  CNN Confidence    : {cnn_conf*100:.1f}%\n"
        f"📐  Geometric Class   : {geo}\n"
        f"{'✅' if val=='Supported' else '❌'}  Validation        : {val}\n\n"
        f"── FACIAL RATIOS (dlib landmarks) ──\n"
        f"  Face Index (L/CW)     : {fmt(ratios.get('face_index','–'))}\n"
        f"  Forehead-to-Cheek     : {fmt(ratios.get('forehead_to_cheek','–'))}\n"
        f"  Jaw-to-Cheek          : {fmt(ratios.get('jaw_to_cheek','–'))}\n"
        f"  Forehead-to-Jaw       : {fmt(ratios.get('forehead_to_jaw','–'))}\n\n"
        f"── CNN PROBABILITIES ──\n"
    )
    for i, cls in enumerate(class_names):
        p   = float(probs[i])
        bar = "█" * int(p * 20)
        summary += f"  {cls:8s}: {p*100:5.1f}%  {bar}\n"
    summary += (
        f"\n{'='*48}\n"
        f"  Final Decision: ŷ_Final = ŷ_CNN\n"
        f"{'='*48}"
    )

    # Recommendations text
    rec = RECS.get(cnn_cls, {})
    def fmt_list(items):
        return "\n".join(f"  {i+1}. {it}" for i, it in enumerate(items))

    recs_text = (
        f"{'='*48}\n"
        f"  PERSONALIZED RECOMMENDATIONS\n"
        f"  for {cnn_cls.upper()} face shape\n"
        f"{'='*48}\n\n"
        f"💇  HAIRCUTS:\n{fmt_list(rec.get('Haircuts', []))}\n\n"
        f"👓  EYEGLASSES:\n{fmt_list(rec.get('Eyeglasses', []))}\n\n"
        f"💎  EARRINGS:\n{fmt_list(rec.get('Earrings', []))}"
    )

    return out_img, summary, recs_text


# ── Gradio UI ─────────────────────────────────────────────────
with gr.Blocks() as demo:
    gr.Markdown("""
    # 🔬 Face Shape Analyzer
    **EfficientNetV2S · dlib 68-point Landmarks · Anthropometric Validation**

    *Israt Jahan Mim · ID: 0242220005101637 · Daffodil International University · 2026*
    *Supervised by: Partha Dip Sarkar · Co-supervised by: Amir Sohel*
    """)

    with gr.Row():
        with gr.Column(scale=1):
            inp_img = gr.Image(type="pil", label="📷 Upload a clear front-facing photo")
            btn     = gr.Button("🔍 Analyse Face Shape", variant="primary", size="lg")
            gr.Markdown("> **Tip:** Front-facing photo with good lighting works best.")
        with gr.Column(scale=1):
            out_img = gr.Image(label="📌 Annotated Result (dlib landmarks)")

    with gr.Row():
        with gr.Column():
            out_sum = gr.Textbox(label="📊 Analysis Result", lines=22)
        with gr.Column():
            out_rec = gr.Textbox(label="✨ Personalised Recommendations", lines=22)

    btn.click(fn=analyze, inputs=inp_img, outputs=[out_img, out_sum, out_rec])

    gr.Markdown("""
    ---
    **Methodology:** EfficientNetV2S is the primary and final classifier.
    Anthropometric Logistic Regression provides supporting validation only.
    **ŷ_Final = ŷ_CNN** — the CNN prediction is never overridden.
    """)

# ── Launch ────────────────────────────────────────────────────
demo.launch(server_name="0.0.0.0", server_port=7860)
