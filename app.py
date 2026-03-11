import streamlit as st
import requests
import base64
import json
import io
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from datetime import datetime

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Vision AI · Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS (votre style industriel conservé)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif; background-color: #0d0d0f; color: #e8e6e1; }
section[data-testid="stSidebar"] { background: #111114; border-right: 1px solid #2a2a35; }
.stButton > button { background: linear-gradient(135deg, #f4a100, #e05c00); color: #0d0d0f; font-weight: 800; border-radius: 6px; border:none; }
[data-testid="metric-container"] { background: #16161e; border: 1px solid #2a2a35; border-radius: 10px; padding: 1rem; }
.title-block { border-left: 4px solid #f4a100; padding-left: 1rem; margin-bottom: 2rem; }
.section-tag { font-family: 'Space Mono', monospace; font-size: 0.7rem; color: #555; text-transform: uppercase; letter-spacing: 0.12em; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HELPERS (ADAPTÉS POUR LA DÉTECTION)
# ─────────────────────────────────────────────

def encode_image_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def call_roboflow_detection_api(image: Image.Image, api_key: str, project: str, version: str, confidence: float, overlap: float) -> dict:
    """Appel l'API de détection d'objets (Object Detection)."""
    b64 = encode_image_b64(image)
    # Notez le changement d'URL : 'detect.roboflow.com' au lieu de 'classify'
    url = f"https://detect.roboflow.com/{project}/{version}"
    params = {
        "api_key": api_key, 
        "confidence": confidence,
        "overlap": overlap # NMS (Non-Maximum Suppression)
    }
    response = requests.post(url, params=params, data=b64, timeout=30)
    response.raise_for_status()
    return response.json()

def annotate_detection_image(image: Image.Image, predictions: list) -> Image.Image:
    """Dessine les boîtes englobantes et les étiquettes."""
    img = image.copy().convert("RGBA")
    draw = ImageDraw.Draw(img)
    
    # Tentative de chargement de police
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
    except:
        font = ImageFont.load_default()

    for pred in predictions:
        # Roboflow renvoie x, y (centre) et width, height
        x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
        left = x - w / 2
        top = y - h / 2
        right = x + w / 2
        bottom = y + h / 2
        
        # Dessiner le rectangle
        color = "#f4a100"
        draw.rectangle([left, top, right, bottom], outline=color, width=3)
        
        # Dessiner l'étiquette
        label = f"{pred['class']} {pred['confidence']:.1%}"
        text_bbox = draw.textbbox((left, top), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((left, top), label, fill="black", font=font)

    return img.convert("RGB")

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Config Détection")
    api_key = st.text_input("🔑 Roboflow API Key", type="password")
    # ID mis à jour selon votre demande
    project = st.text_input("📁 Project ID", value="my-second-project_jdaidi")
    version = st.text_input("🔢 Model Version", value="1")
    
    st.markdown("---")
    conf_threshold = st.slider("Confiance (%)", 0, 100, 40)
    overlap_threshold = st.slider("Chevauchement (IoU) (%)", 0, 100, 30)

# ─────────────────────────────────────────────
#  MAIN CONTENT
# ─────────────────────────────────────────────
st.markdown("""
<div class='title-block'>
  <h1>🎯 Vision AI · Object Detector</h1>
  <p>Projet : my-second-project_jdaidi · Object Detection</p>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader("", type=["jpg", "jpeg", "png", "webp"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.image(image, caption="Image originale", use_container_width=True)
    
    with col2:
        st.markdown("<div class='section-tag'>Info Fichier</div>", unsafe_allow_html=True)
        st.metric("Résolution", f"{image.width}x{image.height}")
        
        ready = api_key and project and version
        if st.button("🚀 Détecter les Objets", disabled=not ready, use_container_width=True):
            with st.spinner("Analyse en cours..."):
                try:
                    # Appel API
                    res = call_roboflow_detection_api(
                        image, api_key, project, version, conf_threshold, overlap_threshold
                    )
                    st.session_state["det_result"] = res
                    st.session_state["orig_img"] = image
                except Exception as e:
                    st.error(f"Erreur: {e}")

# ─────────────────────────────────────────────
#  RESULTS
# ─────────────────────────────────────────────
if "det_result" in st.session_state:
    result = st.session_state["det_result"]
    img_orig = st.session_state["orig_img"]
    preds = result.get("predictions", [])

    st.markdown("---")
    res_col1, res_col2 = st.columns([2, 1])

    with res_col1:
        st.markdown("<div class='section-tag'>Visualisation</div>", unsafe_allow_html=True)
        annotated = annotate_detection_image(img_orig, preds)
        st.image(annotated, use_container_width=True)

    with res_col2:
        st.markdown("<div class='section-tag'>Objets trouvés</div>", unsafe_allow_html=True)
        st.metric("Total", len(preds))
        
        if preds:
            # Petit tableau des scores
            for p in preds[:15]:
                st.markdown(f"**{p['class']}** : `{p['confidence']:.2%}`")
        else:
            st.info("Aucun objet détecté.")

    with st.expander("Raw Data (JSON)"):
        st.json(result)
