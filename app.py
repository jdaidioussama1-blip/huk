import streamlit as st
import requests
import base64
import io
from PIL import Image, ImageDraw, ImageFont

# ─────────────────────────────────────────────
#  CONFIGURATION FIXE (MODIFIEZ ICI)
# ─────────────────────────────────────────────
ROBOFLOW_API_KEY = "VOTRE_CLE_API_ICI"  # Remplacez par votre vraie clé
PROJECT_ID = "my-second-project_jdaidi"
MODEL_VERSION = "1"

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Vision AI · Détection Fixe",
    page_icon="🎯",
    layout="wide"
)

# Style Industriel (CSS)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono&family=Syne:wght@800&display=swap');
    html, body, [class*="css"] { font-family: 'Syne', sans-serif; background-color: #0d0d0f; color: #e8e6e1; }
    .title-block { border-left: 4px solid #f4a100; padding-left: 1rem; margin-bottom: 2rem; }
    .stButton > button { background: linear-gradient(135deg, #f4a100, #e05c00); color: #0d0d0f; font-weight: 800; border:none; width: 100%; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  FONCTIONS
# ─────────────────────────────────────────────

def call_roboflow_detection(image: Image.Image, conf: float):
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    
    url = f"https://detect.roboflow.com/{PROJECT_ID}/{MODEL_VERSION}"
    params = {"api_key": ROBOFLOW_API_KEY, "confidence": conf}
    
    response = requests.post(url, params=params, data=b64, timeout=30)
    response.raise_for_status()
    return response.json()

def annotate_image(image: Image.Image, predictions: list):
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    for pred in predictions:
        x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
        draw.rectangle([x-w/2, y-h/2, x+w/2, y+h/2], outline="#f4a100", width=3)
        draw.text((x-w/2, y-h/2 - 10), f"{pred['class']} {pred['confidence']:.1%}", fill="#f4a100")
    return img

# ─────────────────────────────────────────────
#  INTERFACE PRINCIPALE
# ─────────────────────────────────────────────
st.markdown(f"""
<div class='title-block'>
  <h1>🎯 Détecteur Automatique</h1>
  <p>Projet : {PROJECT_ID} (Mode Fixe)</p>
</div>
""", unsafe_allow_html=True)

# Barre latérale simplifiée
conf_threshold = st.sidebar.slider("Seuil de Confiance (%)", 0, 100, 40)

uploaded = st.file_uploader("Charger une image...", type=["jpg", "png", "jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Image source", use_container_width=True)
    
    if st.button("🚀 Lancer l'analyse"):
        with st.spinner("Analyse en cours sur Roboflow..."):
            try:
                result = call_roboflow_detection(image, conf_threshold)
                preds = result.get("predictions", [])
                
                if preds:
                    st.success(f"✅ {len(preds)} objets détectés")
                    annotated = annotate_image(image, preds)
                    st.image(annotated, caption="Résultat", use_container_width=True)
                else:
                    st.warning("Aucun objet détecté avec ce seuil.")
                
                with st.expander("Détails JSON"):
                    st.json(result)
            except Exception as e:
                st.error(f"Erreur : {e}")
