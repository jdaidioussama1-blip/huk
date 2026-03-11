import streamlit as st
import requests
import base64
import json
import io
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from datetime import datetime

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Vision AI · Roboflow",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS  (dark industrial aesthetic)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0d0d0f;
    color: #e8e6e1;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #111114;
    border-right: 1px solid #2a2a35;
}
section[data-testid="stSidebar"] * { color: #c9c7c0 !important; }

/* Inputs */
input, textarea, select {
    background: #1a1a22 !important;
    border: 1px solid #3a3a50 !important;
    border-radius: 6px !important;
    color: #e8e6e1 !important;
    font-family: 'Space Mono', monospace !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #f4a100, #e05c00);
    color: #0d0d0f;
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 0.95rem;
    border: none;
    border-radius: 6px;
    padding: 0.6rem 2rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    transition: opacity .2s;
}
.stButton > button:hover { opacity: 0.85; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #16161e;
    border: 1px solid #2a2a35;
    border-radius: 10px;
    padding: 1rem;
}
[data-testid="metric-container"] label { color: #888 !important; font-size: 0.75rem; }
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #f4a100 !important;
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
}

/* Upload zone */
[data-testid="stFileUploader"] {
    background: #111114;
    border: 2px dashed #3a3a50;
    border-radius: 10px;
    padding: 1.5rem;
}

/* Title accent */
.title-block { border-left: 4px solid #f4a100; padding-left: 1rem; margin-bottom: 2rem; }
.title-block h1 { font-size: 2.2rem; font-weight: 800; color: #f4f2ec; margin: 0; }
.title-block p  { color: #666; font-size: 0.85rem; font-family: 'Space Mono', monospace; margin: 0; }

/* Result badge */
.badge {
    display: inline-block;
    background: linear-gradient(135deg, #f4a100, #e05c00);
    color: #0d0d0f;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    font-size: 1.1rem;
    padding: 0.4rem 1.2rem;
    border-radius: 4px;
    letter-spacing: 0.06em;
}
.conf-bar-bg {
    background: #1a1a22;
    border-radius: 20px;
    height: 10px;
    margin-top: 4px;
}
.conf-bar-fill {
    height: 10px;
    border-radius: 20px;
    background: linear-gradient(90deg, #f4a100, #e05c00);
}
.pred-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.5rem 0;
    border-bottom: 1px solid #1e1e28;
}
.pred-label { font-size: 0.9rem; font-weight: 600; }
.pred-conf  { font-family: 'Space Mono', monospace; font-size: 0.8rem; color: #f4a100; }
.section-tag {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.5rem;
}
hr { border-color: #2a2a35; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def encode_image_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def call_roboflow_api(image: Image.Image, api_key: str, project: str, version: str, confidence: float) -> dict:
    """Call Roboflow hosted inference API."""
    b64 = encode_image_b64(image)
    url = f"https://classify.roboflow.com/{project}/{version}"
    params = {"api_key": api_key, "confidence": int(confidence)}
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    response = requests.post(url, params=params, data=b64, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


def annotate_image(image: Image.Image, predictions: list) -> Image.Image:
    """Overlay top prediction label on the image."""
    img = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    if not predictions:
        return img.convert("RGB")

    top = predictions[0]
    label = f"{top['class']}  {top['confidence']*100:.1f}%"

    # semi-transparent banner at bottom
    banner_h = max(40, img.height // 12)
    draw.rectangle([(0, img.height - banner_h), (img.width, img.height)],
                   fill=(20, 20, 20, 200))

    # try to load a font, fall back gracefully
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                                  size=max(16, banner_h // 2))
    except Exception:
        font = ImageFont.load_default()

    draw.text((12, img.height - banner_h + 8), label, fill=(244, 161, 0, 255), font=font)
    combined = Image.alpha_composite(img, overlay)
    return combined.convert("RGB")


def make_bar_chart(predictions: list) -> plt.Figure:
    labels = [p["class"] for p in predictions]
    scores = [p["confidence"] * 100 for p in predictions]
    colors = ["#f4a100" if i == 0 else "#2a2a40" for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(6, max(2, len(labels) * 0.55)))
    fig.patch.set_facecolor("#0d0d0f")
    ax.set_facecolor("#0d0d0f")

    bars = ax.barh(labels[::-1], scores[::-1], color=colors[::-1], height=0.55, edgecolor="none")

    ax.set_xlim(0, 105)
    ax.set_xlabel("Confidence (%)", color="#666", fontsize=9)
    ax.tick_params(colors="#aaa", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2a35")

    for bar, score in zip(bars, scores[::-1]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{score:.1f}%", va="center", ha="left", color="#f4a100",
                fontsize=8, fontfamily="monospace")

    plt.tight_layout()
    return fig


def pil_to_download_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


# ─────────────────────────────────────────────
#  SIDEBAR — CONFIGURATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    api_key = st.text_input("🔑 Roboflow API Key", type="password",
                            placeholder="rf-xxxxxxxxxxxxxxxxx")
    project  = st.text_input("📁 Project ID", placeholder="my-project-slug")
    version  = st.text_input("🔢 Model Version", value="1", placeholder="1")
    confidence = st.slider("Seuil de confiance (%)", 0, 100, 40)

    st.markdown("---")
    st.markdown("""
    <div class='section-tag'>Comment trouver vos infos ?</div>
    <small>
    1. <b>API Key</b> → roboflow.com → Settings → API Keys<br>
    2. <b>Project ID</b> → URL de votre projet<br>
    3. <b>Version</b> → Numéro de votre version entraînée
    </small>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MAIN CONTENT
# ─────────────────────────────────────────────
st.markdown("""
<div class='title-block'>
  <h1>🔍 Vision AI · Classifier</h1>
  <p>Powered by Roboflow · Image Classification</p>
</div>
""", unsafe_allow_html=True)

# Upload
st.markdown("<div class='section-tag'>① Charger une image</div>", unsafe_allow_html=True)
uploaded = st.file_uploader("", type=["jpg", "jpeg", "png", "webp"],
                             accept_multiple_files=False)

if uploaded:
    image = Image.open(uploaded).convert("RGB")

    col_img, col_meta = st.columns([2, 1])
    with col_img:
        st.image(image, caption="Image originale", use_container_width=True)
    with col_meta:
        st.metric("Largeur", f"{image.width} px")
        st.metric("Hauteur", f"{image.height} px")
        st.metric("Format", uploaded.type.split("/")[-1].upper())
        st.metric("Taille", f"{len(uploaded.getvalue())/1024:.1f} KB")

    st.markdown("---")

    # Predict button
    ready = api_key and project and version
    if not ready:
        st.warning("⚠️ Veuillez remplir la clé API, le Project ID et la Version dans la barre latérale.")

    if st.button("🚀 Lancer la Prédiction", disabled=not ready):
        with st.spinner("Envoi à Roboflow..."):
            try:
                result = call_roboflow_api(image, api_key, project, version, confidence)
                st.session_state["result"] = result
                st.session_state["image"]  = image
                st.success("✅ Prédiction réussie !")
            except requests.exceptions.HTTPError as e:
                st.error(f"❌ Erreur API : {e.response.status_code} — {e.response.text}")
            except Exception as e:
                st.error(f"❌ Erreur : {e}")

# ─────────────────────────────────────────────
#  RESULTS
# ─────────────────────────────────────────────
if "result" in st.session_state and "image" in st.session_state:
    result = st.session_state["result"]
    image  = st.session_state["image"]

    predictions = result.get("predictions", [])

    if not predictions:
        st.warning("Aucune prédiction retournée. Essayez de baisser le seuil de confiance.")
    else:
        # Sort by confidence
        predictions = sorted(predictions, key=lambda x: x["confidence"], reverse=True)
        top = predictions[0]

        st.markdown("## 📊 Résultats")

        # ── Top prediction banner
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("🏆 Classe prédite", top["class"])
        with c2:
            st.metric("📈 Confiance", f"{top['confidence']*100:.2f}%")
        with c3:
            st.metric("🔢 Nb de classes", len(predictions))

        st.markdown("---")

        left, right = st.columns(2)

        # ── Annotated image
        with left:
            st.markdown("<div class='section-tag'>② Image annotée</div>", unsafe_allow_html=True)
            annotated = annotate_image(image, predictions)
            st.image(annotated, use_container_width=True)

            dl_bytes = pil_to_download_bytes(annotated)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label="⬇️ Télécharger l'image annotée",
                data=dl_bytes,
                file_name=f"prediction_{top['class']}_{ts}.png",
                mime="image/png",
            )

        # ── Bar chart + detail list
        with right:
            st.markdown("<div class='section-tag'>③ Statistiques de détection</div>",
                        unsafe_allow_html=True)
            fig = make_bar_chart(predictions[:10])
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            st.markdown("<div class='section-tag'>Détail des scores</div>",
                        unsafe_allow_html=True)
            for p in predictions[:10]:
                pct = p["confidence"] * 100
                fill_w = int(pct)
                st.markdown(f"""
                <div class='pred-row'>
                  <span class='pred-label'>{p['class']}</span>
                  <span class='pred-conf'>{pct:.2f}%</span>
                </div>
                <div class='conf-bar-bg'>
                  <div class='conf-bar-fill' style='width:{fill_w}%'></div>
                </div>
                """, unsafe_allow_html=True)

        # ── Raw JSON expander
        with st.expander("🔎 Réponse JSON brute de l'API"):
            st.json(result)

        # ── Download JSON report
        report = {
            "timestamp": datetime.now().isoformat(),
            "model": {"project": project, "version": version},
            "top_prediction": {"class": top["class"], "confidence": top["confidence"]},
            "all_predictions": predictions,
        }
        st.download_button(
            "⬇️ Télécharger le rapport JSON",
            data=json.dumps(report, indent=2, ensure_ascii=False),
            file_name=f"report_{ts}.json",
            mime="application/json",
        )

else:
    if not uploaded:
        st.markdown("""
        <div style='text-align:center; padding:4rem 2rem; color:#444;'>
            <div style='font-size:3rem;'>📂</div>
            <p style='font-family: Space Mono, monospace; font-size:0.9rem;'>
                Chargez une image pour commencer
            </p>
        </div>
        """, unsafe_allow_html=True)
