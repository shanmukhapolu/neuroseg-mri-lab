import streamlit as st

from ui.style import load_css
from ui.sidebar import render_sidebar
from ui.viewer import display_workspace
from ui.dashboard import metric_card

from preprocessing.synthetic import generate_synthetic_mri
from utils.helpers import load_image

st.set_page_config(
    page_title="NeuroSeg",
    layout="wide"
)

load_css()

overlay = None
gradient = None
heatmap = None

st.markdown(
    """
    <div class="main-title">
    NeuroSeg
    </div>

    <div class="subtitle">
    Medical Image Segmentation Workstation
    </div>
    """,
    unsafe_allow_html=True
)

settings = render_sidebar()

uploaded = st.file_uploader(
    "Upload MRI",
    type=["png","jpg","jpeg"]
)

if uploaded:
    image = load_image(uploaded)
else:
    image = generate_synthetic_mri()

left,right = st.columns([3,1])

with left:

    st.subheader("MRI Viewer")

    display_workspace(
    original=image,
    overlay=overlay,
    gradient=gradient,
    heatmap=heatmap
)

with right:

    st.subheader("Analytics")

    metric_card(
        "Algorithm",
        settings["algorithm"]
    )

    metric_card(
        "Resolution",
        f"{image.shape[0]}×{image.shape[1]}"
    )

    metric_card(
        "Mean Intensity",
        f"{image.mean():.3f}"
    )