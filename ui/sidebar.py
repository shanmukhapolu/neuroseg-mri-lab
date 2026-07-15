import streamlit as st

def render_sidebar():

    st.sidebar.title("Controls")

    algo = st.sidebar.selectbox(
        "Segmentation Algorithm",
        [
            "Active Contour",
            "Region Growing",
            "Watershed",
            "Chan-Vese",
            "Random Walker",
            "K-Means"
        ]
    )

    opacity = st.sidebar.slider(
        "Overlay Opacity",
        0.0,
        1.0,
        0.5
    )

    return {
        "algorithm":algo,
        "opacity":opacity
    }