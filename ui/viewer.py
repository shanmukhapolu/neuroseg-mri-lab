import streamlit as st


def display_workspace(
    original,
    overlay=None,
    gradient=None,
    heatmap=None
):
    """
    Display the NeuroSeg imaging workstation.

    Parameters
    ----------
    original : ndarray
        Original grayscale MRI.

    overlay : ndarray
        Segmentation overlay.

    gradient : ndarray
        Gradient magnitude image.

    heatmap : ndarray
        Colored heatmap.
    """

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            original,
            caption="Original MRI",
            use_container_width=True,
            clamp=True
        )

    with col2:

        if overlay is not None:
            st.image(
                overlay,
                caption="Segmentation Overlay",
                use_container_width=True
            )
        else:
            st.info("Run a segmentation algorithm.")

    col3, col4 = st.columns(2)

    with col3:

        if gradient is not None:
            st.image(
                gradient,
                caption="Gradient Magnitude",
                use_container_width=True,
                clamp=True
            )
        else:
            st.empty()

    with col4:

        if heatmap is not None:
            st.image(
                heatmap,
                caption="Heatmap",
                use_container_width=True
            )
        else:
            st.empty()