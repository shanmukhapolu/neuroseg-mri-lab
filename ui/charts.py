import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import streamlit as st


def histogram(image, mask):

    pixels = image[mask > 0]

    fig = px.histogram(
        x=pixels,
        nbins=40,
        template="plotly_dark"
    )

    fig.update_layout(
        title="Tumor Intensity Distribution",
        margin=dict(l=5, r=5, t=35, b=5),
        height=300,
        paper_bgcolor="#0B1220",
        plot_bgcolor="#0B1220"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


def comparison_chart(results):

    names = list(results.keys())
    areas = [results[k]["Area (px²)"] for k in names]

    fig = go.Figure()

    fig.add_bar(
        x=names,
        y=areas
    )

    fig.update_layout(
        title="Algorithm Comparison",
        template="plotly_dark",
        height=350,
        paper_bgcolor="#0B1220",
        plot_bgcolor="#0B1220"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


def radar_chart(metrics):

    labels = [
        "Circularity",
        "Solidity",
        "Convexity",
        "Extent",
        "Rectangularity"
    ]

    values = [
        metrics["Circularity"],
        metrics["Solidity"],
        metrics["Convexity"],
        metrics["Extent"],
        metrics["Rectangularity"]
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=labels,
            fill="toself"
        )
    )

    fig.update_layout(
        template="plotly_dark",
        height=420,
        polar=dict(radialaxis=dict(range=[0,1]))
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


def intensity_profile(image):

    center = image.shape[0] // 2

    profile = image[center]

    fig = px.line(
        y=profile,
        template="plotly_dark"
    )

    fig.update_layout(
        title="Center Line Intensity",
        height=300,
        paper_bgcolor="#0B1220",
        plot_bgcolor="#0B1220"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )