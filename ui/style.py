import streamlit as st

def load_css():
    st.markdown("""
    <style>

    .stApp{
        background:#0B1220;
        color:#F8FAFC;
    }

    .main-title{
        font-size:3rem;
        font-weight:700;
        margin-bottom:0;
    }

    .subtitle{
        color:#94A3B8;
        margin-top:0;
        margin-bottom:2rem;
    }

    .metric-card{
        background:#111827;
        padding:20px;
        border-radius:16px;
        border:1px solid rgba(255,255,255,0.05);
        text-align:center;
    }

    .metric-title{
        color:#94A3B8;
        font-size:0.9rem;
    }

    .metric-value{
        font-size:2rem;
        font-weight:700;
    }

    div[data-testid="stSidebar"]{
        background:#111827;
    }

    </style>
    """, unsafe_allow_html=True)