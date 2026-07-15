import streamlit as st


def metric_card(title, value, color="#4F46E5"):
    st.markdown(
        f"""
        <div style="
            background:linear-gradient(145deg,#151B28,#111827);
            border-radius:18px;
            padding:20px;
            margin-bottom:12px;
            border:1px solid rgba(255,255,255,.06);
            border-left:6px solid {color};
            box-shadow:0 8px 18px rgba(0,0,0,.25);
        ">

            <div style="
                color:#94A3B8;
                font-size:13px;
                margin-bottom:6px;
                text-transform:uppercase;
                letter-spacing:.08em;
            ">
                {title}
            </div>

            <div style="
                color:white;
                font-size:30px;
                font-weight:700;
            ">
                {value}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


def status_card(title, value, status):

    colors = {
        "good": "#22C55E",
        "warning": "#FACC15",
        "bad": "#EF4444",
    }

    metric_card(title, value, colors.get(status, "#4F46E5"))