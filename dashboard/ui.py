import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def inject_css():
    st.markdown(
        """
        <style>
        /* ---------- Global shell ---------- */
        .stApp {
            background:
                radial-gradient(circle at 90% 0%, rgba(99,70,255,.18), transparent 30%),
                #080d1a;
            color: #eef2ff;
        }

        /* Streamlit top chrome: keep it dark instead of the default white bar */
        header[data-testid="stHeader"] {
            background: #080d1a !important;
            border-bottom: 1px solid #1d2742 !important;
        }

        header[data-testid="stHeader"] * {
            color: #dbe4ff !important;
        }

        header[data-testid="stHeader"] svg {
            fill: #dbe4ff !important;
            stroke: #dbe4ff !important;
        }

        [data-testid="stToolbar"] {
            background: transparent !important;
        }

        [data-testid="stDecoration"] {
            background: transparent !important;
        }

        /* ---------- Sidebar ---------- */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #080d1a, #11162a);
            border-right: 1px solid #1d2742;
        }

        [data-testid="stSidebar"] * {
            color: #dbe4ff;
        }

        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p {
            color: #aebbd7 !important;
        }

        [data-testid="stSidebar"] input {
            color: #eef2ff !important;
            background: #151d31 !important;
            border: 1px solid #34415f !important;
        }

        /* ---------- Main layout ---------- */
        .block-container {
            max-width: 1500px;
            padding-top: 1.8rem;
            padding-bottom: 3rem;
        }

        .hero {
            background: linear-gradient(135deg, rgba(26,34,61,.98), rgba(15,20,38,.98));
            border: 1px solid #303b63;
            border-radius: 24px;
            padding: 28px;
            box-shadow: 0 0 55px rgba(99,70,255,.14);
            margin-bottom: 22px;
        }

        .kicker {
            color: #a78bfa;
            font-size: .76rem;
            font-weight: 800;
            letter-spacing: .16em;
            text-transform: uppercase;
        }

        .hero h1 {
            color: #f5f7ff !important;
            font-size: 2.45rem;
            line-height: 1.15;
            margin: 4px 0;
        }

        .subtitle {
            color: #c4cee4;
            font-size: 1rem;
        }

        .pill {
            display: inline-block;
            margin: 10px 6px 0 0;
            padding: 6px 11px;
            border-radius: 999px;
            background: rgba(139,124,255,.13);
            border: 1px solid rgba(139,124,255,.3);
            color: #d8d1ff;
            font-size: .70rem;
            font-weight: 700;
        }

        /* ---------- Streamlit text / labels ---------- */
        .stMarkdown,
        .stMarkdown p,
        .stMarkdown li {
            color: #dbe4ff;
        }

        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] label,
        .stSelectbox label,
        .stNumberInput label,
        .stCheckbox label,
        .stTextInput label {
            color: #aebbd7 !important;
        }

        .stCaption,
        [data-testid="stCaptionContainer"] {
            color: #8f9dbb !important;
        }

        /* ---------- Inputs ---------- */
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-baseweb="base-input"] {
            background: #f5f7fb !important;
            border-color: #cbd3e1 !important;
        }

        div[data-baseweb="select"] *,
        div[data-baseweb="input"] input,
        div[data-baseweb="base-input"] input {
            color: #172033 !important;
        }

        /* Checkbox text must remain readable on the dark canvas */
        [data-testid="stCheckbox"] label p {
            color: #dbe4ff !important;
        }

        /* ---------- Buttons ---------- */
        button[kind="primary"] {
            color: #ffffff !important;
            font-weight: 700 !important;
        }

        /* ---------- Cards ---------- */
        .metric-card {
            background: linear-gradient(145deg, #121a30, #0f1628);
            border: 1px solid #283454;
            border-radius: 18px;
            padding: 18px;
            box-shadow: 0 8px 30px rgba(0,0,0,.18);
        }

        .metric-label {
            color: #aebbd7;
            text-transform: uppercase;
            letter-spacing: .08em;
            font-size: .68rem;
        }

        .metric-value {
            color: #f5f7ff;
            font-size: 1.7rem;
            font-weight: 800;
            margin-top: 4px;
        }

        .metric-note {
            color: #8f9dbb;
            font-size: .75rem;
            margin-top: 3px;
        }

        .section {
            color: #f1f5ff;
            font-size: 1.15rem;
            font-weight: 750;
            margin: 25px 0 10px;
        }

        /* Alert/info boxes */
        [data-testid="stAlert"] p {
            color: #dbe4ff !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title, subtitle, pills=None):
    pills = pills or []
    tags = "".join(f'<span class="pill">{p}</span>' for p in pills)
    st.markdown(
        f'<div class="hero"><div class="kicker">ACDX • ADAPTIVE CUSTOMER DECISION ENGINE</div>'
        f'<h1>{title}</h1><div class="subtitle">{subtitle}</div>{tags}</div>',
        unsafe_allow_html=True,
    )


def metric_card(label, value, note=""):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def plotly_bar(df, x, y, title, color=None):
    fig = px.bar(
        df,
        x=x,
        y=y,
        color=color if color else None,
        title=title,
        template="plotly_dark",
        text_auto=True,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=55, b=10),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def plotly_donut(labels, values, title):
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=.62,
            textinfo="label+percent",
            marker=dict(line=dict(color="#0b1020", width=2)),
        )
    )
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=55, b=10),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def plotly_radar(categories, values, title):
    fig = go.Figure(
        go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            name="Customer profile",
        )
    )
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        polar=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=30, r=30, t=55, b=20),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
