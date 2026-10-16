"""ml_components.py - styling, KPI cards and Plotly defaults."""
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

NAVY, SLATE, TEAL, BLUE = "#0B2545", "#5B6B82", "#13A89E", "#2F6FDE"
AMBER, RED, GREEN, GRID = "#C77D0A", "#C0392B", "#1E8E5A", "#E3E8EF"
SEVERITY_COLORS = {"Low": "#9FB3C8", "Medium": BLUE, "High": AMBER, "Critical": RED}

pio.templates["civic"] = go.layout.Template(layout=dict(
    font=dict(family="Inter, Segoe UI, Arial, sans-serif", size=13, color="#1B2A41"),
    paper_bgcolor="white", plot_bgcolor="white",
    colorway=[BLUE, TEAL, NAVY, AMBER, "#8E6CBF", RED],
    xaxis=dict(gridcolor=GRID, linecolor="#C5CEDA", zerolinecolor=GRID),
    yaxis=dict(gridcolor=GRID, linecolor="#C5CEDA", zerolinecolor=GRID),
    margin=dict(l=48, r=20, t=48, b=44), title=dict(font=dict(size=15, color=NAVY), x=0.01),
    legend=dict(orientation="h", y=-0.2),
))
pio.templates.default = "civic"

CSS = f"""
<style>
.stApp {{ background:#F4F6FA; color:#1B2A41; }}
header[data-testid="stHeader"] {{ background:#F4F6FA; }}
.block-container {{ padding-top:3.2rem; padding-bottom:3rem; max-width:1280px; }}
section[data-testid="stSidebar"] {{ background:{NAVY}; }}
section[data-testid="stSidebar"] * {{ color:#E6ECF5 !important; }}
section[data-testid="stSidebar"] hr {{ border-color:#27406B; }}
h1,h2,h3,h4 {{ color:{NAVY}; letter-spacing:-0.01em; }}
p, li, label, .stMarkdown, [data-testid="stCaptionContainer"] {{ color:#1B2A41; }}
[data-testid="stCaptionContainer"] {{ color:{SLATE}; }}
.page-title {{ font-size:1.55rem; font-weight:700; color:{NAVY}; margin:0 0 .15rem 0; }}
.page-sub {{ color:{SLATE}; font-size:.95rem; margin-bottom:1.1rem; }}
.kpi {{ background:#fff; border:1px solid {GRID}; border-radius:6px; padding:.8rem 1rem; box-shadow:0 1px 2px rgba(11,37,69,.05); }}
.kpi .l {{ font-size:.72rem; text-transform:uppercase; letter-spacing:.06em; color:{SLATE}; font-weight:600; }}
.kpi .v {{ font-size:1.6rem; font-weight:700; color:{NAVY}; line-height:1.25; }}
.kpi .s {{ font-size:.78rem; color:{SLATE}; }}
.panel {{ background:#fff; border:1px solid {GRID}; border-radius:6px; padding:1rem 1.2rem; box-shadow:0 1px 2px rgba(11,37,69,.05); }}
.note {{ border-left:4px solid {SLATE}; background:#fff; border-radius:4px; padding:.7rem 1rem; margin:.6rem 0; color:#1B2A41; }}
.note.good {{ border-color:{GREEN}; }} .note.bad {{ border-color:{RED}; }} .note.mixed, .note.warn {{ border-color:{AMBER}; }}
.note b {{ color:{NAVY}; }}
.result {{ background:#fff; border:1px solid {GRID}; border-top:3px solid {TEAL}; border-radius:6px; padding:1.2rem 1.4rem; }}
.result .big {{ font-size:2.6rem; font-weight:700; color:{NAVY}; line-height:1.1; }}
div[data-testid="stForm"] {{ background:#fff; border:1px solid {GRID}; border-radius:6px; }}
.stButton > button, div[data-testid="stFormSubmitButton"] > button {{ border-radius:4px; font-weight:600; }}
button[data-testid="stBaseButton-primaryFormSubmit"], button[data-testid="stBaseButton-primary"] {{ background:{TEAL} !important; border-color:{TEAL} !important; }}
button[data-testid="stBaseButton-primaryFormSubmit"] *, button[data-testid="stBaseButton-primary"] * {{ color:#fff !important; }}
button[data-testid="stBaseButton-secondaryFormSubmit"], button[data-testid="stBaseButton-secondary"] {{ background:#fff !important; border-color:#B8C4D3 !important; }}
button[data-testid="stBaseButton-secondaryFormSubmit"] *, button[data-testid="stBaseButton-secondary"] * {{ color:{NAVY} !important; }}
.stApp [role="radiogroup"] label div[data-baseweb="radio"] > div:first-child {{ border-color:{TEAL} !important; }}
.stApp input, .stApp [data-baseweb="select"] > div {{ background:#fff !important; color:#1B2A41 !important; }}
[data-testid="stDataFrame"] {{ border:1px solid {GRID}; border-radius:6px; }}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = ""):
    st.markdown(f'<div class="page-title">{title}</div><div class="page-sub">{subtitle}</div>', unsafe_allow_html=True)


def kpi(label: str, value: str, sub: str = ""):
    st.markdown(f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div>'
                f'<div class="s">{sub or "&nbsp;"}</div></div>', unsafe_allow_html=True)


def kpi_row(items):
    cols = st.columns(len(items))
    for col, it in zip(cols, items):
        with col:
            kpi(*it)


def note(html: str, kind: str = ""):
    st.markdown(f'<div class="note {kind}">{html}</div>', unsafe_allow_html=True)


def plot(fig, height=360):
    fig.update_layout(height=height)
    st.plotly_chart(fig, width="stretch", theme=None)
