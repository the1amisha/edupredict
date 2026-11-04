"""EduPredict — Behance-inspired Streamlit dashboard and ML teaching prototype.

The included dataset and any trained model are SYNTHETIC demonstrations.
Do not use outputs to make decisions about real students.
"""
from __future__ import annotations

import calendar
from datetime import datetime, timedelta
from html import escape
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / 'data' / 'demo_students.csv'
MODEL_PATH = ROOT / 'models' / 'risk_model.joblib'
FEATURES = ['attendance', 'assignment', 'internal']

st.set_page_config(page_title='EduPredict · Student risk dashboard', page_icon='🎓', layout='wide', initial_sidebar_state='collapsed')

@st.cache_data
def load_students() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model(model_mtime: float | None):
    if model_mtime is None:
        return None
    return joblib.load(MODEL_PATH)


def rule_band(row) -> str:
    """Rule-based placeholder used only until a model has been trained."""
    score = .35*row['attendance'] + .30*row['assignment'] + .35*row['internal']
    return 'High' if score < 50 else 'Medium' if score < 70 else 'Low'


def label_from_probability(p: float) -> str:
    return 'High' if p >= .65 else 'Medium' if p >= .35 else 'Low'


def with_predictions(df: pd.DataFrame, bundle) -> pd.DataFrame:
    df = df.copy()
    if bundle is None:
        df['risk'] = df.apply(rule_band, axis=1)
        df['fail_probability'] = np.nan
    else:
        model = bundle['model']
        fail_idx = list(model.classes_).index(1)
        df['fail_probability'] = model.predict_proba(df[FEATURES])[:, fail_idx]
        df['risk'] = df['fail_probability'].map(label_from_probability)
    return df


def icon_svg(kind: str, size: int = 34) -> str:
    paths = {
        'students': '<circle cx="10" cy="9" r="3"/><circle cx="23" cy="9" r="3"/><path d="M3 26c0-5 3-8 7-8s7 3 7 8M16 22c1-3 4-4 7-4 4 0 7 3 7 8"/>',
        'assignment': '<rect x="7" y="5" width="19" height="24" rx="3"/><path d="M12 12h10M12 18h10M12 24h6"/><path d="M12 4h9"/>',
        'assessment': '<rect x="5" y="7" width="23" height="21" rx="3"/><path d="M11 14h11M11 20h7"/><path d="M13 4h7v5h-7z"/>',
        'alert': '<path d="M16 4 3 27h26L16 4Z"/><path d="M16 12v7M16 23v.3"/>',
    }
    return f'<svg width="{size}" height="{size}" viewBox="0 0 32 32" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{paths[kind]}</svg>'

HERO_ART = '''<svg viewBox="0 0 280 164" width="100%" height="150" aria-hidden="true">
<ellipse cx="145" cy="146" rx="127" ry="12" fill="#e6decc"/>
<rect x="17" y="94" width="242" height="8" rx="4" fill="#9a824c"/>
<rect x="37" y="102" width="7" height="42" rx="2" fill="#786a50"/><rect x="230" y="102" width="7" height="42" rx="2" fill="#786a50"/>
<rect x="69" y="77" width="75" height="49" rx="5" fill="#575750"/><rect x="75" y="83" width="63" height="36" rx="2" fill="#d6c89f"/><path d="M60 126h93" stroke="#575750" stroke-width="5" stroke-linecap="round"/>
<rect x="198" y="72" width="39" height="24" rx="5" fill="#d1b46f"/><path d="M207 72V58h20v14" fill="none" stroke="#685d3f" stroke-width="4"/>
<circle cx="183" cy="44" r="18" fill="#d7a579"/><path d="M165 39c-7-20 20-30 32-12 3 6 3 15-1 21l-4-11-12-6-15 8Z" fill="#454640"/>
<path d="M159 81c3-20 13-27 25-27s25 9 30 27v17h-59Z" fill="#b18d3d"/>
<path d="M168 68c-16 8-22 19-26 29M197 70l-7 23-35 4" stroke="#d7a579" stroke-width="10" stroke-linecap="round"/>
<path d="M172 96h38v40h-50v-14" fill="#6c705d"/><path d="M209 132h11v14M173 132l-7 14" stroke="#565950" stroke-width="7" stroke-linecap="round"/>
<rect x="21" y="77" width="30" height="17" rx="2" fill="#e3cc84"/><rect x="26" y="68" width="25" height="9" rx="2" fill="#b79551"/><rect x="31" y="60" width="20" height="8" rx="2" fill="#737b63"/>
<path d="M248 37l6-10 6 10M251 49h14" stroke="#b99f62" stroke-width="3" stroke-linecap="round"/>
</svg>'''

CSS = r'''
<style>
:root {--ink:#3f403d;--gold:#9b7c31;--gold2:#c2a65e;--cream:#f7f5f0;--warm:#ece9e2;--line:#dcd7cb;--muted:#8d8b86;}
html, body, [data-testid="stAppViewContainer"], .stApp {background:#ded9d0!important;color:var(--ink);font-family:'Segoe UI',-apple-system,BlinkMacSystemFont,sans-serif;}
#MainMenu,footer,[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stDecoration"],[data-testid="stSidebar"]{display:none!important;}
.block-container{max-width:1470px!important;padding:18px 24px 34px!important;}
.st-key-app_shell{background:#f4f2ed;border:1px solid rgba(255,255,255,.7);border-radius:28px;padding:17px 18px 22px!important;box-shadow:0 18px 60px rgba(50,43,31,.13);}
.st-key-app_shell [data-testid="stHorizontalBlock"] {align-items:stretch;}
.st-key-sidebar{background:#50514e;border-radius:22px;min-height:750px;padding:20px 13px 17px!important;box-shadow:0 6px 12px rgba(0,0,0,.04);}
.st-key-sidebar .stMarkdown p{margin:0;}
.brand-name{font-family:Georgia,serif;color:#faf7ef;font-size:26px;font-weight:700;letter-spacing:-.6px;line-height:1.1;padding:8px 5px 0;}
.brand-sub{color:#d4cfbf;font-size:10px;letter-spacing:2px;padding:7px 6px 20px;}
.nav-section{font-size:10px;letter-spacing:1.7px;font-weight:700;color:#a9aaa6;padding:6px 12px 7px;}
.sidebar-bottom{padding:48px 10px 0;color:#c7c8c3;font-size:12px;line-height:1.7;}
.sidebar-bottom strong{color:#e3d5ad;font-size:11px;letter-spacing:1px;}
.st-key-sidebar [data-testid="stButton"] button{width:100%;border:0!important;box-shadow:none!important;display:flex;justify-content:flex-start;padding:10px 11px;border-radius:10px;background:transparent;color:#eeeae3;font-weight:500;font-size:13px;min-height:40px;margin:0;transition:background .2s;}
.st-key-sidebar [data-testid="stButton"] button:hover{background:#666762;color:white;}
.st-key-sidebar [data-testid="stButton"] button p{font-size:13px;}
.st-key-topbar{background:#f9f7f2;border:1px solid #e1ddd3;border-radius:17px;padding:7px 14px!important;min-height:64px;}
.st-key-topbar .stTextInput{margin-top:3px;}
.st-key-topbar [data-testid="stTextInputRootElement"]{background:#f1efeb;border:1px solid #dedbd5;border-radius:10px;}
.st-key-topbar input{font-size:13px;color:#494941;}
.st-key-topbar [data-testid="stHorizontalBlock"]{align-items:center;}
.topbar-icon{font-size:21px;text-align:center;color:#585a55;padding-top:8px;}
.teacher-pill{display:flex;align-items:center;gap:9px;padding:5px 0;}
.teacher-avatar{background:#d7c39b;display:grid;place-items:center;border-radius:50%;width:34px;height:34px;font-size:19px;flex-shrink:0;}
.teacher-pill strong{font-size:12px;white-space:nowrap;display:block;color:#43443e}.teacher-pill small{font-size:10px;color:#99948b;display:block;}
.st-key-hero{background:#f0ede6;border:1px solid #ddd8ce;border-radius:20px;min-height:169px;padding:20px 21px 12px!important;overflow:hidden;}
.st-key-hero [data-testid="stHorizontalBlock"]{align-items:center;}
.eyebrow{font-size:10px;color:#9d7f39;font-weight:700;letter-spacing:1.6px;margin-bottom:7px;}
.hero-title{font-family:Georgia,serif;font-weight:700;font-size:28px;line-height:1.2;color:#87702e;margin:0 0 8px;}
.hero-sub{color:#7a715d;font-size:12px;line-height:1.6;max-width:480px;margin:0;}
.st-key-hero_action button{background:transparent!important;padding:0!important;margin:1px 0 0;border:0!important;color:#9c813f!important;font-size:12px!important;text-decoration:underline!important;box-shadow:none!important;}
.st-key-hero_art{display:flex;align-items:center;justify-content:flex-end;}
.feature-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:11px 0 13px;}
.feature-card{border:1px solid #ded9cc;border-radius:18px;min-height:149px;background:#f8f6f1;padding:13px 15px 12px;position:relative;overflow:hidden;}
.feature-head{font-family:Georgia,serif;font-size:13px;font-weight:700;color:#997e37;text-align:center;border-bottom:1px solid #e1ded4;padding-bottom:8px;margin-bottom:8px;}
.feature-body{display:flex;align-items:center;justify-content:space-between;gap:7px;}
.feature-value{color:#454540;font-size:29px;font-weight:750;letter-spacing:-1px;line-height:1;}
.feature-note{font-size:10px;color:#8a8982;margin-top:8px;max-width:130px;line-height:1.5;}
.feature-illustration{background:#eadfca;color:#9c7e36;border-radius:50%;width:66px;height:66px;display:grid;place-items:center;flex-shrink:0;}
.feature-card.risky .feature-illustration{background:#f1ddd1;color:#b46e50;}
.chart-card,.data-card,.side-card{background:#faf8f3;border:1px solid #ded9ce;border-radius:18px;overflow:hidden;box-sizing:border-box;}
.chart-card{padding:0 0 12px;min-height:229px;}
.card-title{font-family:Georgia,serif;color:#8f742f;font-weight:700;font-size:14px;text-align:center;padding:11px 10px 10px;border-bottom:1px solid #e5e0d6;}
.card-sub{font-size:10px;color:#a29b8f;}
.donut-row{display:flex;align-items:center;justify-content:center;gap:14px;min-height:158px;padding:10px 7px 2px;}
.donut{height:122px;width:122px;border-radius:50%;position:relative;display:grid;place-items:center;flex-shrink:0;}
.donut:before{content:'';position:absolute;inset:15px;background:#faf8f3;border-radius:50%;}
.donut-center{z-index:1;text-align:center;line-height:1.1;color:#565249;}
.donut-center strong{font-size:25px;display:block;}.donut-center span{font-size:9px;color:#9a968b;}
.legend{font-size:10px;color:#777269;line-height:2;}
.dot{width:9px;height:9px;border-radius:50%;display:inline-block;margin-right:5px;}
.bars{height:161px;display:flex;justify-content:space-around;align-items:flex-end;gap:8px;padding:20px 17px 4px;background:repeating-linear-gradient(to top,transparent,transparent 38px,#ebe7df 39px);margin:9px 10px 0;}
.bar-item{display:flex;flex:1;max-width:50px;min-width:0;flex-direction:column;align-items:center;justify-content:flex-end;height:100%;}
.bar-fill{width:62%;max-width:32px;min-width:10px;background:#b49a5b;border-radius:5px 5px 0 0;position:relative;}
.bar-fill:nth-child(odd){background:#b49a5b;}
.bar-value{font-size:9px;color:#777265;text-align:center;margin-bottom:4px;}
.bar-name{font-size:9px;color:#827c71;margin-top:6px;text-align:center;width:100%;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;}
.data-card{margin-top:12px;padding-bottom:4px;}
.table-heading{display:flex;justify-content:space-between;gap:12px;align-items:center;padding:13px 17px 11px;border-bottom:1px solid #e6e1d7;}
.table-heading strong{color:#8a7030;font-family:Georgia,serif;font-size:15px;}.table-heading small{color:#aaa397;font-size:10px;}
.student-table{width:100%;border-collapse:collapse;font-size:11px;color:#514f48;}
.student-table th{text-align:left;color:#aaa398;font-size:10px;font-weight:600;padding:10px 15px;white-space:nowrap;}
.student-table td{padding:10px 15px;border-top:1px solid #ede9e0;white-space:nowrap;}
.student-table .student-name{font-weight:650;color:#484741;}.student-table small{display:block;font-size:9px;color:#a8a195;}
.risk-chip{font-size:9px;border-radius:8px;padding:4px 8px;font-weight:650;}
.risk-High{color:#9d513e;background:#f6dfd6;}.risk-Medium{color:#9b7630;background:#f4ecd5;}.risk-Low{color:#65734c;background:#e3e9d9;}
.mini-track{width:72px;height:6px;border-radius:10px;background:#e8e3d8;overflow:hidden;display:inline-block;vertical-align:middle;margin-left:5px;}.mini-fill{height:100%;border-radius:10px;background:#ae9253;display:block;}
.side-card{margin-bottom:12px;padding:0 0 12px;}
.side-title{font-family:Georgia,serif;font-weight:700;color:#917633;font-size:13px;padding:10px 10px;text-align:center;border-bottom:1px solid #e6e1d7;}
.side-meta{text-align:right;color:#aaa08a;font-size:9px;margin:10px 13px 3px;}
.calendar-head{display:flex;justify-content:space-between;align-items:center;padding:3px 14px 7px;font-size:11px;font-weight:700;color:#6c6250;}
.calendar-grid{display:grid;grid-template-columns:repeat(7,1fr);gap:4px;padding:1px 13px;}
.calendar-weekday{text-align:center;color:#aba49a;font-size:9px;font-weight:700;padding-bottom:6px;}
.calendar-day{text-align:center;font-size:10px;color:#69665f;height:24px;display:grid;place-items:center;border-radius:7px;}
.calendar-day.muted{opacity:.28}.calendar-day.today{background:#a88d4a;color:white;font-weight:800}.calendar-day.event{outline:1px solid #bda15b;outline-offset:-2px;color:#917431;font-weight:700;}
.side-list{padding:7px 12px 0;}
.side-event{padding:10px 7px;border-bottom:1px solid #ebe6db;display:flex;gap:10px;align-items:center;}
.side-event:last-child{border-bottom:0;}
.side-event .date-box{min-width:34px;text-align:center;border-radius:8px;background:#f1e6cc;padding:6px 3px;color:#957936;font-weight:700;font-size:10px;}
.side-event strong{display:block;font-size:10px;color:#5f5b52;}.side-event small{display:block;color:#999187;font-size:9px;margin-top:3px;}
.alert-box{padding:12px 15px;display:flex;gap:10px;align-items:flex-start;border-bottom:1px solid #ece8df;}
.alert-box:last-child{border:0}.alert-circle{width:25px;height:25px;border-radius:50%;display:grid;place-items:center;background:#f2e3d8;color:#a86144;font-size:12px;flex-shrink:0;}
.alert-box strong{display:block;font-size:11px;color:#56524b;}.alert-box small{font-size:10px;line-height:1.5;color:#99948b;display:block;margin-top:2px;}
.hint-strip{font-size:10px;color:#999184;margin:10px 1px 0;line-height:1.5;}
.page-heading{font-family:Georgia,serif;font-size:27px;color:#8b722f;margin:7px 0 3px;}.page-description{font-size:12px;color:#898278;margin-bottom:18px;}
.st-key-page_panel{background:#faf8f3;border:1px solid #ded9ce;border-radius:19px;padding:18px 19px!important;}
.st-key-page_panel h3{font-family:Georgia,serif;color:#8e7536;font-size:19px;}
.st-key-page_panel [data-testid="stDataFrame"]{border-radius:10px;overflow:hidden;}
.stButton button[kind="primary"]{background:#a28746;color:white;border-color:#a28746;}
.stButton button:hover{color:#786028;border-color:#aa8c4b;}
[data-testid="stForm"]{border:1px solid #ded9ce!important;border-radius:15px!important;background:#f7f5ef!important;}
.st-key-app_shell p{line-height:1.5;}
@media(max-width:900px){.block-container{padding:10px 8px 24px!important}.st-key-app_shell{padding:8px!important}.st-key-sidebar{min-height:500px}.feature-grid{grid-template-columns:1fr}.hero-title{font-size:21px}.student-table th,.student-table td{padding:8px 4px;font-size:9px}.mini-track{display:none}.donut-row{gap:6px}.donut{height:89px;width:89px}.donut:before{inset:12px}.sidebar-bottom{padding-top:12px;}}
</style>
'''

st.markdown(CSS, unsafe_allow_html=True)

NAV = [
    ('Overview', ':material/dashboard:'),
    ('Students', ':material/groups:'),
    ('Predict', ':material/insights:'),
    ('Analytics', ':material/bar_chart:'),
    ('Reports', ':material/description:'),
    ('Settings', ':material/settings:'),
]
if 'page' not in st.session_state:
    st.session_state.page = 'Overview'
active = st.session_state.page
st.markdown(f'<style>.st-key-nav_{active} button{{background:linear-gradient(105deg,#90722f,#b39751)!important;color:white!important;font-weight:650!important;}}</style>', unsafe_allow_html=True)

raw = load_students()
bundle = load_model(MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else None)
students = with_predictions(raw, bundle)
mode = 'ML demo' if bundle else 'Rule-based demo'
risk_counts = students['risk'].value_counts()
high = int(risk_counts.get('High', 0))
medium = int(risk_counts.get('Medium', 0))
low = int(risk_counts.get('Low', 0))
now = datetime.now(ZoneInfo('Asia/Kolkata'))
today = now.date()

def student_table(frame: pd.DataFrame, limit: int = 6) -> str:
    lines = []
    for _, row in frame.head(limit).iterrows():
        name = escape(str(row['name']))
        rid = escape(str(row['student_id']))
        risk = row['risk']
        att = int(row['attendance'])
        lines.append(f'''<tr><td class="student-name">{name}<small>{rid}</small></td>
        <td>{att}% <span class="mini-track"><span class="mini-fill" style="width:{att}%"></span></span></td>
        <td>{int(row['assignment'])}%</td><td>{int(row['internal'])}%</td>
        <td><span class="risk-chip risk-{risk}">{risk}</span></td></tr>''')
    return f'''<div class="data-card"><div class="table-heading"><strong>Student risk overview</strong><small>{len(frame)} records</small></div>
    <div style="overflow-x:auto"><table class="student-table"><thead><tr><th>STUDENT</th><th>ATTENDANCE</th><th>ASSIGNMENT</th><th>INTERNAL</th><th>RISK</th></tr></thead>
    <tbody>{''.join(lines) if lines else '<tr><td colspan="5">No matching students.</td></tr>'}</tbody></table></div></div>'''


def metric_cards() -> str:
    return f'''<div class="feature-grid">
     <div class="feature-card"><div class="feature-head">Students Monitored</div><div class="feature-body"><div><div class="feature-value">{len(students)}</div><div class="feature-note">Across the demo class</div></div><div class="feature-illustration">{icon_svg('students', 39)}</div></div></div>
     <div class="feature-card risky"><div class="feature-head">Students At Risk</div><div class="feature-body"><div><div class="feature-value">{high}</div><div class="feature-note">High-priority review</div></div><div class="feature-illustration">{icon_svg('alert', 36)}</div></div></div>
     <div class="feature-card"><div class="feature-head">Average Attendance</div><div class="feature-body"><div><div class="feature-value">{students['attendance'].mean():.0f}%</div><div class="feature-note">Based on available records</div></div><div class="feature-illustration">{icon_svg('assessment', 38)}</div></div></div>
     </div>'''


def donut_chart() -> str:
    n = max(1, len(students))
    pct_h = 100 * high/n
    pct_m = 100 * medium/n
    return f'''<div class="chart-card"><div class="card-title">Performance outlook</div><div class="donut-row">
      <div class="donut" style="background:conic-gradient(#b96d55 0 {pct_h:.2f}%,#c4a15d {pct_h:.2f}% {pct_h+pct_m:.2f}%,#8f9a74 {pct_h+pct_m:.2f}% 100%)"><div class="donut-center"><strong>{len(students)}</strong><span>STUDENTS</span></div></div>
      <div class="legend"><div><span class="dot" style="background:#b96d55"></span>High&nbsp;&nbsp; {high}</div>
      <div><span class="dot" style="background:#c4a15d"></span>Medium&nbsp;&nbsp; {medium}</div>
      <div><span class="dot" style="background:#8f9a74"></span>Low&nbsp;&nbsp; {low}</div></div></div></div>'''


def bar_chart(frame: pd.DataFrame) -> str:
    bars = []
    for _, row in frame.head(7).iterrows():
        v = int(row['attendance'])
        short_name = escape(str(row['name']).split()[0])
        bars.append(f'<div class="bar-item"><div class="bar-value">{v}%</div><div class="bar-fill" style="height:{max(6,int(1.12*v))}px"></div><div class="bar-name">{short_name}</div></div>')
    return f'''<div class="chart-card"><div class="card-title">Attendance by student <span class="card-sub">· varied sample of seven</span></div>
      <div class="bars">{''.join(bars)}</div></div>'''


def right_panel() -> None:
    cal = calendar.Calendar(firstweekday=6)
    weeks = cal.monthdayscalendar(today.year, today.month)
    milestones = [today + timedelta(days=d) for d in (3, 6, 9)]
    events = {d.day for d in milestones if (d.month, d.year) == (today.month, today.year)}
    header = ''.join(f'<div class="calendar-weekday">{d}</div>' for d in ['Su','Mo','Tu','We','Th','Fr','Sa'])
    days = ''
    for week in weeks:
        for day in week:
            classes = ' muted' if day == 0 else ' today' if day == today.day else ' event' if day in events else ''
            days += f'<div class="calendar-day{classes}">{day if day else ""}</div>'
    st.markdown(f'''<div class="side-card"><div class="side-title">Academic Calendar</div>
    <div class="side-meta">Sample schedule</div><div class="calendar-head"><span>{today.strftime('%B %Y')}</span><span style="font-weight:400;color:#9a927c">Demo</span></div>
    <div class="calendar-grid">{header}{days}</div></div>''', unsafe_allow_html=True)
    events_html = ''
    for date, title, sub in zip(milestones,['Attendance review','Assignment check','Practice assessment'],['Check missing classes','Review submissions','Sample class exercise']):
        events_html += f'<div class="side-event"><div class="date-box">{date.strftime("%d")}<br>{date.strftime("%b").upper()}</div><div><strong>{title}</strong><small>{sub}</small></div></div>'
    st.markdown(f'<div class="side-card"><div class="side-title">Upcoming Reviews</div><div class="side-list">{events_html}</div></div>',unsafe_allow_html=True)
    missed = int((students['attendance'] < 60).sum())
    st.markdown(f'''<div class="side-card"><div class="side-title">Student Alerts</div>
     <div class="alert-box"><div class="alert-circle">!</div><div><strong>{high} high-risk students</strong><small>Prioritise academic support</small></div></div>
     <div class="alert-box"><div class="alert-circle">↘</div><div><strong>{missed} below 60% attendance</strong><small>Consider attendance follow-up</small></div></div></div>''', unsafe_allow_html=True)


def student_view(df: pd.DataFrame):
    st.markdown('<div class="page-heading">Students</div><div class="page-description">Explore the demo class, search records and review their risk indicators.</div>', unsafe_allow_html=True)
    with st.container(key='page_panel'):
        left, right = st.columns([2,1])
        with left:
            st.markdown('### Class records')
        with right:
            selected = st.selectbox('Risk category', ['All','High','Medium','Low'], label_visibility='collapsed', key='risk_filter')
        filtered = df if selected == 'All' else df[df['risk'] == selected]
        st.dataframe(filtered[['student_id','name','attendance','assignment','internal','risk']], hide_index=True, use_container_width=True, height=480,
            column_config={'student_id':'Student ID','name':'Student','attendance':'Attendance (%)','assignment':'Assignments (%)','internal':'Internal (%)','risk':'Risk'} )
        st.caption(f'{len(filtered)} synthetic records · {mode}')


def prediction_view():
    st.markdown('<div class="page-heading">Individual Prediction</div><div class="page-description">Enter academic indicators to see an early-warning demonstration.</div>', unsafe_allow_html=True)
    with st.container(key='page_panel'):
        st.markdown('### Check student risk')
        st.caption('All three values are percentages from 0 to 100.')
        with st.form('prediction_form'):
            a = st.slider('Attendance (%)', 0, 100, 75)
            b = st.slider('Assignment score (%)', 0, 100, 65)
            c = st.slider('Internal assessment (%)', 0, 100, 70)
            submitted = st.form_submit_button('Assess student', type='primary', use_container_width=True)
        if submitted:
            if bundle:
                fail_idx = list(bundle['model'].classes_).index(1)
                point = pd.DataFrame([{'attendance':a, 'assignment':b, 'internal':c}])
                p = float(bundle['model'].predict_proba(point)[:,fail_idx][0])
                risk = label_from_probability(p)
                message = f'{risk} risk · estimated failure probability {p:.0%} (synthetic-data demo)'
            else:
                risk = rule_band({'attendance':a,'assignment':b,'internal':c})
                message = f'{risk} risk · rule-based demonstration, not an ML prediction'
            if risk == 'High':
                st.error(message)
            elif risk == 'Medium':
                st.warning(message)
            else:
                st.success(message)
            tips = []
            if a < 75: tips.append('Offer attendance support and check barriers to participation.')
            if b < 60: tips.append('Review assignment completion and offer targeted practice.')
            if c < 60: tips.append('Plan revision for weak internal-assessment topics.')
            if not tips: tips.append('Continue regular academic follow-up.')
            st.markdown('**Suggested academic support**')
            for tip in tips: st.write('• ' + tip)
        st.info('This is a teaching prototype. A model trained on synthetic data is not validated for real educational decisions.')


def analytics_view():
    st.markdown('<div class="page-heading">Model Analytics</div><div class="page-description">Compare the demonstration screening and ML evaluation.</div>', unsafe_allow_html=True)
    with st.container(key='page_panel'):
        if bundle is None:
            st.markdown('### Model not trained yet')
            st.write('The dashboard is currently using a simple rule-based screening demonstration. To train two ML baselines, run:')
            st.code('python train_model.py', language='bash')
            st.write('Then refresh this page. We will display held-out evaluation on synthetic data.')
        else:
            m = bundle['metrics']
            st.markdown(f"### Selected model: {m['model']}")
            c1,c2,c3 = st.columns(3)
            c1.metric('Test accuracy', f"{m['accuracy']:.1%}")
            c2.metric('Failure recall', f"{m['fail_recall']:.1%}")
            c3.metric('Failure F1', f"{m['fail_f1']:.1%}")
            st.write('Confusion matrix (rows: actual, columns: predicted; Pass then Fail):')
            st.dataframe(pd.DataFrame(m['confusion_matrix'], index=['Actual Pass','Actual Fail'], columns=['Pred Pass','Pred Fail']))
            st.warning('These metrics were measured on a synthetic holdout dataset. They do not establish real-world accuracy or fairness.')


def reports_view():
    st.markdown('<div class="page-heading">Reports</div><div class="page-description">Export demo records and risk-screening results.</div>', unsafe_allow_html=True)
    with st.container(key='page_panel'):
        st.markdown('### Export student screening')
        columns = ['student_id','name','attendance','assignment','internal','risk']
        if bundle is not None: columns.append('fail_probability')
        out = students[columns].copy()
        st.download_button('Download CSV report', out.to_csv(index=False).encode('utf-8'), file_name='edupredict_demo_risk_report.csv', mime='text/csv', type='primary')
        st.caption(f'{len(students)} synthetic records · {mode}; do not treat these as actual student outcomes.')
        st.dataframe(out.head(12), use_container_width=True, hide_index=True)


def settings_view():
    st.markdown('<div class="page-heading">About EduPredict</div><div class="page-description">Project information and usage guidance.</div>', unsafe_allow_html=True)
    with st.container(key='page_panel'):
        st.markdown('### Smart student early-warning system')
        st.write('**Inputs:** attendance, assignment score and internal assessment score.')
        st.write(f'**Current prediction mode:** {mode}.')
        st.write('**Purpose:** demonstrate how teachers might identify academic support needs before final examinations.')
        st.write('**Privacy:** this starter contains synthetic student records only. Never upload identifiable student data to a public repository.')
        st.write('**Next step:** develop and validate your model with appropriately permitted, real labelled data before real-world use.')

with st.container(key='app_shell'):
    sidebar_col, workspace_col = st.columns([0.19, 0.81], gap='medium')
    with sidebar_col:
        with st.container(key='sidebar'):
            st.markdown('<div class="brand-name">EduPredict<span style="color:#c9b26c">.</span></div><div class="brand-sub">SMART EDUCATION</div>', unsafe_allow_html=True)
            st.markdown('<div class="nav-section">MENU</div>', unsafe_allow_html=True)
            for page, icon in NAV:
                if st.button(page, icon=icon, key=f'nav_{page}', use_container_width=True):
                    st.session_state.page = page
                    st.rerun()
            st.markdown('<div class="sidebar-bottom"><strong>● DEMO ENVIRONMENT</strong><div>Academic risk insights<br>for learning and research.</div></div>',unsafe_allow_html=True)
    with workspace_col:
        with st.container(key='topbar'):
            search_col, mail_col, notification_col, teacher_col = st.columns([0.58, 0.10, 0.10, 0.22], gap='small')
            with search_col:
                query = st.text_input('Search students', placeholder='⌕  Search students...', label_visibility='collapsed', key='top_search')
            with mail_col:
                st.markdown('<div class="topbar-icon" title="Messages">✉</div>', unsafe_allow_html=True)
            with notification_col:
                st.markdown(f'<div class="topbar-icon" title="High-risk alerts">♧ <sup style="color:#ab624e;font-size:9px">{high}</sup></div>',unsafe_allow_html=True)
            with teacher_col:
                st.markdown('<div class="teacher-pill"><span class="teacher-avatar">👩🏻‍🏫</span><div><strong>Class Teacher</strong><small>Educator dashboard</small></div></div>',unsafe_allow_html=True)
        st.markdown('<div style="height:11px"></div>',unsafe_allow_html=True)
        main_col, right_col = st.columns([0.755, 0.245], gap='small')
        with main_col:
            if active == 'Overview':
                with st.container(key='hero'):
                    hero_copy, hero_art = st.columns([0.69, 0.31], gap='small')
                    with hero_copy:
                        st.markdown('<div class="eyebrow">YOUR CLASSROOM AT A GLANCE</div><h1 class="hero-title">Hello, Teacher!</h1><p class="hero-sub">Stay ahead of learning gaps. Spot students who may need extra support before their final examination.</p>', unsafe_allow_html=True)
                        with st.container(key='hero_action'):
                            if st.button('Review student risk  →', key='review_action'):
                                st.session_state.page = 'Students'
                                st.rerun()
                    with hero_art:
                        st.markdown('<div class="hero-art">'+HERO_ART+'</div>',unsafe_allow_html=True)
                st.markdown(metric_cards(),unsafe_allow_html=True)
                chart_l, chart_r = st.columns([0.4, 0.6], gap='small')
                with chart_l: st.markdown(donut_chart(),unsafe_allow_html=True)
                with chart_r: st.markdown(bar_chart(students.sort_values('attendance').iloc[np.linspace(0, len(students)-1, 7, dtype=int)]), unsafe_allow_html=True)
                filtered = students[students['name'].str.contains(query, case=False, na=False)] if query else students
                preview = filtered.assign(_order=filtered['risk'].map({'High':0,'Medium':1,'Low':2})).sort_values(['_order','attendance']).drop(columns='_order')
                st.markdown(student_table(preview, limit=6),unsafe_allow_html=True)
            elif active == 'Students':
                filtered = students[students['name'].str.contains(query, case=False, na=False)] if query else students
                student_view(filtered)
            elif active == 'Predict': prediction_view()
            elif active == 'Analytics': analytics_view()
            elif active == 'Reports': reports_view()
            else: settings_view()
            st.markdown(f'<div class="hint-strip">{mode} · Synthetic student data only. Screening is meant for academic assistance, not punitive decisions.</div>',unsafe_allow_html=True)
        with right_col:
            right_panel()
