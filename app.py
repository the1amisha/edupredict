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
from faculty_auth import require_faculty_login, logout_faculty

from student_profile import subject_profile_rows, academic_support, academic_indicators, indicator_review_summary

ROOT = Path(__file__).resolve().parent
PRIVATE_DATA = ROOT / 'data' / 'private'
USE_LOCAL_ROSTER = all((PRIVATE_DATA / file).exists() for file in ('mca_students.csv','mca_subject_records.csv'))
DATA_PATH = (PRIVATE_DATA if USE_LOCAL_ROSTER else ROOT / 'data') / 'mca_students.csv'
SUBJECTS_PATH = ROOT / 'data' / 'mca_subjects.csv'
SUBJECT_RECORDS_PATH = (PRIVATE_DATA if USE_LOCAL_ROSTER else ROOT / 'data') / 'mca_subject_records.csv'
IDENTITY_NOTE = ('Names and generated SBU IDs exactly as supplied in the class directory; attendance and marks are simulated.'
                 if USE_LOCAL_ROSTER else 'Fictional identities and simulated academic data.')
MODEL_PATH = ROOT / 'models' / 'risk_model.joblib'
FEATURES = ['attendance', 'assignment', 'internal']

st.set_page_config(page_title='EduPredict · Student risk dashboard', page_icon='🎓', layout='wide', initial_sidebar_state='collapsed')

# Authentication gate: no student data or ML code runs until signed in.
faculty_name = require_faculty_login()

@st.cache_data
def load_students() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)

@st.cache_data
def load_subjects() -> pd.DataFrame:
    return pd.read_csv(SUBJECTS_PATH)


@st.cache_data
def load_subject_records() -> pd.DataFrame:
    df = pd.read_csv(SUBJECT_RECORDS_PATH)
    for col in ['classes_attended','classes_conducted','classes_needed_to_75','assignment','internal']:
        df[col] = df[col].astype('Int64')
    return df


@st.cache_resource
def load_model(model_mtime: float | None):
    if model_mtime is None:
        # Create the synthetic ML demonstration model
        # when running on a fresh cloud deployment.
        from train_model import main as train_demo_model
        train_demo_model()

    return joblib.load(MODEL_PATH)


def rule_band(row) -> str:
    """Rule-based placeholder used only until a model has been trained."""
    score = .35*row['attendance'] + .30*row['assignment'] + .35*row['internal']
    return 'High' if score < 50 else 'Medium' if score < 70 else 'Low'


def label_from_probability(p: float, decision_threshold: float = 0.50) -> str:
    """Make High risk match the model's predicted Fail class.

    Medium marks students approaching the failure decision threshold,
    not a second classification learned by the ML model.
    """
    caution_threshold = max(0.0, decision_threshold - 0.15)
    if p >= decision_threshold:
        return 'High'
    if p >= caution_threshold:
        return 'Medium'
    return 'Low'


def with_predictions(df: pd.DataFrame, bundle) -> pd.DataFrame:
    df = df.copy()
    if bundle is None:
        df['risk'] = df.apply(rule_band, axis=1)
        df['fail_probability'] = np.nan
    else:
        model = bundle['model']
        fail_idx = list(model.classes_).index(1)
        df['fail_probability'] = model.predict_proba(df[FEATURES])[:, fail_idx]
        df['risk'] = df['fail_probability'].map(lambda p: label_from_probability(p, bundle.get('decision_threshold', 0.50)))
    return df


def icon_svg(kind: str, size: int = 34) -> str:
    paths = {
        'students': '<circle cx="10" cy="9" r="3"/><circle cx="23" cy="9" r="3"/><path d="M3 26c0-5 3-8 7-8s7 3 7 8M16 22c1-3 4-4 7-4 4 0 7 3 7 8"/>',
        'assignment': '<rect x="7" y="5" width="19" height="24" rx="3"/><path d="M12 12h10M12 18h10M12 24h6"/><path d="M12 4h9"/>',
        'assessment': '<rect x="5" y="7" width="23" height="21" rx="3"/><path d="M11 14h11M11 20h7"/><path d="M13 4h7v5h-7z"/>',
        'alert': '<path d="M16 4 3 27h26L16 4Z"/><path d="M16 12v7M16 23v.3"/>',
    }
    return f'<svg width="{size}" height="{size}" viewBox="0 0 32 32" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{paths[kind]}</svg>'

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

/* Student Profiles page: match the charcoal/cream/gold dashboard */
.profile-banner {border:1px solid #e1dbcf;border-radius:16px;background:#f2efe8;padding:16px 19px;margin:7px 0 17px;}
.profile-banner strong{font-family:Georgia,serif;font-size:21px;color:#867039;display:block;}
.profile-banner span{color:#898278;font-size:12px;display:block;margin-top:5px;}
.profile-section-title{font-family:Georgia,serif;color:#987b37;font-size:19px;font-weight:700;padding:8px 0;}
.profile-smallprint{color:#898278;font-size:12px;line-height:1.7;margin:6px 0 14px;}

/* Academic signals: descriptive, NOT ML feature attribution */
.indicator-row{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:5px 0;}
.indicator-chip{display:inline-flex;align-items:center;gap:5px;background:#e8ecdf;color:#516243;border:1px solid #d3dcc6;border-radius:9px;padding:5px 8px;font-size:10px;font-weight:650;white-space:nowrap;}
.indicator-chip.indicator-watch{background:#f6e4d9;color:#a05c41;border-color:#efd3c6;}
.indicator-review{background:#f5f1e8;border:1px solid #e0d7c3;padding:12px 15px;border-radius:14px;margin:12px 0;}
.indicator-review strong{font-family:Georgia,serif;font-size:15px;color:#947633;display:block;margin-bottom:8px;}
.indicator-review .indicator-caption{font-size:11px;color:#82796b;line-height:1.55;margin-top:6px;}
.student-table .indicator-chip{padding:4px 6px;font-size:9px;}
/* First-time visitor guidance */
.guide-head{font-family:Georgia,serif;color:#8c7134;font-size:19px;font-weight:700;margin:4px 0 3px;}
.guide-sub{font-size:12px;color:#847e72;line-height:1.5;margin-bottom:10px;}
.st-key-step_find,.st-key-step_subject,.st-key-step_predict{
 background:#faf8f3;border:1px solid #ded9ce;border-radius:16px;padding:13px 14px 11px!important;
 min-height:145px;
}
.guide-number{background:#e8dcc1;color:#816527;border-radius:50%;width:27px;height:27px;
 display:inline-grid;place-items:center;font-size:12px;font-weight:750;margin-bottom:8px;}
.guide-title{font-family:Georgia,serif;font-weight:700;color:#685737;font-size:14px;margin-bottom:4px;}
.guide-desc{font-size:11px;color:#888177;line-height:1.5;min-height:35px;}
.st-key-step_find button,.st-key-step_subject button,.st-key-step_predict button{
 width:100%;background:#a58a49!important;border-color:#a58a49!important;color:white!important;
 border-radius:10px!important;font-size:11px!important;min-height:36px!important;
}
.guide-distinction{margin:12px 0 14px;border-left:3px solid #aa8f50;background:#f1eee5;
 padding:10px 13px;border-radius:0 12px 12px 0;color:#736a58;font-size:12px;line-height:1.55;}
.task-hint{border:1px solid #e2d9c4;border-radius:12px;background:#f4efe3;
 padding:12px 15px;color:#77674c;font-size:12px;line-height:1.6;margin:8px 0 16px;}
.task-hint strong{color:#83692e;}
.topbar-context{font-size:13px;font-weight:650;color:#71623f;padding:10px 3px;}
.topbar-context small{display:block;font-size:10px;font-weight:400;color:#928b7c;margin-top:2px;}
@media(max-width:900px){.st-key-step_find,.st-key-step_subject,.st-key-step_predict{min-height:115px}}

/* EDUPREDICT-UI-POLISH-2026-11 */
/* Let the application occupy the entire available page, rather than a floating card. */
html, body, .stApp, [data-testid="stAppViewContainer"] {
    background: #f4f2ed !important;
}
.block-container, [data-testid="stAppViewBlockContainer"] {
    max-width: 100% !important;
    width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
}
.st-key-app_shell {
    width: 100% !important;
    min-height: 100dvh !important;
    padding: 8px 12px 18px !important;
    margin: 0 !important;
    border: 0 !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    background: #f4f2ed !important;
}
/* A charcoal navigation panel that continues to the bottom of the viewport.
   Keep it visible during scrolling; its own contents remain scrollable on small screens. */
.st-key-sidebar {
    position: sticky !important;
    top: 0 !important;
    height: 100dvh !important;
    min-height: 0 !important;
    max-height: 100dvh !important;
    overflow-y: auto !important;
    overscroll-behavior: contain;
    border-radius: 14px !important;
    padding: 20px 13px 16px !important;
}
.st-key-sidebar::-webkit-scrollbar { width: 5px; }
.st-key-sidebar::-webkit-scrollbar-thumb {
    background: #7b7b72;
    border-radius: 10px;
}
.sidebar-bottom {
    margin-top: 14px !important;
    padding: 14px 8px 6px !important;
    border-top: 1px solid #6b6b64;
}
/* A compact academic-performance illustration, rather than a distorted character. */
.st-key-hero {
    padding: 17px 21px 13px !important;
    min-height: 178px !important;
}
.st-key-hero [data-testid="stImage"] {
    width: 100% !important;
    max-width: 238px !important;
    margin: 0 0 0 auto !important;
}
.st-key-hero [data-testid="stImage"] img {
    width: 100% !important;
    height: 174px !important;
    object-fit: contain !important;
}
.hero-fallback {
    min-height: 140px;
    display: grid;
    place-items: center;
    color: #8f793b;
    font-family: Georgia, serif;
    font-size: 17px;
}
@media (max-width: 900px) {
    .st-key-app_shell {padding: 5px 6px 12px !important;}
    .st-key-sidebar {padding: 12px 8px !important;}
    .st-key-hero [data-testid="stImage"] img {height: 115px !important;}
    .st-key-hero [data-testid="stImage"] {max-width: 155px !important;}
}


/* EDUPREDICT-SIDEBAR-RAIL-2026-11-V2 */
/* IMPORTANT: The sticky element must be the sidebar COLUMN, not an inner
   container. Otherwise the inner container sticks only within its short
   wrapper, leaving the empty strip shown while scrolling down. */
@media (min-width: 901px) {
  div[data-testid="stColumn"]:has(.st-key-sidebar) {
    position: sticky !important;
    position: -webkit-sticky !important;
    top: 0 !important;
    align-self: flex-start !important;
    height: 100vh !important;
    height: 100dvh !important;
    min-height: 0 !important;
    max-height: 100dvh !important;
    overflow: visible !important;
    z-index: 40 !important;
  }
  .st-key-sidebar {
    /* The column now owns sticky positioning. */
    position: relative !important;
    top: auto !important;
    height: 100vh !important;
    height: 100dvh !important;
    min-height: 0 !important;
    max-height: 100dvh !important;
    box-sizing: border-box !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    border-radius: 0 14px 14px 0 !important;
    margin: 0 !important;
  }
  /* Keep the footer at the bottom, instead of stopping after the last link.
     Use a normal flow footer on shorter screens so nothing overlaps. */
  .st-key-sidebar .sidebar-bottom {
    position: absolute !important;
    bottom: 18px !important;
    left: 12px !important;
    right: 12px !important;
    margin: 0 !important;
    padding: 11px 7px 0 !important;
    border-top: 1px solid #77776f !important;
  }
}
@media (min-width: 901px) and (max-height: 800px) {
  .st-key-sidebar .sidebar-bottom {
    position: static !important;
    margin-top: 18px !important;
    padding-top: 13px !important;
  }
}
@media (max-width: 900px) {
  /* Small screens: scrolling as normal is less intrusive than a tall sticky rail. */
  div[data-testid="stColumn"]:has(.st-key-sidebar) {
    position: relative !important;
    top: auto !important;
    height: auto !important;
    max-height: none !important;
    overflow: visible !important;
  }
  .st-key-sidebar {
    position: relative !important;
    top: auto !important;
    height: auto !important;
    min-height: 0 !important;
    max-height: none !important;
    overflow: visible !important;
  }
  .st-key-sidebar .sidebar-bottom {
    position: static !important;
    margin-top: 18px !important;
  }
}


/* EDUPREDICT-CLEAN-UI-2026-11-V1 */
/* Flush the dashboard to the top of the app viewport. The CSS-only st.html()
   calls do not create blank Streamlit layout blocks above the shell. */
[data-testid="stMainBlockContainer"],
[data-testid="stAppViewBlockContainer"],
.block-container {
    padding-top: 0 !important;
    margin-top: 0 !important;
}
.st-key-app_shell {padding-top: 0 !important;}
/* No fake calendar: these styles belong to counts derived from student records. */
.review-stat{display:flex;justify-content:space-between;gap:12px;align-items:center;
  margin:0 13px;padding:11px 3px;border-bottom:1px solid #ebe6db;}
.review-stat:last-child{border-bottom:0;}
.review-stat strong{display:block;font-size:12px;color:#5a554b;}
.review-stat small{display:block;font-size:10px;color:#918a7f;line-height:1.45;margin-top:3px;}
.review-stat .review-num{font-family:Georgia,serif;font-size:20px;font-weight:700;color:#927430;
  background:#f1e8d2;border-radius:11px;padding:8px;min-width:40px;text-align:center;}
.summary-note{font-size:11px;color:#776d5a;line-height:1.5;
  margin:9px 14px 3px;padding:10px 11px;background:#f2eee5;border-radius:10px;}
@media (max-width:900px){.st-key-app_shell{padding-top:0!important;}}

/* EDUPREDICT-STUDENT-SEARCH-STICKY-2026-11-V2 */
/* The header remains visible as the user scrolls the main workspace. */
.st-key-topbar {
    position: sticky !important;
    top: 0 !important;
    z-index: 1001 !important;
    background: #f9f7f2 !important;
    box-shadow: 0 7px 18px rgba(64, 59, 47, .10) !important;
    overflow: visible !important;
}
/* Streamlit wraps keyed containers in an element. Make that wrapper sticky,
   otherwise the child may be constrained by a short wrapper during scrolling.
   Exclude the outer app-shell container, which includes the sidebar. */
.st-key-app_shell [data-testid="stElementContainer"]:has(.st-key-topbar):not(:has(.st-key-sidebar)) {
    position: sticky !important;
    top: 0 !important;
    z-index: 1000 !important;
    align-self: stretch !important;
}
.st-key-topbar [data-testid="stSelectbox"] {
    margin-top: 3px !important;
    margin-bottom: 0 !important;
}
.st-key-topbar [data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background: #f1efeb !important;
    border: 1px solid #dedbd5 !important;
    border-radius: 10px !important;
    min-height: 42px !important;
}
.st-key-topbar [data-testid="stSelectbox"] input {
    color: #494941 !important;
    font-size: 13px !important;
}
@media(max-width:900px) {
    .st-key-topbar {top: 0 !important;}
}
</style>
'''

st.html(CSS)

# Human-readable labels. Internal page keys stay unchanged for compatibility.
NAV = [
    ('Overview', 'Home · Start here', ':material/home:'),
    ('Students', 'Find students', ':material/groups:'),
    ('Profiles', 'Student profile', ':material/person:'),
    ('Subjects', 'Attendance by subject', ':material/menu_book:'),
    ('Predict', 'Try a prediction', ':material/insights:'),
    ('Analytics', 'How the ML works', ':material/bar_chart:'),
    ('Reports', 'Download reports', ':material/download:'),
    ('Settings', 'Help & about', ':material/help:'),
]
PAGE_LABELS = {key: label for key, label, _ in NAV}
if 'page' not in st.session_state:
    st.session_state.page = 'Overview'
active = st.session_state.page
st.html(f'<style>.st-key-nav_{active} button{{background:linear-gradient(105deg,#90722f,#b39751)!important;color:white!important;font-weight:650!important;}}</style>')

raw = load_students()
subject_catalog = load_subjects()
subject_records = load_subject_records()
bundle = load_model(MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else None)
students = with_predictions(raw, bundle)
mode = 'ML demo' if bundle else 'Rule-based demo'
risk_counts = students['risk'].value_counts()
high = int(risk_counts.get('High', 0))
medium = int(risk_counts.get('Medium', 0))
low = int(risk_counts.get('Low', 0))
now = datetime.now(ZoneInfo('Asia/Kolkata'))
today = now.date()


def indicator_chips(attendance, assignment, internal, *, subject_level=False) -> str:
    signals = academic_indicators(attendance, assignment, internal, subject_level=subject_level)
    chips = []
    for signal in signals:
        klass = ' indicator-watch' if signal['flag'] else ''
        label = escape(str(signal['short']))
        note = escape(str(signal['note']), quote=True)
        chips.append(f'<span class="indicator-chip{klass}" title="{note}">{label}: {float(signal["value"]):.0f}%</span>')
    return '<div class="indicator-row">' + ''.join(chips) + '</div>'


def indicators_panel(attendance, assignment, internal, *, subject_level=False) -> str:
    chips = indicator_chips(attendance, assignment, internal, subject_level=subject_level)
    summary = escape(indicator_review_summary(
        attendance, assignment, internal, subject_level=subject_level))
    return (
        '<div class="indicator-review"><strong>Academic indicators to review</strong>'
        + chips + '<div class="indicator-caption">' + summary
        + '. Highlighted values suggest areas for review. '
        + 'The 60% marks benchmarks are not university passing marks. '
        + 'These descriptive indicators are not causal explanations of the ML model.'
        + '</div></div>'
    )


def student_table(frame: pd.DataFrame, limit: int = 6) -> str:
    lines = []
    for _, row in frame.head(limit).iterrows():
        name = escape(str(row['name']))
        rid = escape(str(row['student_id']))
        risk = row['risk']
        lines.append(f'''<tr><td class="student-name">{name}<small>{rid}</small></td>
        <td><span class="risk-chip risk-{risk}">{risk}</span>{indicator_chips(row['attendance'], row['assignment'], row['internal'])}</td>
        <td>{'Subject shortage' if int(row['subjects_below_75']) else 'All meet 75%'}</td></tr>''')
    return f'''<div class="data-card"><div class="table-heading"><strong>MCA Semester 3 student risk overview</strong><small>{len(frame)} records</small></div>
    <div style="overflow-x:auto"><table class="student-table"><thead><tr><th>STUDENT</th><th>ML RISK · ACADEMIC INDICATORS</th><th>ATTENDANCE RULE</th></tr></thead>
    <tbody>{''.join(lines) if lines else '<tr><td colspan="3">No matching students.</td></tr>'}</tbody></table></div></div>'''


def metric_cards() -> str:
    return f'''<div class="feature-grid">
     <div class="feature-card"><div class="feature-head">Students Monitored</div><div class="feature-body"><div><div class="feature-value">{len(students)}</div><div class="feature-note">Across the demo class</div></div><div class="feature-illustration">{icon_svg('students', 39)}</div></div></div>
     <div class="feature-card risky"><div class="feature-head">Students At Risk</div><div class="feature-body"><div><div class="feature-value">{high}</div><div class="feature-note">High-priority review</div></div><div class="feature-illustration">{icon_svg('alert', 36)}</div></div></div>
     <div class="feature-card"><div class="feature-head">Average Attendance</div><div class="feature-body"><div><div class="feature-value">{students['attendance'].mean():.0f}%</div><div class="feature-note">7 attendance-required subjects</div></div><div class="feature-illustration">{icon_svg('assessment', 38)}</div></div></div>
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
    """Show class-derived information, not invented calendar appointments."""
    attendance_shortage = int((students['subjects_below_75'] > 0).sum())
    marks_review = int(((students['assignment'] < 60) | (students['internal'] < 60)).sum())
    attendance_subjects = int(subject_catalog['attendance_required'].sum())

    st.markdown(f'''<div class="side-card"><div class="side-title">Class Overview</div>
    <div class="review-stat"><div><strong>Students</strong><small>Student profiles in the class</small></div><span class="review-num">{len(students)}</span></div>
    <div class="review-stat"><div><strong>Subjects</strong><small>{attendance_subjects} with attendance requirements</small></div><span class="review-num">{len(subject_catalog)}</span></div>
    <div class="summary-note">75% attendance is required in each applicable subject. Mini Project is exempt.</div>
    </div>''', unsafe_allow_html=True)

    st.markdown(f'''<div class="side-card"><div class="side-title">Needs Attention</div>
    <div class="review-stat"><div><strong>Attendance shortages</strong><small>Below 75% in at least one subject</small></div><span class="review-num">{attendance_shortage}</span></div>
    <div class="review-stat"><div><strong>High ML risk</strong><small>Academic follow-up suggested</small></div><span class="review-num">{high}</span></div>
    <div class="review-stat"><div><strong>Marks to review</strong><small>Assignment or internal average below 60%</small></div><span class="review-num">{marks_review}</span></div>
    </div>''', unsafe_allow_html=True)

    st.markdown('''<div class="side-card"><div class="side-title">Understanding the Results</div>
    <div class="summary-note"><strong>Attendance eligibility</strong> uses the subject-wise 75% rule.<br>
    <strong>ML risk</strong> is a separate early-warning estimate. They are not the same result.</div>
    </div>''', unsafe_allow_html=True)

def task_hint(title: str, instructions: str) -> None:
    """A simple, safe one-sentence instruction for the current view."""
    st.markdown(
        '<div class="task-hint"><strong>' + escape(title) + '</strong> · '
        + escape(instructions) + '</div>', unsafe_allow_html=True
    )


def guided_start() -> None:
    """Three real buttons to lead a first-time visitor through the product."""
    st.markdown('<div class="guide-head">Start here — three simple steps</div>'
                '<div class="guide-sub">You can explore without entering any real student data.</div>',
                unsafe_allow_html=True)
    cards = [
        ('step_find', '1', 'Find a student', 'Browse the class and choose someone to review.',
         'Find students', 'Students'),
        ('step_subject', '2', 'Check attendance', 'See the 75% rule separately for each subject.',
         'Check subjects', 'Subjects'),
        ('step_predict', '3', 'Try the ML demo', 'Move three sliders and view a model risk score.',
         'Try prediction', 'Predict'),
    ]
    columns = st.columns(3, gap='small')
    for column, (key, number, title, description, button_text, target) in zip(columns, cards):
        with column:
            with st.container(key=key):
                st.markdown(
                    f'<div class="guide-number">{number}</div>'
                    f'<div class="guide-title">{escape(title)}</div>'
                    f'<div class="guide-desc">{escape(description)}</div>',
                    unsafe_allow_html=True,
                )
                if st.button(button_text, key='guide_to_' + target, use_container_width=True):
                    st.session_state.page = target
                    st.rerun()
    st.markdown(
        '<div class="guide-distinction"><strong>Two different checks:</strong> '
        '<strong>Attendance eligibility</strong> uses 75% in each applicable subject. '
        '<strong>ML risk</strong> is a demonstration score learned from fabricated outcomes, '
        'not a certain prediction of exam failure. Scores below 60% suggest further academic review, '
        'but 60% is not an official passing mark.</div>',
        unsafe_allow_html=True,
    )


def student_view(df: pd.DataFrame):
    st.markdown(f'<div class="page-heading">MCA Semester 3 · Students</div><div class="page-description">{escape(IDENTITY_NOTE)} Academic risk and attendance eligibility are separate indicators.</div>', unsafe_allow_html=True)
    task_hint('Start with the class list', 'Search for a student, filter by risk, then open their detailed profile to review subjects.')
    with st.container(key='page_panel'):
        left, right = st.columns([2, 1])
        with left:
            st.markdown('### Class records')
        with right:
            selected = st.selectbox('Risk category', ['All','High','Medium','Low'], key='risk_filter')
        filtered = (df if selected == 'All' else df[df['risk'] == selected]).copy()
        filtered['review_indicators'] = filtered.apply(lambda r: indicator_review_summary(r['attendance'], r['assignment'], r['internal']), axis=1)
        columns = ['student_id','name','attendance','assignment','internal','risk','review_indicators','subjects_below_75','attendance_eligibility']
        st.dataframe(filtered[columns], hide_index=True, use_container_width=True, height=320,
            column_config={'student_id':'SBU ID' if USE_LOCAL_ROSTER else 'Demo ID','name':'Student','attendance':'Overall attendance (%)',
                           'assignment':'Assignment mean (%)','internal':'Internal mean (%)',
                           'risk':'ML risk','review_indicators':'Academic review flags',
                           'subjects_below_75':'Subjects below 75%',
                           'attendance_eligibility':'Attendance status'})
        st.caption(f'{len(filtered)} class records · {mode}. Marks and attendance are simulated, not official. Overall attendance does NOT replace subject-specific checks.')

        st.markdown('### View a student’s subject-wise record')
        choices = students.set_index('student_id')['name'].to_dict()
        selected_id = st.selectbox('Student', options=list(choices),
                                  format_func=lambda sid: f'{choices[sid]} ({sid})', key='student_details')
        pupil = students.loc[students['student_id'] == selected_id].iloc[0]
        p1, p2, p3 = st.columns(3)
        p1.metric('Overall attendance', f"{int(pupil['attendance'])}%")
        p2.metric('Subjects below 75%', int(pupil['subjects_below_75']))
        p3.metric('Model risk band', str(pupil['risk']))
        st.markdown(indicators_panel(pupil['attendance'], pupil['assignment'], pupil['internal']), unsafe_allow_html=True)
        if st.button('Open detailed student profile  →', type='primary', key='open_full_profile'):
            st.session_state.profile_student_id = selected_id
            st.session_state.page = 'Profiles'
            st.rerun()
        detail = subject_records[subject_records['student_id'] == selected_id].copy()
        detail['attendance_pct'] = detail['attendance_pct'].round(1)
        st.dataframe(detail[['subject','classes_attended','classes_conducted','attendance_pct',
                             'attendance_status','classes_needed_to_75','assignment','internal']],
                     hide_index=True, use_container_width=True,
                     column_config={'subject':'Subject','classes_attended':'Attended',
                                    'classes_conducted':'Conducted','attendance_pct':'Attendance (%)',
                                    'attendance_status':'Attendance rule',
                                    'classes_needed_to_75':'Classes to reach 75%*',
                                    'assignment':'Assignment (%)','internal':'Internal (%)'})
        st.caption('*Minimum consecutive future classes the student would need to attend, assuming those classes are held. Mini Project is exempt (blank attendance).')



def profile_view():
    st.markdown('<div class="page-heading">Individual Student Profile</div><div class="page-description">MCA Semester 3 · Subject-wise attendance, simulated marks, and exploratory ML risk estimates.</div>', unsafe_allow_html=True)
    task_hint('Choose a student above', 'Check their attendance by subject, then choose one subject for a closer look. Overall attendance does not override a subject shortage.')
    with st.container(key='page_panel'):
        identifiers = students['student_id'].astype(str).tolist()
        names = dict(zip(students['student_id'].astype(str), students['name']))
        if st.session_state.get('profile_student_id') not in identifiers:
            st.session_state.profile_student_id = identifiers[0]
        sid = st.selectbox(
            'Choose a student', options=identifiers,
            format_func=lambda selected: f'{names[selected]} ({selected})',
            key='profile_student_id',
        )
        pupil = students.loc[students['student_id'].astype(str) == sid].iloc[0]
        details = subject_profile_rows(subject_records, subject_catalog, sid, bundle)
        examined = details.loc[details['attendance_required']]
        attendance_shortages = int((examined['verified_attendance_status'] == 'Below 75%').sum())
        high_subject_risks = int((examined['subject_risk'] == 'High').sum())

        safe_name = escape(str(pupil['name']))
        safe_id = escape(str(sid))
        st.markdown(f'<div class="profile-banner"><strong>{safe_name}</strong><span>{safe_id} · MCA Semester 3 · 8 subjects · All attendance and marks are synthetic</span></div>', unsafe_allow_html=True)
        a, b, c = st.columns(3)
        a.metric('Overall attendance · 7 subjects', f"{int(pupil['attendance'])}%")
        b.metric('Subjects below 75%', attendance_shortages)
        c.metric('Subjects flagged High (ML demo)', high_subject_risks if bundle is not None else '—')

        if attendance_shortages:
            st.warning(f'Attendance eligibility: {attendance_shortages} subject(s) below 75%. This is separate from ML risk.')
        else:
            st.success('Attendance eligibility: minimum 75% met in all seven applicable subjects.')

        st.markdown(indicators_panel(pupil['attendance'], pupil['assignment'], pupil['internal']), unsafe_allow_html=True)
        st.caption('Overall attendance is descriptive; subject eligibility is checked separately.')
        st.markdown('<div class="profile-section-title">Subject-wise academic record</div>', unsafe_allow_html=True)
        display_rows = []
        for _, record in details.iterrows():
            exempt = not bool(record['attendance_required'])
            display_rows.append({
                'Subject': record['subject'],
                'Faculty': record['professor'],
                'Attendance': 'Exempt' if exempt else f"{float(record['attendance_pct']):.1f}%",
                'Eligibility': record['verified_attendance_status'],
                'Classes to 75%': None if exempt else int(record['verified_classes_to_75']),
                'Assignment': '—' if exempt else f"{int(record['assignment'])}%",
                'Internal': '—' if exempt else f"{int(record['internal'])}%",
                'ML demo risk': record['subject_risk'],
                'Review indicators': 'Exempt' if exempt else indicator_review_summary(record['attendance_pct'], record['assignment'], record['internal'], subject_level=True),
                'Risk score': '—' if exempt or pd.isna(record['risk_score_pct']) else f"{float(record['risk_score_pct']):.1f}%",
            })
        st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True, height=345)
        st.caption('Mini Project is exempt from attendance and excluded from subject exam-risk estimates. The 75% status is calculated from attended/conducted classes, independently of ML.')
        st.info('**Research limitation:** The current model was trained on separate synthetic aggregate scores, not on real subject-level exam outcomes. Its subject-level results are exploratory model scores—not validated failure probabilities or official eligibility decisions.')

        st.markdown('<div class="profile-section-title">Examine a subject</div>', unsafe_allow_html=True)
        chosen_code = st.selectbox('Subject', options=details['subject_code'].tolist(),
            format_func=lambda code: str(details.loc[details['subject_code'] == code, 'subject'].iloc[0]),
            key='profile_subject_select')
        subject = details.loc[details['subject_code'] == chosen_code].iloc[0]
        st.write(f"**Professor:** {subject['professor']}  ·  **Credits:** {int(subject['credits'])}")
        if not bool(subject['attendance_required']):
            st.info('Mini Project is attendance-exempt; no attendance recovery or exam-risk estimate applies.')
        else:
            x, y, z = st.columns(3)
            x.metric('Attendance', f"{float(subject['attendance_pct']):.1f}%")
            y.metric('Assignment', f"{int(subject['assignment'])}%")
            z.metric('Internal assessment', f"{int(subject['internal'])}%")
            st.progress(min(100, max(0, int(round(float(subject['attendance_pct']))))), text='Attendance progress · minimum 75%')
            if subject['verified_attendance_status'] == 'Below 75%':
                st.warning(f"Attendance shortage: {int(subject['verified_classes_to_75'])} consecutive future attended class(es) needed to reach 75%, assuming each class is held and attended.")
            else:
                st.success('Attendance requirement met for this subject.')
            if bundle is not None:
                st.markdown(f"**Exploratory ML risk band:** {subject['subject_risk']} · **uncalibrated demo score:** {float(subject['risk_score_pct']):.1f}%")
            else:
                st.caption('Train the demonstration ML model to see subject-level scores; attendance eligibility works without it.')
            st.markdown(indicators_panel(subject['attendance_pct'], subject['assignment'], subject['internal'], subject_level=True), unsafe_allow_html=True)

        st.markdown('**Suggested academic support**')
        for note in academic_support(subject):
            st.write('• ' + note)
        st.caption('Suggestions are based on attendance and marks thresholds, not a causal explanation of the ML model. These are fictional academic records.')


def subjects_view():
    st.markdown('<div class="page-heading">Subjects & Attendance</div><div class="page-description">Seven subjects require 75% attendance. Mini Project is exempt. All attendance and marks are synthetic.</div>', unsafe_allow_html=True)
    task_hint('Pick one subject', 'See which demo students meet 75% attendance and how many future classes could resolve a shortage. Mini Project is exempt.')
    with st.container(key='page_panel'):
        st.markdown('### MCA Semester 3 subjects')
        st.dataframe(subject_catalog[['subject_code','subject','professor','credits','attendance_required']],
                     hide_index=True, use_container_width=True,
                     column_config={'subject_code':'Paper code','subject':'Subject',
                                    'professor':'Faculty','credits':'Credits',
                                    'attendance_required':'75% attendance required'})
        selected_code = st.selectbox('Select a subject', options=subject_catalog['subject_code'].tolist(),
            format_func=lambda code: str(subject_catalog.loc[subject_catalog['subject_code'] == code, 'subject'].iloc[0]),
            key='subject_select')
        meta = subject_catalog.loc[subject_catalog['subject_code'] == selected_code].iloc[0]
        st.markdown(f"**Faculty:** {escape(str(meta['professor']))}  ·  **Credits:** {int(meta['credits'])}")
        rows = subject_records[subject_records['subject_code'] == selected_code].copy()
        if not bool(meta['attendance_required']):
            st.info('Mini Project has no attendance requirement. All 34 students are marked Exempt; no 0% attendance is assigned.')
        else:
            eligible_count = int((rows['attendance_status'] == 'Meets 75%').sum())
            c1, c2, c3 = st.columns(3)
            c1.metric('Meets 75%', eligible_count)
            c2.metric('Below 75%', len(rows) - eligible_count)
            c3.metric('Avg. attendance', f"{rows['attendance_pct'].mean():.1f}%")
        st.dataframe(rows[['student_id','name','classes_attended','classes_conducted',
                           'attendance_pct','attendance_status','classes_needed_to_75','assignment','internal']],
                     hide_index=True, use_container_width=True, height=400,
                     column_config={'student_id':'SBU ID' if USE_LOCAL_ROSTER else 'Demo ID','name':'Student',
                                    'classes_attended':'Attended','classes_conducted':'Conducted',
                                    'attendance_pct':'Attendance (%)','attendance_status':'Eligibility',
                                    'classes_needed_to_75':'Classes to 75%*',
                                    'assignment':'Assignment (%)','internal':'Internal (%)'})
        st.caption('*Future attendance recovery is illustrative. University decisions can involve additional policies.')


def prediction_view():
    st.markdown('<div class="page-heading">Try a Prediction</div><div class="page-description">Explore how the synthetic ML demo responds to attendance and assessment marks.</div>', unsafe_allow_html=True)
    task_hint('Try your own example', 'Set attendance, assignment and internal scores with the sliders, then click Assess student. This does not determine exam eligibility.')
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
                risk = label_from_probability(p, bundle.get('decision_threshold', 0.50))
                message = f'{risk} risk · model risk score {p:.0%} (synthetic-data demo; not a calibrated probability)'
            else:
                risk = rule_band({'attendance':a,'assignment':b,'internal':c})
                message = f'{risk} risk · rule-based demonstration, not an ML prediction'
            if risk == 'High':
                st.error(message)
            elif risk == 'Medium':
                st.warning(message)
            else:
                st.success(message)
            st.markdown(indicators_panel(a, b, c), unsafe_allow_html=True)
            st.caption('ML output and descriptive academic review flags are separate; low marks do not automatically determine the ML label.')
            tips = []
            if a < 75: tips.append('Offer attendance support and check barriers to participation.')
            if b < 60: tips.append('Review assignment completion and offer targeted practice.')
            if c < 60: tips.append('Plan revision for weak internal-assessment topics.')
            if not tips: tips.append('Continue regular academic follow-up.')
            st.markdown('**Suggested academic support**')
            for tip in tips: st.write('• ' + tip)
        st.info('This is an ML teaching demo. In the Subjects page, attendance eligibility is checked separately for each of the seven non-exempt subjects. A model risk score is not proof of exam ineligibility.')


def analytics_view():
    st.markdown('<div class="page-heading">How the ML Model Works</div><div class="page-description">Understand test results on synthetic examples, not real exam predictions.</div>', unsafe_allow_html=True)
    task_hint('Understand the ML result', 'Accuracy measures overall correct predictions; recall measures how many simulated failing students were flagged; F1 balances recall and precision.')
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
            st.caption(f"Decision threshold: {bundle.get('decision_threshold', 0.50):.0%}. The High-risk label matches the model's predicted Fail class; Medium means close to the threshold.")
            st.write('Confusion matrix (rows: actual, columns: predicted; Pass then Fail):')
            st.caption('Read the matrix by row: actual Pass/Fail, and by column: model-predicted Pass/Fail. A missed failure appears in the Actual Fail / Pred Pass cell.')
            st.dataframe(pd.DataFrame(m['confusion_matrix'], index=['Actual Pass','Actual Fail'], columns=['Pred Pass','Pred Fail']))
            st.warning('Model metrics are from the separate 280-record synthetic training dataset, not the 34 roster-linked MCA demo profiles. They do not establish real-world accuracy or fairness.')


def reports_view():
    st.markdown('<div class="page-heading">Reports</div><div class="page-description">Export demo records and risk-screening results.</div>', unsafe_allow_html=True)
    task_hint('Download demonstration data', 'Export class risk or subject attendance records. Do not share locally exported reports containing classmates’ names or IDs.')
    with st.container(key='page_panel'):
        st.markdown('### Export student screening')
        columns = ['student_id','name','attendance','assignment','internal','risk','subjects_below_75','attendance_eligibility']
        if bundle is not None: columns.append('fail_probability')
        out = students[columns].copy()
        st.download_button('Download CSV report', out.to_csv(index=False).encode('utf-8'), file_name='edupredict_demo_risk_report.csv', mime='text/csv', type='primary')
        st.caption(f'{len(students)} class profiles · {mode}; all attendance and marks are simulated. Downloaded reports contain names and IDs; do not post them publicly.')
        st.dataframe(out.head(12), use_container_width=True, hide_index=True)
        st.download_button('Download subject-wise attendance CSV',
            subject_records.to_csv(index=False).encode('utf-8'),
            file_name='edupredict_mca_subject_attendance_DEMO.csv', mime='text/csv')


def settings_view():
    st.markdown('<div class="page-heading">About EduPredict</div><div class="page-description">Project information and usage guidance.</div>', unsafe_allow_html=True)
    task_hint('How to use EduPredict', 'Begin at Home, open Find students for class profiles, choose Attendance by subject for the 75% rule, or use Try a prediction for the ML demo.')
    with st.container(key='page_panel'):
        st.markdown('### Smart student early-warning system')
        st.write('**Inputs:** attendance, assignment score and internal assessment score.')
        st.write(f'**Current prediction mode:** {mode}.')
        st.write('**Class:** MCA Semester 3 · 34 student profiles · 8 subjects (7 attendance-required).')
        st.write('**Attendance rule:** minimum 75% in each of seven attendance-required subjects; Mini Project exempt.')
        st.write('**Purpose:** demonstrate subject attendance eligibility and ML-based academic support screening before final examinations.')
        st.write('**Data:** ' + IDENTITY_NOTE + ' Email addresses are not used. Local student names and IDs must not be committed to a public repository.')
        st.write('**Next step:** develop and validate your model with appropriately permitted, real labelled data before real-world use.')

def open_student_from_header_search() -> None:
    """Open the selected student's profile before Streamlit reruns the page."""
    student_id = st.session_state.get('header_student_picker')
    if student_id is None:
        return
    known_ids = set(students['student_id'].astype(str))
    if str(student_id) not in known_ids:
        return
    st.session_state['profile_student_id'] = str(student_id)
    st.session_state['page'] = 'Profiles'
    # Clearing here (in a widget callback) permits immediately searching again.
    st.session_state['header_student_picker'] = None


with st.container(key='app_shell'):
    sidebar_col, workspace_col = st.columns([0.19, 0.81], gap='medium')
    with sidebar_col:
        with st.container(key='sidebar'):
            st.markdown('<div class="brand-name">EduPredict<span style="color:#c9b26c">.</span></div><div class="brand-sub">SMART EDUCATION</div>', unsafe_allow_html=True)
            st.markdown('<div class="nav-section">GET STARTED</div>', unsafe_allow_html=True)
            for page, label, icon in NAV:
                if page == 'Profiles':
                    st.markdown('<div class="nav-section">EXPLORE</div>', unsafe_allow_html=True)
                if page == 'Analytics':
                    st.markdown('<div class="nav-section">LEARN & EXPORT</div>', unsafe_allow_html=True)
                if st.button(label, icon=icon, key=f'nav_{page}', use_container_width=True):
                    st.session_state.page = page
                    st.rerun()
            st.markdown('<div class="nav-section">ACCOUNT</div>', unsafe_allow_html=True)
            if st.button('Log out', icon=':material/logout:', key='faculty_logout', use_container_width=True):
                logout_faculty()
    with workspace_col:
        with st.container(key='topbar'):
            search_col, teacher_col = st.columns([0.68, 0.32], gap='small')
            with search_col:
                # This native picker supports suggestions by student name or SBU ID.
                student_names = dict(zip(students['student_id'].astype(str),
                                         students['name'].astype(str)))
                student_options = sorted(student_names, key=lambda sid: (student_names[sid].casefold(), sid))
                st.selectbox(
                    'Find a student by name or SBU ID',
                    options=student_options,
                    index=None,
                    format_func=lambda sid: f'{student_names[sid]} ({sid})',
                    placeholder='Search a name or SBU ID — select a suggestion',
                    label_visibility='collapsed',
                    key='header_student_picker',
                    on_change=open_student_from_header_search,
                    help='Click to see students, or type a name/ID. Enter selects the highlighted match.',
                )
                query = ''  # Selected suggestions navigate to a profile; do not filter this page.
            with teacher_col:
                st.markdown(f'<div class="teacher-pill"><span class="teacher-avatar">👩🏻‍🏫</span><div><strong>{escape(faculty_name)}</strong><small>Faculty account</small></div></div>',unsafe_allow_html=True)
        if active != 'Overview':
            if st.button('← Back to Home', key='back_to_home'):
                st.session_state.page = 'Overview'
                st.rerun()
        st.markdown('<div style="height:11px"></div>',unsafe_allow_html=True)
        if active == 'Overview':
            main_col, right_col = st.columns([0.755, 0.245], gap='small')
        else:
            main_col = st.container()  # More room for subject and student tables.
            right_col = None
        with main_col:
            if active == 'Overview':
                with st.container(key='hero'):
                    hero_copy, hero_art = st.columns([0.69, 0.31], gap='small')
                    with hero_copy:
                        st.markdown('<div class="eyebrow">SBU · MCA SEMESTER 3 · SIMULATED ACADEMIC DATA</div><h1 class="hero-title">Hello, Teacher!</h1><p class="hero-sub">Monitor academic risk across 34 student profiles and review 75% attendance eligibility in seven subjects. Mini Project is exempt.</p>', unsafe_allow_html=True)
                        with st.container(key='hero_action'):
                            if st.button('Review student risk  →', key='review_action'):
                                st.session_state.page = 'Students'
                                st.rerun()
                    with hero_art:
                        hero_path = ROOT / 'assets' / 'hero_academic.png'
                        if hero_path.exists():
                            st.image(str(hero_path), use_container_width=True)
                        else:
                            st.markdown('<div class="hero-fallback">Progress · Insights · Support</div>', unsafe_allow_html=True)
                guided_start()
                st.markdown(metric_cards(),unsafe_allow_html=True)
                attendance_shortage = int((students['subjects_below_75'] > 0).sum())
                st.caption(f'Attendance eligibility: {attendance_shortage} of 34 student profiles are below 75% in at least one subject. This is separate from ML risk. Attendance and marks are synthetic; IDs and names follow the local roster when available.')
                chart_l, chart_r = st.columns([0.4, 0.6], gap='small')
                with chart_l: st.markdown(donut_chart(),unsafe_allow_html=True)
                with chart_r: st.markdown(bar_chart(students.sort_values('attendance').iloc[np.linspace(0, len(students)-1, 7, dtype=int)]), unsafe_allow_html=True)
                filtered = students[students['name'].str.contains(query, case=False, na=False)] if query else students
                preview = filtered.assign(_order=filtered['risk'].map({'High':0,'Medium':1,'Low':2})).sort_values(['_order','attendance']).drop(columns='_order')
                st.markdown(student_table(preview, limit=6),unsafe_allow_html=True)
            elif active == 'Students':
                filtered = students[students['name'].str.contains(query, case=False, na=False)] if query else students
                student_view(filtered)
            elif active == 'Profiles': profile_view()
            elif active == 'Subjects': subjects_view()
            elif active == 'Predict': prediction_view()
            elif active == 'Analytics': analytics_view()
            elif active == 'Reports': reports_view()
            else: settings_view()
            identity_label = "Local roster identities" if USE_LOCAL_ROSTER else "Fictional identities"
            st.markdown(f'<div class="hint-strip">{mode} · {identity_label} · Synthetic marks/attendance · 8 subjects · 75% attendance check.</div>',unsafe_allow_html=True)
        if right_col is not None:
            with right_col:
                right_panel()
