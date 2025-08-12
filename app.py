# app.py (The new, unified, single-page application)

# ==============================================================================
# LIBRARIES & IMPORTS (All imports are here)
# ==============================================================================
import streamlit as st
import numpy as np
import pandas as pd
import io
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import math
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import beta, norm, t, f
from scipy.optimize import curve_fit
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import silhouette_score 
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.cluster import KMeans
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.inspection import PartialDependenceDisplay
from lifelines.statistics import logrank_test
from lifelines import KaplanMeierFitter
from PIL import Image
import shap
import xgboost as xgb
from scipy.special import logsumexp

# ==============================================================================
# APP CONFIGURATION
# ==============================================================================
st.set_page_config(
    layout="wide",
    page_title="Biotech V&V Analytics Toolkit",
    page_icon="🔬"
)

# --- ADD THESE COLOR CONSTANTS BENEATH YOUR CONFIG ---
PRIMARY_COLOR = "#0068C9"
DARK_GREY = "#333333"
SUCCESS_GREEN = "#2ca02c"

# ==============================================================================
# CSS STYLES
# ==============================================================================
st.markdown("""
<style>
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        color: #333;
    }
    .main .block-container {
        padding: 2rem 5rem;
        max-width: 1600px;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 2px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; background-color: #F0F2F6; border-radius: 4px 4px 0px 0px;
        padding: 0px 24px; border-bottom: 2px solid transparent; transition: all 0.3s;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF; font-weight: 600; border-bottom: 2px solid #0068C9;
    }
    [data-testid="stMetric"] {
        background-color: #FFFFFF; border: 1px solid #E0E0E0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04); padding: 15px 20px; border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# ALL HELPER & PLOTTING FUNCTIONS
# ==============================================================================
@st.cache_data
def plot_v_model():
    """
    Generates a high-quality, static V-Model diagram with generic biotech examples.
    """
    fig = go.Figure()
    
    # --- CORRECTED: Use the consistent BASE_STAGES dictionary for a clean, static plot ---
    # This ensures the main diagram is simple and generic, while the table holds the details.
    v_model_stages = {
        'URS': {'name': 'User Requirements', 'icon': '🎯', 'x': 0, 'y': 5, 'question': 'What does the user/business need?', 'tools': 'Business Case, Target Product Profile'},
        'FS':  {'name': 'Functional Specs', 'icon': '⚙️', 'x': 1, 'y': 4, 'question': 'What must the system *do*?', 'tools': 'Assay Specs, Throughput Goals, User Roles'},
        'DS':  {'name': 'Design Specs', 'icon': '✍️', 'x': 2, 'y': 3, 'question': 'How will it be built/configured?', 'tools': 'DOE, Component Selection, Architecture'},
        'BUILD': {'name': 'Implementation', 'icon': '🛠️', 'x': 3, 'y': 2, 'question': 'Build, Code, Write SOPs, Train', 'tools': 'Physical Transfer, Coding, Training'},
        'IQOQ':{'name': 'IQ / OQ', 'icon': '🔌', 'x': 4, 'y': 3, 'question': 'Is it installed and operating correctly?', 'tools': 'Calibration, Unit & Integration Tests'},
        'PQ':  {'name': 'Performance Qualification', 'icon': '📈', 'x': 5, 'y': 4, 'question': 'Does it perform reliably in its environment?', 'tools': 'Gage R&R, Method Comp, PPQ Runs'},
        'VAL': {'name': 'Final Validation', 'icon': '✅', 'x': 6, 'y': 5, 'question': 'Does it meet the original user need?', 'tools': 'Validation Report, UAT, Final Release'}
    }
    # --- END OF CORRECTION ---
    
    verification_color = '#008080'
    validation_color = '#0068C9'
    path_keys = ['URS', 'FS', 'DS', 'BUILD', 'IQOQ', 'PQ', 'VAL']
    path_x = [v_model_stages[p]['x'] for p in path_keys]
    path_y = [v_model_stages[p]['y'] for p in path_keys]
    
    fig.add_trace(go.Scatter(x=path_x, y=path_y, mode='lines', line=dict(color='lightgrey', width=5), hoverinfo='none'))

    for i in range(3):
        start_key, end_key = path_keys[i], path_keys[-(i+1)]
        fig.add_shape(type="line", x0=v_model_stages[start_key]['x'], y0=v_model_stages[start_key]['y'],
                      x1=v_model_stages[end_key]['x'], y1=v_model_stages[end_key]['y'],
                      line=dict(color="darkgrey", width=2, dash="dot"))

    for i, (key, stage) in enumerate(v_model_stages.items()):
        color = verification_color if i < 3 else validation_color if i > 3 else '#636EFA'
        fig.add_shape(type="rect", x0=stage['x']-0.5, y0=stage['y']-0.4,
                      x1=stage['x']+0.5, y1=stage['y']+0.4,
                      line=dict(color="black", width=2), fillcolor=color, layer='above')
        fig.add_annotation(x=stage['x'], y=stage['y']+0.15, text=f"{stage['icon']} <b>{stage['name']}</b>",
                           showarrow=False, font=dict(color='white', size=12), align='center')
        hover_text = (f"<b>{stage['icon']} {stage['name']}</b><br><br>"
                      f"<i>{stage['question']}</i><br><br>"
                      f"<b>Generic Examples:</b><br>{stage.get('tools', 'N/A')}")
        fig.add_trace(go.Scatter(
            x=[stage['x']], y=[stage['y']], mode='markers',
            marker=dict(color='rgba(0,0,0,0)', size=100),
            hoverinfo='text', text=hover_text,
            hoverlabel=dict(bgcolor="white", font_size=14, font_family="Arial", bordercolor="black")
        ))
        
    fig.add_annotation(x=1, y=5.2, text="<span style='color:#008080; font-size: 20px;'><b>VERIFICATION</b></span><br><span style='font-size: 12px;'>(Are we building the system right?)</span>",
                       showarrow=False, align='center')
    fig.add_annotation(x=5, y=5.2, text="<span style='color:#0068C9; font-size: 20px;'><b>VALIDATION</b></span><br><span style='font-size: 12px;'>(Are we building the right system?)</span>",
                       showarrow=False, align='center')

    fig.update_layout(
        title_text="<b>The V-Model for Technology Transfer (Hover for Details)</b>",
        title_x=0.5, showlegend=False,
        xaxis=dict(visible=False, range=[-0.7, 6.7]), yaxis=dict(visible=False, range=[1.5, 6.0]),
        height=650, margin=dict(l=20, r=20, t=80, b=20),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig
    
@st.cache_data
def create_v_model_summary_table():
    """
    Creates a pandas DataFrame summarizing the V-Model contexts for display as a table.
    """
    # UPDATED: Added IVD and Pharma, and made other column names more specific.
    all_contexts = {
        'R&D Assay Method': {
            'URS': {'tools': 'Target Product Profile (TPP), Required sensitivity/specificity'},
            'FS':  {'tools': 'Assay type (e.g., ELISA, HPLC), Linearity, LOD/LOQ goals'},
            'DS':  {'tools': 'Reagent selection, SOP drafting, Robustness DOE plan'},
            'BUILD': {'tools': 'Method development experiments, SOP finalization, Analyst training'},
            'IQOQ':{'tools': 'Reagent qualification, Instrument calibration for this assay'},
            'PQ':  {'tools': 'Intermediate Precision, Gage R&R, Method Comparison'},
            'VAL': {'tools': 'Final Validation Report, Control charting plan'}
        },
        'Instrument': {
            'URS': {'tools': 'Required throughput, Sample types, User skill level'},
            'FS':  {'tools': 'Automation level, Data output format (LIMS), Footprint'},
            'DS':  {'tools': 'Vendor selection, Site prep requirements, Service contract'},
            'BUILD': {'tools': 'Purchase, Delivery, Physical installation'},
            'IQOQ':{'tools': 'Utility connections check (IQ), Factory tests run (OQ)'},
            'PQ':  {'tools': 'Performance on representative samples, Throughput testing'},
            'VAL': {'tools': 'System meets all URS criteria, Final release for GMP use'}
        },
        'Software System': {
            'URS': {'tools': 'Business process map, 21 CFR Part 11 requirements'},
            'FS':  {'tools': 'User roles, Required calculations, Audit trail specifications'},
            'DS':  {'tools': 'System architecture, Database schema, UI mockups'},
            'BUILD': {'tools': 'Coding, Configuration of COTS system, Writing user manuals'},
            'IQOQ':{'tools': 'Server setup validation (IQ), Unit & Integration testing (OQ)'},
            'PQ':  {'tools': 'User Acceptance Testing (UAT) with real-world scenarios'},
            'VAL': {'tools': 'System Validation Report, Release notes, Go-live approval'}
        },
        'IVD Development': {
            'URS': {'tools': 'Diagnostic claim, Clinical sensitivity/specificity reqs, Sample matrix'},
            'FS':  {'tools': 'Assay principle (e.g., PCR), Analytical performance goals (LOD, precision)'},
            'DS':  {'tools': 'Antibody/reagent selection, Protocol optimization, Verification/Validation plan'},
            'BUILD': {'tools': 'Prototype kit assembly, Reagent formulation, SOP drafting'},
            'IQOQ':{'tools': 'Manufacturing equipment qualification, QC test method validation'},
            'PQ':  {'tools': 'Clinical performance studies, Stability (shelf-life), Shipping validation'},
            'VAL': {'tools': '510(k)/PMA submission to FDA, Design History File (DHF) finalization'}
        },
        'Pharma Process': {
            'URS': {'tools': 'Target Product Profile (TPP), Critical Quality Attributes (CQAs), Cost of goods'},
            'FS':  {'tools': 'Unit operations (e.g., cell culture), In-Process Controls (IPCs)'},
            'DS':  {'tools': 'Critical Process Parameters (CPPs), Bill of Materials, Scale-down model design'},
            'BUILD': {'tools': 'Engineering runs, Master Batch Record (MBR) authoring'},
            'IQOQ':{'tools': 'Facility/utility qualification, Equipment commissioning'},
            'PQ':  {'tools': 'Process Performance Qualification (PPQ) runs, Cpk analysis'},
            'VAL': {'tools': 'Final PPQ report, Submission to regulatory agency'}
        }
    }
    
    df_data = {}
    for context, stages in all_contexts.items():
        df_data[context] = {stage: data['tools'] for stage, data in stages.items()}

    # Ensure a consistent column order
    column_order = ['R&D Assay Method', 'Instrument', 'Software System', 'IVD Development', 'Pharma Process']
    df = pd.DataFrame(df_data)[column_order]
    
    stage_order = ['URS', 'FS', 'DS', 'BUILD', 'IQOQ', 'PQ', 'VAL']
    stage_names = {
        'URS': 'User Requirements', 'FS': 'Functional Specs', 'DS': 'Design Specs',
        'BUILD': 'Implementation', 'IQOQ': 'IQ / OQ', 'PQ': 'Performance Qualification',
        'VAL': 'Final Validation'
    }
    
    df = df.reindex(stage_order)
    df.index = df.index.map(lambda key: f"{stage_names[key]} ({key})")
    df.index.name = "V-Model Stage"
    
    return df

@st.cache_data
def create_styled_v_model_table(df):
    """
    Transforms the V-Model summary DataFrame into a styled Plotly Table.
    """
    # UPDATED: Added colors for IVD and Pharma, and renamed other keys to match the new DataFrame.
    app_colors = {
        'R&D Assay Method': {'header': '#0068C9', 'cell': 'rgba(0, 104, 201, 0.1)'},
        'Instrument': {'header': '#008080', 'cell': 'rgba(0, 128, 128, 0.1)'},
        'Software System': {'header': '#636EFA', 'cell': 'rgba(99, 110, 250, 0.1)'},
        'IVD Development': {'header': '#2ca02c', 'cell': 'rgba(44, 160, 44, 0.1)'}, # New color for IVD
        'Pharma Process': {'header': '#FF7F0E', 'cell': 'rgba(255, 127, 14, 0.1)'}
    }
    
    # Prepare header values and colors
    header_values = [f"<b>{df.index.name}</b>"] + [f"<b>{col}</b>" for col in df.columns]
    header_fill_colors = ['#F0F2F6'] + [app_colors[col]['header'] for col in df.columns]
    header_font_colors = ['black'] + ['white'] * len(df.columns)
    
    # Prepare cell values and colors, transposing the dataframe
    cell_values = [df.index.tolist()] + [df[col].tolist() for col in df.columns]
    cell_fill_colors = [['#F8F9FA'] * len(df)] + [[app_colors[col]['cell']] * len(df) for col in df.columns]

    fig = go.Figure(data=[go.Table(
        # Increase column widths to better accommodate the new content
        columnwidth = [120] + [200]*len(df.columns),
        header=dict(
            values=header_values,
            fill_color=header_fill_colors,
            font=dict(color=header_font_colors, size=14),
            align=['left', 'center'],
            height=40,
            line_color='darkslategray'
        ),
        cells=dict(
            values=cell_values,
            fill_color=cell_fill_colors,
            align=['left', 'left'],
            font_size=12,
            height=30,
            line_color='darkslategray'
        ))
    ])

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        # Increase height slightly for better viewing
        height=500
    )
    
    return fig
    
# --- RESTORED PLOTTING FUNCTION 2 ---
# FIX: Replace the entire plot_act_grouped_timeline function with this new, complete version.
@st.cache_data
def plot_act_grouped_timeline():
    """Generates the project-based timeline with all tools, including Act 0."""
    all_tools_data = [
        # --- ACT 0 ---
        {'name': 'TPP & CQA Cascade', 'act': 0, 'year': 2009, 'inventor': 'ICH Q8', 'desc': 'Defines the "golden thread" of Quality by Design.'},
        {'name': 'Analytical Target Profile', 'act': 0, 'year': 2012, 'inventor': 'FDA/AAPS', 'desc': 'Creates the "contract" for a new analytical method.'},
        {'name': 'Quality Risk Management (QRM) Suite', 'act': 0, 'year': 1949, 'inventor': 'US Military', 'desc': 'Proactively identifies and mitigates process risks (FMEA, FTA, etc.).'},
        {'name': 'Design for Excellence (DfX)', 'act': 0, 'year': 1980, 'inventor': 'Concurrent Engineering', 'desc': 'Optimizing design for the entire product lifecycle.'},
        {'name': 'Validation Master Plan', 'act': 0, 'year': 1990, 'inventor': 'GAMP Forum', 'desc': 'The master project plan for any validation effort.'},
        {'name': 'Requirements Traceability Matrix', 'act': 0, 'year': 1980, 'inventor': 'Systems Engineering', 'desc': 'Ensures all requirements are built and tested.'},
        # --- ACT I ---
        {'name': 'Exploratory Data Analysis (EDA)', 'act': 1, 'year': 1977, 'inventor': 'John Tukey', 'desc': 'The critical first step of understanding any dataset.'},
        {'name': 'Confidence Interval Concept', 'act': 1, 'year': 1937, 'inventor': 'Jerzy Neyman', 'desc': 'Neyman formalizes the frequentist confidence interval.'},
        {'name': 'Confidence Intervals for Proportions', 'act': 1, 'year': 1927, 'inventor': 'Edwin B. Wilson', 'desc': 'Wilson develops a superior confidence interval for pass/fail data.'},
        {'name': 'Core Validation Parameters', 'act': 1, 'year': 1980, 'inventor': 'ICH / FDA', 'desc': 'Accuracy, Precision, Specificity codified.'},
        {'name': 'LOD & LOQ', 'act': 1, 'year': 1968, 'inventor': 'Lloyd Currie (NIST)', 'desc': 'Currie at NIST formalizes the statistical basis.'},
        {'name': 'Linearity & Range', 'act': 1, 'year': 1805, 'inventor': 'Legendre/Gauss', 'desc': 'Using linear regression to verify proportionality.'},
        {'name': 'Non-Linear Regression (4PL/5PL)', 'act': 1, 'year': 1975, 'inventor': 'Bioassay Field', 'desc': 'Models for sigmoidal curves common in immunoassays.'},
        {'name': 'Gage R&R / VCA', 'act': 1, 'year': 1982, 'inventor': 'AIAG', 'desc': 'AIAG codifies Measurement Systems Analysis (MSA).'},
        {'name': 'Attribute Agreement Analysis', 'act': 1, 'year': 1960, 'inventor': 'Cohen/Fleiss', 'desc': 'Validating human inspectors for pass/fail decisions.'},
        {'name': 'Comprehensive Diagnostic Validation', 'act': 1, 'year': 1950, 'inventor': 'Multi-Disciplinary', 'desc': 'A full suite of metrics for validating IVDs.'},
        {'name': 'ROC Curve Analysis', 'act': 1, 'year': 1945, 'inventor': 'Signal Processing Labs', 'desc': 'Developed for radar, now the standard for diagnostic tests.'},
        {'name': 'Assay Robustness (DOE)', 'act': 1, 'year': 1926, 'inventor': 'R.A. Fisher', 'desc': 'Fisher publishes his work on Design of Experiments.'},
        {'name': 'Mixture Design (Formulations)', 'act': 1, 'year': 1958, 'inventor': 'Henry Scheffé', 'desc': 'Specialized DOE for optimizing formulations and blends.'},
        {'name': 'Process Optimization: From DOE to AI', 'act': 1, 'year': 2017, 'inventor': 'Modern Synthesis', 'desc': 'Combining classic DOE with modern ML for deep optimization.'},
        {'name': 'Split-Plot Designs', 'act': 1, 'year': 1930, 'inventor': 'R.A. Fisher & F. Yates', 'desc': 'Specialized DOE for factors that are "hard-to-change".'},
        {'name': 'Causal Inference', 'act': 1, 'year': 2018, 'inventor': 'Judea Pearl et al.', 'desc': 'Moving beyond correlation to identify root causes.'},
        # --- ACT II ---
        {'name': 'Sample Size for Qualification', 'act': 2, 'year': 1940, 'inventor': 'Dodge/Romig', 'desc': 'Statistically justifying the number of samples for validation.'},
        {'name': 'Advanced Stability Design', 'act': 2, 'year': 2003, 'inventor': 'ICH Q1D', 'desc': 'Using Bracketing & Matrixing to create efficient stability studies.'},
        {'name': 'Method Comparison', 'act': 2, 'year': 1986, 'inventor': 'Bland & Altman', 'desc': 'Bland & Altman revolutionize method agreement studies.'},
        {'name': 'Equivalence Testing (TOST)', 'act': 2, 'year': 1987, 'inventor': 'Donald Schuirmann', 'desc': 'Schuirmann proposes TOST for bioequivalence.'},
        {'name': 'Statistical Equivalence for Process Transfer', 'act': 2, 'year': 1990, 'inventor': 'Modern Synthesis', 'desc': 'Proving two processes perform equivalently after a transfer.'},
        {'name': 'Process Stability (SPC)', 'act': 2, 'year': 1924, 'inventor': 'Walter Shewhart', 'desc': 'Shewhart invents the control chart at Bell Labs.'},
        {'name': 'Process Capability (Cpk)', 'act': 2, 'year': 1986, 'inventor': 'Bill Smith (Motorola)', 'desc': 'Motorola popularizes Cpk with the Six Sigma initiative.'},
        {'name': 'First Time Yield & Cost of Quality', 'act': 2, 'year': 1980, 'inventor': 'Six Sigma/TQM', 'desc': 'Quantifying the business impact of process performance.'},
        {'name': 'Tolerance Intervals', 'act': 2, 'year': 1942, 'inventor': 'Abraham Wald', 'desc': 'Wald develops intervals to cover a proportion of a population.'},
        {'name': 'Bayesian Inference', 'act': 2, 'year': 1990, 'inventor': 'Metropolis et al.', 'desc': 'Computational methods (MCMC) make Bayes practical.'},
        # --- ACT III ---
        {'name': 'Process Control Plan Builder', 'act': 3, 'year': 1980, 'inventor': 'Automotive Industry', 'desc': 'Creating the operational playbook for process monitoring.'},
        {'name': 'Run Validation (Westgard)', 'act': 3, 'year': 1981, 'inventor': 'James Westgard', 'desc': 'Westgard publishes his multi-rule QC system.'},
        {'name': 'Small Shift Detection', 'act': 3, 'year': 1954, 'inventor': 'Page/Roberts', 'desc': 'Charts for faster detection of small process drifts.'},
        {'name': 'Multivariate SPC', 'act': 3, 'year': 1931, 'inventor': 'Harold Hotelling', 'desc': 'Hotelling develops the multivariate analog to the t-test.'},
        {'name': 'Stability Analysis (Shelf-Life)', 'act': 3, 'year': 1993, 'inventor': 'ICH', 'desc': 'ICH guidelines formalize statistical shelf-life estimation.'},
        {'name': 'Reliability / Survival Analysis', 'act': 3, 'year': 1958, 'inventor': 'Kaplan & Meier', 'desc': 'Kaplan-Meier estimator for time-to-event data.'},
        {'name': 'Time Series Analysis', 'act': 3, 'year': 1970, 'inventor': 'Box & Jenkins', 'desc': 'Box & Jenkins publish their seminal work on ARIMA models.'},
        {'name': 'Multivariate Analysis (MVA)', 'act': 3, 'year': 1975, 'inventor': 'Herman Wold', 'desc': 'Partial Least Squares for modeling complex process data.'},
        {'name': 'Predictive QC (Classification)', 'act': 3, 'year': 1958, 'inventor': 'David Cox', 'desc': 'Cox develops Logistic Regression for binary outcomes.'},
        {'name': 'Explainable AI (XAI)', 'act': 3, 'year': 2017, 'inventor': 'Lundberg et al.', 'desc': 'Methods like SHAP to open the AI "black box".'},
        {'name': 'Clustering (Unsupervised)', 'act': 3, 'year': 1957, 'inventor': 'Stuart Lloyd', 'desc': 'Algorithm for finding hidden groups in data.'},
        {'name': 'Anomaly Detection', 'act': 3, 'year': 2008, 'inventor': 'Liu et al.', 'desc': 'Using Isolation Forests to find novel failures.'},
        {'name': 'Advanced AI Concepts', 'act': 3, 'year': 2017, 'inventor': 'Vaswani et al.', 'desc': 'Transformers and other advanced architectures emerge.'},
        {'name': 'MEWMA + XGBoost Diagnostics', 'act': 3, 'year': 1992, 'inventor': 'Lowry et al.', 'desc': 'Multivariate EWMA for sensitive drift detection, enhanced with modern AI for diagnosis.'},
        {'name': 'BOCPD + ML Features', 'act': 3, 'year': 2007, 'inventor': 'Adams & MacKay', 'desc': 'Probabilistic real-time detection of process changes (changepoints).'},
        {'name': 'Kalman Filter + Residual Chart', 'act': 3, 'year': 1960, 'inventor': 'Rudolf E. Kálmán', 'desc': 'Optimal state estimation for dynamic systems, used for intelligent fault detection.'},
        {'name': 'RL for Chart Tuning', 'act': 3, 'year': 2005, 'inventor': 'RL Community', 'desc': 'Using AI to economically optimize control chart parameters, balancing risk and cost.'},
        {'name': 'TCN + CUSUM', 'act': 3, 'year': 2018, 'inventor': 'Bai, Kolter & Koltun', 'desc': 'Hybrid model using AI to de-seasonalize data for ultra-sensitive drift detection.'},
        {'name': 'LSTM Autoencoder + Hybrid Monitoring', 'act': 3, 'year': 1997, 'inventor': 'Hochreiter/Schmidhuber', 'desc': 'Unsupervised anomaly detection by learning a process\'s normal dynamic fingerprint.'},
    ]
    all_tools_data.sort(key=lambda x: (x['act'], x['year']))
    act_ranges = {0: (-5, 20), 1: (25, 65), 2: (70, 90), 3: (95, 140)}
    tools_by_act = {0: [], 1: [], 2: [], 3: []}
    for tool in all_tools_data: tools_by_act[tool['act']].append(tool)
    for act_num, tools_in_act in tools_by_act.items():
        start, end = act_ranges[act_num]
        x_coords = np.linspace(start, end, len(tools_in_act))
        for i, tool in enumerate(tools_in_act):
            tool['x'] = x_coords[i]
    y_offsets = [3.0, -3.0, 3.5, -3.5, 2.5, -2.5, 4.0, -4.0, 2.0, -2.0, 4.5, -4.5, 1.5, -1.5]
    for i, tool in enumerate(all_tools_data):
        tool['y'] = y_offsets[i % len(y_offsets)]
    
    fig = go.Figure()
    acts = {
        0: {'name': 'Act 0: Planning & Strategy', 'color': 'rgba(128, 128, 128, 0.9)', 'boundary': (-10, 23)},
        1: {'name': 'Act I: Characterization', 'color': 'rgba(0, 128, 128, 0.9)', 'boundary': (23, 68)},
        2: {'name': 'Act II: Qualification & Transfer', 'color': 'rgba(0, 104, 201, 0.9)', 'boundary': (68, 93)},
        3: {'name': 'Act III: Lifecycle Management', 'color': 'rgba(100, 0, 100, 0.9)', 'boundary': (93, 145)}
    }
    
    for act_info in acts.values():
        x0, x1 = act_info['boundary']
        fig.add_shape(type="rect", x0=x0, y0=-5.0, x1=x1, y1=5.0, line=dict(width=0), fillcolor='rgba(230, 230, 230, 0.7)', layer='below')
        fig.add_annotation(x=(x0 + x1) / 2, y=7.0, text=f"<b>{act_info['name']}</b>", showarrow=False, font=dict(size=20, color="#555"))

    fig.add_shape(type="line", x0=-5, y0=0, x1=140, y1=0, line=dict(color="black", width=3), layer='below')

    for act_num, act_info in acts.items():
        act_tools = [tool for tool in all_tools_data if tool['act'] == act_num]
        fig.add_trace(go.Scatter(x=[tool['x'] for tool in act_tools], y=[tool['y'] for tool in act_tools], mode='markers',
                                 marker=dict(size=12, color=act_info['color'], symbol='circle', line=dict(width=2, color='black')),
                                 hoverinfo='text', text=[f"<b>{tool['name']} ({tool['year']})</b><br><i>{tool['desc']}</i>" for tool in act_tools], name=act_info['name']))

    for tool in all_tools_data:
        fig.add_shape(type="line", x0=tool['x'], y0=0, x1=tool['x'], y1=tool['y'], line=dict(color='grey', width=1))
        fig.add_annotation(x=tool['x'], y=tool['y'], text=f"<b>{tool['name'].replace(': ', ':<br>')}</b><br><i>({tool['year']})</i>",
                           showarrow=False, yshift=40 if tool['y'] > 0 else -40, font=dict(size=11, color=acts[tool['act']]['color']), align="center")

    fig.update_layout(title_text='<b>The V&V Analytics Toolkit: A Project-Based View</b>', title_font_size=28, title_x=0.5,
                      xaxis=dict(visible=False), yaxis=dict(visible=False, range=[-8, 8]), plot_bgcolor='white', paper_bgcolor='white',
                      height=900, margin=dict(l=20, r=20, t=140, b=20), showlegend=True,
                      legend=dict(title_text="<b>Project Phase</b>", title_font_size=16, font_size=14, orientation="h",
                                  yanchor="bottom", y=1.02, xanchor="center", x=0.5))
    return fig

# FIX: Replace the entire plot_chronological_timeline function with this new, complete version.
@st.cache_data
def plot_chronological_timeline():
    # SME Note: Added all new tools with their historical context and 'reason for invention'.
    all_tools_data = [
        {'name': 'Linearity & Range', 'year': 1805, 'inventor': 'Legendre/Gauss', 'reason': 'To predict the orbits of celestial bodies from a limited number of observations.'},
        {'name': 'Process Stability', 'year': 1924, 'inventor': 'Walter Shewhart', 'reason': 'The dawn of mass manufacturing (telephones) required new methods for controlling process variation.'},
        {'name': 'Assay Robustness (DOE)', 'year': 1926, 'inventor': 'R.A. Fisher', 'reason': 'To revolutionize agricultural science by efficiently testing multiple factors (fertilizers, varieties) at once.'},
        {'name': 'Confidence Intervals for Proportions', 'year': 1927, 'inventor': 'Edwin B. Wilson', 'reason': 'To solve the poor performance of the standard binomial confidence interval, especially for small samples.'},
        {'name': 'Split-Plot Designs', 'year': 1930, 'inventor': 'R.A. Fisher & F. Yates', 'reason': 'To solve agricultural experiments with factors that were difficult or expensive to change on a small scale.'},
        {'name': 'Multivariate SPC', 'year': 1931, 'inventor': 'Harold Hotelling', 'reason': 'To generalize the t-test and control charts to monitor multiple correlated variables simultaneously.'},
        {'name': 'Confidence Interval Concept', 'year': 1937, 'inventor': 'Jerzy Neyman', 'reason': 'A need for rigorous, objective methods in the growing field of mathematical statistics.'},
        {'name': 'Sample Size for Qualification', 'year': 1940, 'inventor': 'Dodge & Romig', 'reason': 'WWII demanded a statistical basis for accepting or rejecting massive lots of military supplies.'},
        {'name': 'Tolerance Intervals', 'year': 1942, 'inventor': 'Abraham Wald', 'reason': 'WWII demanded mass production of interchangeable military parts that would fit together reliably.'},
        {'name': 'ROC Curve Analysis', 'year': 1945, 'inventor': 'Signal Processing Labs', 'reason': 'Developed during WWII to distinguish enemy radar signals from noise, a classic signal detection problem.'},
        {'name': 'Quality Risk Management (FMEA)', 'year': 1949, 'inventor': 'US Military', 'reason': 'To proactively assess and mitigate reliability risks in complex military systems.'},
        {'name': 'Comprehensive Diagnostic Validation', 'year': 1950, 'inventor': 'Multi-Disciplinary', 'reason': 'The post-war boom in epidemiology required a full suite of metrics to validate new disease screening tests.'},
        {'name': 'Assay Robustness (RSM)', 'year': 1951, 'inventor': 'Box & Wilson', 'reason': 'The post-war chemical industry boom created demand for efficient process optimization techniques.'},
        {'name': 'Small Shift Detection', 'year': 1954, 'inventor': 'Page (CUSUM) & Roberts (EWMA)', 'reason': 'Maturing industries required charts more sensitive to small, slow process drifts than Shewhart\'s original design.'},
        {'name': 'Clustering (Unsupervised)', 'year': 1957, 'inventor': 'Stuart Lloyd', 'reason': 'The advent of early digital computing at Bell Labs made iterative, data-driven grouping algorithms feasible.'},
        {'name': 'Predictive QC (Classification)', 'year': 1958, 'inventor': 'David Cox', 'reason': 'A need to model binary outcomes (pass/fail, live/die) in a regression framework.'},
        {'name': 'Reliability / Survival Analysis', 'year': 1958, 'inventor': 'Kaplan & Meier', 'reason': 'The rise of clinical trials necessitated a formal way to handle \'censored\' data.'},
        {'name': 'Mixture Design (Formulations)', 'year': 1958, 'inventor': 'Henry Scheffé', 'reason': 'To provide a systematic way for chemists and food scientists to optimize recipes and formulations.'},
        {'name': 'Attribute Agreement Analysis', 'year': 1960, 'inventor': 'Cohen/Fleiss', 'reason': 'Psychologists needed to measure the reliability of judgments between raters, corrected for chance agreement.'},
        {'name': 'Kalman Filter + Residual Chart', 'year': 1960, 'inventor': 'Rudolf E. Kálmán', 'reason': 'The Apollo program needed a way to navigate to the moon using noisy sensor data, requiring optimal state estimation.'},
        {'name': 'LOD & LOQ', 'year': 1968, 'inventor': 'Lloyd Currie (NIST)', 'reason': 'To create a harmonized, statistically rigorous framework for defining the sensitivity of analytical methods.'},
        {'name': 'Time Series Analysis', 'year': 1970, 'inventor': 'Box & Jenkins', 'reason': 'To provide a comprehensive statistical methodology for forecasting and control in industrial and economic processes.'},
        {'name': 'Multivariate Analysis (MVA)', 'year': 1975, 'inventor': 'Herman Wold', 'reason': 'To model "data-rich but theory-poor" systems in social science, later adapted for chemometrics.'},
        {'name': 'Core Validation Parameters', 'year': 1980, 'inventor': 'ICH / FDA', 'reason': 'Globalization of the pharmaceutical industry required harmonized standards for drug approval.'},
        {'name': 'Run Validation (Westgard)', 'year': 1981, 'inventor': 'James Westgard', 'reason': 'The automation of clinical labs demanded a more sensitive, diagnostic system for daily quality control.'},
        {'name': 'Gage R&R / VCA', 'year': 1982, 'inventor': 'AIAG', 'reason': 'The US auto industry, facing a quality crisis, needed to formalize the analysis of their measurement systems.'},
        {'name': 'Method Comparison', 'year': 1986, 'inventor': 'Bland & Altman', 'reason': 'A direct response to the widespread misuse of correlation for comparing clinical measurement methods.'},
        {'name': 'Process Capability (Cpk)', 'year': 1986, 'inventor': 'Bill Smith (Motorola)', 'reason': 'The Six Sigma quality revolution at Motorola popularized a simple metric to quantify process capability.'},
        {'name': 'Equivalence Testing (TOST)', 'year': 1987, 'inventor': 'Donald Schuirmann', 'reason': 'The rise of the generic drug industry created a regulatory need to statistically *prove* equivalence.'},
        {'name': 'Validation Master Plan', 'year': 1990, 'inventor': 'GAMP Forum', 'reason': 'Increasingly complex computerized systems required a high-level strategic plan for validation.'},
        {'name': 'Bayesian Inference', 'year': 1990, 'inventor': 'Metropolis et al.', 'reason': 'The explosion in computing power made simulation-based methods (MCMC) practical, unlocking Bayesian inference.'},
        {'name': 'MEWMA + XGBoost Diagnostics', 'year': 1992, 'inventor': 'Lowry et al.', 'reason': 'A need to generalize the sensitive EWMA chart to monitor multiple correlated variables at once.'},
        {'name': 'Stability Analysis (Shelf-Life)', 'year': 1993, 'inventor': 'ICH', 'reason': 'To harmonize global pharmaceutical regulations for determining a product\'s shelf-life.'},
        {'name': 'LSTM Autoencoder + Hybrid Monitoring', 'year': 1997, 'inventor': 'Hochreiter/Schmidhuber', 'reason': 'A need to model long-range temporal dependencies in data, later adapted for unsupervised anomaly detection.'},
        {'name': 'Advanced Stability Design', 'year': 2003, 'inventor': 'ICH Q1D', 'reason': 'To provide a risk-based statistical framework for reducing the cost of complex stability studies.'},
        {'name': 'RL for Chart Tuning', 'year': 2005, 'inventor': 'RL Community', 'reason': 'A desire to move beyond purely statistical chart design to an economically optimal framework balancing risk and cost.'},
        {'name': 'BOCPD + ML Features', 'year': 2007, 'inventor': 'Adams & MacKay', 'reason': 'A need for a more robust, probabilistic method for detecting changepoints in real-time streaming data.'},
        {'name': 'Anomaly Detection', 'year': 2008, 'inventor': 'Liu et al.', 'reason': 'A need for a fast, efficient algorithm (Isolation Forest) to find outliers in high-dimensional data.'},
        {'name': 'TPP & CQA Cascade', 'year': 2009, 'inventor': 'ICH Q8', 'reason': 'The Quality by Design movement required a formal framework to link patient needs to process controls.'},
        {'name': 'Analytical Target Profile', 'year': 2012, 'inventor': 'FDA/AAPS', 'reason': 'To extend QbD principles to the lifecycle of analytical methods, defining a "contract" for method performance.'},
        {'name': 'Explainable AI (XAI)', 'year': 2017, 'inventor': 'Lundberg et al.', 'reason': 'The rise of powerful but opaque "black box" models necessitated methods to explain their reasoning (XAI).'},
        {'name': 'Advanced AI Concepts', 'year': 2017, 'inventor': 'Vaswani et al.', 'reason': 'The Deep Learning revolution produced powerful new architectures like Transformers for sequence modeling.'},
        {'name': 'TCN + CUSUM', 'year': 2018, 'inventor': 'Bai, Kolter & Koltun', 'reason': 'A need for a faster, more effective deep learning architecture for sequence modeling to rival LSTMs.'},
        {'name': 'Causal Inference', 'year': 2018, 'inventor': 'Judea Pearl et al.', 'reason': 'The limitations of purely predictive models spurred a "causal revolution" to answer "why" questions.'},
    ]
    all_tools_data.sort(key=lambda x: x['year'])
    y_offsets = [3.0, -3.0, 3.5, -3.5, 2.5, -2.5, 4.0, -4.0, 2.0, -2.0, 4.5, -4.5, 1.5, -1.5]
    for i, tool in enumerate(all_tools_data):
        tool['y'] = y_offsets[i % len(y_offsets)]
    
    fig = go.Figure()
    eras = {
        'The Foundations (1920-1949)': {'color': 'rgba(0, 128, 128, 0.7)', 'boundary': (1920, 1949)},
        'Post-War & Industrial Boom (1950-1979)': {'color': 'rgba(0, 104, 201, 0.7)', 'boundary': (1950, 1979)},
        'The Quality Revolution (1980-1999)': {'color': 'rgba(100, 0, 100, 0.7)', 'boundary': (1980, 1999)},
        'The AI & Data Era (2000-Present)': {'color': 'rgba(214, 39, 40, 0.7)', 'boundary': (2000, 2025)}
    }
    
    for era_name, era_info in eras.items():
        x0, x1 = era_info['boundary']
        fig.add_shape(type="rect", x0=x0, y0=-5.5, x1=x1, y1=5.5, line=dict(width=0), fillcolor=era_info['color'], opacity=0.15, layer='below')
        fig.add_annotation(x=(x0 + x1) / 2, y=6.5, text=f"<b>{era_name}</b>", showarrow=False, font=dict(size=18, color=era_info['color']))

    fig.add_shape(type="line", x0=1920, y0=0, x1=2025, y1=0, line=dict(color="black", width=3), layer='below')

    for tool in all_tools_data:
        x_coord, y_coord = tool['year'], tool['y']
        tool_color = next((era['color'] for era in eras.values() if era['boundary'][0] <= x_coord <= era['boundary'][1]), 'grey')
        fig.add_trace(go.Scatter(x=[x_coord], y=[y_coord], mode='markers', marker=dict(size=12, color=tool_color, line=dict(width=2, color='black')),
                                 hoverinfo='text', text=f"<b>{tool['name']} ({tool['year']})</b><br><i>Inventor(s): {tool['inventor']}</i><br><br><b>Reason:</b> {tool['reason']}"))
        fig.add_shape(type="line", x0=x_coord, y0=0, x1=x_coord, y1=y_coord, line=dict(color='grey', width=1))
        fig.add_annotation(x=x_coord, y=y_coord, text=f"<b>{tool['name']}</b>", showarrow=False, yshift=25 if y_coord > 0 else -25, font=dict(size=11, color=tool_color), align="center")

    fig.update_layout(title_text='<b>A Chronological Timeline of V&V Analytics</b>', title_font_size=28, title_x=0.5,
                      xaxis=dict(range=[1920, 2025], showgrid=True), yaxis=dict(visible=False, range=[-8, 8]),
                      plot_bgcolor='white', paper_bgcolor='white', height=700, margin=dict(l=20, r=20, t=100, b=20), showlegend=False)
    return fig

# FIX: Replace the entire create_toolkit_conceptual_map function with this new, complete version.
@st.cache_data
def create_toolkit_conceptual_map():
    # SME Note: Completely re-architected the map to reflect the new four-act structure and all tools.
    structure = {
        'Validation Planning & Strategy': ['Risk Management', 'Requirements Definition', 'Design Principles'],
        'Method & Process Characterization': ['Foundational Statistics', 'Measurement Systems Analysis', 'Experimental Design'],
        'Process & Lifecycle Management': ['Statistical Process Control', 'Validation & Qualification'],
        'Advanced Analytics (ML/AI)': ['Predictive Modeling', 'Unsupervised Learning', 'Time Series & Sequential']
    }
    sub_structure = {
        'Risk Management': ['Quality Risk Management (QRM) Suite'],
        'Requirements Definition': ['TPP & CQA Cascade', 'Analytical Target Profile (ATP) Builder', 'Requirements Traceability Matrix (RTM)'],
        'Design Principles': ['Design for Excellence (DfX)'],
        'Foundational Statistics': ['Confidence Interval Concept', 'Confidence Intervals for Proportions', 'Bayesian Inference', 'Comprehensive Diagnostic Validation'],
        'Measurement Systems Analysis': ['Gage R&R / VCA', 'Attribute Agreement Analysis', 'Method Comparison', 'LOD & LOQ'],
        'Experimental Design': ['Assay Robustness (DOE)', 'Process Optimization: From DOE to AI', 'Mixture Design (Formulations)', 'Split-Plot Designs', 'Causal Inference'],
        'Statistical Process Control': ['Process Stability (SPC)', 'Small Shift Detection', 'Multivariate SPC', 'MEWMA + XGBoost Diagnostics', 'Run Validation (Westgard)'],
        'Validation & Qualification': ['Process Capability (Cpk)', 'Tolerance Intervals', 'Reliability / Survival Analysis', 'Stability Analysis (Shelf-Life)', 'Sample Size for Qualification', 'Statistical Equivalence for Process Transfer', 'Advanced Stability Design', 'First Time Yield & Cost of Quality'],
        'Predictive Modeling': ['Linearity & Range', 'Non-Linear Regression (4PL/5PL)', 'Multivariate Analysis (MVA)', 'Predictive QC (Classification)', 'Explainable AI (XAI)'],
        'Unsupervised Learning': ['Clustering (Unsupervised)', 'Anomaly Detection', 'LSTM Autoencoder'],
        'Time Series & Sequential': ['Time Series Analysis', 'BOCPD + ML Features', 'Kalman Filter + Residual Chart', 'TCN + CUSUM', 'RL for Chart Tuning', 'Advanced AI Concepts']
    }
    tool_origins = {
        'TPP & CQA Cascade': 'Biostatistics', 'Analytical Target Profile (ATP) Builder': 'Biostatistics', 'Quality Risk Management (QRM) Suite': 'Industrial Quality Control', 'Design for Excellence (DfX)': 'Industrial Quality Control', 'Validation Master Plan (VMP) Builder': 'Industrial Quality Control', 'Requirements Traceability Matrix (RTM)': 'Industrial Quality Control',
        'Confidence Interval Concept': 'Statistics', 'Confidence Intervals for Proportions': 'Statistics', 'Equivalence Testing (TOST)': 'Biostatistics', 'Bayesian Inference': 'Statistics', 'ROC Curve Analysis': 'Statistics', 'Comprehensive Diagnostic Validation': 'Biostatistics',
        'Gage R&R / VCA': 'Industrial Quality Control', 'Attribute Agreement Analysis': 'Statistics', 'Method Comparison': 'Biostatistics', 'LOD & LOQ': 'Statistics',
        'Assay Robustness (DOE)': 'Statistics', 'Process Optimization: From DOE to AI': 'Data Science / ML', 'Mixture Design (Formulations)': 'Statistics', 'Split-Plot Designs': 'Statistics', 'Causal Inference': 'Data Science / ML',
        'Process Stability (SPC)': 'Industrial Quality Control', 'Small Shift Detection': 'Industrial Quality Control', 'Multivariate SPC': 'Industrial Quality Control', 'MEWMA + XGBoost Diagnostics': 'Data Science / ML', 'Run Validation (Westgard)': 'Biostatistics',
        'Process Capability (Cpk)': 'Industrial Quality Control', 'Tolerance Intervals': 'Statistics', 'Reliability / Survival Analysis': 'Biostatistics', 'Stability Analysis (Shelf-Life)': 'Biostatistics', 'Sample Size for Qualification': 'Industrial Quality Control', 'Statistical Equivalence for Process Transfer': 'Biostatistics', 'Advanced Stability Design': 'Biostatistics', 'First Time Yield & Cost of Quality': 'Industrial Quality Control',
        'Linearity & Range': 'Statistics', 'Non-Linear Regression (4PL/5PL)': 'Biostatistics', 'Multivariate Analysis (MVA)': 'Data Science / ML', 'Predictive QC (Classification)': 'Data Science / ML', 'Explainable AI (XAI)': 'Data Science / ML',
        'Clustering (Unsupervised)': 'Data Science / ML', 'Anomaly Detection': 'Data Science / ML', 'LSTM Autoencoder': 'Data Science / ML',
        'Time Series Analysis': 'Statistics', 'BOCPD + ML Features': 'Data Science / ML', 'Kalman Filter + Residual Chart': 'Statistics', 'TCN + CUSUM': 'Data Science / ML', 'RL for Chart Tuning': 'Data Science / ML', 'Advanced AI Concepts': 'Data Science / ML'
    }
    origin_colors = {'Statistics': '#1f77b4', 'Biostatistics': '#2ca02c', 'Industrial Quality Control': '#ff7f0e', 'Data Science / ML': '#d62728', 'Structure': '#6A5ACD'}

    nodes = {}
    vertical_spacing = 1.8
    all_tools_flat = [tool for sublist in sub_structure.values() for tool in sublist]
    y_coords = np.linspace(len(all_tools_flat) * vertical_spacing, -len(all_tools_flat) * vertical_spacing, len(all_tools_flat))
    x_positions = [4, 5]
    for i, tool_key in enumerate(all_tools_flat):
        short_name = tool_key.replace(' +', '<br>+').replace(' (', '<br>(').replace('Comprehensive ', 'Comprehensive<br>').replace(': From', ':<br>From')
        nodes[tool_key] = {'x': x_positions[i % 2], 'y': y_coords[i], 'name': tool_key, 'short': short_name, 'origin': tool_origins.get(tool_key)}

    for l2_key, l3_keys in sub_structure.items():
        child_ys = [nodes[child_key]['y'] for child_key in l3_keys]
        nodes[l2_key] = {'x': 2.5, 'y': np.mean(child_ys), 'name': l2_key, 'short': l2_key.replace(' ', '<br>'), 'origin': 'Structure'}

    for l1_key, l2_keys in structure.items():
        child_ys = [nodes[child_key]['y'] for child_key in l2_keys]
        nodes[l1_key] = {'x': 1, 'y': np.mean(child_ys), 'name': l1_key, 'short': l1_key.replace(' ', '<br>').replace('Validation Planning', 'Validation<br>Planning'), 'origin': 'Structure'}

    nodes['CENTER'] = {'x': -0.5, 'y': 0, 'name': 'V&V Analytics Toolkit', 'short': 'V&V Analytics<br>Toolkit', 'origin': 'Structure'}
    
    fig = go.Figure()
    all_edges = [('CENTER', l1) for l1 in structure.keys()] + \
                [(l1, l2) for l1, l2s in structure.items() for l2 in l2s] + \
                [(l2, l3) for l2, l3s in sub_structure.items() for l3 in l3s]
    for start_key, end_key in all_edges:
        fig.add_shape(type="line", x0=nodes[start_key]['x'], y0=nodes[start_key]['y'],
                      x1=nodes[end_key]['x'], y1=nodes[end_key]['y'], line=dict(color="lightgrey", width=1.5))
    
    data_by_origin = {name: {'x': [], 'y': [], 'short': [], 'full': [], 'size': [], 'font_size': []} for name in origin_colors.keys()}
    size_map = {'CENTER': 150, 'Level1': 130, 'Level2': 110, 'Tool': 90}
    font_map = {'CENTER': 16, 'Level1': 14, 'Level2': 12, 'Tool': 11}

    for key, data in nodes.items():
        if key == 'CENTER': level = 'CENTER'
        elif key in structure: level = 'Level1'
        elif key in sub_structure: level = 'Level2'
        else: level = 'Tool'
        if data['origin']: # Check if origin exists
            data_by_origin[data['origin']]['x'].append(data['x'])
            data_by_origin[data['origin']]['y'].append(data['y'])
            data_by_origin[data['origin']]['short'].append(data['short'])
            data_by_origin[data['origin']]['full'].append(data['name'])
            data_by_origin[data['origin']]['size'].append(size_map[level])
            data_by_origin[data['origin']]['font_size'].append(font_map[level])
        
    for origin_name, data in data_by_origin.items():
        if not data['x']: continue
        is_structure = origin_name == 'Structure'
        fig.add_trace(go.Scatter(
            x=data['x'], y=data['y'], text=data['short'], mode='markers+text', textposition="middle center",
            marker=dict(size=data['size'], color=origin_colors[origin_name], symbol='circle',
                        line=dict(width=2, color='black' if not is_structure else origin_colors[origin_name])),
            textfont=dict(size=data['font_size'], color='white', family="Arial, sans-serif"),
            hovertext=[f"<b>{name}</b><br>Origin: {origin_name}" for name in data['full']], hoverinfo='text',
            name=origin_name))

    fig.update_layout(
        title_text='<b>Conceptual Map of the V&V Analytics Toolkit</b>', showlegend=True,
        legend=dict(title="<b>Tool Origin</b>", x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.7)', bordercolor="Black", borderwidth=1),
        xaxis=dict(visible=False, range=[-1, 6]),
        yaxis=dict(visible=False, range=[-len(all_tools_flat)*1.2, len(all_tools_flat)*1.2]),
        height=len(all_tools_flat) * 65, # Adjusted height
        margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor='#FFFFFF', paper_bgcolor='#f0f2f6'
    )
    return fig
#==============================================================================================================================================================================================================
#=========================================================================================== ACT 0 BEGGINNG ===================================================================================================
#==============================================================================================================================================================================================================

@st.cache_data
def plot_tpp_cqa_cascade(project_type, target1_val, target2_val, target1_tag, target2_tag):
    """
    Generates a professional, interactive Sankey diagram for TPP -> CQA -> CPP cascade for multiple project types.
    """
    cascade_data = {
        "Monoclonal Antibody": {
            "TPP": "A safe, effective, and stable MAb therapeutic.", "CQAs": {"Purity > 99%": {"link": "Efficacy"}, "Aggregate < 1%": {"link": "Safety"}, "Potency 80-120%": {"link": "Efficacy"}, "Charge Variants": {"link": "Efficacy"}, "Stability": {"link": "Shelf-Life"}},
            "CPPs": {"Bioreactor pH": ["Potency 80-120%", "Charge Variants"], "Column Load": ["Purity > 99%", "Aggregate < 1%"], "Formulation Buffer": ["Stability"]}
        },
        "IVD Kit": {
            "TPP": "A reliable and accurate diagnostic kit.", "CQAs": {"Sensitivity > 98%": {"link": "Efficacy"}, "Specificity > 99%": {"link": "Efficacy"}, "Precision < 15% CV": {"link": "Reliability"}, "Shelf-Life": {"link": "Shelf-Life"}},
            "CPPs": {"Antibody Conc.": ["Sensitivity > 98%"], "Blocking Buffer": ["Specificity > 99%"], "Lyophilization Cycle": ["Shelf-Life"]}
        },
        "Pharma Process (Small Molecule)": {
            "TPP": "A robust and efficient process for Drug Substance X.", "CQAs": {"Impurity Profile < 0.1%": {"link": "Purity"}, "Process Yield > 85%": {"link": "Yield"}, "Correct Polymorph (Form II)": {"link": "Efficacy"}, "Particle Size (D90 < 20μm)": {"link": "Efficacy"}},
            "CPPs": {"Reaction Temperature": ["Impurity Profile < 0.1%"], "Reagent Stoichiometry": ["Process Yield > 85%"], "Crystallization Solvent": ["Correct Polymorph (Form II)"], "Milling Speed": ["Particle Size (D90 < 20μm)"]}
        },
        "Instrument Qualification": {
            "TPP": "A high-throughput, reliable liquid handler for automated sample prep.", "CQAs": {"Dispense Accuracy & Precision": {"link": "Reliability"}, "Throughput (Plates/hr)": {"link": "Throughput"}, "Cross-Contamination < 0.01%": {"link": "Reliability"}, "Uptime > 99%": {"link": "Reliability"}},
            "CPPs": {"Pump Calibration Algorithm": ["Dispense Accuracy & Precision"], "Robotic Arm Speed": ["Throughput (Plates/hr)"], "Tip Wash Protocol": ["Cross-Contamination < 0.01%"], "Preventive Maintenance": ["Uptime > 99%"]}
        },
        "Computer System Validation": {
            "TPP": "A 21 CFR Part 11 compliant LIMS for managing lab data.", "CQAs": {"Audit Trail Integrity": {"link": "Compliance"}, "User Access Control": {"link": "Compliance"}, "Report Generation Time < 5s": {"link": "Performance"}, "Data Backup & Recovery": {"link": "Compliance"}},
            "CPPs": {"Database Schema": ["Audit Trail Integrity"], "Role-Based Permission Matrix": ["User Access Control"], "Query Optimization": ["Report Generation Time < 5s"], "Replication Strategy": ["Data Backup & Recovery"]}
        }
    }
    
    data = cascade_data[project_type]
    
    labels = [f"<b>TPP:</b><br>{data['TPP']}"]
    labels.extend([f"<b>CQA:</b> {cqa}" for cqa in data['CQAs']])
    labels.extend([f"<b>CPP:</b> {cpp}" for cpp in data['CPPs']])
    node_colors = [PRIMARY_COLOR] + [SUCCESS_GREEN]*len(data['CQAs']) + ['#636EFA']*len(data['CPPs'])

    for i, (key, props) in enumerate(data['CQAs'].items()):
        is_active = (target1_tag in props['link'] and target1_val) or \
                    (target2_tag in props['link'] and target2_val)
        if is_active:
            node_colors[i + 1] = '#FFBF00'
            
    sources, targets, values = [], [], []
    tpp_idx = 0
    for i, cqa in enumerate(data['CQAs']):
        cqa_idx = labels.index(f"<b>CQA:</b> {cqa}")
        sources.append(tpp_idx)
        targets.append(cqa_idx)
        values.append(1)
    for cpp, cqa_links in data['CPPs'].items():
        cpp_idx = labels.index(f"<b>CPP:</b> {cpp}")
        for link in cqa_links:
            cqa_idx = labels.index(f"<b>CQA:</b> {link}")
            sources.append(cqa_idx)
            targets.append(cpp_idx)
            values.append(1)

    fig = go.Figure(data=[go.Sankey(
        node=dict(pad=25, thickness=20, line=dict(color="black", width=0.5), label=labels, color=node_colors),
        link=dict(source=sources, target=targets, value=values, color='rgba(200,200,200,0.5)')
    )])
    
    fig.update_layout(title_text="<b>The 'Golden Thread': TPP → CQA → CPP Cascade</b>", font_size=12, height=600)
    return fig

@st.cache_data
def plot_atp_radar_chart(project_type, atp_values, achieved_values=None):
    """
    Generates a professional-grade radar chart for an Analytical Target Profile,
    now supporting multiple project types with unique performance criteria.
    """
    profiles = {
        "Pharma Assay (HPLC)": {
            'categories': ['Accuracy<br>(%Rec)', 'Precision<br>(%CV)', 'Linearity<br>(R²)', 'Range<br>(Turn-down)', 'Sensitivity<br>(LOD)'],
            'ranges': [[95, 102], [5, 1], [0.99, 0.9999], [10, 100], [1, 50]],
            'direction': [1, -1, 1, 1, 1]
        },
        "IVD Kit (ELISA)": {
            'categories': ['Clinical<br>Sensitivity', 'Clinical<br>Specificity', 'Precision<br>(%CV)', 'Robustness', 'Shelf-Life<br>(Months)'],
            'ranges': [[90, 100], [90, 100], [20, 10], [1, 10], [6, 24]],
            'direction': [1, 1, -1, 1, 1]
        },
        "Instrument Qualification": {
            'categories': ['Accuracy<br>(Bias %)', 'Precision<br>(%CV)', 'Throughput<br>(Samples/hr)', 'Uptime<br>(%)', 'Footprint<br>(m²)'],
            'ranges': [[5, 0.1], [5, 0.5], [10, 200], [95, 99.9], [5, 1]],
            'direction': [-1, -1, 1, 1, -1]
        },
        "Software System (LIMS)": {
            'categories': ['Reliability<br>(Uptime %)', 'Performance<br>(Query Time)', 'Security<br>(Compliance)', 'Usability<br>(User Score)', 'Scalability<br>(Users)'],
            'ranges': [[99, 99.999], [10, 0.5], [1, 10], [1, 10], [50, 5000]],
            'direction': [1, -1, 1, 1, 1]
        },
        "Pharma Process (MAb)": {
            'categories': ['Yield<br>(g/L)', 'Purity<br>(%)', 'Process<br>Consistency', 'Robustness<br>(PAR Size)', 'Cycle Time<br>(Days)'],
            'ranges': [[1, 10], [98, 99.9], [1, 10], [1, 10], [20, 10]],
            'direction': [1, 1, 1, 1, -1]
        }
    }
    
    profile = profiles[project_type]
    categories = profile['categories']
    
    def normalize_value(val, val_range, direction):
        low, high = val_range
        if direction == -1: low, high = high, low
        scaled = ((val - low) / (high - low)) * 100 if (high - low) != 0 else 50
        return np.clip(scaled, 0, 100)

    scaled_atp = [normalize_value(atp_values[i], profile['ranges'][i], profile['direction'][i]) for i in range(len(categories))]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=scaled_atp, theta=categories, fill='toself', name='ATP (Target)',
        line=dict(color=PRIMARY_COLOR), fillcolor='rgba(0, 104, 201, 0.4)'
    ))

    if achieved_values:
        scaled_achieved = [normalize_value(achieved_values[i], profile['ranges'][i], profile['direction'][i]) for i in range(len(categories))]
        fig.add_trace(go.Scatterpolar(
            r=scaled_achieved, theta=categories, fill='toself', name='Achieved Performance',
            line=dict(color=SUCCESS_GREEN), fillcolor='rgba(44, 160, 44, 0.4)'
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], showticklabels=False, ticks='', gridcolor='lightgrey'),
            angularaxis=dict(tickfont=dict(size=12), linecolor='grey', gridcolor='lightgrey')
        ),
        showlegend=True,
        legend=dict(yanchor="bottom", y=-0.2, xanchor="center", x=0.5, orientation="h"),
        title=f"<b>Target Profile for {project_type}</b>",
        margin=dict(t=80)
    )
    
    return fig

@st.cache_data
def plot_pha_matrix(pha_data, project_type):
    """Generates a professional-grade Preliminary Hazard Analysis (PHA) risk matrix."""
    df = pd.DataFrame(pha_data)
    if df.empty:
        return go.Figure().update_layout(title_text="No PHA data available for this scenario.")
        
    fig = go.Figure()
    # Risk regions defined for a 5x4 matrix
    severity_levels = {'Negligible': 1, 'Minor': 2, 'Serious': 3, 'Critical': 4, 'Catastrophic': 5}
    likelihood_levels = {'Improbable': 1, 'Remote': 2, 'Occasional': 3, 'Frequent': 4}

    # Define risk regions with explicit coordinates for clarity
    # High Risk (Red)
    fig.add_shape(type="rect", x0=3.5, y0=2.5, x1=5.5, y1=4.5, fillcolor='rgba(239, 83, 80, 0.3)', line_width=0, layer='below')
    # Medium Risk (Yellow)
    fig.add_shape(type="rect", x0=0.5, y0=2.5, x1=3.5, y1=4.5, fillcolor='rgba(255, 193, 7, 0.2)', line_width=0, layer='below')
    fig.add_shape(type="rect", x0=3.5, y0=0.5, x1=5.5, y1=2.5, fillcolor='rgba(255, 193, 7, 0.2)', line_width=0, layer='below')
    # Low Risk (Green)
    fig.add_shape(type="rect", x0=0.5, y0=0.5, x1=3.5, y1=2.5, fillcolor='rgba(44, 160, 44, 0.2)', line_width=0, layer='below')

    fig.add_trace(go.Scatter(
        x=df['Severity'], y=df['Likelihood'], mode='markers+text',
        text=df.index + 1, textposition='top center',
        marker=dict(color=PRIMARY_COLOR, size=20, symbol='diamond'),
        hovertext='<b>Hazard:</b> ' + df['Hazard'] + '<br><b>Severity:</b> ' + df['Severity'].astype(str) + '<br><b>Likelihood:</b> ' + df['Likelihood'].astype(str),
        hoverinfo='text'
    ))
    
    fig.update_layout(
        title=f'<b>Preliminary Hazard Analysis (PHA) Risk Matrix for {project_type}</b>',
        xaxis_title='<b>Severity of Harm</b>', yaxis_title='<b>Likelihood of Occurrence</b>',
        xaxis=dict(tickvals=list(severity_levels.values()), ticktext=list(severity_levels.keys()), range=[0.5, 5.5]),
        yaxis=dict(tickvals=list(likelihood_levels.values()), ticktext=list(likelihood_levels.keys()), range=[0.5, 4.5]),
        margin=dict(l=40, r=40, b=40, t=60)
    )
    return fig

@st.cache_data
def plot_fmea_dashboard(fmea_df, project_type):
    """
    Generates a professional-grade FMEA dashboard with a Risk Matrix and Pareto Chart from a dataframe.
    This version is corrected to use a standard 1-10 scale for S, O, and D.
    """
    df = fmea_df.copy()
    if df.empty:
        return go.Figure().update_layout(title_text="No FMEA data."), go.Figure().update_layout(title_text="No FMEA data.")
        
    df['RPN_Initial'] = df['S'] * df['O_Initial'] * df['D_Initial']
    df['RPN_Final'] = df['S'] * df['O_Final'] * df['D_Final']
    
    # --- 1. Risk Matrix (S vs O) on a 10x10 grid ---
    fig_matrix = go.Figure()
    # Define risk regions for a 10x10 grid
    # High Risk (Red)
    fig_matrix.add_shape(type="rect", x0=7.5, y0=7.5, x1=10.5, y1=10.5, fillcolor='rgba(239, 83, 80, 0.3)', line_width=0, layer='below')
    # Medium Risk (Yellow)
    fig_matrix.add_shape(type="rect", x0=0.5, y0=7.5, x1=7.5, y1=10.5, fillcolor='rgba(255, 193, 7, 0.2)', line_width=0, layer='below')
    fig_matrix.add_shape(type="rect", x0=7.5, y0=0.5, x1=10.5, y1=7.5, fillcolor='rgba(255, 193, 7, 0.2)', line_width=0, layer='below')
    # Low Risk (Green)
    fig_matrix.add_shape(type="rect", x0=0.5, y0=0.5, x1=7.5, y1=7.5, fillcolor='rgba(44, 160, 44, 0.2)', line_width=0, layer='below')
    
    fig_matrix.add_trace(go.Scatter(x=df['S'], y=df['O_Initial'], mode='markers+text', text=df.index + 1, textposition='top center',
        marker=dict(color='black', size=15, symbol='circle'), name='Initial Risk', 
        hovertext='<b>Mode:</b> ' + df['Failure Mode'] + '<br><b>S:</b> ' + df['S'].astype(str) + ' | <b>O:</b> ' + df['O_Initial'].astype(str), hoverinfo='text'))
    
    fig_matrix.add_trace(go.Scatter(x=df['S'], y=df['O_Final'], mode='markers+text', text=df.index + 1, textposition='bottom center',
        marker=dict(color=PRIMARY_COLOR, size=15, symbol='diamond-open'), name='Final Risk', 
        hovertext='<b>Mode:</b> ' + df['Failure Mode'] + '<br><b>S:</b> ' + df['S'].astype(str) + ' | <b>O:</b> ' + df['O_Final'].astype(str), hoverinfo='text'))
    
    # Add arrows to show mitigation
    for i in df.index:
        if df.loc[i, 'O_Initial'] != df.loc[i, 'O_Final'] or df.loc[i, 'D_Initial'] != df.loc[i, 'D_Final']:
            fig_matrix.add_annotation(x=df.loc[i, 'S'], y=df.loc[i, 'O_Final'], ax=df.loc[i, 'S'], ay=df.loc[i, 'O_Initial'],
                                      xref='x', yref='y', axref='x', ayref='y', showarrow=True, arrowhead=2, arrowcolor=PRIMARY_COLOR, arrowwidth=2)
    
    fig_matrix.update_layout(title=f'<b>1. Risk Matrix (S vs. O) for {project_type}</b>',
                             xaxis_title='<b>Severity (S)</b>', yaxis_title='<b>Occurrence (O)</b>',
                             xaxis=dict(tickvals=list(range(1,11)), range=[0.5, 10.5]), yaxis=dict(tickvals=list(range(1,11)), range=[0.5, 10.5]),
                             legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    # --- 2. RPN Pareto Chart ---
    df_pareto = df[['Failure Mode', 'RPN_Initial', 'RPN_Final']].copy().sort_values('RPN_Initial', ascending=False)
    df_pareto = df_pareto.melt(id_vars='Failure Mode', var_name='Stage', value_name='RPN')
    df_pareto['Stage'] = df_pareto['Stage'].replace({'RPN_Initial': 'Initial', 'RPN_Final': 'Final'})
    
    fig_pareto = px.bar(df_pareto, x='Failure Mode', y='RPN', color='Stage', barmode='group',
                        title='<b>2. Risk Priority Number (RPN) Pareto</b>',
                        labels={'Failure Mode': 'Potential Failure Mode', 'RPN': 'Risk Priority Number (S x O x D)'},
                        color_discrete_map={'Initial': 'grey', 'Final': PRIMARY_COLOR})
    
    # Update action threshold for a 1-10 scale (max RPN = 1000)
    action_threshold = 100
    fig_pareto.add_hline(y=action_threshold, line_dash="dash", line_color="red", annotation_text="Action Threshold", annotation_position="bottom right")

    return fig_matrix, fig_pareto

@st.cache_data
def plot_fta_diagram(fta_data, project_type):
    """Generates a professional-grade Fault Tree Analysis (FTA) diagram from a data structure."""
    fig = go.Figure()
    nodes, links = fta_data['nodes'], fta_data['links']
    
    # Draw links first to be in the background
    for link in links:
        fig.add_trace(go.Scatter(x=[nodes[link[0]]['pos'][0], nodes[link[1]]['pos'][0]], 
                                 y=[nodes[link[0]]['pos'][1], nodes[link[1]]['pos'][1]], 
                                 mode='lines', line=dict(color='grey', width=2)))

    # Prepare node data for plotting
    node_x, node_y, node_text, node_color, node_symbols, node_size = [], [], [], [], [], []
    for name, attrs in nodes.items():
        node_x.append(attrs['pos'][0])
        node_y.append(attrs['pos'][1])
        full_label = f"<b>{attrs['label']}</b>" + (f"<br>P={attrs['prob']:.4f}" if attrs.get('prob') is not None else "")
        node_text.append(full_label)
        node_symbols.append(attrs['shape'])
        
        if attrs.get('type') == 'top':
            node_color.append('salmon')
            node_size.append(70)
        elif attrs.get('type') == 'gate':
            node_color.append('lightgrey')
            node_size.append(50)
        else: # Basic Event
            node_color.append('skyblue')
            node_size.append(70)
            
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, text=node_text, mode='markers+text', textposition="middle center",
        marker=dict(size=node_size, symbol=node_symbols, color=node_color, line=dict(width=2, color='black')),
        textfont=dict(size=10, color='black'),
        hoverinfo='text'
    ))
    
    fig.update_layout(
        title=f"<b>Fault Tree Analysis (FTA) for '{fta_data['title']}'</b>",
        xaxis=dict(visible=False, range=[-0.1, 1.1]), 
        yaxis=dict(visible=False, range=[-0.1, 1.1]), 
        showlegend=False,
        margin=dict(l=20, r=20, b=20, t=60)
    )
    return fig

@st.cache_data
def plot_eta_diagram(eta_data, project_type):
    """Generates a professional-grade Event Tree Analysis (ETA) diagram."""
    fig = go.Figure()
    nodes, paths = eta_data['nodes'], eta_data['paths']
    
    # Draw paths
    for path in paths:
        fig.add_trace(go.Scatter(x=path['x'], y=path['y'], mode='lines', line=dict(color=path['color'], width=2, dash=path.get('dash', 'solid'))))

    # Draw nodes and barriers
    node_x, node_y, node_text = [], [], []
    for name, attrs in nodes.items():
        node_x.append(attrs['pos'][0])
        node_y.append(attrs['pos'][1])
        node_text.append(f"<b>{attrs['label']}</b>")
        # Add probability annotations for barriers
        if 'prob_success' in attrs:
            fig.add_annotation(x=attrs['pos'][0], y=attrs['pos'][1], text=f"Success<br>P={attrs['prob_success']}", showarrow=False, yshift=15, font=dict(size=9))
            fig.add_annotation(x=attrs['pos'][0], y=attrs['pos'][1], text=f"Failure<br>P={1-attrs['prob_success']}", showarrow=False, yshift=-15, font=dict(size=9))

    fig.add_trace(go.Scatter(x=node_x, y=node_y, text=node_text, mode='markers+text', textposition="top center",
                             marker=dict(color='skyblue', size=15, line=dict(width=2, color='black')),
                             textfont=dict(size=10, color='black'), hoverinfo='text'))
    
    # Add outcomes
    for name, attrs in eta_data['outcomes'].items():
         fig.add_annotation(x=attrs['pos'][0], y=attrs['pos'][1],
                           text=f"<b>{name}</b><br>P={attrs['prob']:.5f}",
                           showarrow=False, bgcolor=attrs['color'], borderpad=4, font=dict(color='white'))

    fig.update_layout(
        title=f"<b>Event Tree Analysis (ETA) for '{eta_data['title']}'</b>",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        showlegend=False, margin=dict(l=20, r=20, b=20, t=60)
    )
    return fig
    
@st.cache_data
def plot_vmp_flow(project_type):
    """
    Generates a dynamic V-Model or flowchart for a selected validation project type,
    highlighting key tools from the toolkit for each stage.
    """
    plans = {
        "Analytical Method Validation": {
            'title': "V-Model for Analytical Method Validation",
            'stages': {
                'URS': {'name': 'Assay Requirements', 'tools': 'ATP Builder'},
                'FS': {'name': 'Performance Specs', 'tools': 'Core Validation, LOD & LOQ'},
                'DS': {'name': 'Method Design', 'tools': 'Mixture Design, DOE'},
                'BUILD': {'name': 'Method Development', 'tools': 'Linearity, 4PL Regression'},
                'IQ/OQ': {'name': 'Reagent & Inst. Qual', 'tools': 'Gage R&R / VCA'},
                'PQ': {'name': 'Method Performance Qual', 'tools': 'Comprehensive Diagnostic Validation'},
                'VAL': {'name': 'Final Validation Report', 'tools': 'Method Comparison, Equivalence'}
            }
        },
        "Instrument Qualification": {
            'title': "V-Model for Instrument Qualification",
            'stages': {
                'URS': {'name': 'User Needs (URS)', 'tools': 'ATP Builder'},
                'FS': {'name': 'Functional Specs (FS)', 'tools': 'Quality Risk Management (FMEA)'},
                'DS': {'name': 'Vendor Selection/Config', 'tools': ''},
                'BUILD': {'name': 'Purchase & Install', 'tools': ''},
                'IQ/OQ': {'name': 'IQ & OQ Execution', 'tools': 'Attribute Agreement Analysis'},
                'PQ': {'name': 'Performance Qualification', 'tools': 'Gage R&R, Process Stability (SPC)'},
                'VAL': {'name': 'Final IQ/OQ/PQ Report', 'tools': ''}
            }
        },
        "Pharma Process (PPQ)": {
            'title': "Workflow for Process Performance Qualification",
            'stages': {
                'PLAN': {'name': 'PPQ Protocol', 'tools': 'FMEA, Sample Size for Qualification'},
                'EXEC': {'name': 'Execute PPQ Runs', 'tools': 'Process Stability (SPC)'},
                'EVAL': {'name': 'Assess Capability', 'tools': 'Process Capability (Cpk), Tolerance Intervals'},
                'REPORT': {'name': 'Final PPQ Report', 'tools': 'Statistical Equivalence for Transfer'}
            },
            'type': 'flowchart'
        },
        "Software System (CSV)": {
            'title': "V-Model for GxP Software Validation",
            'stages': {
                'URS': {'name': 'User Requirements', 'tools': 'RTM Builder'},
                'FS': {'name': 'Functional Specs', 'tools': 'FMEA'},
                'DS': {'name': 'Design Specs', 'tools': 'Advanced AI Concepts'},
                'BUILD': {'name': 'Coding & Configuration', 'tools': 'Explainable AI (XAI)'},
                'IQ/OQ': {'name': 'Installation & Unit Test', 'tools': 'Anomaly Detection'},
                'PQ': {'name': 'Performance Testing (UAT)', 'tools': 'Predictive QC (Classification)'},
                'VAL': {'name': 'Final Validation Summary', 'tools': 'Clustering'}
            }
        }
    }
    
    plan = plans[project_type]
    fig = go.Figure()

    if plan.get('type') == 'flowchart':
        keys = list(plan['stages'].keys())
        for i, key in enumerate(keys):
            stage = plan['stages'][key]
            fig.add_shape(type="rect", x0=i*2, y0=0.4, x1=i*2+1.5, y1=0.6, fillcolor=PRIMARY_COLOR, line=dict(color='black'))
            fig.add_annotation(x=i*2+0.75, y=0.5, text=f"<b>{stage['name']}</b>", font=dict(color='white'), showarrow=False)
            fig.add_annotation(x=i*2+0.75, y=0.3, text=f"<i>Key Tools:<br>{stage['tools']}</i>", showarrow=False)
            if i < len(keys) - 1:
                fig.add_annotation(x=i*2+1.75, y=0.5, text="►", font=dict(size=30, color='grey'), showarrow=False)
        fig.update_layout(title=f'<b>{plan["title"]}</b>', xaxis=dict(visible=False), yaxis=dict(visible=False, range=[0,1]))
    else: # V-Model
        v_model_stages = plan['stages']
        path_keys = list(v_model_stages.keys())
        path_x = [0, 1, 2, 3, 4, 5, 6]
        path_y = [5, 4, 3, 2, 3, 4, 5]
        fig.add_trace(go.Scatter(x=path_x, y=path_y, mode='lines', line=dict(color='lightgrey', width=5), hoverinfo='none'))
        for i in range(3):
            fig.add_shape(type="line", x0=path_x[i], y0=path_y[i], x1=path_x[-(i+1)], y1=path_y[-(i+1)], line=dict(color="darkgrey", width=2, dash="dot"))
        for i, key in enumerate(path_keys):
            stage = v_model_stages[key]
            fig.add_shape(type="rect", x0=path_x[i]-0.6, y0=path_y[i]-0.4, x1=path_x[i]+0.6, y1=path_y[i]+0.4, fillcolor=PRIMARY_COLOR, line=dict(color="black"))
            fig.add_annotation(x=path_x[i], y=path_y[i], text=f"<b>{stage['name']}</b>", font=dict(color='white'), showarrow=False)
            fig.add_annotation(x=path_x[i], y=path_y[i]-0.6, text=f"<i>Key Tool: {stage['tools']}</i>" if stage['tools'] else "", showarrow=False)
        fig.update_layout(title=f'<b>{plan["title"]}</b>', xaxis=dict(visible=False), yaxis=dict(visible=False, range=[1, 6]))
        
    fig.update_layout(height=500)
    return fig

@st.cache_data
def plot_rtm_sankey(completed_streams):
    """
    Generates a professional-grade, multi-stream Sankey diagram simulating a complex tech transfer project.
    """
    # Master data structure for the entire project
    nodes_data = {
        # ID: [Label, Stream, x-pos]
        'GOAL': ["Project Goal:<br>Successful Tech Transfer", "Project", 0],
        'PROC-CQA': ["CQA: Purity > 99%", "Process", 1],
        'PROC-CPP': ["CPP: Column Load", "Process", 2],
        'PROC-TEST': ["PPQ Run Results", "Process", 3],
        'ASSAY-ATP': ["ATP: Accuracy 98-102%", "Assay", 1],
        'ASSAY-VAL': ["Assay Validation Report", "Assay", 2],
        'INST-URS': ["URS: HPLC Throughput", "Instrument", 1],
        'INST-OQ': ["Instrument OQ", "Instrument", 2],
        'INST-PQ': ["Instrument PQ", "Instrument", 3],
        'SOFT-URS': ["URS: LIMS Data Integrity", "Software", 1],
        'SOFT-FS': ["FS: Audit Trail Spec", "Software", 2],
        'SOFT-TEST': ["CSV Test Scripts", "Software", 3],
    }
    
    links_data = [
        # Source, Target, Stream
        ('GOAL', 'PROC-CQA', 'Process'),
        ('PROC-CQA', 'PROC-CPP', 'Process'),
        ('PROC-CPP', 'PROC-TEST', 'Process'),
        ('PROC-CQA', 'ASSAY-ATP', 'Cross-Stream'), # Dependency: Process CQA requires an assay
        ('ASSAY-ATP', 'ASSAY-VAL', 'Assay'),
        ('ASSAY-VAL', 'PROC-TEST', 'Cross-Stream'), # Dependency: Process test relies on validated assay
        ('ASSAY-VAL', 'INST-URS', 'Cross-Stream'), # Dependency: Assay validation requires a qualified instrument
        ('INST-URS', 'INST-OQ', 'Instrument'),
        ('INST-OQ', 'INST-PQ', 'Instrument'),
        ('PROC-TEST', 'SOFT-URS', 'Cross-Stream'), # Dependency: Process results must be stored in LIMS
        ('SOFT-URS', 'SOFT-FS', 'Software'),
        ('SOFT-FS', 'SOFT-TEST', 'Software')
    ]

    all_nodes_ids = list(nodes_data.keys())
    all_nodes_labels = [v[0] for v in nodes_data.values()]
    
    # --- Interactivity Logic ---
    link_colors = []
    for source, target, stream in links_data:
        if stream in completed_streams or nodes_data[source][1] in completed_streams:
            link_colors.append(SUCCESS_GREEN)
        else:
            link_colors.append('rgba(200,200,200,0.5)')

    node_colors = []
    for node_id, values in nodes_data.items():
        if values[1] in completed_streams:
            node_colors.append(SUCCESS_GREEN)
        else:
            node_colors.append(PRIMARY_COLOR)

    fig = go.Figure(data=[go.Sankey(
        arrangement='snap',
        node=dict(
            pad=25, thickness=20, line=dict(color="black", width=0.5),
            label=[k for k in all_nodes_ids], # Use short IDs
            hovertemplate='%{customdata}<extra></extra>',
            customdata=all_nodes_labels, # Full labels on hover
            color=node_colors,
            x=[v[2] / 3 for v in nodes_data.values()], # Force horizontal position for swimlanes
            y=[['Project', 'Process', 'Assay', 'Instrument', 'Software'].index(v[1]) * 0.22 for v in nodes_data.values()]
        ),
        link=dict(
            source=[all_nodes_ids.index(l[0]) for l in links_data],
            target=[all_nodes_ids.index(l[1]) for l in links_data],
            value=[1]*len(links_data),
            color=link_colors
        )
    )])
    
    fig.update_layout(
        title_text="<b>Integrated RTM for Technology Transfer Project</b>", font_size=12, height=600
    )
    return fig, links_data, nodes_data

@st.cache_data
def plot_dfx_dashboard(project_type, mfg_effort, quality_effort, sustainability_effort, ux_effort):
    """
    Generates a professional-grade DfX dashboard with a Performance Radar Chart and a Cost Structure Pie Chart comparison.
    """
    profiles = {
        "Pharma Assay (ELISA)": {
            'categories': ['Robustness', 'Run Time (hrs)', 'Reagent Cost (RCU)', 'Precision (%CV)', 'Ease of Use'], 'baseline': [5, 4.0, 25.0, 18.0, 5], 'direction': [1, -1, -1, -1, 1], 'reliability_idx': 0,
            'impact': {'mfg': [0.1, -0.1, -0.2, 0, 0.1], 'quality': [0.5, -0.05, 0, -0.6, 0.2], 'sustainability': [0, 0, -0.3, 0, 0], 'ux': [0.1, -0.2, 0, 0, 0.7]}
        },
        "Instrument (Liquid Handler)": {
            'categories': ['Throughput<br>(plates/hr)', 'Uptime (%)', 'Footprint (m²)', 'Service Cost<br>(RCU/yr)', 'Precision (%CV)'], 'baseline': [20, 95.0, 2.5, 5000, 5.0], 'direction': [1, 1, -1, -1, -1], 'reliability_idx': 1,
            'impact': {'mfg': [0.2, 0.1, -0.2, -0.1, 0], 'quality': [0.1, 0.8, 0, -0.2, -0.6], 'sustainability': [0, 0.1, -0.1, -0.4, 0], 'ux': [0, 0.2, 0, -0.6, 0]}
        },
        "Software (LIMS)": {
            'categories': ['Performance<br>(Query Time s)', 'Scalability<br>(Users)', 'Reliability<br>(Uptime %)', 'Compliance<br>Score', 'Dev Cost (RCU)'], 'baseline': [8.0, 100, 99.5, 6, 500], 'direction': [-1, 1, 1, 1, -1], 'reliability_idx': 2,
            'impact': {'mfg': [-0.1, 0.2, 0.2, 0, -0.4], 'quality': [-0.2, 0.1, 0.7, 0.8, 0.2], 'sustainability': [0, 0.5, 0.1, 0, -0.1], 'ux': [-0.4, 0.2, 0, 0.5, 0.1]}
        },
        "Pharma Process (MAb)": {
            'categories': ['Yield (g/L)', 'Cycle Time<br>(days)', 'COGS (RCU/g)', 'Purity (%)', 'Robustness<br>(PAR Size)'], 'baseline': [3.0, 18, 100, 98.5, 5], 'direction': [1, -1, -1, 1, 1], 'reliability_idx': 4,
            'impact': {'mfg': [0.3, -0.2, -0.4, 0.1, 0.2], 'quality': [0.1, 0, -0.1, 0.6, 0.8], 'sustainability': [0.05, -0.1, -0.2, 0, 0.1], 'ux': [0, 0, 0, 0, 0]}
        }
    }
    profile = profiles[project_type]
    efforts = {'mfg': mfg_effort, 'quality': quality_effort, 'sustainability': sustainability_effort, 'ux': ux_effort}
    
    optimized_kpis = profile['baseline'].copy()
    for i in range(len(profile['categories'])):
        total_impact = sum(efforts[k] * profile['impact'][k][i] for k in efforts)
        optimized_kpis[i] += total_impact * (1 if profile['direction'][i] == 1 else -1)
        if '%' in profile['categories'][i]: optimized_kpis[i] = np.clip(optimized_kpis[i], 0, 100)
        if 'Score' in profile['categories'][i]: optimized_kpis[i] = np.clip(optimized_kpis[i], 0, 10)
        
    kpis = {'baseline': profile['baseline'], 'optimized': optimized_kpis}

    # Simplified but more realistic Cost Model
    def get_cost_breakdown(kpis, p_type):
        if p_type == "Pharma Assay (ELISA)": return [kpis[2], 10, 5, kpis[1]*2]
        elif p_type == "Instrument (Liquid Handler)": return [15000, 2000, kpis[3], 1000 * (5/kpis[4])]
        elif p_type == "Software (LIMS)": return [kpis[4], 150, 100, 50]
        elif p_type == "Pharma Process (MAb)": return [kpis[2]*0.4, kpis[2]*0.3, kpis[2]*0.2, kpis[2]*0.1]
        return [1,1,1,1]

    cost_labels = ['Core (Material/Dev)', 'Manufacturing/Labor', 'Service/Validation', 'QC/Consumables']
    base_costs = get_cost_breakdown(profile['baseline'], project_type)
    optimized_costs = get_cost_breakdown(optimized_kpis, project_type)
    
    # --- Create Plots ---
    # Plot 1: Radar Chart for Performance Profile
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(r=profile['baseline'], theta=profile['categories'], fill='toself', name='Baseline Design', line=dict(color='grey')))
    fig_radar.add_trace(go.Scatterpolar(r=optimized_kpis, theta=profile['categories'], fill='toself', name='DfX Optimized Design', line=dict(color=SUCCESS_GREEN)))
    fig_radar.update_layout(title="<b>1. Project Performance Profile</b>", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    # Plot 2: Cost Structure Pie Charts
    fig_cost = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]], subplot_titles=['<b>Baseline Cost Structure</b>', '<b>Optimized Cost Structure</b>'])
    fig_cost.add_trace(go.Pie(labels=cost_labels, values=base_costs, name="Base", marker_colors=['#636EFA', '#EF553B', '#00CC96', '#AB63FA']), 1, 1)
    fig_cost.add_trace(go.Pie(labels=cost_labels, values=optimized_costs, name="Optimized", marker_colors=['#636EFA', '#EF553B', '#00CC96', '#AB63FA']), 1, 2)
    fig_cost.update_traces(hole=.4, hoverinfo="label+percent+value", textinfo='percent', textfont_size=14)
    fig_cost.update_layout(title_text="<b>2. Lifecycle Cost Breakdown (Total RCU)</b>", showlegend=False, annotations=[
        dict(text=f'{sum(base_costs):,.0f} RCU', x=0.18, y=0.5, font_size=20, showarrow=False),
        dict(text=f'{sum(optimized_costs):,.0f} RCU', x=0.82, y=0.5, font_size=20, showarrow=False)
    ])
    
    return fig_radar, fig_cost, kpis, profile['categories'], base_costs, optimized_costs
#==================================================================ACT 0 END ==============================================================================================================================
#==========================================================================================================================================================================================================
@st.cache_data
def plot_eda_dashboard(df, numeric_cols, cat_cols):
    """
    Generates a professional-grade, multi-part EDA dashboard from an uploaded dataframe.
    """
    # Plot 1: Correlation Heatmap
    corr_matrix = df[numeric_cols].corr()
    fig_heatmap = px.imshow(corr_matrix, text_auto=".2f", aspect="auto",
                            color_continuous_scale='RdBu_r', range_color=[-1,1],
                            title="<b>1. Correlation Heatmap</b>")

    # Plot 2: Scatter Plot Matrix (Pair Plot)
    fig_pairplot = px.scatter_matrix(df, dimensions=numeric_cols, color=cat_cols[0] if cat_cols else None,
                                     title="<b>2. Scatter Plot Matrix (Pair Plot)</b>")
    fig_pairplot.update_traces(diagonal_visible=False)

    # Plot 3: Univariate Distribution Plots
    fig_hist = make_subplots(rows=len(numeric_cols), cols=1, subplot_titles=[f"Distribution of {col}" for col in numeric_cols])
    for i, col in enumerate(numeric_cols):
        fig_hist.add_trace(go.Histogram(x=df[col], name=col), row=i+1, col=1)
    fig_hist.update_layout(title_text="<b>3. Univariate Distributions</b>", showlegend=False, height=250*len(numeric_cols))
    
    return fig_heatmap, fig_pairplot, fig_hist

@st.cache_data
def plot_ci_concept(n=30):
    """
    Generates enhanced, more realistic, and pedagogically sound plots for the
    confidence interval concept module.
    """
    np.random.seed(42)
    pop_mean, pop_std = 100, 15
    
    # --- Plot 1: Population vs. Sampling Distribution ---
    x_range = np.linspace(pop_mean - 4*pop_std, pop_mean + 4*pop_std, 400)
    pop_dist = norm.pdf(x_range, pop_mean, pop_std)
    
    sampling_dist_std = pop_std / np.sqrt(n)
    sampling_dist = norm.pdf(x_range, pop_mean, sampling_dist_std)
    
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=x_range, y=pop_dist, fill='tozeroy', name='Population Distribution (Unseen "Truth")', line=dict(color='skyblue', width=3)))
    fig1.add_trace(go.Scatter(x=x_range, y=sampling_dist, fill='tozeroy', name=f'Sampling Distribution of Mean (n={n})', line=dict(color='orange', width=3)))
    fig1.add_vline(x=pop_mean, line=dict(color='black', dash='dash', width=2), annotation_text="<b>True Population Mean (μ)</b>", annotation_position="top left")

    # --- Simulation for Plot 2 and Highlight for Plot 1 ---
    n_sims = 1000
    samples = np.random.normal(pop_mean, pop_std, size=(n_sims, n))
    sample_means = samples.mean(axis=1)
    sample_stds = samples.std(axis=1, ddof=1)
    
    t_crit = t.ppf(0.975, df=n-1)
    margin_of_error = t_crit * sample_stds / np.sqrt(n)
    
    ci_lowers = sample_means - margin_of_error
    ci_uppers = sample_means + margin_of_error
    
    capture_mask = (ci_lowers <= pop_mean) & (ci_uppers >= pop_mean)
    capture_count = np.sum(capture_mask)
    avg_width = np.mean(ci_uppers - ci_lowers)
    
    # --- SME ENHANCEMENT: Visually link the two plots ---
    # Select one random sample from our simulation to visualize its origin
    highlight_idx = 15 # A good example that is slightly off-center
    highlight_sample = samples[highlight_idx, :]
    highlight_mean = sample_means[highlight_idx]
    
    # Add the individual data points from this sample to Plot 1
    fig1.add_trace(go.Scatter(
        x=highlight_sample, y=np.zeros_like(highlight_sample),
        mode='markers', name=f'One Sample (n={n})',
        marker=dict(color='darkred', size=8, symbol='line-ns-open', line_width=2)
    ))
    # Add the sample mean from this sample to Plot 1
    fig1.add_trace(go.Scatter(
        x=[highlight_mean], y=[0], mode='markers', name='Sample Mean (x̄)',
        marker=dict(color='red', size=12, symbol='x', line_width=3)
    ))
    fig1.update_layout(title_text=f"<b>The Theory: Where a Sample Comes From (n={n})</b>", showlegend=True,
                      legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.7)'),
                      xaxis_title="Measured Value", yaxis_title="Probability Density")

    # --- Plot 2: CI Simulation ---
    fig2 = go.Figure()
    
    # Plot first 100 CIs for visualization
    n_to_plot = min(n_sims, 100)
    for i in range(n_to_plot):
        is_captured = capture_mask[i]
        color = '#636EFA' if is_captured else '#EF553B' # Streamlit's default blue/red
        is_highlight = (i == highlight_idx)
        
        # Create a hover text for each interval
        hover_text = (f"<b>Experiment #{i}</b><br>"
                      f"Sample Mean: {sample_means[i]:.2f}<br>"
                      f"CI: [{ci_lowers[i]:.2f}, {ci_uppers[i]:.2f}]<br>"
                      f"<b>Captured True Mean? {'Yes' if is_captured else 'No'}</b>")
                      
        fig2.add_trace(go.Scatter(
            x=[ci_lowers[i], ci_uppers[i]], y=[i, i], mode='lines',
            line=dict(color=color, width=4 if is_highlight else 2),
            name=f'CI #{i}', showlegend=False,
            hoverinfo='text', hovertext=hover_text
        ))
        fig2.add_trace(go.Scatter(
            x=[sample_means[i]], y=[i], mode='markers',
            marker=dict(color=color, size=6, symbol='line-ns-open', line_width=2),
            showlegend=False, hoverinfo='none'
        ))

    # Add specific traces for the legend to make them interactive
    fig2.add_trace(go.Scatter(x=[None], y=[None], mode='lines', name='Captured Mean (Success)', line=dict(color='#636EFA', width=3)))
    fig2.add_trace(go.Scatter(x=[None], y=[None], mode='lines', name='Missed Mean (Failure)', line=dict(color='#EF553B', width=3)))

    fig2.add_vline(x=pop_mean, line=dict(color='black', dash='dash', width=2), annotation_text="<b>True Population Mean (μ)</b>", annotation_position="bottom right")
    
    # Highlight the specific interval corresponding to the sample in Plot 1
    fig2.add_annotation(x=ci_uppers[highlight_idx], y=highlight_idx,
                        text="This CI comes<br>from the sample<br>shown above",
                        showarrow=True, arrowhead=2, ax=40, ay=-40, font_size=12,
                        bordercolor="black", borderwidth=1, bgcolor="white")

    fig2.update_layout(title_text=f"<b>The Practice: {n_to_plot} Simulated Experiments</b>", yaxis_visible=False,
                      xaxis_title="Measured Value", showlegend=True, legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.7)'))
    
    return fig1, fig2, capture_count, n_sims, avg_width

@st.cache_data
def plot_core_validation_params(bias_pct=1.5, repeat_cv=1.5, intermed_cv=2.5, interference_effect=8.0):
    """
    Generates enhanced, more realistic dynamic plots for the core validation module.
    """
    # --- 1. Accuracy (Bias) Data ---
    # SME Enhancement: The accuracy simulation now incorporates the precision (random error)
    # parameters, which is more realistic.
    np.random.seed(42)
    true_values = np.array([50, 100, 150])
    df_accuracy_list = []
    
    # Calculate overall bias and recovery for annotation
    all_measured = []
    all_true = []

    for val in true_values:
        # Simulate data with both systematic bias and random error (%CV)
        # Use the intermediate precision CV as it represents the total expected random error
        random_error_sd = val * (intermed_cv / 100)
        measurements = np.random.normal(val * (1 + bias_pct / 100), random_error_sd, 10)
        all_measured.extend(measurements)
        all_true.extend([val] * 10)
        for m in measurements:
            df_accuracy_list.append({'True Value': f'Level {val}', 'Measured Value': m, 'Nominal': val})
    
    df_accuracy = pd.DataFrame(df_accuracy_list)
    
    # Calculate overall %Bias
    overall_bias = (np.mean(all_measured) - np.mean(all_true)) / np.mean(all_true) * 100
    
    fig1 = px.box(df_accuracy, x='True Value', y='Measured Value', 
                  title=f'<b>1. Accuracy (Systematic Error) | Overall Bias: {overall_bias:.2f}%</b>',
                  points='all', color_discrete_sequence=['#1f77b4'])
    
    for val in true_values:
        fig1.add_hline(y=val, line_dash="dash", line_color="black", annotation_text=f"True Value = {val}", annotation_position="bottom right")
    fig1.update_layout(xaxis_title="Reference Material Concentration", yaxis_title="Measured Concentration")

    # --- 2. Precision (Random Error) Data ---
    # SME Enhancement: Simulate a more realistic intermediate precision study with multiple
    # days and analysts to better visualize the components of random error.
    np.random.seed(123)
    n_days, n_analysts_per_day, n_reps = 3, 2, 5
    center_val = 100.0
    
    # Define variance components from CV sliders
    var_repeat = (center_val * repeat_cv / 100)**2
    # Intermediate precision includes repeatability PLUS between-analyst/day variance
    var_between_cond = max(0, (center_val * intermed_cv / 100)**2 - var_repeat)
    
    precision_data = []
    for day in range(1, n_days + 1):
        for analyst in range(1, n_analysts_per_day + 1):
            # Each day/analyst combination has its own small random bias
            condition_bias = np.random.normal(0, np.sqrt(var_between_cond))
            # Measurements within that condition have repeatability error
            measurements = np.random.normal(center_val + condition_bias, np.sqrt(var_repeat), n_reps)
            for m in measurements:
                precision_data.append({'value': m, 'condition': f'Day {day}, Analyst {analyst}'})
    
    df_precision = pd.DataFrame(precision_data)
    
    # Calculate overall precision from the simulated data
    total_cv_calc = df_precision['value'].std() / df_precision['value'].mean() * 100
    
    fig2 = px.box(df_precision, x='condition', y='value', points="all",
                     title=f'<b>2. Intermediate Precision (Random Error) | Overall %CV: {total_cv_calc:.2f}%</b>',
                     labels={'value': 'Measured Value @ 100', 'condition': 'Run Condition (Day, Analyst)'})
    fig2.update_xaxes(tickangle=45)

    # --- 3. Specificity (Interference Error) Data ---
    np.random.seed(2023)
    analyte_signal = np.random.normal(1.0, 0.05, 15)
    matrix_blank_signal = np.random.normal(0.02, 0.01, 15)
    
    # The signal of the combined sample is now controlled by the interference slider
    interfered_signal = analyte_signal * (1 + interference_effect / 100)
    
    # SME Enhancement: Add pass/fail color and KPI to the title
    is_significant_interference = abs(interference_effect) > 5 # Example threshold
    interference_color = '#EF553B' if is_significant_interference else '#00CC96'
    
    df_specificity = pd.DataFrame({
        'Analyte Only': analyte_signal,
        'Matrix Blank': matrix_blank_signal,
        'Analyte + Interferent': interfered_signal
    }).melt(var_name='Sample Type', value_name='Signal Response')

    fig3 = px.box(df_specificity, x='Sample Type', y='Signal Response', points='all',
                  title=f'<b>3. Specificity (Interference Error) | Effect: {interference_effect:.1f}%</b>',
                  color='Sample Type',
                  color_discrete_map={
                      'Analyte Only': '#636EFA',
                      'Matrix Blank': 'grey',
                      'Analyte + Interferent': interference_color
                  })
    fig3.update_layout(xaxis_title="Sample Composition", yaxis_title="Assay Signal (e.g., Absorbance)", showlegend=False)

    return fig1, fig2, fig3

@st.cache_data
def plot_diagnostic_dashboard(sensitivity, specificity, prevalence, n_total=10000):
    """
    Generates a comprehensive, multi-plot dashboard for diagnostic test validation,
    and calculates a full suite of 24 performance metrics.
    """
    # 1. --- Calculate the Confusion Matrix values from inputs ---
    n_diseased = int(n_total * prevalence)
    n_healthy = n_total - n_diseased
    
    tp = int(n_diseased * sensitivity)
    fn = n_diseased - tp
    
    tn = int(n_healthy * specificity)
    fp = n_healthy - tn
    
    # 2. --- Calculate ALL 24 derived metrics systematically ---
    tpr, tnr = (tp / (tp + fn) if (tp + fn) > 0 else 0), (tn / (tn + fp) if (tn + fp) > 0 else 0)
    fpr, fnr = 1 - tnr, 1 - tpr
    ppv, npv = (tp / (tp + fp) if (tp + fp) > 0 else 0), (tn / (tn + fn) if (tn + fn) > 0 else 0)
    fdr, for_val = 1 - ppv, 1 - npv
    accuracy = (tp + tn) / n_total
    f1_score = 2 * (ppv * tpr) / (ppv + tpr) if (ppv + tpr) > 0 else 0
    youdens_j = tpr + tnr - 1
    p_obs = accuracy
    p_exp = ((tp + fp) / n_total) * ((tp + fn) / n_total) + ((fn + tn) / n_total) * ((fp + tn) / n_total)
    kappa = (p_obs - p_exp) / (1 - p_exp) if (1 - p_exp) != 0 else 0
    mcc_denom = np.sqrt(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn))
    mcc = (tp * tn - fp * fn) / mcc_denom if mcc_denom > 0 else 0
    lr_plus, lr_minus = (tpr / fpr if fpr > 0 else float('inf')), (fnr / tnr if tnr > 0 else 0)
    
    sens_clipped = np.clip(sensitivity, 0.00001, 0.99999)
    spec_clipped = np.clip(specificity, 0.00001, 0.99999)
    separation = norm.ppf(sens_clipped) + norm.ppf(spec_clipped)
    scores_diseased = np.random.normal(separation, 1, 5000)
    scores_healthy = np.random.normal(0, 1, 5000)
    y_true_roc, y_scores_roc = np.concatenate([np.ones(5000), np.zeros(5000)]), np.concatenate([scores_diseased, scores_healthy])
    fpr_roc, tpr_roc, _ = roc_curve(y_true_roc, y_scores_roc)
    auc_val = auc(fpr_roc, tpr_roc)
    prob_scores = 1 / (1 + np.exp(-y_scores_roc + separation/2))
    log_loss = -np.mean(y_true_roc * np.log(prob_scores + 1e-15) + (1 - y_true_roc) * np.log(1 - prob_scores + 1e-15))

    # --- PLOTTING ---
    # Plot 1: Professional Confusion Matrix
    fig_cm = go.Figure(data=go.Heatmap(z=[[fn, tp], [tn, fp]], x=['Actual Diseased', 'Actual Healthy'], y=['Predicted Negative', 'Predicted Positive'],
                                       colorscale=[[0, '#e3f2fd'], [1, '#0d47a1']], showscale=False, textfont={"size":16}))
    z_values = [[f'<b>FN</b><br>{fn}', f'<b>TP</b><br>{tp}'], [f'<b>TN</b><br>{tn}', f'<b>FP</b><br>{fp}']]
    for i, row in enumerate(z_values):
        for j, val in enumerate(row):
            fig_cm.add_annotation(x=j, y=i, text=val, showarrow=False, font=dict(color='white' if [[fn, tp], [tn, fp]][i][j] > n_total/3 else 'black'))
    fig_cm.add_annotation(x=0, y=1.15, text=f"Sensitivity (TPR)<br><b>{tpr:.1%}</b>", showarrow=False, yshift=30)
    fig_cm.add_annotation(x=1, y=-0.15, text=f"Specificity (TNR)<br><b>{tnr:.1%}</b>", showarrow=False, yshift=-30)
    fig_cm.add_annotation(x=-0.2, y=1, text=f"PPV<br><b>{ppv:.1%}</b>", showarrow=False, xshift=-30)
    fig_cm.add_annotation(x=-0.2, y=0, text=f"NPV<br><b>{npv:.1%}</b>", showarrow=False, xshift=-30)
    fig_cm.update_layout(title="<b>1. Confusion Matrix & Key Rates</b>", xaxis_title="Actual Condition", yaxis_title="Predicted Outcome", margin=dict(t=50, l=50, r=10, b=50))

    # Plot 2: Professional ROC Curve
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr_roc, y=tpr_roc, mode='lines', name=f'AUC = {auc_val:.3f}', line=dict(color=PRIMARY_COLOR, width=3)))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Chance', line=dict(color='grey', width=2, dash='dash')))
    fig_roc.add_trace(go.Scatter(x=[fpr], y=[tpr], mode='markers', name="Current Threshold", marker=dict(color=SUCCESS_GREEN, size=15, symbol='star', line=dict(width=2, color='black'))))
    youden_idx = np.argmax(tpr_roc - fpr_roc)
    fig_roc.add_trace(go.Scatter(x=[fpr_roc[youden_idx]], y=[tpr_roc[youden_idx]], mode='markers', name="Optimal (Youden's J)", marker=dict(color='#FFBF00', size=12, symbol='diamond')))
    fig_roc.update_layout(title="<b>2. Receiver Operating Characteristic (ROC)</b>", xaxis_title="False Positive Rate (1 - Specificity)", yaxis_title="True Positive Rate (Sensitivity)", legend=dict(x=0.01, y=0.01, yanchor='bottom'))

    # Plot 3: Predictive Values vs. Prevalence
    prevalence_range = np.linspace(0.001, 1, 100)
    ppv_curve = (sensitivity * prevalence_range) / (sensitivity * prevalence_range + (1 - specificity) * (1 - prevalence_range))
    npv_curve = (specificity * (1 - prevalence_range)) / (specificity * (1 - prevalence_range) + (1 - sensitivity) * prevalence_range)
    fig_pv = go.Figure()
    fig_pv.add_trace(go.Scatter(x=prevalence_range, y=ppv_curve, mode='lines', name='PPV (Precision)', line=dict(color='red', width=3)))
    fig_pv.add_trace(go.Scatter(x=prevalence_range, y=npv_curve, mode='lines', name='NPV', line=dict(color='blue', width=3)))
    fig_pv.add_vline(x=prevalence, line=dict(color='black', dash='dash'), annotation_text=f"Current Prevalence ({prevalence:.1%})")
    fig_pv.update_layout(title="<b>3. The Prevalence Effect on Predictive Values</b>", xaxis_title="Disease Prevalence in Population", yaxis_title="Predictive Value", xaxis_tickformat=".0%", yaxis_tickformat=".0%", legend=dict(x=0.01, y=0.01, yanchor='bottom'))
    
    metrics = {
        "True Positive (TP)": tp, "True Negative (TN)": tn, "False Positive (FP)": fp, "False Negative (FN)": fn,
        "Prevalence": prevalence, "Sensitivity (TPR / Power)": tpr, "Specificity (TNR)": tnr,
        "False Positive Rate (α)": fpr, "False Negative Rate (β)": fnr,
        "PPV (Precision)": ppv, "NPV": npv, "False Discovery Rate (FDR)": fdr, "False Omission Rate (FOR)": for_val,
        "Accuracy": accuracy, "F1 Score": f1_score, "Youden's Index (J)": youdens_j,
        "Matthews Correlation Coefficient (MCC)": mcc, "Cohen’s Kappa (κ)": kappa,
        "Positive Likelihood Ratio (LR+)": lr_plus, "Negative Likelihood Ratio (LR-)": lr_minus,
        "Area Under Curve (AUC)": auc_val, "Log-Loss (Cross-Entropy)": log_loss
    }
    other_concepts = {
        "Bias": "Systematic deviation from a true value. In diagnostics, this could be a technology that consistently over- or under-estimates a biomarker value.",
        "Error": "The difference between a measured value and the true value. Comprises both random error (imprecision) and systematic error (bias).",
        "Precision": "The closeness of repeated measurements to each other. Poor precision widens the score distributions, reducing the separation between healthy and diseased populations.",
        "Prevalence Threshold": "A policy decision, not a calculation. It's the prevalence at which you decide the PPV is too low for the test to be useful as a screening tool."
    }
    
    return fig_cm, fig_roc, fig_pv, metrics, other_concepts

@st.cache_data
def plot_attribute_agreement(n_parts, n_replicates, prevalence, skilled_accuracy, uncertain_accuracy, biased_accuracy, bias_strength):
    """
    Generates a professional-grade dashboard for Attribute Agreement Analysis with realistic inspector archetypes.
    """
    np.random.seed(42)
    # 1. Create the "gold standard" reference parts, including "borderline" cases
    n_defective = int(n_parts * prevalence)
    n_good = n_parts - n_defective
    reference = np.array([1] * n_defective + [0] * n_good) # 1=Defective, 0=Good
    
    # Add borderline parts that are harder to classify
    n_borderline = int(n_parts * 0.2)
    borderline_indices = np.random.choice(n_parts, n_borderline, replace=False)
    is_borderline = np.zeros(n_parts, dtype=bool)
    is_borderline[borderline_indices] = True
    
    # 2. Simulate inspector assessments with different archetypes
    inspectors = {
        'Inspector A (Skilled)': {'acc': skilled_accuracy, 'bias': 0.5},
        'Inspector B (Uncertain)': {'acc': uncertain_accuracy, 'bias': 0.5},
        'Inspector C (Biased)': {'acc': biased_accuracy, 'bias': bias_strength}
    }
    
    assessments = []
    for name, params in inspectors.items():
        for part_idx in range(n_parts):
            for _ in range(n_replicates):
                true_status = reference[part_idx]
                accuracy = params['acc']
                # The "Uncertain" inspector is less accurate on borderline parts
                if "Uncertain" in name and is_borderline[part_idx]:
                    accuracy *= 0.7 
                
                is_correct = np.random.rand() < accuracy
                if is_correct:
                    assessment = true_status
                else: # They got it wrong
                    assessment = 1 - true_status
                    # The "Biased" inspector is more likely to make a false positive error
                    if "Biased" in name and true_status == 0:
                        assessment = 1 if np.random.rand() < params['bias'] else 0
                
                assessments.append([name, f'Part_{part_idx+1}', true_status, assessment])

    df = pd.DataFrame(assessments, columns=['Inspector', 'Part', 'Reference', 'Assessment'])

    # 3. Calculate Key Metrics
    # Fleiss' Kappa for overall agreement
    # Create a contingency table: rows are parts, columns are Good/Defective ratings by inspectors
    contingency_table = pd.crosstab(df['Part'], df['Assessment'])
    if 0 not in contingency_table.columns: contingency_table[0] = 0
    if 1 not in contingency_table.columns: contingency_table[1] = 0
    
    N = len(contingency_table) # Number of subjects
    n = contingency_table.sum(axis=1).iloc[0] # Number of ratings per subject
    p_j = contingency_table.sum(axis=0) / (N * n) # Proportions of each category
    P_i = ( (contingency_table**2).sum(axis=1) - n ) / (n * (n-1))
    P_bar = P_i.mean()
    P_e_bar = (p_j**2).sum()
    kappa = (P_bar - P_e_bar) / (1 - P_e_bar) if (1-P_e_bar) != 0 else 0


    # Individual inspector effectiveness
    effectiveness = {}
    for name in df['Inspector'].unique():
        sub = df[df['Inspector'] == name]
        cm = pd.crosstab(sub['Reference'], sub['Assessment'])
        tn = cm.loc[0,0] if 0 in cm.index and 0 in cm.columns else 0
        fp = cm.loc[0,1] if 0 in cm.index and 1 in cm.columns else 0
        fn = cm.loc[1,0] if 1 in cm.index and 0 in cm.columns else 0
        tp = cm.loc[1,1] if 1 in cm.index and 1 in cm.columns else 0
        
        miss_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        effectiveness[name] = {'Miss Rate': miss_rate, 'False Alarm Rate': false_alarm_rate, 'Accuracy': accuracy}
    
    df_eff = pd.DataFrame(effectiveness).T.reset_index().rename(columns={'index':'Inspector'})

    # 4. Generate Plots
    fig_eff = go.Figure()
    fig_eff.add_trace(go.Scatter(
        x=df_eff['False Alarm Rate'], y=df_eff['Miss Rate'],
        mode='markers+text', text=df_eff['Inspector'], textposition='top center',
        marker=dict(
            size=(df_eff['Miss Rate'] + df_eff['False Alarm Rate']) * 200 + 15, # Bubble size reflects total error
            color=df_eff['Accuracy'], colorscale='RdYlGn', cmin=0.7, cmax=1.0,
            showscale=True, colorbar_title='Accuracy'
        ),
    ))
    fig_eff.add_shape(type="rect", x0=0, y0=0, x1=0.05, y1=0.1, fillcolor='rgba(44, 160, 44, 0.2)', line_width=0, layer='below')
    fig_eff.add_annotation(x=0.025, y=0.05, text="<b>Ideal Zone</b>", showarrow=False)
    fig_eff.update_layout(title="<b>1. Inspector Effectiveness Report</b>",
                          xaxis_title="False Alarm Rate (Good parts failed)", yaxis_title="Miss Rate (Bad parts passed)",
                          xaxis_tickformat=".1%", yaxis_tickformat=".1%",
                          xaxis_range=[-0.02, max(0.2, df_eff['False Alarm Rate'].max()*1.2)],
                          yaxis_range=[-0.02, max(0.2, df_eff['Miss Rate'].max()*1.2)])

    # Plot 2: Cohen's Kappa Matrix
    from sklearn.metrics import cohen_kappa_score
    kappa_matrix = pd.DataFrame(index=inspectors.keys(), columns=inspectors.keys())
    for name1 in inspectors.keys():
        for name2 in inspectors.keys():
            d1 = df[df['Inspector'] == name1]['Assessment']
            d2 = df[df['Inspector'] == name2]['Assessment']
            kappa_matrix.loc[name1, name2] = cohen_kappa_score(d1, d2)
    kappa_matrix = kappa_matrix.astype(float)
    
    fig_kappa = px.imshow(kappa_matrix, text_auto=".2f", aspect="auto",
                          color_continuous_scale='Blues',
                          title="<b>2. Inter-Inspector Agreement (Cohen's Kappa)</b>",
                          labels=dict(color="Kappa"))
    
    return fig_eff, fig_kappa, kappa, df_eff

@st.cache_data    
def plot_gage_rr(part_sd=5.0, repeatability_sd=1.5, operator_sd=0.75, interaction_sd=0.5):
    """
    Generates dynamic and more realistic plots for the Gage R&R module,
    including sorted parts and operator-part interaction.
    """
    np.random.seed(10)
    n_operators, n_samples, n_replicates = 3, 10, 3
    operators = ['Alice', 'Bob', 'Charlie']
    
    # --- SME ENHANCEMENT 1: Create structured, sorted parts that span a realistic process range ---
    # Instead of random sampling, create parts that are intentionally spread out, as in a real study.
    center = 100
    # Use part_sd to define the spread of parts, e.g., covering a +/- 2.5 sigma range
    part_spread = 2.5 * part_sd 
    true_part_values_sorted = np.linspace(center - part_spread, center + part_spread, n_samples)
    part_names_sorted = [f'Part-{i+1:02d}' for i in range(n_samples)]
    part_map = dict(zip(part_names_sorted, true_part_values_sorted))

    # Generate consistent operator biases
    operator_biases = np.random.normal(0, operator_sd, n_operators)
    operator_bias_map = {op: bias for op, bias in zip(operators, operator_biases)}
    
    # --- SME ENHANCEMENT 2: Simulate Operator-Part Interaction ---
    # Create a consistent random effect for each specific operator-part combination.
    interaction_effects = {}
    for op in operators:
        for part_name in part_names_sorted:
            interaction_effects[(op, part_name)] = np.random.normal(0, interaction_sd)

    data = []
    for operator in operators:
        for part_name, true_value in part_map.items():
            # The final "true" value for this measurement includes the main effects and the new interaction effect
            effective_true_value = true_value + operator_bias_map[operator] + interaction_effects[(operator, part_name)]
            
            # Generate measurements with repeatability (instrument noise) around this effective value
            measurements = np.random.normal(effective_true_value, repeatability_sd, n_replicates)
            
            for m in measurements:
                data.append([operator, part_name, m])
    
    df = pd.DataFrame(data, columns=['Operator', 'Part', 'Measurement'])
    
    # Perform ANOVA (the model correctly captures the new interaction term)
    model = ols('Measurement ~ C(Part) + C(Operator) + C(Part):C(Operator)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    
    ms_operator = anova_table.loc['C(Operator)', 'sum_sq'] / anova_table.loc['C(Operator)', 'df']
    ms_part = anova_table.loc['C(Part)', 'sum_sq'] / anova_table.loc['C(Part)', 'df']
    ms_interaction = anova_table.loc['C(Part):C(Operator)', 'sum_sq'] / anova_table.loc['C(Part):C(Operator)', 'df']
    ms_error = anova_table.loc['Residual', 'sum_sq'] / anova_table.loc['Residual', 'df']
    
    # Calculate variance components from Mean Squares
    var_repeatability = ms_error
    var_operator = max(0, (ms_operator - ms_interaction) / (n_samples * n_replicates))
    var_interaction = max(0, (ms_interaction - ms_error) / n_replicates)
    var_reproducibility = var_operator + var_interaction
    var_part = max(0, (ms_part - ms_interaction) / (n_operators * n_replicates))
    var_rr = var_repeatability + var_reproducibility
    var_total = var_rr + var_part
    
    # Calculate final KPIs
    pct_rr = (var_rr / var_total) * 100 if var_total > 0 else 0
    ndc = int(1.41 * np.sqrt(var_part / var_rr)) if var_rr > 0 else 10
    
    # Plotting (updated titles and layout for clarity)
    fig = make_subplots(rows=2, cols=2, column_widths=[0.7, 0.3], row_heights=[0.5, 0.5],
                        specs=[[{"rowspan": 2}, {}], [None, {}]],
                        subplot_titles=("<b>Measurement by Part (Grouped by Operator)</b>",
                                        "<b>Overall Variation by Operator</b>",
                                        "<b>Final Verdict: Variation Contribution</b>"))

    # Main plot: Box plot for each part, colored by operator
    fig_box = px.box(df, x='Part', y='Measurement', color='Operator', color_discrete_sequence=px.colors.qualitative.Plotly)
    for trace in fig_box.data: fig.add_trace(trace, row=1, col=1)
    
    # Add operator mean lines to show trends and interactions
    for i, operator in enumerate(operators):
        operator_df = df[df['Operator'] == operator]
        part_means = operator_df.groupby('Part')['Measurement'].mean().reindex(part_names_sorted) # Ensure correct order
        fig.add_trace(go.Scatter(x=part_means.index, y=part_means.values, mode='lines',
                                 line=dict(width=2), name=f'{operator} Mean',
                                 showlegend=False, marker_color=fig_box.data[i].marker.color), row=1, col=1)

    # Top-right plot: Box plot of overall measurements for each operator
    fig_op_box = px.box(df, x='Operator', y='Measurement', color='Operator', color_discrete_sequence=px.colors.qualitative.Plotly)
    for trace in fig_op_box.data: fig.add_trace(trace, row=1, col=2)
    
    # Bottom-right plot: Final verdict bar chart
    pct_part = (var_part / var_total) * 100 if var_total > 0 else 0
    fig.add_trace(go.Bar(x=['% Gage R&R', '% Part Var.'], y=[pct_rr, pct_part],
                         marker_color=['#EF553B' if pct_rr > 30 else ('#FECB52' if pct_rr > 10 else '#00CC96'), '#636EFA'],
                         text=[f'{pct_rr:.1f}%', f'{pct_part:.1f}%'], textposition='auto'), row=2, col=2)
    
    # Add acceptance criteria lines to the verdict plot
    fig.add_hline(y=10, line_dash="dash", line_color="darkgreen", annotation_text="Acceptable < 10%", annotation_position="bottom right", row=2, col=2)
    fig.add_hline(y=30, line_dash="dash", line_color="darkorange", annotation_text="Unacceptable > 30%", annotation_position="top right", row=2, col=2)

    fig.update_layout(title_text='<b>Gage R&R Study: Is the Measurement System Fit for Purpose?</b>', title_x=0.5, height=800,
                      boxmode='group', showlegend=True, legend_title_text='Operator')
    fig.update_xaxes(tickangle=0, row=1, col=1) # No longer need tickangle with sorted part names
    fig.update_yaxes(title_text="Measurement", row=1, col=1)
    
    return fig, pct_rr, ndc

@st.cache_data
def plot_lod_loq(slope=0.02, baseline_sd=0.01):
    """
    Generates enhanced, more realistic dynamic plots for the LOD & LOQ module.
    """
    np.random.seed(3)
    
    # --- 1. Simulate a more realistic dataset ---
    # Multiple blank lots and multiple low-concentration spikes
    n_blanks, n_spikes = 60, 20
    blank_signals = np.random.normal(0.05, baseline_sd, n_blanks)
    
    # Use the calculated LOQ (from the sliders) to determine a realistic spike level
    # This creates a circular but effective demonstration
    approx_loq_conc = (10 * baseline_sd) / slope
    spike_signals = np.random.normal(0.05 + approx_loq_conc * slope, baseline_sd * 1.5, n_spikes)

    df_dist = pd.concat([
        pd.DataFrame({'Signal': blank_signals, 'Sample Type': 'Blanks'}), 
        pd.DataFrame({'Signal': spike_signals, 'Sample Type': 'Low Conc Spikes'})
    ])
    
    # --- 2. Low-Level Calibration Curve ---
    concentrations = np.array([0, 0, 0, 0, 0, 0.5, 0.5, 1, 1, 2, 2, 5, 5, 10, 10])
    signals = 0.05 + slope * concentrations + np.random.normal(0, baseline_sd, len(concentrations))
    df_cal = pd.DataFrame({'Concentration': concentrations, 'Signal': signals})
    
    # --- 3. Fit Model and Calculate Key Values ---
    # Use the blank measurements to get a robust estimate of the blank mean and SD
    mean_blank = np.mean(blank_signals)
    sd_blank = np.std(blank_signals, ddof=1)
    
    X = sm.add_constant(df_cal['Concentration'])
    model = sm.OLS(df_cal['Signal'], X).fit()
    
    fit_slope = model.params['Concentration'] if model.params['Concentration'] > 0.001 else 0.001
    
    # Calculate the full LOB/LOD/LOQ hierarchy
    LOB = mean_blank + 1.645 * sd_blank
    LOD_signal = LOB + 1.645 * sd_blank # Signal at the LOD
    LOD_conc = (LOD_signal - model.params['const']) / fit_slope
    LOQ_conc = (10 * sd_blank) / fit_slope # ICH Signal/Noise method
    
    # --- 4. Plotting ---
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=("<b>1. Signal Distributions: Defining LOB & LOD</b>",
                                        "<b>2. Calibration Curve: Defining LOQ</b>"),
                        vertical_spacing=0.15)
    
    # Top Plot: Signal Distributions
    fig_hist = px.histogram(df_dist, x='Signal', color='Sample Type', barmode='overlay',
                            opacity=0.7, nbins=30,
                            color_discrete_map={'Blanks': 'skyblue', 'Low Conc Spikes': 'lightgreen'})
    for trace in fig_hist.data:
        fig.add_trace(trace, row=1, col=1)
    
    fig.add_vline(x=LOB, line_dash="dot", line_color="purple", row=1, col=1,
                  annotation_text=f"<b>LOB = {LOB:.3f}</b>", annotation_position="top")
    fig.add_vline(x=LOD_signal, line_dash="dash", line_color="orange", row=1, col=1,
                  annotation_text=f"<b>LOD Signal = {LOD_signal:.3f}</b>", annotation_position="top")
    
    # Bottom Plot: Calibration Curve with Prediction Intervals
    fig.add_trace(go.Scatter(x=df_cal['Concentration'], y=df_cal['Signal'], mode='markers',
                             name='Calibration Points', marker=dict(color='darkblue', size=8)), row=2, col=1)
    
    x_pred = pd.DataFrame({'const': 1, 'Concentration': np.linspace(-1, df_cal['Concentration'].max()*1.1, 100)})
    pred_summary = model.get_prediction(x_pred).summary_frame(alpha=0.05)
    
    fig.add_trace(go.Scatter(x=x_pred['Concentration'], y=pred_summary['mean'], mode='lines',
                             name='Regression Line', line=dict(color='red', dash='dash')), row=2, col=1)
    # SME Enhancement: Show Prediction Interval, which is key for LOQ
    fig.add_trace(go.Scatter(x=x_pred['Concentration'], y=pred_summary['obs_ci_upper'],
                             fill=None, mode='lines', line_color='rgba(255,0,0,0.2)', showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_pred['Concentration'], y=pred_summary['obs_ci_lower'],
                             fill='tonexty', mode='lines', line_color='rgba(255,0,0,0.2)',
                             name='95% Prediction Interval'), row=2, col=1)

    fig.add_vline(x=LOD_conc, line_dash="dash", line_color="orange", row=2, col=1,
                  annotation_text=f"<b>LOD = {LOD_conc:.2f}</b>", annotation_position="bottom")
    fig.add_vline(x=LOQ_conc, line_dash="dot", line_color="red", row=2, col=1,
                  annotation_text=f"<b>LOQ = {LOQ_conc:.2f}</b>", annotation_position="top")
    
    fig.update_layout(title_text='<b>Assay Sensitivity Analysis: The LOB, LOD, & LOQ Hierarchy</b>',
                      title_x=0.5, height=800, legend=dict(x=0.01, y=0.45))
    fig.update_yaxes(title_text="Assay Signal (e.g., Absorbance)", row=1, col=1)
    fig.update_xaxes(title_text="Signal", row=1, col=1)
    fig.update_yaxes(title_text="Assay Signal", row=2, col=1)
    fig.update_xaxes(title_text="Concentration (ng/mL)", row=2, col=1, range=[-1, df_cal['Concentration'].max()*1.1])
    
    return fig, LOD_conc, LOQ_conc
    
# ==============================================================================
# HELPER & PLOTTING FUNCTION (Linearity) - SME ENHANCED
# ==============================================================================
@st.cache_data
def plot_linearity(curvature=-1.0, random_error=1.0, proportional_error=2.0, use_wls=False):
    """
    Generates enhanced, more realistic dynamic plots for the Linearity module,
    including replicates and optional Weighted Least Squares (WLS) regression.
    """
    np.random.seed(42)
    nominal_levels = np.array([10, 25, 50, 100, 150, 200, 250])
    n_replicates = 3 # SME Enhancement: Real studies use replicates
    
    data = []
    for nom in nominal_levels:
        for _ in range(n_replicates):
            curvature_effect = curvature * (nom / 150)**3
            error = np.random.normal(0, random_error + nom * (proportional_error / 100))
            measured = nom + curvature_effect + error
            data.append({'Nominal': nom, 'Measured': measured})
    
    df = pd.DataFrame(data)
    
    # --- SME Enhancement: Implement OLS vs. WLS Regression ---
    X = sm.add_constant(df['Nominal'])
    
    if use_wls and proportional_error > 0:
        # For WLS, weights are typically the inverse of the variance at each level.
        # We approximate variance from the nominal concentration.
        df['weights'] = 1 / (random_error + df['Nominal'] * (proportional_error / 100))**2
        model = sm.WLS(df['Measured'], X, weights=df['weights']).fit()
        model_type = "Weighted Least Squares (WLS)"
    else:
        model = sm.OLS(df['Measured'], X).fit()
        model_type = "Ordinary Least Squares (OLS)"

    df['Predicted'] = model.predict(X)
    df['Residual'] = model.resid
    df['Recovery'] = (df['Measured'] / df['Nominal']) * 100
    
    # --- Plotting ---
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{}, {}], [{"colspan": 2}, None]],
        subplot_titles=(f"<b>1. Linearity Plot (R² = {model.rsquared:.4f})</b>",
                        "<b>2. Residuals vs. Nominal</b>",
                        "<b>3. Percent Recovery vs. Nominal</b>"),
        vertical_spacing=0.2
    )
    
    # Plot 1: Linearity Plot
    fig.add_trace(go.Scatter(x=df['Nominal'], y=df['Measured'], mode='markers', name='Measured Values',
                             marker=dict(size=8, color='blue', opacity=0.7)), row=1, col=1)
    
    # Plot the regression line using a smooth range
    x_range = np.linspace(0, 260, 100)
    y_range = model.predict(sm.add_constant(x_range))
    fig.add_trace(go.Scatter(x=x_range, y=y_range, mode='lines', name=f'{model_type} Fit',
                             line=dict(color='red')), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0, 260], y=[0, 260], mode='lines', name='Line of Identity (y=x)',
                             line=dict(dash='dash', color='black')), row=1, col=1)
    
    # Plot 2: Residual Plot (SME Enhancement: Use box plots for clarity)
    fig.add_trace(px.box(df, x='Nominal', y='Residual').data[0].update(marker_color='green', name='Residuals'), row=1, col=2)
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=2)
    
    # Plot 3: Recovery Plot (SME Enhancement: Use box plots for clarity)
    fig.add_trace(px.box(df, x='Nominal', y='Recovery').data[0].update(marker_color='purple', name='Recovery'), row=2, col=1)
    fig.add_hrect(y0=80, y1=120, fillcolor="green", opacity=0.1, layer="below", line_width=0, row=2, col=1)
    fig.add_hline(y=100, line_dash="dash", line_color="black", row=2, col=1)
    fig.add_hline(y=80, line_dash="dot", line_color="red", row=2, col=1, annotation_text="80% Limit")
    fig.add_hline(y=120, line_dash="dot", line_color="red", row=2, col=1, annotation_text="120% Limit")
    
    fig.update_layout(title_text=f'<b>Assay Linearity Dashboard (Model: {model_type})</b>',
                      title_x=0.5, height=800, showlegend=True,
                      legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.7)'))
    fig.update_xaxes(title_text="Nominal Concentration", row=1, col=1); fig.update_yaxes(title_text="Measured Concentration", row=1, col=1)
    fig.update_xaxes(title_text="Nominal Concentration", row=1, col=2); fig.update_yaxes(title_text="Residual (Error)", row=1, col=2)
    fig.update_xaxes(title_text="Nominal Concentration", row=2, col=1); fig.update_yaxes(title_text="% Recovery", range=[70, 130], row=2, col=1)
    
    return fig, model

# ==============================================================================
# HELPER & PLOTTING FUNCTION (4PL) - SME ENHANCED
# ==============================================================================
@st.cache_data
def plot_4pl_regression(a_true=1.5, b_true=1.2, c_true=10.0, d_true=0.05, noise_sd=0.05, proportional_noise=1.0, use_irls=True):
    """
    Generates enhanced, more realistic dynamic plots for the 4PL regression module,
    including heteroscedastic noise and weighted fitting (IRLS).
    """
    # 4PL logistic function
    def four_pl(x, a, b, c, d):
        return d + (a - d) / (1 + (x / c)**b)

    # SME Enhancement: More realistic data generation with replicates and proportional noise
    np.random.seed(42)
    conc_levels = np.logspace(-2, 3, 8)
    conc_replicates = np.repeat(conc_levels, 2) # Duplicates for replicates

    signal_true = four_pl(conc_replicates, a_true, b_true, c_true, d_true)
    
    # Noise model: combination of constant and signal-proportional noise
    total_noise_sd = noise_sd + (signal_true * (proportional_noise / 100))
    signal_measured = signal_true + np.random.normal(0, total_noise_sd)
    
    df = pd.DataFrame({'Concentration': conc_replicates, 'Signal': signal_measured})

    # SME Enhancement: Implement Iteratively Reweighted Least Squares (IRLS) for fitting
    model_type = "Least Squares"
    try:
        p0 = [a_true, b_true, c_true, d_true]
        if use_irls and proportional_noise > 0:
            model_type = "Weighted (IRLS)"
            # Use sigma parameter in curve_fit for weighting (inverse of variance)
            # Estimate variance at each point
            sigma_weights = noise_sd + (df['Signal'] * (proportional_noise / 100))
            params, cov = curve_fit(four_pl, df['Concentration'], df['Signal'], p0=p0, sigma=sigma_weights, absolute_sigma=True, maxfev=10000)
        else:
            params, cov = curve_fit(four_pl, df['Concentration'], df['Signal'], p0=p0, maxfev=10000)
    except RuntimeError:
        params, cov = p0, np.full((4, 4), np.inf)
        
    a_fit, b_fit, c_fit, d_fit = params
    perr = np.sqrt(np.diag(cov)) # Standard errors of parameters

    # Calculate residuals
    df['Predicted'] = four_pl(df['Concentration'], *params)
    df['Residual'] = df['Signal'] - df['Predicted']

    # --- Plotting ---
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=(f"<b>1. 4-Parameter Logistic Fit (Model: {model_type})</b>",
                                        "<b>2. Residuals vs. Concentration</b>"),
                        vertical_spacing=0.15)

    # Plot 1: 4PL Curve
    fig.add_trace(go.Scatter(x=df['Concentration'], y=df['Signal'], mode='markers',
                             name='Measured Data', marker=dict(size=8, color='blue')), row=1, col=1)
    
    x_fit = np.logspace(-2, 3, 100)
    y_fit = four_pl(x_fit, *params)
    fig.add_trace(go.Scatter(x=x_fit, y=y_fit, mode='lines',
                             name='4PL Fit', line=dict(color='red', dash='dash')), row=1, col=1)
    
    # Add annotations for key fitted parameters
    fig.add_hline(y=d_fit, line_dash='dot', annotation_text=f"d={d_fit:.2f}", row=1, col=1)
    fig.add_hline(y=a_fit, line_dash='dot', annotation_text=f"a={a_fit:.2f}", row=1, col=1)
    fig.add_vline(x=c_fit, line_dash='dot', annotation_text=f"EC50={c_fit:.2f}", row=1, col=1)
    
    # Plot 2: Residuals
    fig.add_trace(go.Scatter(x=df['Concentration'], y=df['Residual'], mode='markers',
                             name='Residuals', marker=dict(color='green', size=8)), row=2, col=1)
    fig.add_hline(y=0, line_dash='dash', line_color='black', row=2, col=1)

    fig.update_layout(height=800, title_text='<b>Non-Linear Regression for Bioassay Potency</b>', title_x=0.5,
                      legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.7)'))
    fig.update_xaxes(type="log", title_text="Concentration (log scale)", row=1, col=1)
    fig.update_yaxes(title_text="Signal Response", row=1, col=1)
    fig.update_xaxes(type="log", title_text="Concentration (log scale)", row=2, col=1)
    fig.update_yaxes(title_text="Residual (Signal - Predicted)", row=2, col=1)
    
    return fig, params, perr
    
# ==============================================================================
# HELPER & PLOTTING FUNCTION (ROC Curve) - SME ENHANCED
# ==============================================================================
@st.cache_data
def plot_roc_curve(diseased_mean=65, population_sd=10, cutoff=55):
    """
    Generates enhanced, more realistic, and interactive plots for the ROC curve module,
    including a dynamic cutoff point and visualized confusion matrix.
    """
    np.random.seed(0)
    
    healthy_mean = 45
    scores_diseased = np.random.normal(loc=diseased_mean, scale=population_sd, size=200)
    scores_healthy = np.random.normal(loc=healthy_mean, scale=population_sd, size=200)
    
    y_true = np.concatenate([np.ones(200), np.zeros(200)]) # 1 for diseased, 0 for healthy
    y_scores = np.concatenate([scores_diseased, scores_healthy])
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc_value = auc(fpr, tpr)

    # --- Calculations for dynamic cutoff point ---
    TP = np.sum((scores_diseased >= cutoff))
    FN = np.sum((scores_diseased < cutoff))
    TN = np.sum((scores_healthy < cutoff))
    FP = np.sum((scores_healthy >= cutoff))
    
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 1
    current_fpr = 1 - specificity
    
    ppv = TP / (TP + FP) if (TP + FP) > 0 else 0 # Positive Predictive Value
    npv = TN / (TN + FN) if (TN + FN) > 0 else 0 # Negative Predictive Value
    
    # --- Plotting ---
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("<b>1. Score Distributions & Interactive Cutoff</b>",
                        f"<b>2. Receiver Operating Characteristic (ROC) Curve | AUC = {auc_value:.3f}</b>"),
        vertical_spacing=0.15,
        row_heights=[0.6, 0.4]
    )

    # Plot 1: Distributions
    # Use density plots for a smoother look
    from scipy.stats import gaussian_kde
    x_range = np.linspace(min(scores_healthy.min(), scores_diseased.min()), max(scores_healthy.max(), scores_diseased.max()), 200)
    healthy_kde = gaussian_kde(scores_healthy)
    diseased_kde = gaussian_kde(scores_diseased)

    fig.add_trace(go.Scatter(x=x_range, y=healthy_kde(x_range), fill='tozeroy', name='Healthy', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_range, y=diseased_kde(x_range), fill='tozeroy', name='Diseased', line=dict(color='red')), row=1, col=1)
    
    # SME Enhancement: Shade the TP/FP/TN/FN areas
    fig.add_vrect(x0=cutoff, x1=x_range.max(), fillcolor="rgba(255, 0, 0, 0.2)", layer="below", line_width=0,
                  annotation_text="Positive Calls", annotation_position="top right", row=1, col=1)
    fig.add_vrect(x0=x_range.min(), x1=cutoff, fillcolor="rgba(0, 0, 255, 0.2)", layer="below", line_width=0,
                  annotation_text="Negative Calls", annotation_position="top left", row=1, col=1)

    fig.add_vline(x=cutoff, line_width=3, line_color='black', name='Cutoff',
                  annotation_text=f"Cutoff = {cutoff}", annotation_position="bottom right", row=1, col=1)

    # Plot 2: ROC Curve
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve', line=dict(color='darkorange', width=3)), row=2, col=1)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='No-Discrimination Line', line=dict(color='grey', width=2, dash='dash')), row=2, col=1)
    
    # SME Enhancement: Add the dynamic point for the selected cutoff
    fig.add_trace(go.Scatter(x=[current_fpr], y=[sensitivity], mode='markers',
                             marker=dict(color='black', size=15, symbol='x', line=dict(width=3)),
                             name='Current Cutoff Performance'), row=2, col=1)

    fig.update_layout(height=800, title_text="<b>Diagnostic Assay Performance Dashboard</b>", title_x=0.5,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_xaxes(title_text="Assay Score", row=1, col=1)
    fig.update_yaxes(title_text="Density", row=1, col=1)
    fig.update_xaxes(title_text="False Positive Rate (1 - Specificity)", range=[-0.05, 1.05], row=2, col=1)
    fig.update_yaxes(title_text="True Positive Rate (Sensitivity)", range=[-0.05, 1.05], row=2, col=1)
    
    return fig, auc_value, sensitivity, specificity, ppv, npv
    
@st.cache_data
def plot_tost(delta=5.0, true_diff=1.0, std_dev=5.0, n_samples=50):
    """
    Generates an enhanced, 3-plot dashboard for the TOST module that visually
    connects the raw data to the final statistical conclusion.
    """
    np.random.seed(1)
    data_A = np.random.normal(loc=100, scale=std_dev, size=n_samples)
    data_B = np.random.normal(loc=100 + true_diff, scale=std_dev, size=n_samples)
    
    mean_A, var_A = np.mean(data_A), np.var(data_A, ddof=1)
    mean_B, var_B = np.mean(data_B), np.var(data_B, ddof=1)
    diff_mean = mean_B - mean_A
    
    if n_samples <= 1:
        return go.Figure(), 1.0, False, 0, 0, 0, 0, 0

    std_err_diff = np.sqrt(var_A/n_samples + var_B/n_samples)
    df_welch = (std_err_diff**4) / (((var_A/n_samples)**2 / (n_samples-1)) + ((var_B/n_samples)**2 / (n_samples-1)))
    
    t_lower = (diff_mean - (-delta)) / std_err_diff
    t_upper = (diff_mean - delta) / std_err_diff
    p_lower, p_upper = stats.t.sf(t_lower, df_welch), stats.t.cdf(t_upper, df_welch)
    p_tost = max(p_lower, p_upper)
    is_equivalent = p_tost < 0.05
    
    ci_margin = t.ppf(0.95, df_welch) * std_err_diff
    ci_lower, ci_upper = diff_mean - ci_margin, diff_mean + ci_margin
    
    # --- THIS LINE WAS MOVED HERE ---
    ci_color = '#00CC96' if is_equivalent else '#EF553B' # Green for pass, Red for fail
    # --- END OF MOVE ---
    
    # --- PLOTTING ---
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.4, 0.4, 0.2],
        vertical_spacing=0.1,
        subplot_titles=("<b>1. Raw Data Distributions (The Samples)</b>",
                        "<b>2. Distribution of the Difference in Means (The Evidence)</b>",
                        "<b>3. Equivalence Test (The Verdict)</b>")
    )

    # Plot 1: Raw Data Distributions
    from scipy.stats import gaussian_kde
    x_range1 = np.linspace(min(data_A.min(), data_B.min()), max(data_A.max(), data_B.max()), 200)
    kde_A, kde_B = gaussian_kde(data_A), gaussian_kde(data_B)
    fig.add_trace(go.Scatter(x=x_range1, y=kde_A(x_range1), fill='tozeroy', name='Method A', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_range1, y=kde_B(x_range1), fill='tozeroy', name='Method B', line=dict(color='green')), row=1, col=1)
    fig.add_vline(x=mean_A, line=dict(color='royalblue', dash='dash'), row=1, col=1)
    fig.add_vline(x=mean_B, line=dict(color='darkgreen', dash='dash'), row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=1, col=1)
    
    # Plot 2: THE VISUAL BRIDGE - Distribution of the Difference
    x_range2 = np.linspace(diff_mean - 4*std_err_diff, diff_mean + 4*std_err_diff, 200)
    diff_pdf = stats.t.pdf(x_range2, df=df_welch, loc=diff_mean, scale=std_err_diff)
    fig.add_trace(go.Scatter(x=x_range2, y=diff_pdf, fill='tozeroy', name='Sampling Dist.', line=dict(color='grey')), row=2, col=1)
    
    # Shade the 90% CI area
    x_fill = np.linspace(ci_lower, ci_upper, 100)
    y_fill = stats.t.pdf(x_fill, df=df_welch, loc=diff_mean, scale=std_err_diff)
    fig.add_trace(go.Scatter(x=x_fill, y=y_fill, fill='tozeroy', name='90% CI', line=dict(color=ci_color), fillcolor=ci_color), row=2, col=1)
    fig.add_vrect(x0=-delta, x1=delta, fillcolor="rgba(0,128,0,0.1)", layer="below", line_width=0, row=2, col=1)
    fig.add_vline(x=-delta, line=dict(color="red", dash="dash"), row=2, col=1)
    fig.add_vline(x=delta, line=dict(color="red", dash="dash"), row=2, col=1)
    fig.update_yaxes(showticklabels=False, row=2, col=1)

    # Plot 3: The Verdict CI Bar
    fig.add_trace(go.Scatter(
        x=[diff_mean], y=[0], error_x=dict(type='data', array=[ci_upper-diff_mean], arrayminus=[diff_mean-ci_lower], thickness=15),
        mode='markers', name='90% CI for Diff.', marker=dict(color=ci_color, size=18, line=dict(width=2, color='black'))
    ), row=3, col=1)
    fig.add_vrect(x0=-delta, x1=delta, fillcolor="rgba(0,128,0,0.1)", layer="below", line_width=0, row=3, col=1)
    fig.add_vline(x=-delta, line=dict(color="red", dash="dash"), row=3, col=1)
    fig.add_vline(x=delta, line=dict(color="red", dash="dash"), row=3, col=1)
    fig.update_yaxes(showticklabels=False, range=[-1, 1], row=3, col=1)
    
    fig.update_layout(height=800, title_text="<b>Equivalence Testing: From Raw Data to Verdict</b>", title_x=0.5, showlegend=False)
    fig.update_xaxes(title_text="Measured Value", row=1, col=1)
    fig.update_xaxes(title_text="Difference in Means", row=2, col=1)
    fig.update_xaxes(title_text="Difference in Means", row=3, col=1)
    
    return fig, p_tost, is_equivalent, ci_lower, ci_upper, mean_A, mean_B, diff_mean

# ==============================================================================
# HELPER & PLOTTING FUNCTION (DOE/RSM) - SME ENHANCED
# ==============================================================================
@st.cache_data
def plot_doe_robustness(ph_effect=2.0, temp_effect=5.0, interaction_effect=0.0, ph_quad_effect=-5.0, temp_quad_effect=-5.0, noise_sd=1.0):
    """
    Generates enhanced, more realistic dynamic RSM plots for the DOE module,
    including a Pareto plot of effects and ANOVA table.
    """
    np.random.seed(42)
    
    # 1. Design the experiment in coded units (-alpha to +alpha)
    alpha = 1.414
    design_coded = {
        'pH_coded':  [-1, 1, -1, 1, -alpha, alpha, 0, 0, 0, 0, 0, 0, 0],
        'Temp_coded':[-1, -1, 1, 1, 0, 0, -alpha, alpha, 0, 0, 0, 0, 0]
    }
    df = pd.DataFrame(design_coded)
    
    # Map coded units to realistic "real" units
    df['pH'] = df['pH_coded'] * 0.5 + 7.0   # e.g., pH 6.5 to 7.5
    df['Temp'] = df['Temp_coded'] * 5 + 30  # e.g., Temp 25 to 35 C

    # 2. Simulate the response using the full quadratic model in CODED units
    true_response = 100 + \
                    ph_effect * df['pH_coded'] + \
                    temp_effect * df['Temp_coded'] + \
                    interaction_effect * df['pH_coded'] * df['Temp_coded'] + \
                    ph_quad_effect * (df['pH_coded']**2) + \
                    temp_quad_effect * (df['Temp_coded']**2)
    
    df['Response'] = true_response + np.random.normal(0, noise_sd, len(df))

    # 3. Analyze the results with a quadratic OLS model using coded variables
    model = ols('Response ~ pH_coded + Temp_coded + I(pH_coded**2) + I(Temp_coded**2) + pH_coded:Temp_coded', data=df).fit()
    
    # SME Enhancement: Create a user-friendly ANOVA summary table
    anova_table = sm.stats.anova_lm(model, typ=2)
    anova_summary = anova_table[['sum_sq', 'df', 'F', 'PR(>F)']].reset_index()
    anova_summary.columns = ['Term', 'Sum of Squares', 'df', 'F-value', 'p-value']
    anova_summary['Term'] = anova_summary['Term'].str.replace('_coded', '') # Clean up names
    
    # 4. Create the prediction grid for the surfaces (in coded units)
    x_range_coded = np.linspace(-1.5, 1.5, 50)
    y_range_coded = np.linspace(-1.5, 1.5, 50)
    xx, yy = np.meshgrid(x_range_coded, y_range_coded)
    grid = pd.DataFrame({'pH_coded': xx.ravel(), 'Temp_coded': yy.ravel()})
    pred = model.predict(grid).values.reshape(xx.shape)

    # 5. Find the optimum point from the model
    max_idx = np.unravel_index(np.argmax(pred), pred.shape)
    opt_ph_coded = x_range_coded[max_idx[1]]
    opt_temp_coded = y_range_coded[max_idx[0]]
    max_response = np.max(pred)
    # Convert optimum back to real units for reporting
    opt_ph_real = opt_ph_coded * 0.5 + 7.0
    opt_temp_real = opt_temp_coded * 5 + 30
    
    # 6. Create the 2D Contour Plot (in real units for interpretation)
    x_range_real = x_range_coded * 0.5 + 7.0
    y_range_real = y_range_coded * 5 + 30
    fig_contour = go.Figure(data=[
        go.Contour(z=pred, x=x_range_real, y=y_range_real, colorscale='Viridis',
                    contours=dict(coloring='lines', showlabels=True, labelfont=dict(size=12, color='white'))),
        go.Scatter(x=df['pH'], y=df['Temp'], mode='markers',
                   marker=dict(color='red', size=12, line=dict(width=2, color='black')), name='Design Points')
    ])
    fig_contour.add_trace(go.Scatter(x=[opt_ph_real], y=[opt_temp_real], mode='markers',
                                     marker=dict(color='gold', size=18, symbol='star', line=dict(width=2, color='black')),
                                     name='Predicted Optimum'))
    fig_contour.update_layout(title='<b>2D Response Surface (Contour Plot)</b>',
                              xaxis_title="pH (Real Units)", yaxis_title="Temperature (°C, Real Units)")

    # 7. Create the 3D Surface Plot
    fig_3d = go.Figure(data=[
        go.Surface(z=pred, x=x_range_real, y=y_range_real, colorscale='Viridis', opacity=0.8),
        go.Scatter3d(x=df['pH'], y=df['Temp'], z=df['Response'], mode='markers', 
                      marker=dict(color='red', size=5, line=dict(width=2, color='black')), name='Design Points')
    ])
    fig_3d.update_layout(title='<b>3D Response Surface Plot</b>',
                         scene=dict(xaxis_title='pH', yaxis_title='Temp (°C)', zaxis_title='Response'),
                         margin=dict(l=0, r=0, b=0, t=40))

    # 8. SME Enhancement: Create a Pareto Plot of Standardized Effects
    effects = model.params[1:] # Exclude intercept
    std_errs = model.bse[1:]
    t_values = np.abs(effects / std_errs)
    
    # Get p-values from ANOVA table to color the bars
    p_values_map = anova_summary.set_index('Term')['p-value']
    effect_names = ['pH', 'Temp', 'I(pH**2)', 'I(Temp**2)', 'pH:Temp']
    p_values = [p_values_map.get(name, 1.0) for name in effect_names]
    
    effects_df = pd.DataFrame({'Effect': effect_names, 't-value': t_values, 'p-value': p_values})
    effects_df = effects_df.sort_values(by='t-value', ascending=False)
    
    fig_pareto = px.bar(effects_df, x='Effect', y='t-value',
                        title='<b>Pareto Plot of Standardized Effects</b>',
                        labels={'Effect': 'Model Term', 't-value': 'Absolute t-value (Effect Magnitude)'},
                        color=effects_df['p-value'] < 0.05,
                        color_discrete_map={True: '#00CC96', False: '#636EFA'},
                        template='plotly_white')
    # Add significance threshold line
    t_crit_pareto = stats.t.ppf(1 - 0.05 / 2, df=model.df_resid)
    fig_pareto.add_hline(y=t_crit_pareto, line_dash="dash", line_color="red",
                         annotation_text=f"Significance (p=0.05)", annotation_position="bottom right")
    fig_pareto.update_layout(showlegend=False)

    return fig_contour, fig_3d, fig_pareto, anova_summary, opt_ph_real, opt_temp_real, max_response

@st.cache_data
def plot_mixture_design(a_effect, b_effect, c_effect, ab_interaction, ac_interaction, bc_interaction, noise_sd, response_threshold):
    """
    Generates a professional-grade dashboard for a mixture design of experiments.
    """
    # 1. Define the experimental design points (Simplex-Lattice Design)
    points = [[1,0,0], [0,1,0], [0,0,1], [0.5,0.5,0], [0.5,0,0.5], [0,0.5,0.5], [1/3,1/3,1/3]]
    df = pd.DataFrame(points, columns=['A', 'B', 'C'])

    # 2. Simulate the response using the Scheffé quadratic model
    a, b, c = df['A'], df['B'], df['C']
    true_response = (a_effect * a) + (b_effect * b) + (c_effect * c) + \
                    (ab_interaction * a * b) + (ac_interaction * a * c) + (bc_interaction * b * c)
    df['Response'] = true_response + np.random.normal(0, noise_sd, len(df))

    # 3. Fit the model
    model = ols('Response ~ A + B + C + A:B + A:C + B:C - 1', data=df).fit()
    
    # 4. Create a dense grid of points for the ternary plot surface
    @st.cache_data
    def generate_ternary_grid(n=40):
        import itertools
        s = itertools.product(range(n + 1), repeat=3)
        points = np.array(list(s))
        points = points[points.sum(axis=1) == n] / n
        return pd.DataFrame(points, columns=['A', 'B', 'C'])
        
    grid = generate_ternary_grid()
    grid['Predicted_Response'] = model.predict(grid)
    
    # 5. Generate Plot 1: Model Effects
    effects = model.params
    effect_df = pd.DataFrame({'Term': effects.index, 'Coefficient': effects.values})
    effect_df['Type'] = ['Interaction' if ':' in term else 'Main Effect' for term in effect_df['Term']]
    fig_effects = px.bar(effect_df, x='Coefficient', y='Term', color='Type', orientation='h',
                         title='<b>1. Model Effects Plot</b>',
                         labels={'Coefficient': 'Coefficient Value (Impact on Response)'})
    fig_effects.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False)

    # 6. Generate Plot 2: Professional Ternary Map
    opt_idx = grid['Predicted_Response'].idxmax()
    opt_blend = grid.loc[opt_idx]
    
    fig_ternary = go.Figure()

    # --- THIS IS THE CORRECTED PLOTTING LOGIC ---
    # Layer 1: The heatmap surface, created with a scatterternary trace
    fig_ternary.add_trace(go.Scatterternary(
        a=grid['A'], b=grid['B'], c=grid['C'],
        mode='markers',
        marker=dict(
            color=grid['Predicted_Response'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Response<br>(e.g., Solubility)'),
            size=5
        ),
        hoverinfo='none'
    ))

    # Layer 2: The "Sweet Spot" / Design Space boundary
    sweet_spot_grid = grid[grid['Predicted_Response'] >= response_threshold]
    fig_ternary.add_trace(go.Scatterternary(
        a=sweet_spot_grid['A'], b=sweet_spot_grid['B'], c=sweet_spot_grid['C'],
        mode='markers',
        marker=dict(color=SUCCESS_GREEN, size=5, opacity=0.6),
        hoverinfo='none',
        name='Design Space'
    ))
    # --- END OF CORRECTION ---

    # Layer 3: The experimental points
    fig_ternary.add_trace(go.Scatterternary(
        a=df['A'], b=df['B'], c=df['C'], mode='markers',
        marker=dict(symbol='circle', color='red', size=12, line=dict(width=2, color='black')),
        name='DOE Runs'
    ))
    
    # Layer 4: The predicted optimum
    fig_ternary.add_trace(go.Scatterternary(
        a=[opt_blend['A']], b=[opt_blend['B']], c=[opt_blend['C']], mode='markers',
        marker=dict(symbol='star', color='white', size=18, line=dict(width=2, color='black')),
        name='Predicted Optimum'
    ))
    
    fig_ternary.update_layout(
        title='<b>2. Formulation Design Space Map</b>',
        ternary_sum=1,
        ternary_aaxis_title_text='<b>Component A (%)</b>',
        ternary_baxis_title_text='<b>Component B (%)</b>',
        ternary_caxis_title_text='<b>Component C (%)</b>',
        margin=dict(l=40, r=40, b=40, t=60), showlegend=False
    )
    
    return fig_effects, fig_ternary, model, opt_blend
    
@st.cache_data
def plot_doe_optimization_suite(ph_effect, temp_effect, interaction_effect, ph_quad_effect, temp_quad_effect, asymmetry_effect, noise_sd, yield_threshold):
    """
    Generates a full suite of professional-grade plots: Pareto, 3D RSM, 2D RSM Map, and 2D ML PDP.
    """
    np.random.seed(42)
    # 1. Simulate a realistic, asymmetric process
    design_coded = {'pH_coded': [-1, 1, -1, 1, -1.414, 1.414, 0, 0, 0, 0, 0], 'Temp_coded': [-1, -1, 1, 1, 0, 0, -1.414, 1.414, 0, 0, 0]}
    df = pd.DataFrame(design_coded)
    df['pH_real'] = df['pH_coded'] * 0.5 + 7.2
    df['Temp_real'] = df['Temp_coded'] * 5 + 37
    true_response = 100 + ph_effect*df['pH_coded'] + temp_effect*df['Temp_coded'] + interaction_effect*df['pH_coded']*df['Temp_coded'] + \
                    ph_quad_effect*(df['pH_coded']**2) + temp_quad_effect*(df['Temp_coded']**2) + asymmetry_effect*(df['pH_coded']**3)
    df['Response'] = true_response + np.random.normal(0, noise_sd, len(df))

    # 2. Fit both RSM and ML models
    X = df[['pH_coded', 'Temp_coded']]
    y = df['Response']
    rsm_model = ols('Response ~ pH_coded + Temp_coded + I(pH_coded**2) + I(Temp_coded**2) + pH_coded:Temp_coded', data=df).fit()
    ml_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42).fit(X, y)

    # 3. Create prediction grids
    x_range_coded = np.linspace(-2, 2, 100); y_range_coded = np.linspace(-2, 2, 100)
    xx_c, yy_c = np.meshgrid(x_range_coded, y_range_coded)
    grid = pd.DataFrame({'pH_coded': xx_c.ravel(), 'Temp_coded': yy_c.ravel()})
    pred_rsm = rsm_model.predict(grid).values.reshape(xx_c.shape)
    pred_ml = ml_model.predict(grid).reshape(xx_c.shape)
    x_range_real = x_range_coded * 0.5 + 7.2; y_range_real = y_range_coded * 5 + 37

    # 4. Find optimum and define NOR
    max_idx = np.unravel_index(np.argmax(pred_rsm), pred_rsm.shape)
    opt_temp_real, opt_ph_real = y_range_real[max_idx[1]], x_range_real[max_idx[0]]
    max_response = np.max(pred_rsm)
    nor = {'x0': opt_temp_real - 1, 'y0': opt_ph_real - 0.1, 'x1': opt_temp_real + 1, 'y1': opt_ph_real + 0.1}

    # 5. Generate Pareto Plot
    anova_for_pareto = sm.stats.anova_lm(rsm_model, typ=2)
    anova_filtered = anova_for_pareto.drop('Residual', errors='ignore')
    p_values_map = anova_filtered['PR(>F)']
    effects = rsm_model.params[1:]; std_errs = rsm_model.bse[1:]
    t_values = np.abs(effects / std_errs)
    effect_names = ['pH', 'Temp', 'pH²', 'Temp²', 'pH:Temp']
    effects_df = pd.DataFrame({'Effect': effect_names, 't-value': t_values.values, 'p-value': p_values_map.values}).sort_values('t-value', ascending=False)
    
    fig_pareto = px.bar(effects_df, x='Effect', y='t-value', title='<b>1. Pareto Plot of Effects</b>',
                        color=effects_df['p-value'] < 0.05,
                        color_discrete_map={True: SUCCESS_GREEN, False: PRIMARY_COLOR}, template='plotly_white')
    t_crit = stats.t.ppf(1 - 0.05 / 2, df=rsm_model.df_resid)
    fig_pareto.add_hline(y=t_crit, line_dash="dash", line_color="red", annotation_text="Significance (p=0.05)")
    fig_pareto.update_layout(showlegend=False)

    # 6. Generate Professional 3D Surface Plot (Code is correct)
    fig_rsm_3d = go.Figure(data=[go.Surface(z=pred_rsm, x=y_range_real, y=x_range_real, colorscale='Plasma', cmin=80, cmax=100, opacity=0.9, colorbar_title='Yield')])
    fig_rsm_3d.add_trace(go.Scatter3d(x=df['Temp_real'], y=df['pH_real'], z=df['Response'], mode='markers', marker=dict(color='red', size=5, line=dict(width=2, color='black')), name='DOE Runs'))
    fig_rsm_3d.update_layout(title='<b>2a. RSM Model (3D Surface)</b>', scene=dict(xaxis_title='Temp', yaxis_title='pH', zaxis_title='Yield'), margin=dict(l=0,r=0,b=0,t=40))

    # 7. Generate Professional 2D Topographic Map (Code is correct)
    fig_rsm_2d = go.Figure()
    fig_rsm_2d.add_trace(go.Contour(z=pred_rsm, x=y_range_real, y=x_range_real, colorscale='Geyser', contours_coloring='fill', showscale=False))
    fig_rsm_2d.add_trace(go.Contour(z=(pred_rsm >= yield_threshold).astype(int), x=y_range_real, y=x_range_real, contours_coloring='fill', showscale=False, colorscale=[[0, 'rgba(239, 83, 80, 0.4)'], [1, 'rgba(44, 160, 44, 0.4)']], line_width=0))
    fig_rsm_2d.add_shape(type="rect", x0=nor['x0'], y0=nor['y0'], x1=nor['x1'], y1=nor['y1'], line=dict(color='white', width=3, dash='dash'))
    fig_rsm_2d.add_trace(go.Scatter(x=[opt_temp_real], y=[opt_ph_real], mode='markers', marker=dict(color='white', size=18, symbol='star', line=dict(width=2, color='black'))))
    fig_rsm_2d.add_trace(go.Scatter(x=df['Temp_real'], y=df['pH_real'], mode='markers', marker=dict(color='red', size=8, line=dict(width=1, color='black'))))
    fig_rsm_2d.add_annotation(x=np.mean([nor['x0'], nor['x1']]), y=np.mean([nor['y0'], nor['y1']]), text="<b>NOR</b>", showarrow=False, font=dict(color='white', size=16))
    fig_rsm_2d.update_layout(title='<b>2b. RSM Topographic Map (PAR & NOR)</b>', xaxis_title='Temperature (°C)', yaxis_title='pH', margin=dict(l=0,r=0,b=0,t=40), showlegend=False)
    
    # 8. Generate 2D PDP Heatmap from ML Model (Code is correct)
    fig_pdp, ax_pdp = plt.subplots(figsize=(8, 6))
    display = PartialDependenceDisplay.from_estimator(estimator=ml_model, X=X, features=[(0, 1)], feature_names=['pH (coded)', 'Temp (coded)'], kind="average", ax=ax_pdp, contour_kw={"cmap": "viridis"})
    ax_pdp.set_title("3. ML Model (2D Partial Dependence)", fontsize=16)
    pdp_buffer = io.BytesIO()
    fig_pdp.savefig(pdp_buffer, format='png', bbox_inches='tight')
    plt.close(fig_pdp)
    pdp_buffer.seek(0)

    # --- THIS IS THE SECOND KEY FIX ---
    # Create the final anova_table for display with the correct column names
    anova_display_table = sm.stats.anova_lm(rsm_model, typ=2).reset_index()
    anova_display_table.columns = ['Term', 'Sum of Squares', 'df', 'F-value', 'p-value']
    # --- END OF FIX ---
    
    return fig_pareto, fig_rsm_3d, fig_rsm_2d, pdp_buffer, anova_display_table, opt_ph_real, opt_temp_real, max_response
    
# ==============================================================================
# HELPER & PLOTTING FUNCTION (Split-Plot) - SME ENHANCED
# ==============================================================================
@st.cache_data
def plot_split_plot_doe(lot_variation_sd=0.5, interaction_effect=0.0):
    """
    Generates enhanced, more realistic dynamic plots for a Split-Plot DOE,
    including a controllable interaction effect and a dedicated interaction plot.
    """
    np.random.seed(42)
    
    # --- Define the Experimental Design ---
    lots = ['Lot A', 'Lot B']
    concentrations = [10, 20, 30] # mg/L
    n_replicates = 4 

    # --- Dynamic Data Generation ---
    data = []
    # Simulate a "true" effect for the lots
    # To make the effect consistent, we'll use a fixed shift, scaled by the SD slider
    lot_effects = {'Lot A': 0, 'Lot B': -2 * lot_variation_sd}
    
    # SME Enhancement: Add a controllable interaction effect
    # The effect of the supplement is now different for Lot B
    for lot in lots:
        for conc in concentrations:
            supplement_effect = (conc - 10) * 0.5
            current_interaction = 0
            if lot == 'Lot B':
                # Interaction term: scales with supplement concentration
                current_interaction = interaction_effect * (conc - 10) / 10 

            true_mean = 100 + supplement_effect + lot_effects[lot] + current_interaction
            measurements = np.random.normal(true_mean, 1.5, n_replicates)
            for m in measurements:
                data.append([lot, conc, m])

    df = pd.DataFrame(data, columns=['Lot', 'Supplement', 'Response'])

    # --- Analyze the data with ANOVA ---
    # Note: A proper split-plot uses a mixed model, but for visualization and p-values,
    # a standard ANOVA is a reasonable approximation here.
    model = ols('Response ~ C(Lot) * C(Supplement)', data=df).fit() # Use '*' for interaction
    anova_table = sm.stats.anova_lm(model, typ=2).reset_index()
    anova_table.columns = ['Term', 'Sum of Squares', 'df', 'F-value', 'p-value']
    
    # --- Plotting ---
    fig_main = px.box(df, x='Lot', y='Response', color=df['Supplement'].astype(str),
                 title='<b>1. Split-Plot Experimental Results</b>',
                 labels={
                     "Lot": "Base Media Lot (Hard-to-Change)",
                     "Response": "Cell Viability (%)",
                     "color": "Supplement Conc. (mg/L)"
                 },
                 points='all')
    
    # SME Enhancement: Add mean lines to the box plot for clarity
    mean_data = df.groupby(['Lot', 'Supplement'])['Response'].mean().reset_index()
    for conc in concentrations:
        subset = mean_data[mean_data['Supplement'] == conc]
        fig_main.add_trace(go.Scatter(x=subset['Lot'], y=subset['Response'], mode='lines',
                                      line=dict(width=3, dash='dash'), showlegend=False))

    fig_main.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))

    # SME Enhancement: Create a dedicated Interaction Plot
    fig_interaction = px.line(mean_data, x='Supplement', y='Response', color='Lot',
                              title='<b>2. Interaction Plot</b>',
                              labels={'Supplement': 'Supplement Conc. (mg/L)', 'Response': 'Mean Cell Viability (%)'},
                              markers=True)
    fig_interaction.update_layout(xaxis=dict(tickvals=concentrations),
                                  legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99))
    fig_interaction.add_annotation(x=20, y=mean_data['Response'].min(),
                                   text="<i>Parallel lines = No Interaction<br>Non-parallel lines = Interaction</i>",
                                   showarrow=False, yshift=-40)

    return fig_main, fig_interaction, anova_table

# ==============================================================================
# HELPER & PLOTTING FUNCTION (Causal Inference) - SME ENHANCED
# ==============================================================================
@st.cache_data
def plot_causal_inference(confounding_strength=5.0):
    """
    Generates enhanced, more realistic dynamic plots for the Causal Inference module,
    featuring a professionally redesigned Directed Acyclic Graph (DAG).
    """
    # --- 1. The Causal Map (DAG) - Redesigned for professional rendering ---
    fig_dag = go.Figure()

    # Node positions and properties
    nodes = {
        'Sensor Reading': {'pos': (0, 0), 'color': '#636EFA'},
        'Product Purity': {'pos': (3, 0), 'color': '#00CC96'},
        'Calibration Age': {'pos': (1.5, 1.5), 'color': '#EF553B'}
    }
    
    # --- THIS IS THE CORRECTED BLOCK ---
    # Add Edges (Arrows) with color-coding for paths
    # The arguments `axshift` and `ayshift` have been removed.
    # The arrow's tail is now precisely controlled by `ax` and `ay` in data coordinates.
    # Causal Path (Green)
    fig_dag.add_annotation(x=nodes['Product Purity']['pos'][0] - 0.45, y=nodes['Product Purity']['pos'][1], # Arrow head
                           ax=nodes['Sensor Reading']['pos'][0] + 0.45, ay=nodes['Sensor Reading']['pos'][1], # Arrow tail
                           xref='x', yref='y', axref='x', ayref='y', showarrow=True,
                           arrowhead=2, arrowwidth=4, arrowcolor='#00CC96')
    
    # Backdoor Path (Red)
    fig_dag.add_annotation(x=nodes['Sensor Reading']['pos'][0] + 0.15, y=nodes['Sensor Reading']['pos'][1] + 0.4,
                           ax=nodes['Calibration Age']['pos'][0] - 0.15, ay=nodes['Calibration Age']['pos'][1] - 0.4,
                           xref='x', yref='y', axref='x', ayref='y', showarrow=True,
                           arrowhead=2, arrowwidth=3, arrowcolor='#EF553B')
    fig_dag.add_annotation(x=nodes['Product Purity']['pos'][0] - 0.15, y=nodes['Product Purity']['pos'][1] + 0.4,
                           ax=nodes['Calibration Age']['pos'][0] + 0.15, ay=nodes['Calibration Age']['pos'][1] - 0.4,
                           xref='x', yref='y', axref='x', ayref='y', showarrow=True,
                           arrowhead=2, arrowwidth=3, arrowcolor='#EF553B')
    # --- END OF CORRECTION ---

    # Add Nodes (Circles)
    for name, attrs in nodes.items():
        fig_dag.add_shape(type="circle", xref="x", yref="y",
                          x0=attrs['pos'][0] - 0.5, y0=attrs['pos'][1] - 0.5,
                          x1=attrs['pos'][0] + 0.5, y1=attrs['pos'][1] + 0.5,
                          line_color="Black", fillcolor=attrs['color'], line_width=2)
        fig_dag.add_annotation(x=attrs['pos'][0], y=attrs['pos'][1], text=f"<b>{name.replace(' ', '<br>')}</b>",
                               showarrow=False, font=dict(color='white', size=12))

    # Add Path Labels
    fig_dag.add_annotation(x=1.5, y=-0.3, text="<b><span style='color:#00CC96'>Direct Causal Path</span></b>",
                           showarrow=False, font_size=14)
    fig_dag.add_annotation(x=1.5, y=0.8, text="<b><span style='color:#EF553B'>Confounding 'Backdoor' Path</span></b>",
                           showarrow=False, font_size=14)

    fig_dag.update_layout(
        title="<b>1. The Causal Map (DAG): Calibration Drift Scenario</b>",
        showlegend=False, xaxis=dict(visible=False, showgrid=False, range=[-1, 4]),
        yaxis=dict(visible=False, showgrid=False, range=[-1, 2.5]),
        height=400, margin=dict(t=50, b=20), plot_bgcolor='rgba(0,0,0,0)'
    )

    # --- 2. Simulate data demonstrating Simpson's Paradox ---
    np.random.seed(42)
    n_samples = 200
    cal_age = np.random.randint(0, 2, n_samples)
    
    true_causal_effect_sensor_on_purity = 0.8
    true_effect_age_on_purity = -confounding_strength
    true_effect_age_on_sensor = confounding_strength
    
    sensor = 50 + true_effect_age_on_sensor * cal_age + np.random.normal(0, 5, n_samples)
    purity = 90 + true_causal_effect_sensor_on_purity * (sensor - 50) + true_effect_age_on_purity * cal_age + np.random.normal(0, 5, n_samples)
    
    df = pd.DataFrame({'SensorReading': sensor, 'Purity': purity, 'CalibrationAge': cal_age})
    df['calibration_status'] = df['CalibrationAge'].apply(lambda x: 'Old' if x == 1 else 'New')

    # --- 3. Calculate effects ---
    naive_model = ols('Purity ~ SensorReading', data=df).fit()
    naive_effect = naive_model.params['SensorReading']
    
    adjusted_model = ols('Purity ~ SensorReading + C(calibration_status)', data=df).fit()
    adjusted_effect = adjusted_model.params['SensorReading']

    # --- 4. Create the scatter plot ---
    fig_scatter = px.scatter(df, x='SensorReading', y='Purity', color='calibration_status',
                             title="<b>2. Simpson's Paradox: The Danger of Confounding</b>",
                             color_discrete_map={'New': 'blue', 'Old': 'red'},
                             labels={'SensorReading': 'In-Process Sensor Reading', 'calibration_status': 'Calibration Status'})
    
    x_range = np.array([df['SensorReading'].min(), df['SensorReading'].max()])
    
    fig_scatter.add_trace(go.Scatter(x=x_range, y=naive_model.predict({'SensorReading': x_range}), mode='lines', 
                                     name='Naive Correlation (Misleading)', line=dict(color='orange', width=4, dash='dash')))
    
    intercept_new = adjusted_model.params['Intercept']
    intercept_old = intercept_new + adjusted_model.params['C(calibration_status)[T.Old]']
    fig_scatter.add_trace(go.Scatter(x=x_range, y=intercept_new + adjusted_effect * x_range, mode='lines', 
                                     name='True Causal Effect (Within Groups)', line=dict(color='darkgreen', width=4)))
    fig_scatter.add_trace(go.Scatter(x=x_range, y=intercept_old + adjusted_effect * x_range, mode='lines', 
                                     showlegend=False, line=dict(color='darkgreen', width=4)))

    fig_scatter.update_layout(height=500, legend=dict(x=0.01, y=0.99))
    
    return fig_dag, fig_scatter, naive_effect, adjusted_effect
##==========================================================================================================================================================================================
##=============================================================================================END ACT I ===================================================================================
##==========================================================================================================================================================================================
@st.cache_data
def plot_sample_size_curves(confidence_level, reliability, lot_size, calc_method, required_n):
    """
    Generates a plot showing the trade-off between sample size and achievable reliability.
    """
    c = confidence_level / 100
    r_req = reliability / 100
    
    # Define the range of sample sizes to plot
    max_n = 3 * required_n if isinstance(required_n, int) and required_n > 50 else 300
    n_range = np.arange(1, max_n)

    # --- Calculate Achievable Reliability for each model ---
    # Binomial Calculation (inverse of the main formula)
    r_binomial = (1 - c)**(1 / n_range)

    # Hypergeometric Calculation (iterative solve for R at each n)
    @st.cache_data
    def solve_hypergeometric_r(n, M, C):
        log_alpha = math.log(1 - C)
        # Iterate backwards from the max possible defects to find the highest D that works
        for D in range(int(0.5 * M), -1, -1):
            if n > M - D: continue # Cannot sample more than the number of good items
            log_prob_zero_defect = (
                math.lgamma(M - D + 1) - math.lgamma(n + 1) - math.lgamma(M - D - n + 1)
            ) - (
                math.lgamma(M + 1) - math.lgamma(n + 1) - math.lgamma(M - n + 1)
            )
            if log_prob_zero_defect <= log_alpha:
                return (M - D) / M
        return 0.5 # Return a baseline if no solution found
    
    r_hypergeometric = [solve_hypergeometric_r(n, lot_size, c) for n in n_range] if lot_size else None

    # --- Create the Plot ---
    fig = go.Figure()

    # Add Binomial Curve
    fig.add_trace(go.Scatter(
        x=n_range, y=r_binomial, mode='lines', name='Binomial Model (Infinite Lot)',
        line=dict(color=PRIMARY_COLOR, width=3)
    ))

    # Add Hypergeometric Curve if applicable
    if r_hypergeometric and "Hypergeometric" in calc_method:
        fig.add_trace(go.Scatter(
            x=n_range, y=r_hypergeometric, mode='lines', name=f'Hypergeometric (Lot Size={lot_size})',
            line=dict(color='#FF7F0E', width=3, dash='dash')
        ))
        
    # Add the user's requirement point and lines
    if isinstance(required_n, int):
        fig.add_trace(go.Scatter(
            x=[required_n], y=[r_req], mode='markers', name='Your Requirement',
            marker=dict(color=SUCCESS_GREEN, size=15, symbol='star', line=dict(width=2, color='black'))
        ))
        # Add dashed lines to the axes
        fig.add_shape(type="line", x0=0, y0=r_req, x1=required_n, y1=r_req, line=dict(color="grey", width=2, dash="dash"))
        fig.add_shape(type="line", x0=required_n, y0=0, x1=required_n, y1=r_req, line=dict(color="grey", width=2, dash="dash"))

    fig.update_layout(
        title="<b>Sample Size vs. Achievable Reliability</b>",
        xaxis_title="Sample Size (n) with Zero Failures",
        yaxis_title=f"Achievable Reliability at {confidence_level:.1f}% Confidence",
        yaxis=dict(tickformat=".2%", range=[min(r_binomial) - 0.01, 1.01]),
        xaxis=dict(range=[0, max_n]),
        legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99)
    )

    return fig

@st.cache_data
def plot_stability_design_comparison(strengths, containers, design_type):
    """
    Generates a professional-grade dashboard comparing full, bracketing, and matrixing stability study designs.
    """
    time_points = ["0", "3", "6", "9", "12", "18", "24", "36"]
    
    # --- Generate the full design matrix ---
    full_design = []
    for s in strengths:
        for c in containers:
            full_design.append({'Strength': s, 'Container': c, 'Timepoints': len(time_points)})
    df_full = pd.DataFrame(full_design)
    
    # --- Determine the reduced design based on user selection ---
    df_reduced = df_full.copy()
    pulls_saved = 0
    
    if design_type == "Bracketing":
        min_s, max_s = min(strengths), max(strengths)
        df_reduced = df_full[df_full['Strength'].isin([min_s, max_s])].copy()
    
    elif design_type == "Matrixing":
        # A simple half-matrix design for demonstration
        df_reduced['Timepoints'] = 0
        for i, row in df_full.iterrows():
            if (i % 2) == 0: # Test every other combination at all timepoints
                df_reduced.loc[i, 'Timepoints'] = len(time_points)
            else: # For the others, test only at key timepoints
                df_reduced.loc[i, 'Timepoints'] = 3 # e.g., 0, 12, 36
    
    total_pulls_full = df_full['Timepoints'].sum()
    total_pulls_reduced = df_reduced['Timepoints'].sum()
    pulls_saved = total_pulls_full - total_pulls_reduced

    # --- Create the Visualization (Heatmap Table) ---
    def create_heatmap_table(df, title):
        pivot = df.pivot(index='Container', columns='Strength', values='Timepoints').fillna(0).astype(int)
        fig = px.imshow(pivot, text_auto=True, aspect="auto",
                        labels=dict(x="Strength (mg)", y="Container Type", color="Pulls"),
                        title=f"<b>{title} (Total Pulls: {df['Timepoints'].sum()})</b>",
                        color_continuous_scale='Greens')
        fig.update_xaxes(side="top")
        return fig

    fig_full = create_heatmap_table(df_full, "Full Study Design")
    fig_reduced = create_heatmap_table(df_reduced, f"Reduced Study Design ({design_type})")
    
    return fig_full, fig_reduced, pulls_saved, total_pulls_full, total_pulls_reduced

@st.cache_data
def plot_spc_charts(scenario='Stable'):
    """
    Generates dynamic SPC charts based on a selected process scenario.
    """
    np.random.seed(42)
    n_points = 25
    
    # --- Generate Base Data ---
    data_i = np.random.normal(loc=100.0, scale=2.0, size=n_points)
    data_xbar = np.random.normal(loc=100, scale=5, size=(n_points, 5))
    data_p_defects = np.random.binomial(n=200, p=0.02, size=n_points)
    
    # --- Inject Special Cause based on Scenario ---
    if scenario == 'Sudden Shift':
        data_i[15:] += 8
        data_xbar[15:, :] += 6
        data_p_defects[15:] = np.random.binomial(n=200, p=0.08, size=10)
    elif scenario == 'Gradual Trend':
        trend = np.linspace(0, 10, n_points)
        data_i += trend
        data_xbar += trend[:, np.newaxis]
        data_p_defects += np.random.binomial(n=200, p=trend/200, size=n_points)
    elif scenario == 'Increased Variability':
        data_i[15:] = np.random.normal(loc=100.0, scale=6.0, size=10)
        data_xbar[15:, :] = np.random.normal(loc=100, scale=15, size=(10, 5))
        data_p_defects[15:] = np.random.binomial(n=200, p=0.02, size=10) # Less obvious on p-chart

    # --- I-MR Chart ---
    x_i = np.arange(1, len(data_i) + 1)
    limit_data_i = data_i[:15] if scenario != 'Stable' else data_i
    mean_i = np.mean(limit_data_i)
    mr = np.abs(np.diff(data_i))
    mr_mean = np.mean(np.abs(np.diff(limit_data_i)))
    sigma_est_i = mr_mean / 1.128
    UCL_I, LCL_I = mean_i + 3 * sigma_est_i, mean_i - 3 * sigma_est_i
    UCL_MR = mr_mean * 3.267
    
    fig_imr = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("I-Chart", "MR-Chart"))
    fig_imr.add_trace(go.Scatter(x=x_i, y=data_i, mode='lines+markers', name='Value'), row=1, col=1)
    fig_imr.add_hline(y=mean_i, line=dict(dash='dash', color='black'), row=1, col=1); fig_imr.add_hline(y=UCL_I, line=dict(color='red'), row=1, col=1); fig_imr.add_hline(y=LCL_I, line=dict(color='red'), row=1, col=1)
    fig_imr.add_trace(go.Scatter(x=x_i[1:], y=mr, mode='lines+markers', name='Range'), row=2, col=1)
    fig_imr.add_hline(y=mr_mean, line=dict(dash='dash', color='black'), row=2, col=1); fig_imr.add_hline(y=UCL_MR, line=dict(color='red'), row=2, col=1)
    fig_imr.update_layout(title_text='<b>1. I-MR Chart</b>', showlegend=False)
    
    # --- X-bar & R Chart ---
    subgroup_means = np.mean(data_xbar, axis=1)
    subgroup_ranges = np.max(data_xbar, axis=1) - np.min(data_xbar, axis=1)
    x_xbar = np.arange(1, n_points + 1)
    limit_data_xbar_means = subgroup_means[:15] if scenario != 'Stable' else subgroup_means
    limit_data_xbar_ranges = subgroup_ranges[:15] if scenario != 'Stable' else subgroup_ranges
    mean_xbar, mean_r = np.mean(limit_data_xbar_means), np.mean(limit_data_xbar_ranges)
    UCL_X, LCL_X = mean_xbar + 0.577 * mean_r, mean_xbar - 0.577 * mean_r
    UCL_R = 2.114 * mean_r; LCL_R = 0 * mean_r

    fig_xbar = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("X-bar Chart", "R-Chart"))
    fig_xbar.add_trace(go.Scatter(x=x_xbar, y=subgroup_means, mode='lines+markers'), row=1, col=1)
    fig_xbar.add_hline(y=mean_xbar, line=dict(dash='dash', color='black'), row=1, col=1); fig_xbar.add_hline(y=UCL_X, line=dict(color='red'), row=1, col=1); fig_xbar.add_hline(y=LCL_X, line=dict(color='red'), row=1, col=1)
    fig_xbar.add_trace(go.Scatter(x=x_xbar, y=subgroup_ranges, mode='lines+markers'), row=2, col=1)
    fig_xbar.add_hline(y=mean_r, line=dict(dash='dash', color='black'), row=2, col=1); fig_xbar.add_hline(y=UCL_R, line=dict(color='red'), row=2, col=1)
    fig_xbar.update_layout(title_text='<b>2. X-bar & R Chart</b>', showlegend=False)

    # --- P-Chart ---
    proportions = data_p_defects / 200
    limit_data_p = proportions[:15] if scenario != 'Stable' else proportions
    p_bar = np.mean(limit_data_p)
    sigma_p = np.sqrt(p_bar * (1-p_bar) / 200)
    UCL_P, LCL_P = p_bar + 3 * sigma_p, max(0, p_bar - 3 * sigma_p)

    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(x=np.arange(1, n_points+1), y=proportions, mode='lines+markers'))
    fig_p.add_hline(y=p_bar, line=dict(dash='dash', color='black')); fig_p.add_hline(y=UCL_P, line=dict(color='red')); fig_p.add_hline(y=LCL_P, line=dict(color='red'))
    fig_p.update_layout(title_text='<b>3. P-Chart</b>', yaxis_tickformat=".0%", showlegend=False, xaxis_title="Batch Number", yaxis_title="Proportion Defective")
    
    return fig_imr, fig_xbar, fig_p
    
@st.cache_data
def plot_capability(scenario='Ideal'):
    """
    Generates enhanced, more realistic dynamic plots for the process capability module,
    including multiple 'out of control' types and a KDE overlay.
    """
    np.random.seed(42)
    n = 150
    LSL, USL, Target = 90, 110, 100
    
    # --- Data Generation based on scenario ---
    mean, std = 100, 1.5
    is_stable = True
    phase1_end = 75

    if scenario == 'Ideal (High Cpk)':
        mean, std = 100, 1.5
    elif scenario == 'Shifted (Low Cpk)':
        mean, std = 104, 1.5
    elif scenario == 'Variable (Low Cpk)':
        mean, std = 100, 3.5
    elif scenario == 'Out of Control (Shift)':
        is_stable = False
        data = np.random.normal(mean, std, n)
        data[phase1_end:] += 6 # Add a shift
    elif scenario == 'Out of Control (Trend)':
        is_stable = False
        data = np.random.normal(mean, std, n)
        data += np.linspace(0, 8, n) # Add a gradual trend
    elif scenario == 'Out of Control (Bimodal)':
        is_stable = False
        data1 = np.random.normal(97, 1.5, n // 2)
        data2 = np.random.normal(103, 1.5, n // 2)
        data = np.concatenate([data1, data2])
        np.random.shuffle(data)

    if is_stable:
        data = np.random.normal(mean, std, n)
        
    # --- Control Chart Calculations ---
    # Use only stable part for limits if a known instability is introduced
    limit_data = data[:phase1_end] if scenario in ['Out of Control (Shift)', 'Out of Control (Trend)'] else data
    center_line = np.mean(limit_data)
    mr_mean = np.mean(np.abs(np.diff(limit_data)))
    sigma_est = mr_mean / 1.128 # d2 for n=2
    UCL_I, LCL_I = center_line + 3 * sigma_est, center_line - 3 * sigma_est
    
    ooc_indices = np.where((data > UCL_I) | (data < LCL_I))[0]

    # --- Capability Calculation ---
    if not is_stable or len(ooc_indices) > 0:
        cpk_val = np.nan # Invalid if not stable
    else:
        # Use overall mean and std for capability calculation if stable
        data_mean, data_std = np.mean(data), np.std(data, ddof=1)
        cpk_upper = (USL - data_mean) / (3 * data_std)
        cpk_lower = (data_mean - LSL) / (3 * data_std)
        cpk_val = min(cpk_upper, cpk_lower)

    # --- Plotting ---
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
        subplot_titles=("<b>1. Control Chart (Is the process stable?)</b>",
                        "<b>2. Capability Histogram (Does it meet specs?)</b>")
    )
    # Control Chart
    fig.add_trace(go.Scatter(x=np.arange(n), y=data, mode='lines+markers', name='Process Data',
                             marker=dict(color='#636EFA')), row=1, col=1)
    fig.add_trace(go.Scatter(x=ooc_indices, y=data[ooc_indices], mode='markers', name='Out of Control',
                             marker=dict(color='#EF553B', size=10, symbol='x')), row=1, col=1)
    fig.add_hline(y=center_line, line_dash="dash", line_color="black", row=1, col=1)
    fig.add_hline(y=UCL_I, line_color="red", row=1, col=1)
    fig.add_hline(y=LCL_I, line_color="red", row=1, col=1)
    if scenario in ['Out of Control (Shift)', 'Out of Control (Trend)']:
        fig.add_vrect(x0=phase1_end - 0.5, x1=n - 0.5, fillcolor="rgba(255,150,0,0.15)", line_width=0,
                      annotation_text="Process Change", annotation_position="top left", row=1, col=1)
    
    # Histogram
    fig.add_trace(go.Histogram(x=data, name='Distribution', nbinsx=25, histnorm='probability density'), row=2, col=1)
    
    # SME Enhancement: Use KDE instead of normal curve for flexibility
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(data)
    x_curve = np.linspace(min(data.min(), LSL-5), max(data.max(), USL+5), 200)
    y_curve = kde(x_curve)
    fig.add_trace(go.Scatter(x=x_curve, y=y_curve, mode='lines', name='Process Voice (KDE)',
                             line=dict(color='darkblue', width=3)), row=2, col=1)
    
    # SME Enhancement: Highlight OOC points on the histogram
    if len(ooc_indices) > 0:
        fig.add_trace(go.Scatter(x=data[ooc_indices], y=np.zeros_like(ooc_indices), mode='markers',
                                 name='OOC Points', marker=dict(color='#EF553B', size=8, symbol='circle')), row=2, col=1)

    # Add Spec Limits
    fig.add_vline(x=LSL, line_dash="dot", line_color="darkred", annotation_text="<b>LSL</b>", row=2, col=1)
    fig.add_vline(x=USL, line_dash="dot", line_color="darkred", annotation_text="<b>USL</b>", row=2, col=1)
    fig.add_vline(x=Target, line_dash="dash", line_color="grey", annotation_text="Target", row=2, col=1)
    
    # SME Enhancement: Add a Cpk Verdict annotation
    if np.isnan(cpk_val):
        verdict_text, verdict_color = "INVALID (Process Unstable)", "#EF553B"
    elif cpk_val < 1.0:
        verdict_text, verdict_color = f"POOR (Cpk = {cpk_val:.2f})", "#EF553B"
    elif cpk_val < 1.33:
        verdict_text, verdict_color = f"MARGINAL (Cpk = {cpk_val:.2f})", "#FECB52"
    else:
        verdict_text, verdict_color = f"GOOD (Cpk = {cpk_val:.2f})", "#00CC96"
        
    fig.add_annotation(x=0.98, y=0.98, xref="x2 domain", yref="y2 domain",
                       text=f"<b>Capability Verdict:<br>{verdict_text}</b>",
                       showarrow=False, font=dict(size=16, color='white'),
                       bgcolor=verdict_color, borderpad=10, bordercolor='black', borderwidth=2)

    fig.update_layout(height=700, showlegend=False, xaxis2_title="Measured Value")
    return fig, cpk_val

@st.cache_data
def plot_proportion_cis(n_samples, n_successes, prior_alpha, prior_beta):
    """
    Calculates and plots a comparison of multiple confidence interval methods for a binomial proportion.
    """
    if n_samples == 0:
        return go.Figure(), {} # Return empty objects if no data

    k, n = n_successes, n_samples
    p_hat = k / n
    
    # --- Calculate all 7 intervals ---
    metrics = {}
    
    # 1. Wald (for demonstration of its flaws)
    if n > 0 and p_hat > 0 and p_hat < 1:
        wald_se = np.sqrt(p_hat * (1 - p_hat) / n)
        metrics['Wald (Approximate)'] = (p_hat - 1.96 * wald_se, p_hat + 1.96 * wald_se)
    else:
        metrics['Wald (Approximate)'] = (0, 0) # Fails at extremes
        
    # 2. Wilson Score
    metrics['Wilson Score'] = wilson_score_interval(p_hat, n)
    
    # 3. Agresti-Coull
    n_adj, k_adj = n + 4, k + 2
    p_adj = k_adj / n_adj
    ac_se = np.sqrt(p_adj * (1 - p_adj) / n_adj)
    metrics['Agresti–Coull'] = (p_adj - 1.96 * ac_se, p_adj + 1.96 * ac_se)
    
    # 4. Clopper-Pearson (Exact)
    metrics['Clopper–Pearson (Exact)'] = stats.beta.interval(0.95, k, n - k + 1)
    
    # 5. Jeffreys Interval (Bayesian)
    metrics['Jeffreys Interval (Bayesian)'] = stats.beta.interval(0.95, k + 0.5, n - k + 0.5)
    
    # 6. Bayesian with Beta Priors
    metrics['Bayesian with Custom Prior'] = stats.beta.interval(0.95, k + prior_alpha, n - k + prior_beta)
    
    # 7. Bootstrapped CI
    @st.cache_data
    def bootstrap_ci(num_samples, num_successes):
        data = np.array([1] * num_successes + [0] * (num_samples - num_successes))
        if len(data) == 0: return (0, 0)
        boot_means = [np.random.choice(data, size=len(data), replace=True).mean() for _ in range(2000)]
        return np.percentile(boot_means, [2.5, 97.5])
    metrics['Bootstrapped CI'] = bootstrap_ci(n, k)
    
    # --- Create the Plot ---
    fig = go.Figure()
    
    # Order the intervals for logical presentation
    order = ['Wald (Approximate)', 'Agresti–Coull', 'Wilson Score', 'Clopper–Pearson (Exact)', 
             'Jeffreys Interval (Bayesian)', 'Bayesian with Custom Prior', 'Bootstrapped CI']
    
    for i, name in enumerate(order):
        lower, upper = metrics[name]
        color = '#EF553B' if name.startswith('Wald') else PRIMARY_COLOR
        fig.add_trace(go.Scatter(
            x=[lower, upper], y=[name, name],
            mode='lines', line=dict(color=color, width=10),
            hovertemplate=f"<b>{name}</b><br>95% CI: [{lower:.3f}, {upper:.3f}]<extra></extra>"
        ))
    
    fig.add_vline(x=p_hat, line=dict(color='black', dash='dash'), annotation_text=f"Observed Rate = {p_hat:.2%}")
    fig.add_vrect(x0=0.95, x1=1.0, fillcolor="rgba(44, 160, 44, 0.1)", layer="below", line_width=0,
                  annotation_text="Target >95% Success", annotation_position="bottom left")

    fig.update_layout(
        title=f'<b>Comparing 95% CIs for {k} Successes in {n} Samples</b>',
        xaxis_title='Success Rate (Proportion)',
        yaxis_title='Confidence Interval Method',
        xaxis_range=[-0.05, 1.05],
        xaxis_tickformat=".0%",
        showlegend=False
    )
    
    return fig, metrics

@st.cache_data
def plot_process_equivalence(cpk_site_a, mean_shift, var_change_factor, n_samples, margin):
    """
    Generates a professional, multi-plot dashboard for demonstrating statistical equivalence between two processes.
    """
    np.random.seed(42)
    lsl, usl = 90, 110
    
    # 1. Define Site A (Original Process) from its Cpk
    mean_a = 100
    std_a = (usl - lsl) / (6 * cpk_site_a)
    data_a = np.random.normal(mean_a, std_a, n_samples)
    
    # 2. Define Site B (New Process) based on shifts
    mean_b = mean_a + mean_shift
    std_b = std_a * var_change_factor
    data_b = np.random.normal(mean_b, std_b, n_samples)
    
    # 3. Calculate Cpk for both samples
    def calculate_cpk(data, lsl, usl):
        m, s = np.mean(data), np.std(data, ddof=1)
        if s == 0: return 10.0 # Handle case of no variation
        return min((usl - m) / (3 * s), (m - lsl) / (3 * s))
    
    cpk_a_sample = calculate_cpk(data_a, lsl, usl)
    cpk_b_sample = calculate_cpk(data_b, lsl, usl)
    diff_cpk = cpk_b_sample - cpk_a_sample

    # 4. Perform Equivalence Test (Bootstrap CI for Cpk difference)
    # --- THIS DECORATOR HAS BEEN REMOVED ---
    def bootstrap_cpk_diff(d1, d2, l, u, n_boot=1000):
        boot_diffs = []
        for _ in range(n_boot):
            s1 = np.random.choice(d1, len(d1), replace=True)
            s2 = np.random.choice(d2, len(d2), replace=True)
            boot_cpk1 = calculate_cpk(s1, l, u)
            boot_cpk2 = calculate_cpk(s2, l, u)
            boot_diffs.append(boot_cpk2 - boot_cpk1)
        return np.array(boot_diffs)
    # --- END OF FIX ---

    boot_diffs = bootstrap_cpk_diff(data_a, data_b, lsl, usl)
    ci_lower, ci_upper = np.percentile(boot_diffs, [5, 95]) # 90% CI for TOST
    is_equivalent = (ci_lower >= -margin) and (ci_upper <= margin)

    # 5. Generate Plots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("<b>1. Process Capability Comparison</b>", "<b>2. Statistical Evidence (Bootstrap Distribution of Difference)</b>", "<b>3. Equivalence Verdict</b>"),
        row_heights=[0.5, 0.3, 0.2], vertical_spacing=0.15
    )
    
    # Plot 1: Smoothed Capability Distributions (KDE)
    x_range = np.linspace(lsl-10, usl+10, 300)
    kde_a = stats.gaussian_kde(data_a)
    kde_b = stats.gaussian_kde(data_b)
    fig.add_trace(go.Scatter(x=x_range, y=kde_a(x_range), fill='tozeroy', name=f'Site A (Cpk={cpk_a_sample:.2f})', line=dict(color=PRIMARY_COLOR)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_range, y=kde_b(x_range), fill='tozeroy', name=f'Site B (Cpk={cpk_b_sample:.2f})', line=dict(color=SUCCESS_GREEN)), row=1, col=1)
    fig.add_vline(x=lsl, line_dash="dot", line_color="darkred", annotation_text="<b>LSL</b>", row=1, col=1)
    fig.add_vline(x=usl, line_dash="dot", line_color="darkred", annotation_text="<b>USL</b>", row=1, col=1)
    fig.update_layout(barmode='overlay', legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.01))
    fig.update_traces(opacity=0.7, row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=1, col=1)

    # Plot 2: Bootstrap Distribution of the Difference (The "Bridge Plot")
    ci_color = SUCCESS_GREEN if is_equivalent else '#EF553B'
    fig.add_trace(go.Histogram(x=boot_diffs, name='Bootstrap Results', marker_color='grey', histnorm='probability density'), row=2, col=1)
    fig.add_vrect(x0=ci_lower, x1=ci_upper, fillcolor=ci_color, opacity=0.3, line_width=0, row=2, col=1)
    fig.add_vline(x=-margin, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_vline(x=margin, line_dash="dash", line_color="red", row=2, col=1)
    fig.update_yaxes(showticklabels=False, row=2, col=1)

    # Plot 3: Equivalence Verdict Bar
    fig.add_vrect(x0=-margin, x1=margin, fillcolor="rgba(44,160,44,0.1)", layer="below", line_width=0, row=3, col=1)
    fig.add_trace(go.Scatter(x=[ci_lower, ci_upper], y=[1, 1], mode='lines', line=dict(color=ci_color, width=10)), row=3, col=1)
    fig.add_trace(go.Scatter(x=[diff_cpk], y=[1], mode='markers', marker=dict(color='white', size=10, line=dict(color='black', width=2))), row=3, col=1)
    fig.add_annotation(x=0, y=1.5, text=f"Equivalence Zone (±{margin})", showarrow=False, row=3, col=1)
    fig.update_yaxes(showticklabels=False, row=3, col=1)
    fig.update_xaxes(title_text="Difference in Cpk (Site B - Site A)", row=3, col=1)
    
    return fig, is_equivalent, diff_cpk, cpk_a_sample, cpk_b_sample, ci_lower, ci_upper
# ==============================================================================
# HELPER & PLOTTING FUNCTION (Tolerance Intervals) - SME ENHANCED
# ==============================================================================
@st.cache_data
def plot_tolerance_intervals(n=30, coverage_pct=99.0):
    """
    Generates enhanced, more realistic, and illustrative dynamic plots for the
    Tolerance Interval module, including a visualization of the true population.
    """
    np.random.seed(42)
    pop_mean, pop_std = 100, 5
    
    # Simulate a single sample from the true population
    data = np.random.normal(pop_mean, pop_std, n)
    mean, std = np.mean(data), np.std(data, ddof=1)
    
    # 95% CI for the mean (n-dependent)
    sem = std / np.sqrt(n) if n > 0 else 0
    ci_margin = t.ppf(0.975, df=n-1) * sem if n > 1 else 0
    ci = (mean - ci_margin, mean + ci_margin)
    
    # Tolerance Interval (n and coverage dependent)
    # Using a more precise calculation instead of a lookup table
    from scipy.stats import chi2
    g = (1 - (1 - 0.95) / 2) # 95% confidence
    p = coverage_pct / 100.0 # e.g., 99% coverage
    
    # Chi-square factor for standard deviation uncertainty
    chi2_val = chi2.ppf(1-g, n-1)
    
    # Z-score for coverage proportion
    z_val = norm.ppf((1+p)/2)
    
    # Combine for the k-factor (Howe's method approximation)
    k_factor = z_val * np.sqrt((n-1)*(1 + 1/n) / chi2_val)
    
    ti_margin = k_factor * std
    ti = (mean - ti_margin, mean + ti_margin)

    # --- Plotting ---
    fig = go.Figure()

    # SME Enhancement: Plot the true population distribution in the background
    x_range = np.linspace(pop_mean - 5 * pop_std, pop_mean + 5 * pop_std, 400)
    pop_pdf = norm.pdf(x_range, pop_mean, pop_std)
    fig.add_trace(go.Scatter(x=x_range, y=pop_pdf, mode='lines',
                             line=dict(color='grey', dash='dash'), name='True Population Distribution'))

    # SME Enhancement: Shade the area of the true population covered by the TI
    x_fill = np.linspace(ti[0], ti[1], 100)
    y_fill = norm.pdf(x_fill, pop_mean, pop_std)
    fig.add_trace(go.Scatter(x=x_fill, y=y_fill, fill='tozeroy',
                             mode='none', fillcolor='rgba(0,128,0,0.3)',
                             name=f'Area Covered by TI'))
    
    # Plot the sample data histogram
    fig.add_trace(go.Histogram(x=data, name='Sample Data', histnorm='probability density',
                               marker_color='#636EFA'))

    # Add annotations for the two intervals
    # Confidence Interval
    fig.add_shape(type="rect", xref="x", yref="paper", x0=ci[0], y0=0.6, x1=ci[1], y1=0.7,
                  fillcolor="rgba(255,165,0,0.7)", line_width=1, line_color="black")
    fig.add_annotation(x=(ci[0]+ci[1])/2, y=0.65, yref="paper",
                       text=f"<b>CI for Mean</b><br>[{ci[0]:.2f}, {ci[1]:.2f}]",
                       showarrow=False, font=dict(color="black"))

    # Tolerance Interval
    fig.add_shape(type="rect", xref="x", yref="paper", x0=ti[0], y0=0.4, x1=ti[1], y1=0.5,
                  fillcolor="rgba(0,128,0,0.7)", line_width=1, line_color="black")
    fig.add_annotation(x=(ti[0]+ti[1])/2, y=0.45, yref="paper",
                       text=f"<b>Tolerance Interval</b><br>[{ti[0]:.2f}, {ti[1]:.2f}]",
                       showarrow=False, font=dict(color="white"))
    
    # Calculate the actual coverage for display
    actual_coverage = norm.cdf(ti[1], pop_mean, pop_std) - norm.cdf(ti[0], pop_mean, pop_std)
    
    fig.update_layout(
        title=f"<b>Confidence vs. Tolerance Interval | Actual Coverage: {actual_coverage:.2%}</b>",
        xaxis_title="Measured Value", yaxis_title="Density", showlegend=False,
        annotations=[
            dict(x=0.02, y=0.98, xref='paper', yref='paper',
                 text="<b>CI asks:</b> Where is the <u>mean</u>?<br><b>TI asks:</b> Where are the <u>individuals</u>?",
                 showarrow=False, align='left', font=dict(size=14),
                 bgcolor='rgba(255,255,255,0.7)')
        ]
    )
    
    return fig, ci, ti

@st.cache_data
def plot_method_comparison(constant_bias=2.0, proportional_bias=3.0, random_error_sd=3.0):
    """
    Generates an enhanced, more realistic, and integrated dashboard for method comparison,
    using Passing-Bablok regression and adding CIs to the Bland-Altman plot.
    """
    np.random.seed(1)
    n_samples = 50
    true_values = np.linspace(20, 200, n_samples)
    
    # Simulate data with different error types
    error_ref = np.random.normal(0, random_error_sd, n_samples)
    error_test = np.random.normal(0, random_error_sd, n_samples)
    
    ref_method = true_values + error_ref
    test_method = constant_bias + true_values * (1 + proportional_bias / 100) + error_test
    
    df = pd.DataFrame({'Reference': ref_method, 'Test': test_method})

    # --- SME Enhancement: Implement Passing-Bablok Regression ---
    # This is a simplified version for demonstration. A real implementation is more complex.
    slopes = []
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            if (df['Reference'][i] - df['Reference'][j]) != 0:
                slope = (df['Test'][i] - df['Test'][j]) / (df['Reference'][i] - df['Reference'][j])
                slopes.append(slope)
    pb_slope = np.median(slopes)
    pb_intercept = np.median(df['Test'] - pb_slope * df['Reference'])
    
    # --- Bland-Altman Calculations with Confidence Intervals ---
    df['Average'] = (df['Reference'] + df['Test']) / 2
    df['Difference'] = df['Test'] - df['Reference']
    mean_diff = df['Difference'].mean()
    std_diff = df['Difference'].std(ddof=1)
    
    # CIs for the mean bias and Limits of Agreement
    ci_bias_margin = 1.96 * std_diff / np.sqrt(n_samples)
    ci_loa_margin = 1.96 * std_diff * np.sqrt(3 / n_samples)
    
    upper_loa = mean_diff + 1.96 * std_diff
    lower_loa = mean_diff - 1.96 * std_diff
    
    # --- Plotting: Integrated 2x2 Dashboard ---
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"rowspan": 2}, {}], [None, {}]],
        subplot_titles=("<b>1. Method Agreement (Passing-Bablok)</b>",
                        "<b>2. Bland-Altman Plot</b>",
                        "<b>3. Residuals vs. Reference</b>"),
        vertical_spacing=0.15, horizontal_spacing=0.1
    )

    # Plot 1 (Main): Passing-Bablok Regression
    fig.add_trace(go.Scatter(x=df['Reference'], y=df['Test'], mode='markers', name='Samples',
                             marker=dict(color='#636EFA')), row=1, col=1)
    x_range = np.array([0, df['Reference'].max() * 1.05])
    y_fit = pb_intercept + pb_slope * x_range
    fig.add_trace(go.Scatter(x=x_range, y=y_fit, mode='lines', name='Passing-Bablok Fit',
                             line=dict(color='red', width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_range, y=x_range, mode='lines', name='Line of Identity (y=x)',
                             line=dict(color='black', dash='dash')), row=1, col=1)
    fig.add_annotation(x=0.05, y=0.95, xref="x domain", yref="y domain",
                       text=f"<b>y = {pb_slope:.2f}x + {pb_intercept:.2f}</b>",
                       showarrow=False, font=dict(size=14, color='red'), row=1, col=1)

    # Plot 2 (Top-Right): Bland-Altman
    fig.add_trace(go.Scatter(x=df['Average'], y=df['Difference'], mode='markers', name='Difference',
                             marker=dict(color='#1f77b4')), row=1, col=2)
    # Mean Bias with CI
    fig.add_hrect(y0=mean_diff - ci_bias_margin, y1=mean_diff + ci_bias_margin,
                  fillcolor='rgba(0,0,255,0.1)', line_width=0, row=1, col=2)
    fig.add_hline(y=mean_diff, line=dict(color='blue'), name='Mean Bias', row=1, col=2,
                  annotation_text=f"Bias: {mean_diff:.2f}", annotation_position="bottom right")
    # Limits of Agreement with CIs
    fig.add_hrect(y0=upper_loa - ci_loa_margin, y1=upper_loa + ci_loa_margin,
                  fillcolor='rgba(255,0,0,0.1)', line_width=0, row=1, col=2)
    fig.add_hrect(y0=lower_loa - ci_loa_margin, y1=lower_loa + ci_loa_margin,
                  fillcolor='rgba(255,0,0,0.1)', line_width=0, row=1, col=2)
    fig.add_hline(y=upper_loa, line=dict(color='red', dash='dash'), name='Upper LoA', row=1, col=2)
    fig.add_hline(y=lower_loa, line=dict(color='red', dash='dash'), name='Lower LoA', row=1, col=2)

    # Plot 3 (Bottom-Right): Residuals vs. Reference (to diagnose proportional bias)
    df['Residuals'] = df['Test'] - (pb_intercept + pb_slope * df['Reference'])
    fig.add_trace(go.Scatter(x=df['Reference'], y=df['Residuals'], mode='markers', name='Residuals',
                             marker=dict(color='#ff7f0e')), row=2, col=2)
    fig.add_hline(y=0, line=dict(color='black', dash='dash'), row=2, col=2)

    fig.update_layout(height=800, title_text="<b>Method Comparison Dashboard</b>", title_x=0.5,
                      legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.7)'))
    fig.update_xaxes(title_text="Reference Method", row=1, col=1); fig.update_yaxes(title_text="Test Method", row=1, col=1)
    fig.update_xaxes(title_text="Average of Methods", row=1, col=2); fig.update_yaxes(title_text="Difference (Test - Ref)", row=1, col=2)
    fig.update_xaxes(title_text="Reference Method", row=2, col=2); fig.update_yaxes(title_text="Residuals from Fit", row=2, col=2)
    
    return fig, pb_slope, pb_intercept, mean_diff, upper_loa, lower_loa

@st.cache_data
def plot_bayesian(prior_type, n_qc=20, k_qc=18, spec_limit=0.90):
    """
    Generates enhanced, more realistic, and interactive plots for the Bayesian inference module,
    including interactive data and visualization of the credible interval.
    """
    # Define Priors based on selection
    if prior_type == "Strong R&D Prior":
        # Corresponds to ~98 successes in 100 trials
        a_prior, b_prior = 98, 2
    elif prior_type == "Skeptical/Regulatory Prior":
        # Weakly centered around 80%, wide uncertainty
        a_prior, b_prior = 4, 1
    else: # "No Prior (Uninformative)"
        # Uninformative Jeffreys prior
        a_prior, b_prior = 0.5, 0.5
        
    # Bayesian Update (Posterior calculation)
    a_post = a_prior + k_qc
    b_post = b_prior + (n_qc - k_qc)
    
    # Calculate key metrics
    prior_mean = a_prior / (a_prior + b_prior)
    mle = k_qc / n_qc if n_qc > 0 else 0
    posterior_mean = a_post / (a_post + b_post)

    # --- Plotting ---
    x = np.linspace(0, 1, 500)
    fig = go.Figure()

    # Plot Prior
    prior_pdf = beta.pdf(x, a_prior, b_prior)
    fig.add_trace(go.Scatter(x=x, y=prior_pdf, mode='lines', name='Prior Belief',
                             line=dict(color='green', dash='dash', width=3)))

    # Plot Posterior
    posterior_pdf = beta.pdf(x, a_post, b_post)
    fig.add_trace(go.Scatter(x=x, y=posterior_pdf, mode='lines', name='Posterior Belief (Updated)',
                             line=dict(color='blue', width=4), fill='tozeroy',
                             fillcolor='rgba(0,0,255,0.1)'))
    
    # SME Enhancement: Calculate and shade the 95% Credible Interval (HDI)
    ci_lower, ci_upper = beta.ppf([0.025, 0.975], a_post, b_post)
    x_fill = np.linspace(ci_lower, ci_upper, 100)
    y_fill = beta.pdf(x_fill, a_post, b_post)
    fig.add_trace(go.Scatter(x=x_fill, y=y_fill, fill='tozeroy', mode='none',
                             fillcolor='rgba(0,0,255,0.3)', name='95% Credible Interval'))

    # SME Enhancement: Show the data/likelihood as a point estimate
    fig.add_trace(go.Scatter(x=[mle], y=[0], mode='markers', name=f'Data Likelihood (k/n = {mle:.2f})',
                             marker=dict(color='red', size=15, symbol='diamond', line=dict(width=2, color='black'))))

    # SME Enhancement: Calculate probability of meeting a spec
    prob_gt_spec = 1.0 - beta.cdf(spec_limit, a_post, b_post)
    fig.add_vline(x=spec_limit, line_dash="dot", line_color="black",
                  annotation_text=f"Spec Limit ({spec_limit:.0%})", annotation_position="top")

    fig.update_layout(
        title=f"<b>Bayesian Update for QC Pass Rate</b>",
        xaxis_title="True Pass Rate Parameter (θ)", yaxis_title="Probability Density",
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.7)'),
        xaxis=dict(range=[0,1], tickformat=".0%"),
        yaxis=dict(showticklabels=False)
    )
    
    return fig, prior_mean, mle, posterior_mean, (ci_lower, ci_upper), prob_gt_spec

@st.cache_data
def plot_fty_coq(project_type, improvement_effort):
    """
    Generates a professional-grade, 4-plot dashboard for FTY and COQ for multiple project types.
    """
    profiles = {
        "Pharma Process (MAb)": {
            'steps': ["Cell Culture", "Harvest", "Purification", "Formulation", "Fill/Finish"],
            'base_fty': [0.98, 0.99, 0.92, 0.97, 0.99],
            'cost_factors': {'internal': 80000, 'external': 200000}
        },
        "Analytical Assay (ELISA)": {
            'steps': ["Coating", "Blocking", "Sample Add", "Detection", "Analysis"],
            'base_fty': [0.99, 0.98, 0.95, 0.96, 0.97],
            'cost_factors': {'internal': 5000, 'external': 20000}
        },
        "Instrument Qualification": {
            'steps': ["URS/FS", "IQ", "OQ", "PQ", "Final Report"],
            'base_fty': [0.95, 0.99, 0.92, 0.96, 0.98],
            'cost_factors': {'internal': 10000, 'external': 40000}
        },
        "Software System (CSV)": {
            'steps': ["Requirements", "Design", "Coding", "Testing", "Deployment"],
            'base_fty': [0.90, 0.95, 0.88, 0.94, 0.99],
            'cost_factors': {'internal': 25000, 'external': 100000}
        }
    }
    profile = profiles[project_type]
    steps, base_fty = profile['steps'], profile['base_fty']
    
    # --- Calculate the Improved ("After") Process ---
    improved_fty = base_fty.copy()
    temp_fty = np.array(improved_fty)
    effort_applied_per_step = np.zeros(len(steps))
    effort_remaining = improvement_effort
    while effort_remaining > 0 and np.min(temp_fty) < 0.999:
        worst_step_idx = np.argmin(temp_fty)
        effort_to_apply = 1
        effort_applied_per_step[worst_step_idx] += effort_to_apply
        improvement = (1 - temp_fty[worst_step_idx]) * 0.2 * effort_to_apply
        temp_fty[worst_step_idx] += improvement
        effort_remaining -= effort_to_apply
    improved_fty = list(temp_fty)

    rty_base, rty_improved = np.prod(base_fty), np.prod(improved_fty)

    # --- Calculate Cost of Quality (COQ) ---
    base_coq = {
        "Prevention": 20000 + (improvement_effort * 2000), "Appraisal": 30000 + (improvement_effort * 1500),
        "Internal Failure": profile['cost_factors']['internal'] * (1 - rty_base) / (1-0.9),
        "External Failure": profile['cost_factors']['external'] * (1 - rty_base)**2 / (1-0.9)**2
    }
    improved_coq = {
        "Prevention": 20000 + (improvement_effort * 2000), "Appraisal": 30000 + (improvement_effort * 1500),
        "Internal Failure": profile['cost_factors']['internal'] * (1 - rty_improved) / (1-0.9),
        "External Failure": profile['cost_factors']['external'] * (1 - rty_improved)**2 / (1-0.9)**2
    }

    # --- PLOTTING ---
    # Plot 1: Pareto Chart of Scrap/Rework
    base_scrap = [1000 * (1 - fty) * np.prod(base_fty[:i]) for i, fty in enumerate(base_fty)]
    improved_scrap = [1000 * (1 - fty) * np.prod(improved_fty[:i]) for i, fty in enumerate(improved_fty)]
    df_pareto = pd.DataFrame({'Step': steps, 'Baseline': base_scrap, 'Optimized': improved_scrap}).sort_values('Baseline', ascending=False)
    
    fig_pareto = go.Figure()
    fig_pareto.add_trace(go.Bar(name='Baseline Loss', x=df_pareto['Step'], y=df_pareto['Baseline'], marker_color='grey'))
    fig_pareto.add_trace(go.Bar(name='Optimized Loss', x=df_pareto['Step'], y=df_pareto['Optimized'], marker_color=SUCCESS_GREEN))
    fig_pareto.update_layout(title="<b>1. Pareto Chart of Yield Loss</b>", yaxis_title="Units Lost per 1000 Started", barmode='group', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    # Plot 2: SPC Chart of the Weakest Step
    worst_step_name = df_pareto['Step'].iloc[0]
    worst_step_idx_orig = steps.index(worst_step_name)
    
    np.random.seed(worst_step_idx_orig)
    base_mean_offset = (1 - base_fty[worst_step_idx_orig]) * 10
    base_std_dev = (1 - base_fty[worst_step_idx_orig]) * 5 + 1
    
    improvement_factor = 1 - (effort_applied_per_step[worst_step_idx_orig] / 10.0) * 0.8
    improved_mean_offset = base_mean_offset * improvement_factor
    improved_std_dev = base_std_dev * improvement_factor
    
    data = np.random.normal(100 - improved_mean_offset, improved_std_dev, 25)
    mean, std = 100, 2
    ucl, lcl = mean + 3*std, mean - 3*std
    
    fig_spc = go.Figure()
    fig_spc.add_trace(go.Scatter(y=data, mode='lines+markers', name='CPP Data', line=dict(color=PRIMARY_COLOR)))
    fig_spc.add_hline(y=ucl, line=dict(color='red', dash='dash')); fig_spc.add_hline(y=lcl, line=dict(color='red', dash='dash'))
    fig_spc.add_hline(y=mean, line=dict(color='black', dash='dot'))
    fig_spc.update_layout(title=f"<b>2. SPC of Critical Parameter for '{worst_step_name}'</b>", yaxis_title="CPP Value", xaxis_title="Batch Number")

    # Plot 3: Yield Funnel Sankey Diagram
    fig_sankey = go.Figure()
    scenarios = {'Baseline': base_fty, 'Optimized': improved_fty}
    for i, (name, ftys) in enumerate(scenarios.items()):
        labels = ["Input"] + steps + ["Final Output"] + [f"Scrap/Rework {j+1}" for j in range(len(steps))]
        sources, targets, values = [], [], []
        units_in = 1000
        for j, fty in enumerate(ftys):
            units_out, scrap = units_in * fty, units_in * (1 - fty)
            sources.extend([j, j]); targets.extend([j + 1, len(steps) + 1 + j]); values.extend([units_out, scrap])
            units_in = units_out
        
        fig_sankey.add_trace(go.Sankey(
            domain={'x': [i*0.5, i*0.5+0.48]},
            node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=labels, color=PRIMARY_COLOR),
            link=dict(source=sources, target=targets, value=values)
        ))
    rty_text = "Right First Time" if "Software" in project_type else "Rolled Throughput Yield"
    fig_sankey.update_layout(title_text=f"<b>3. Process Yield Funnel ({rty_text})</b>",
                             annotations=[dict(x=0.24, y=1.1, text=f"<b>Baseline (RTY: {rty_base:.1%})</b>", showarrow=False),
                                          dict(x=0.74, y=1.1, text=f"<b>Optimized (RTY: {rty_improved:.1%})</b>", showarrow=False)])

    # Plot 4: Cost of Quality "Iceberg" Chart
    fig_iceberg = go.Figure()
    good_quality_base = base_coq['Prevention'] + base_coq['Appraisal']
    poor_quality_base = base_coq['Internal Failure'] + base_coq['External Failure']
    good_quality_improved = improved_coq['Prevention'] + improved_coq['Appraisal']
    poor_quality_improved = improved_coq['Internal Failure'] + improved_coq['External Failure']
    
    fig_iceberg.add_trace(go.Bar(x=['Baseline', 'Optimized'], y=[good_quality_base, good_quality_improved], name='Cost of Good Quality (Visible Investment)', marker_color='skyblue'))
    fig_iceberg.add_trace(go.Bar(x=['Baseline', 'Optimized'], y=[-poor_quality_base, -poor_quality_improved], name='Cost of Poor Quality (Hidden Losses)', marker_color='salmon'))
    fig_iceberg.add_hline(y=0, line_color="darkblue", line_width=3)
    fig_iceberg.update_layout(title="<b>4. The Cost of Quality 'Iceberg'</b>", yaxis_title="Cost (Relative Cost Units)", barmode='relative',
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    
    return fig_pareto, fig_spc, fig_sankey, fig_iceberg, rty_base, rty_improved, base_coq, improved_coq
##=================================================================================================================================================================================================
##=======================================================================================END ACT II ===============================================================================================
##=================================================================================================================================================================================================
@st.cache_data
def plot_control_plan_dashboard(cpp_data, sample_size, frequency, spc_tool):
    """
    Generates a professional-grade Control Plan dashboard including a simulated SPC chart.
    """
    # 1. --- Generate the Control Plan Table ---
    control_plan_data = {
        'Parameter (CPP)': [cpp_data['name']],
        'Specification': [f"{cpp_data['lsl']} - {cpp_data['usl']}"],
        'Measurement System': [cpp_data['method']],
        'Sample Size / Freq.': [f"n={sample_size}, {frequency}"],
        'Control Method': [spc_tool],
        'Reaction Plan': ['Follow OCAP-001']
    }
    df = pd.DataFrame(control_plan_data)
    fig_table = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns), fill_color=PRIMARY_COLOR, font=dict(color='white', size=14), align='left', height=40),
        cells=dict(values=[df[col] for col in df.columns], fill_color='lavender', align='left', font_size=12, height=30)
    )])
    fig_table.update_layout(title_text="<b>1. Process Control Plan Document</b>", margin=dict(l=10, r=10, t=40, b=10))

    # 2. --- Simulate and Plot the SPC Chart ---
    np.random.seed(42)
    process_mean = (cpp_data['usl'] + cpp_data['lsl']) / 2
    process_std = (cpp_data['usl'] - cpp_data['lsl']) / 8 # Simulate a Cpk of 1.33
    
    # More frequent sampling means faster detection of shifts
    shift_detection_point = 25 - int(np.log2(1 if "batch" in frequency else (5 if "hour" in frequency else 10)))
    
    data = np.random.normal(process_mean, process_std, 30)
    data[15:] += process_std * 0.5 # Introduce a small 0.5 sigma shift
    
    fig_spc = go.Figure()
    fig_spc.add_trace(go.Scatter(y=data, mode='lines+markers', name='CPP Data'))
    mean_line = np.mean(data[:15])
    std_line = np.std(data[:15])
    fig_spc.add_hline(y=mean_line + 3*std_line, line=dict(color='red', dash='dash'))
    fig_spc.add_hline(y=mean_line - 3*std_line, line=dict(color='red', dash='dash'))
    fig_spc.add_hline(y=mean_line, line=dict(color='black', dash='dot'))
    fig_spc.add_vrect(x0=14.5, x1=29.5, fillcolor="rgba(255,165,0,0.1)", line_width=0, annotation_text="Process Shift")
    if 15 < shift_detection_point < 30:
        fig_spc.add_vline(x=shift_detection_point, line=dict(color='purple', width=2), annotation_text="Predicted Detection")

    fig_spc.update_layout(title=f"<b>2. Simulated Performance of '{spc_tool}'</b>", yaxis_title=cpp_data['name'])

    # 3. --- Generate the OCAP Flowchart ---
    fig_flowchart = go.Figure(go.Sankey(
        arrangement = "snap",
        node = {"label": ["Process In-Control", f"Signal on {spc_tool}?", "Stop Process", "Notify QA", "Continue Process", "Investigate Cause"], "color": [SUCCESS_GREEN, 'orange', 'red', 'red', SUCCESS_GREEN, 'lightblue']},
        link = {"source": [0, 1, 1, 3], "target": [1, 2, 4, 5], "value": [1, 0.5, 0.5, 0.5], "label": ["Monitor", "YES", "NO", "Action"]}
    ))
    fig_flowchart.update_layout(title_text="<b>3. Out-of-Control Action Plan (OCAP) Flowchart</b>", font_size=12)

    return fig_table, fig_spc, fig_flowchart

@st.cache_data
def plot_westgard_scenario(scenario='Stable'):
    """
    Generates an enhanced, more realistic dynamic Westgard chart, including
    algorithmic rule detection and a Power Functions plot.
    """
    # Establish historical process parameters
    mean, std = 100, 2
    n_points = 30
    
    # --- Generate data based on the selected scenario ---
    np.random.seed(101) # Use a consistent seed for base data
    data = np.random.normal(mean, std, n_points)
    
    # Inject more realistic failures
    if scenario == 'Large Random Error':
        data[20] = 107.5 # A single blunder
    elif scenario == 'Systematic Shift':
        data[18:] = np.random.normal(mean + 2.2*std, std, n_points - 18) # A true shift in the mean
    elif scenario == 'Increased Imprecision':
        data[22:] = np.random.normal(mean, std * 2.5, n_points - 22) # Increased noise
    
    # --- SME ENHANCEMENT: Algorithmic Westgard Rule detection ---
    violations = {}
    limits = {i: mean + i * std for i in [-3, -2, -1, 1, 2, 3]}

    for i in range(1, n_points):
        # 1-3s rule
        if data[i] > limits[3] or data[i] < limits[-3]:
            violations[i] = "1-3s Violation"
        # 2-2s rule
        if (data[i] > limits[2] and data[i-1] > limits[2]) or \
           (data[i] < limits[-2] and data[i-1] < limits[-2]):
            violations[i] = violations.get(i, "") + " 2-2s Violation"
            violations[i-1] = violations.get(i-1, "") + " 2-2s Violation"
        # R-4s rule
        if abs(data[i] - data[i-1]) > 4 * std:
            violations[i] = violations.get(i, "") + " R-4s Violation"
            violations[i-1] = violations.get(i-1, "") + " R-4s Violation"
    # 4-1s rule
    for i in range(3, n_points):
        if (all(d > limits[1] for d in data[i-3:i+1])) or \
           (all(d < limits[-1] for d in data[i-3:i+1])):
            for j in range(i-3, i+1):
                violations[j] = violations.get(j, "") + " 4-1s Violation"
    # 10-x rule (simplified check)
    if n_points >= 10:
        for i in range(9, n_points):
            if (all(d > mean for d in data[i-9:i+1])) or \
               (all(d < mean for d in data[i-9:i+1])):
                for j in range(i-9, i+1):
                    violations[j] = violations.get(j, "") + " 10-x Violation"

    # --- Plotting ---
    fig = go.Figure()
    
    # Add shaded regions for control zones
    for i, color in zip([3, 2, 1], ['rgba(239,83,80,0.1)', 'rgba(254,203,82,0.1)', 'rgba(0,204,150,0.1)']):
        fig.add_hrect(y0=mean - i*std, y1=mean + i*std, line_width=0, fillcolor=color, layer='below')
    
    # Add SD lines with labels
    for i in [-3, -2, -1, 1, 2, 3]:
        fig.add_hline(y=mean + i*std, line=dict(color='grey', dash='dot'),
                      annotation_text=f"{i:+}σ", annotation_position="bottom right")
    fig.add_hline(y=mean, line=dict(color='black', dash='dash'), annotation_text='Mean')

    # Add data trace
    fig.add_trace(go.Scatter(x=np.arange(1, n_points + 1), y=data, mode='lines+markers', name='Control Data',
                             line=dict(color='#636EFA', width=3),
                             marker=dict(size=10, symbol='circle', line=dict(width=2, color='black'))))
    
    # Add violation highlights and hover text
    violation_indices = sorted(violations.keys())
    fig.add_trace(go.Scatter(
        x=np.array(violation_indices) + 1,
        y=data[violation_indices],
        mode='markers', name='Violation',
        marker=dict(color='red', size=16, symbol='diamond', line=dict(width=2, color='black')),
        hoverinfo='text',
        text=[f"<b>Point {i+1}</b><br>Value: {data[i]:.2f}<br>Rules: {violations[i].strip()}" for i in violation_indices]
    ))
        
    fig.update_layout(title=f"<b>Westgard Rules Diagnostic Chart: {scenario} Scenario</b>",
                      xaxis_title="Measurement Number", yaxis_title="Control Value",
                      showlegend=False, height=600)

    # --- SME Enhancement: Power Functions Plot ---
    shifts = np.linspace(0, 4, 50) # Shift size in sigma units
    power = {
        '1-3s': 1 - (norm.cdf(3 - shifts) - norm.cdf(-3 - shifts)),
        '2-2s': 1 - (norm.cdf(2 - shifts)**2 - norm.cdf(-2-shifts)**2), # Approximation
        '4-1s': 1 - (norm.cdf(1 - shifts)**4 - norm.cdf(-1-shifts)**4), # Approximation
        '10-x': 1 - (norm.cdf(0-shifts)**10 - norm.cdf(-0-shifts)**10) # Approximation
    }
    
    fig_power = go.Figure()
    for rule, p_detect in power.items():
        fig_power.add_trace(go.Scatter(x=shifts, y=p_detect, mode='lines', name=rule))
    
    fig_power.update_layout(
        title="<b>Rule Power Functions: Probability of Error Detection</b>",
        xaxis_title="Systematic Shift Size (in multiples of σ)",
        yaxis_title="Probability of Detection (Power)",
        yaxis_tickformat=".0%",
        legend_title_text="Westgard Rule"
    )
    
    return fig, fig_power, violations
    
# ==============================================================================
# HELPER & PLOTTING FUNCTION (Multivariate SPC) - SME ENHANCED
# ==============================================================================
@st.cache_data
def plot_multivariate_spc(scenario='Stable', n_train=100, n_monitor=30, random_seed=42):
    """
    Generates enhanced, more realistic MSPC analysis and plots, including
    labeled confidence ellipses and a more subtle correlation break.
    """
    if scenario == 'Stable':
        np.random.seed(101)
    else:
        np.random.seed(random_seed)

    # 1. --- Data Generation ---
    mean_train = [25, 150]
    cov_train = [[5, 12], [12, 40]] # Strong positive correlation
    df_train = pd.DataFrame(np.random.multivariate_normal(mean_train, cov_train, n_train), columns=['Temperature', 'Pressure'])

    if scenario == 'Stable':
        df_monitor = pd.DataFrame(np.random.multivariate_normal(mean_train, cov_train, n_monitor), columns=['Temperature', 'Pressure'])
    elif scenario == 'Shift in Y Only':
        mean_shift = [25, 165] # Shifted up in Pressure
        df_monitor = pd.DataFrame(np.random.multivariate_normal(mean_shift, cov_train, n_monitor), columns=['Temperature', 'Pressure'])
    elif scenario == 'Correlation Break':
        # More subtle break - correlation weakens but doesn't disappear
        cov_break = [[5, 4], [4, 40]] 
        df_monitor = pd.DataFrame(np.random.multivariate_normal(mean_train, cov_break, n_monitor), columns=['Temperature', 'Pressure'])

    df_full = pd.concat([df_train, df_monitor], ignore_index=True)
    mean_vec = df_train.mean().values

    # 2. --- MSPC Calculations (T² and SPE) ---
    S_inv = np.linalg.inv(df_train.cov())
    diff = df_full[['Temperature', 'Pressure']].values - mean_vec
    df_full['T2'] = [d.T @ S_inv @ d for d in diff]
    
    # SPE is only meaningful if we reduce dimensions (e.g., n_components=1)
    pca_spe = PCA(n_components=1).fit(df_train)
    X_hat_spe = pca_spe.inverse_transform(pca_spe.transform(df_full[['Temperature', 'Pressure']]))
    residuals = df_full[['Temperature', 'Pressure']].values - X_hat_spe
    df_full['SPE'] = np.sum(residuals**2, axis=1)

    # 3. --- Calculate Control Limits ---
    alpha = 0.01
    p = df_train.shape[1]
    t2_ucl = (p * (n_train - 1) / (n_train - p)) * f.ppf(1 - alpha, p, n_train - p)
    spe_ucl = np.percentile(df_full['SPE'].iloc[:n_train], (1 - alpha) * 100)

    # 4. --- OOC Check and Error Type Determination ---
    monitor_data = df_full.iloc[n_train:]
    t2_ooc_points = monitor_data[monitor_data['T2'] > t2_ucl]
    spe_ooc_points = monitor_data[monitor_data['SPE'] > spe_ucl]
    alarm_detected = not t2_ooc_points.empty or not spe_ooc_points.empty
    
    if scenario == 'Stable':
        error_type_str = "Type I Error (False Alarm)" if alarm_detected else "Correct In-Control"
    else:
        error_type_str = "Correct Detection" if alarm_detected else "Type II Error (Missed Signal)"

    # --- PLOTTING ---
    # 5. --- Create Process State Space Plot ---
    fig_scatter = go.Figure()
    
    # Helper for ellipse calculation and plotting
    def get_ellipse_path(mean, cov, conf_level):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]; vals, vecs = vals[order], vecs[:, order]
        theta = np.arctan2(*vecs[:, 0][::-1])
        scale_factor = np.sqrt(stats.chi2.ppf(conf_level, df=2))
        width, height = 2 * scale_factor * np.sqrt(vals)
        
        t = np.linspace(0, 2 * np.pi, 100)
        ellipsis_x_circ = width/2 * np.cos(t)
        ellipsis_y_circ = height/2 * np.sin(t)

        x_rotated = ellipsis_x_circ * np.cos(theta) - ellipsis_y_circ * np.sin(theta)
        y_rotated = ellipsis_x_circ * np.sin(theta) + ellipsis_y_circ * np.cos(theta)
        
        return x_rotated + mean[0], y_rotated + mean[1]

    x99, y99 = get_ellipse_path(mean_vec, cov_train, 0.99)
    x95, y95 = get_ellipse_path(mean_vec, cov_train, 0.95)

    # Plot Ellipses as filled polygons
    fig_scatter.add_trace(go.Scatter(x=x99, y=y99, fill="toself", fillcolor='rgba(239,83,80,0.1)', line=dict(color='rgba(239,83,80,0.5)'), name='99% CI'))
    fig_scatter.add_trace(go.Scatter(x=x95, y=y95, fill="toself", fillcolor='rgba(0,204,150,0.1)', line=dict(color='rgba(0,204,150,0.5)'), name='95% CI'))

    # Add annotations
    fig_scatter.add_annotation(x=np.mean(x95), y=np.max(y95), text="<b>95% CI (Normal Zone)</b>", showarrow=False, font=dict(color='darkgreen'))
    fig_scatter.add_annotation(x=np.mean(x99), y=np.max(y99) + 2, text="<b>99% CI (Control Limit)</b>", showarrow=False, font=dict(color='darkred'))

    # Add data points
    fig_scatter.add_trace(go.Scatter(x=df_train['Temperature'], y=df_train['Pressure'], mode='markers', marker=dict(color='#636EFA', opacity=0.7), name='In-Control (Training)'))
    fig_scatter.add_trace(go.Scatter(x=df_monitor['Temperature'], y=df_monitor['Pressure'], mode='markers', marker=dict(color='black', size=8, symbol='star'), name=f'Monitoring ({scenario})'))
    fig_scatter.update_layout(title=f"<b>Process State Space: Normal Operating Region</b>", xaxis_title="Temperature (°C)", yaxis_title="Pressure (kPa)", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    
    # 6. --- Create Control Charts Plot ---
    fig_charts = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("<b>Hotelling's T² Chart (Distance to Center)</b>", "<b>SPE Chart (Distance to Model)</b>"))
    chart_indices = np.arange(1, len(df_full) + 1)
    fig_charts.add_trace(go.Scatter(x=chart_indices, y=df_full['T2'], mode='lines+markers', name='T² Value'), row=1, col=1)
    fig_charts.add_hline(y=t2_ucl, line_dash="dash", line_color="red", row=1, col=1)
    if not t2_ooc_points.empty: fig_charts.add_trace(go.Scatter(x=t2_ooc_points.index + 1, y=t2_ooc_points['T2'], mode='markers', marker=dict(color='red', size=10, symbol='x')), row=1, col=1)
    
    fig_charts.add_trace(go.Scatter(x=chart_indices, y=df_full['SPE'], mode='lines+markers', name='SPE Value'), row=2, col=1)
    fig_charts.add_hline(y=spe_ucl, line_dash="dash", line_color="red", row=2, col=1)
    if not spe_ooc_points.empty: fig_charts.add_trace(go.Scatter(x=spe_ooc_points.index + 1, y=spe_ooc_points['SPE'], mode='markers', marker=dict(color='red', size=10, symbol='x')), row=2, col=1)
    
    fig_charts.add_vrect(x0=n_train+0.5, x1=n_train+n_monitor+0.5, fillcolor="rgba(255,150,0,0.15)", line_width=0, annotation_text="Monitoring Phase", annotation_position="top left", row='all', col=1)
    fig_charts.update_layout(height=500, title_text="<b>Multivariate Control Charts</b>", showlegend=False, yaxis_title="T² Statistic", yaxis2_title="SPE Statistic", xaxis2_title="Observation Number")

    # 7. --- Create Contribution Plot for Diagnosis ---
    fig_contrib = None
    if alarm_detected:
        if not t2_ooc_points.empty:
            first_ooc_point = df_full.loc[t2_ooc_points.index[0]]
            contributions = (first_ooc_point[['Temperature', 'Pressure']] - mean_vec)**2
            title_text = "<b>T² Alarm Diagnosis: Likely Mean Shift</b>"
        else: # SPE alarm
            first_ooc_idx = spe_ooc_points.index[0]
            contributions = pd.Series(residuals[first_ooc_idx]**2, index=['Temperature', 'Pressure'])
            title_text = "<b>SPE Alarm Diagnosis: Likely Correlation Change</b>"
        
        fig_contrib = px.bar(x=contributions.index, y=contributions.values, title=title_text,
                             labels={'x':'Process Variable', 'y':'Contribution Value'},
                             color=contributions.index, color_discrete_sequence=px.colors.qualitative.Plotly)

    return fig_scatter, fig_charts, fig_contrib, not t2_ooc_points.empty, not spe_ooc_points.empty, error_type_str
    
# ==============================================================================
# HELPER & PLOTTING FUNCTION (Small Shift) - SME ENHANCED & CORRECTED
# ==============================================================================
@st.cache_data
def plot_ewma_cusum_comparison(shift_size=0.75, scenario='Sudden Shift'):
    """
    Generates enhanced, more realistic dynamic I, EWMA, and CUSUM charts,
    including multiple scenarios and a "Time to Detect" KPI.
    """
    np.random.seed(123)
    n_points = 50
    shift_point = 25
    mean, std = 100, 2
    
    # --- Data Generation with dynamic shift scenarios ---
    data = np.random.normal(mean, std, n_points)
    
    if scenario == 'Sudden Shift':
        actual_shift_value = shift_size * std
        data[shift_point:] += actual_shift_value
        scenario_desc = f"{shift_size}σ Sudden Shift"
    elif scenario == 'Gradual Drift':
        drift = np.linspace(0, shift_size * std, n_points - shift_point)
        data[shift_point:] += drift
        scenario_desc = f"{shift_size}σ Gradual Drift"

    # --- Calculations ---
    # I-Chart
    i_ucl, i_lcl = mean + 3 * std, mean - 3 * std
    i_ooc = np.where((data[shift_point:] > i_ucl) | (data[shift_point:] < i_lcl))[0]
    i_detect_time = i_ooc[0] + 1 if len(i_ooc) > 0 else np.nan

    # EWMA
    lam = 0.2
    
    # --- THIS IS THE CORRECTED LINE ---
    ewma = pd.Series(data).ewm(alpha=lam, adjust=False).mean().values
    # --- END OF CORRECTION ---
    
    ewma_ucl = mean + 3 * (std * np.sqrt(lam / (2-lam)))
    ewma_lcl = mean - 3 * (std * np.sqrt(lam / (2-lam)))
    ewma_ooc = np.where((ewma[shift_point:] > ewma_ucl) | (ewma[shift_point:] < ewma_lcl))[0]
    ewma_detect_time = ewma_ooc[0] + 1 if len(ewma_ooc) > 0 else np.nan
    
    # CUSUM
    target = mean; k = 0.5 * std # Slack parameter tuned for 1-sigma shifts
    h = 5 * std # Decision interval
    sh, sl = np.zeros(n_points), np.zeros(n_points)
    for i in range(1, n_points):
        sh[i] = max(0, sh[i-1] + (data[i] - target) - k)
        sl[i] = max(0, sl[i-1] + (target - data[i]) - k)
    cusum_ooc = np.where((sh[shift_point:] > h) | (sl[shift_point:] > h))[0]
    cusum_detect_time = cusum_ooc[0] + 1 if len(cusum_ooc) > 0 else np.nan
        
    # --- Plotting ---
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("<b>I-Chart: The Beat Cop</b> (Detects large, obvious events)",
                                        "<b>EWMA: The Sentinel</b> (Detects small, sustained shifts)",
                                        "<b>CUSUM: The Bloodhound</b> (Fastest detection for specific shifts)"))

    # Plot I-Chart
    fig.add_trace(go.Scatter(x=np.arange(1, n_points+1), y=data, mode='lines+markers', name='Data'), row=1, col=1)
    fig.add_hline(y=i_ucl, line_color='red', line_dash='dash', row=1, col=1)
    fig.add_hline(y=i_lcl, line_color='red', line_dash='dash', row=1, col=1)
    if not np.isnan(i_detect_time):
        idx = shift_point + int(i_detect_time) - 1
        fig.add_trace(go.Scatter(x=[idx+1], y=[data[idx]], mode='markers', name='First Detect',
                                 marker=dict(color='red', size=12, symbol='x')), row=1, col=1)

    # Plot EWMA
    fig.add_trace(go.Scatter(x=np.arange(1, n_points+1), y=ewma, mode='lines+markers', name='EWMA'), row=2, col=1)
    fig.add_hline(y=ewma_ucl, line_color='red', line_dash='dash', row=2, col=1)
    fig.add_hline(y=ewma_lcl, line_color='red', line_dash='dash', row=2, col=1)
    if not np.isnan(ewma_detect_time):
        idx = shift_point + int(ewma_detect_time) - 1
        fig.add_trace(go.Scatter(x=[idx+1], y=[ewma[idx]], mode='markers', name='First Detect',
                                 marker=dict(color='red', size=12, symbol='x')), row=2, col=1)
    
    # Plot CUSUM
    fig.add_trace(go.Scatter(x=np.arange(1, n_points+1), y=sh, mode='lines+markers', name='CUSUM High'), row=3, col=1)
    fig.add_trace(go.Scatter(x=np.arange(1, n_points+1), y=sl, mode='lines+markers', name='CUSUM Low'), row=3, col=1)
    fig.add_hline(y=h, line_color='red', line_dash='dash', row=3, col=1)
    if not np.isnan(cusum_detect_time):
        idx = shift_point + int(cusum_detect_time) - 1
        fig.add_trace(go.Scatter(x=[idx+1], y=[sh[idx] if sh[idx]>0 else sl[idx]], mode='markers', name='First Detect',
                                 marker=dict(color='red', size=12, symbol='x')), row=3, col=1)

    # Add annotation for the process change
    fig.add_vrect(x0=shift_point + 0.5, x1=n_points + 0.5, 
                  fillcolor="rgba(255,150,0,0.15)", line_width=0,
                  annotation_text="Process Change Begins", annotation_position="top left",
                  row='all', col=1)

    fig.update_layout(title=f"<b>Performance Comparison: Detecting a {scenario_desc}</b>",
                      height=800, showlegend=False)
    fig.update_xaxes(title_text="Data Point Number", row=3, col=1)
    
    return fig, i_detect_time, ewma_detect_time, cusum_detect_time
    
# ==============================================================================
# HELPER & PLOTTING FUNCTION (Time Series) - SME ENHANCED
# ==============================================================================
@st.cache_data
def plot_time_series_analysis(trend_strength=10, noise_sd=2, changepoint_strength=0.0):
    """
    Generates an enhanced, more realistic time series dashboard, including a changepoint
    scenario, forecast intervals, and essential ARIMA diagnostics (residuals, ACF).
    """
    np.random.seed(42)
    periods = 104
    changepoint_loc = 60 # Location of the trend changepoint
    dates = pd.date_range(start='2020-01-01', periods=periods, freq='W')
    
    # --- Dynamic Data Generation with Changepoint ---
    trend1 = np.linspace(50, 50 + trend_strength, changepoint_loc)
    end_val = trend1[-1]
    trend2 = np.linspace(end_val, end_val + (trend_strength + changepoint_strength), periods - changepoint_loc)
    trend = np.concatenate([trend1, trend2])
    
    seasonality = 5 * np.sin(np.arange(periods) * (2*np.pi/52.14))
    noise = np.random.normal(0, noise_sd, periods)
    
    y = trend + seasonality + noise
    df = pd.DataFrame({'ds': dates, 'y': y})
    
    train, test = df.iloc[:90], df.iloc[90:]

    # --- Re-fit models on the dynamic data ---
    m_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False).fit(train)
    future = m_prophet.make_future_dataframe(periods=14, freq='W')
    fc_prophet = m_prophet.predict(future)

    m_arima = ARIMA(train['y'], order=(5,1,0)).fit()
    fc_arima = m_arima.get_forecast(steps=14).summary_frame(alpha=0.2) # 80% CI

    # --- Dynamic KPI Calculation (Mean Absolute Error) ---
    mae_prophet = np.mean(np.abs(fc_prophet['yhat'].iloc[-14:].values - test['y'].values))
    mae_arima = np.mean(np.abs(fc_arima['mean'].values - test['y'].values))
    
    # --- ARIMA Diagnostics ---
    from statsmodels.graphics.tsaplots import plot_acf
    residuals_arima = m_arima.resid
    
    # --- Plotting Dashboard ---
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"colspan": 2}, None], [{}, {}]],
        subplot_titles=("<b>1. Forecast vs. Actual Data</b>",
                        "<b>2. ARIMA Residuals Over Time</b>", "<b>3. ARIMA Residuals ACF Plot</b>"),
        vertical_spacing=0.2
    )
    
    # Plot 1: Main Forecast
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Actual Data', line=dict(color='black')), row=1, col=1)
    fig.add_trace(go.Scatter(x=fc_prophet['ds'], y=fc_prophet['yhat_upper'], mode='lines', line=dict(width=0), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=fc_prophet['ds'], y=fc_prophet['yhat_lower'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(255,0,0,0.1)', name='Prophet 80% CI'), row=1, col=1)
    fig.add_trace(go.Scatter(x=test['ds'], y=fc_arima['mean_ci_upper'], mode='lines', line=dict(width=0), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=test['ds'], y=fc_arima['mean_ci_lower'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0,128,0,0.1)', name='ARIMA 80% CI'), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=fc_prophet['ds'], y=fc_prophet['yhat'], mode='lines', name='Prophet Forecast', line=dict(dash='dash', color='red')), row=1, col=1)
    fig.add_trace(go.Scatter(x=test['ds'], y=fc_arima['mean'], mode='lines', name='ARIMA Forecast', line=dict(dash='dash', color='green')), row=1, col=1)
    
    # --- THIS IS THE CORRECTED BLOCK ---
    # Draw the vertical lines WITHOUT annotation text first.
    forecast_start_date = train['ds'].iloc[-1]
    changepoint_date = df['ds'][changepoint_loc]
    fig.add_vline(x=forecast_start_date, line_width=2, line_dash="dash", line_color="grey")
    fig.add_vline(x=changepoint_date, line_width=2, line_dash="dot", line_color="purple")
    
    # Now, add the annotations separately using fig.add_annotation for full control.
    fig.add_annotation(x=forecast_start_date, y=0.05, yref="paper", text="Forecast Start",
                       showarrow=False, xshift=10, font=dict(color="grey"))
    fig.add_annotation(x=changepoint_date, y=0.95, yref="paper", text="Trend Changepoint",
                       showarrow=False, xshift=-10, font=dict(color="purple"))
    # --- END OF CORRECTION ---

    # Plot 2: ARIMA Residuals
    fig.add_trace(go.Scatter(x=train['ds'][1:], y=residuals_arima[1:], mode='lines', name='Residuals'), row=2, col=1)
    fig.add_hline(y=0, line_dash='dash', line_color='black', row=2, col=1)

    # Plot 3: ARIMA ACF
    acf_vals = sm.tsa.acf(residuals_arima, nlags=20)
    fig.add_trace(go.Bar(x=np.arange(1, 21), y=acf_vals[1:], name='ACF'), row=2, col=2)
    fig.add_shape(type='line', x0=0.5, x1=20.5, y0=1.96/np.sqrt(len(residuals_arima)), y1=1.96/np.sqrt(len(residuals_arima)), line=dict(color='red', dash='dash'), row=2, col=2)
    fig.add_shape(type='line', x0=0.5, x1=20.5, y0=-1.96/np.sqrt(len(residuals_arima)), y1=-1.96/np.sqrt(len(residuals_arima)), line=dict(color='red', dash='dash'), row=2, col=2)

    fig.update_layout(height=800, title_text='<b>Time Series Forecasting & Diagnostics Dashboard</b>',
                      legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.01))
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Process Value", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Residual Value", row=2, col=1)
    fig.update_xaxes(title_text="Lag", row=2, col=2)
    fig.update_yaxes(title_text="Autocorrelation", row=2, col=2)
                      
    return fig, mae_arima, mae_prophet

# ==============================================================================
# HELPER & PLOTTING FUNCTION (Stability Analysis) - SME ENHANCED
# ==============================================================================
@st.cache_data
def plot_stability_analysis(degradation_rate=-0.4, noise_sd=0.5, batch_to_batch_sd=0.1):
    """
    Generates enhanced, more realistic dynamic plots for stability analysis,
    including a formal ANCOVA test for poolability and individual batch trend lines,
    as per ICH Q1E guideline principles.
    """
    np.random.seed(1)
    time_points = np.array([0, 3, 6, 9, 12, 18, 24])
    n_batches = 3
    
    # --- Dynamic Data Generation with Batch-to-Batch Variation ---
    df_list = []
    for i in range(n_batches):
        # SME Enhancement: Control batch-to-batch variation for intercepts and slopes
        initial_potency = np.random.normal(102, batch_to_batch_sd * 5)
        batch_degradation_rate = np.random.normal(degradation_rate, batch_to_batch_sd)
        
        noise = np.random.normal(0, noise_sd, len(time_points))
        potency = initial_potency + batch_degradation_rate * time_points + noise
        
        batch_df = pd.DataFrame({'Time': time_points, 'Potency': potency, 'Batch': f'Batch {i+1}'})
        df_list.append(batch_df)
    
    df_melt = pd.concat(df_list, ignore_index=True)

    # --- SME Enhancement: Formal ANCOVA for Poolability ---
    # Fit a full model with interactions to test if slopes are different
    ancova_model = ols('Potency ~ Time * C(Batch)', data=df_melt).fit()
    anova_table = sm.stats.anova_lm(ancova_model, typ=2)
    poolability_p_value = anova_table.loc['Time:C(Batch)', 'PR(>F)']
    is_poolable = poolability_p_value > 0.25 # ICH Q1E criterion

    # --- Fit Pooled Model and Calculate Shelf Life ---
    # This is done regardless of poolability to show the user the result they WOULD get.
    pooled_model = ols('Potency ~ Time', data=df_melt).fit()
    LSL = 95.0
    
    x_pred = pd.DataFrame({'Time': np.linspace(0, 48, 100)})
    predictions = pooled_model.get_prediction(x_pred).summary_frame(alpha=0.05)
    
    shelf_life_df = predictions[predictions['mean_ci_lower'] >= LSL]
    shelf_life = x_pred['Time'][shelf_life_df.index[-1]] if not shelf_life_df.empty else 0
    shelf_life_val = shelf_life if not shelf_life_df.empty else 0
    
    if shelf_life > 47:
        shelf_life_str = ">48 Months"
    else:
        shelf_life_str = f"{shelf_life:.1f} Months"

    # --- Plotting ---
    fig = go.Figure()
    
    # Add shaded acceptable region
    fig.add_hrect(y0=LSL, y1=105, fillcolor="rgba(0,204,150,0.1)", layer="below", line_width=0)

    colors = px.colors.qualitative.Plotly
    # SME Enhancement: Plot individual batch data and trend lines
    for i, batch in enumerate(df_melt['Batch'].unique()):
        batch_df = df_melt[df_melt['Batch'] == batch]
        color = colors[i % len(colors)]
        # Raw data points
        fig.add_trace(go.Scatter(x=batch_df['Time'], y=batch_df['Potency'], mode='markers',
                                 name=batch, marker=dict(color=color, size=8)))
        # Individual batch fit line from ANCOVA model
        pred_batch = ancova_model.predict(batch_df)
        fig.add_trace(go.Scatter(x=batch_df['Time'], y=pred_batch, mode='lines',
                                 line=dict(color=color, dash='dash'), showlegend=False))

    # Plot results from the pooled model
    fig.add_trace(go.Scatter(x=x_pred['Time'], y=predictions['mean'], mode='lines',
                             name='Pooled Mean Trend', line=dict(color='black', width=3)))
    fig.add_trace(go.Scatter(x=x_pred['Time'], y=predictions['mean_ci_lower'], mode='lines',
                             name='95% Lower CI (Pooled)', line=dict(color='red', dash='dot', width=3)))
    
    # Add Limit and Shelf-Life lines
    fig.add_hline(y=LSL, line=dict(color='red', width=2), annotation_text="<b>Specification Limit</b>")
    
    if shelf_life_val > 0 and shelf_life_val < 48:
        fig.add_vline(x=shelf_life_val, line=dict(color='blue', dash='dash'),
                      annotation_text=f'<b>Shelf-Life = {shelf_life_str}</b>', annotation_position="bottom")
    
    title_color = '#00CC96' if is_poolable else '#EF553B'
    title_text = f"<b>Stability Analysis for Shelf-Life | Batches are {'Poolable' if is_poolable else 'NOT Poolable'} (p={poolability_p_value:.3f})</b>"
    
    fig.update_layout(
        title={'text': title_text, 'font': {'color': title_color}},
        xaxis_title="Time (Months)", yaxis_title="Potency (%)",
        xaxis_range=[-1, 48], yaxis_range=[max(85, df_melt['Potency'].min()-2), 105],
        legend=dict(x=0.01, y=0.01, yanchor='bottom', bgcolor='rgba(255,255,255,0.7)')
    )
    
    return fig, shelf_life_str, pooled_model.params['Time'], poolability_p_value

# The @st.cache_data decorator has been removed to allow for dynamic updates from sliders.
# ==============================================================================
# HELPER & PLOTTING FUNCTION (Survival Analysis) - SME ENHANCED & CORRECTED
# ==============================================================================
@st.cache_data
def plot_survival_analysis(group_b_lifetime=30, censor_rate=0.2):
    """
    Generates an enhanced, more realistic survival analysis dashboard, including
    confidence intervals, an "at risk" table, and a proper log-rank test.
    """
    np.random.seed(42)
    n_samples = 50
    
    # Generate time-to-event data from a Weibull distribution
    time_A = stats.weibull_min.rvs(c=1.5, scale=20, size=n_samples)
    time_B = stats.weibull_min.rvs(c=1.5, scale=group_b_lifetime, size=n_samples)
    
    # Generate censoring status
    event_observed_A = 1 - np.random.binomial(1, censor_rate, n_samples)
    event_observed_B = 1 - np.random.binomial(1, censor_rate, n_samples)

    # Use the lifelines library for robust calculations
    kmf_A = KaplanMeierFitter()
    kmf_A.fit(time_A, event_observed=event_observed_A, label='Group A (Old Component)')
    
    kmf_B = KaplanMeierFitter()
    kmf_B.fit(time_B, event_observed=event_observed_B, label='Group B (New Component)')

    # Perform the log-rank test for statistical significance
    results = logrank_test(time_A, time_B, event_observed_A, event_observed_B)
    p_value = results.p_value

    # --- Plotting ---
    # --- THIS IS THE CORRECTED BLOCK ---
    # Specify the `types` for each subplot in the `specs` argument.
    # Row 1 is a standard 'xy' plot. Row 2 is a 'table'.
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.8, 0.2],
        specs=[[{"type": "xy"}],
               [{"type": "table"}]],
        subplot_titles=("<b>Kaplan-Meier Survival Estimates</b>",) # Title only for the first plot
    )
    # --- END OF CORRECTION ---
    
    # Plot curves with confidence intervals
    colors = px.colors.qualitative.Plotly
    for kmf, color_hex in zip([kmf_A, kmf_B], [colors[0], colors[1]]):
        kmf_df = kmf.survival_function_.join(kmf.confidence_interval_)
        fig.add_trace(go.Scatter(x=kmf_df.index, y=kmf_df[kmf.label], mode='lines',
                                 name=kmf.label, line=dict(color=color_hex, shape='hv', width=3)), row=1, col=1)
        
        fill_color_rgba = f'rgba({",".join(str(c) for c in px.colors.hex_to_rgb(color_hex))}, 0.2)'
        
        fig.add_trace(go.Scatter(x=kmf_df.index, y=kmf_df[f'{kmf.label}_upper_0.95'], mode='lines',
                                 line=dict(width=0), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=kmf_df.index, y=kmf_df[f'{kmf.label}_lower_0.95'], mode='lines',
                                 line=dict(width=0), fill='tonexty', fillcolor=fill_color_rgba,
                                 name=f'{kmf.label} 95% CI'), row=1, col=1)

    # Add markers for censored data
    censored_A = time_A[event_observed_A == 0]
    censored_B = time_B[event_observed_B == 0]
    fig.add_trace(go.Scatter(x=censored_A, y=kmf_A.predict(censored_A), mode='markers',
                             marker=dict(color=colors[0], symbol='line-ns-open', size=10),
                             name='Censored (Group A)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=censored_B, y=kmf_B.predict(censored_B), mode='markers',
                             marker=dict(color=colors[1], symbol='line-ns-open', size=10),
                             name='Censored (Group B)'), row=1, col=1)

    # Add "At Risk" table to the second row
    time_bins = np.linspace(0, max(time_A.max(), time_B.max()), 6).astype(int)
    at_risk_A = [np.sum(time_A >= t) for t in time_bins]
    at_risk_B = [np.sum(time_B >= t) for t in time_bins]
    
    fig.add_trace(go.Table(
        header=dict(values=['<b>Time Point</b>'] + [f'<b>{t}</b>' for t in time_bins], align="left"),
        cells=dict(values=[
            ['<b>Group A At Risk</b>', '<b>Group B At Risk</b>'],
            *[[at_risk_A[i], at_risk_B[i]] for i in range(len(time_bins))]
        ], align="left", font=dict(size=12)),
    ), row=2, col=1)
    
    fig.update_layout(
        title='<b>Reliability / Survival Analysis Dashboard</b>',
        xaxis_title='Time to Event (e.g., Days to Failure)',
        yaxis_title='Survival Probability',
        yaxis_range=[0, 1.05],
        legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98, bgcolor='rgba(255,255,255,0.7)'),
        height=800
    )
    
    return fig, kmf_A.median_survival_time_, kmf_B.median_survival_time_, p_value

# ==============================================================================
# HELPER & PLOTTING FUNCTION (MVA/PLS) - SME ENHANCED
# ==============================================================================
@st.cache_data
def plot_mva_pls(signal_strength=2.0, noise_sd=0.2):
    """
    Generates an enhanced, more realistic MVA dashboard with more realistic spectral data,
    a cross-validation plot, and a predicted vs. actual plot.
    """
    np.random.seed(0)
    n_samples = 50
    n_features = 200
    wavelengths = np.linspace(800, 2200, n_features)

    # --- SME Enhancement: More realistic spectral data generation ---
    def make_peak(x, center, width, height):
        return height * np.exp(-((x - center)**2) / (2 * width**2))

    # Generate a curved baseline
    baseline = 0.1 + 5e-5 * (wavelengths - 800) + 1e-7 * (wavelengths - 1500)**2
    X = np.tile(baseline, (n_samples, 1))
    
    # Generate response variable y (e.g., concentration)
    y = np.linspace(5, 25, n_samples) + np.random.normal(0, 0.1, n_samples)
    
    # Add peaks to X that are correlated with y
    for i in range(n_samples):
        # Peak 1 (positively correlated with y)
        X[i, :] += make_peak(wavelengths, 1200, 50, y[i] * 0.005 * signal_strength)
        # Peak 2 (negatively correlated with y)
        X[i, :] += make_peak(wavelengths, 1600, 60, (25 - y[i]) * 0.004 * signal_strength)
        # Add some noise peaks unrelated to y
        X[i, :] += make_peak(wavelengths, 1000, 30, np.random.rand() * 0.05)
    
    # Add random measurement noise
    X += np.random.normal(0, noise_sd * 0.01, X.shape)

    # --- Dynamic Model Fitting & KPI Calculation ---
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import r2_score, mean_squared_error

    max_comps = 8
    r2_cal = []
    r2_cv = [] # This is Q2
    for n_comp in range(1, max_comps + 1):
        pls_cv = PLSRegression(n_components=n_comp)
        pls_cv.fit(X, y)
        y_cv = cross_val_predict(pls_cv, X, y, cv=10)
        r2_cal.append(r2_score(y, pls_cv.predict(X)))
        r2_cv.append(r2_score(y, y_cv))

    # Select the optimal number of components that maximizes Q2
    optimal_n_comp = np.argmax(r2_cv) + 1
    
    # Fit the final model
    pls = PLSRegression(n_components=optimal_n_comp)
    pls.fit(X, y)
    y_pred = pls.predict(X)
    
    model_r2 = pls.score(X, y)
    model_q2 = r2_cv[optimal_n_comp - 1]
    rmsecv = np.sqrt(mean_squared_error(y, cross_val_predict(pls, X, y, cv=10)))

    # VIP score calculation
    T = pls.x_scores_; W = pls.x_weights_; Q = pls.y_loadings_
    p, h = W.shape; VIPs = np.zeros((p,))
    s = np.diag(T.T @ T @ Q.T @ Q).reshape(h, -1)
    total_s = np.sum(s)
    for i in range(p):
        weight = np.array([(W[i,j] / np.linalg.norm(W[:,j]))**2 for j in range(h)])
        VIPs[i] = np.sqrt(p * (s.T @ weight) / total_s)

    # --- Plotting Dashboard ---
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("<b>1. Raw Spectral Data</b> (Color mapped to Y)",
                        "<b>2. Cross-Validation: Choosing Components</b>",
                        "<b>3. Model Performance: Predicted vs. Actual</b>",
                        "<b>4. Model Interpretation: VIP Scores</b>"),
        vertical_spacing=0.2, horizontal_spacing=0.1
    )

    # Plot 1: Raw Spectra
    for i in range(n_samples):
        color = px.colors.sample_colorscale('Viridis', (y[i] - y.min()) / (y.max() - y.min()))[0]
        fig.add_trace(go.Scatter(x=wavelengths, y=X[i,:], mode='lines', line=dict(color=color), showlegend=False), row=1, col=1)

    # Plot 2: Cross-Validation
    fig.add_trace(go.Scatter(x=np.arange(1, max_comps+1), y=r2_cal, mode='lines+markers', name='R² (Fit)', line=dict(color='blue')), row=1, col=2)
    fig.add_trace(go.Scatter(x=np.arange(1, max_comps+1), y=r2_cv, mode='lines+markers', name='Q² (Predict)', line=dict(color='green')), row=1, col=2)
    fig.add_vline(x=optimal_n_comp, line_dash="dash", line_color="black", annotation_text=f"Optimal LV={optimal_n_comp}", row=1, col=2)

    # Plot 3: Predicted vs. Actual
    fig.add_trace(go.Scatter(x=y, y=y_pred.flatten(), mode='markers', name='Samples'), row=2, col=1)
    fig.add_shape(type='line', x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max(),
                  line=dict(color='black', dash='dash'), row=2, col=1)

    # Plot 4: VIP Scores
    fig.add_trace(go.Bar(x=wavelengths, y=VIPs, name='VIP Score', marker_color='orange'), row=2, col=2)
    fig.add_hline(y=1, line=dict(color='red', dash='dash'), name='Significance Threshold', row=2, col=2)
    # Highlight the true peaks
    fig.add_vrect(x0=1150, x1=1250, fillcolor="rgba(0,255,0,0.15)", line_width=0, row=2, col=2, annotation_text="True Signal", annotation_position="top left")
    fig.add_vrect(x0=1550, x1=1650, fillcolor="rgba(0,255,0,0.15)", line_width=0, row=2, col=2)
    
    fig.update_layout(height=800, title_text='<b>Multivariate Analysis (PLS Regression) Dashboard</b>', title_x=0.5, showlegend=False)
    fig.update_xaxes(title_text='Wavelength (nm)', row=1, col=1); fig.update_yaxes(title_text='Absorbance', row=1, col=1)
    fig.update_xaxes(title_text='Number of Latent Variables', row=1, col=2); fig.update_yaxes(title_text='Coefficient of Determination', row=1, col=2)
    fig.update_xaxes(title_text='Actual Value', row=2, col=1); fig.update_yaxes(title_text='Predicted Value', row=2, col=1, scaleanchor="x2", scaleratio=1)
    fig.update_xaxes(title_text='Wavelength (nm)', row=2, col=2); fig.update_yaxes(title_text='VIP Score', row=2, col=2)
    
    return fig, model_r2, model_q2, optimal_n_comp, rmsecv

# ==============================================================================
# HELPER & PLOTTING FUNCTION (Clustering) - SME ENHANCED
# ==============================================================================
@st.cache_data
def plot_clustering(separation=15, spread=2.5, n_true_clusters=3):
    """
    Generates an enhanced, more realistic clustering dashboard, including a variable
    number of true clusters, a Silhouette diagnostic plot, and Voronoi boundaries.
    """
    np.random.seed(42)
    n_points_per_cluster = 50
    
    # --- Dynamic Data Generation ---
    # SME Enhancement: Generate a variable number of true clusters
    centers = []
    angle_step = 360 / n_true_clusters
    for i in range(n_true_clusters):
        angle = np.deg2rad(i * angle_step)
        x_center = 10 + separation * np.cos(angle)
        y_center = 10 + separation * np.sin(angle)
        centers.append((x_center, y_center))

    X_list, Y_list = [], []
    for x_c, y_c in centers:
        X_list.append(np.random.normal(x_c, spread, n_points_per_cluster))
        Y_list.append(np.random.normal(y_c, spread, n_points_per_cluster))
        
    X = np.concatenate(X_list)
    Y = np.concatenate(Y_list)
    df = pd.DataFrame({'X': X, 'Y': Y})
    
    # --- 1. Perform Clustering for a range of k and Diagnostics ---
    k_range = range(2, 9)
    wcss = []
    silhouette_scores = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(df[['X', 'Y']])
        wcss.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(df[['X', 'Y']], kmeans.labels_))
    
    # Find the optimal k based on silhouette score
    optimal_k = k_range[np.argmax(silhouette_scores)]

    # --- 2. Generate Plots based on the OPTIMAL k ---
    kmeans_opt = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto').fit(df)
    df['Cluster'] = kmeans_opt.labels_.astype(str)
    final_silhouette = silhouette_score(df[['X', 'Y']], df['Cluster'])

    # --- Plotting Dashboard ---
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("<b>1. Discovered Process Regimes</b>",
                        "<b>2. Elbow Method for Selecting k</b>",
                        "<b>3. Silhouette Analysis for Selecting k</b>",),
        specs=[[{"rowspan": 2}, {}], [None, {}]],
        vertical_spacing=0.2
    )

    # Plot 1: Main Scatter Plot with Voronoi boundaries
    fig_scatter = px.scatter(df, x='X', y='Y', color='Cluster',
                             labels={'X': 'Process Parameter 1', 'Y': 'Process Parameter 2'})
    for trace in fig_scatter.data:
        fig.add_trace(trace, row=1, col=1)
    
    centers = kmeans_opt.cluster_centers_
    fig.add_trace(go.Scatter(x=centers[:, 0], y=centers[:, 1], mode='markers',
                             marker=dict(color='black', size=15, symbol='x'),
                             name='Centroids'), row=1, col=1)

    # Plot 2: Elbow Method
    fig.add_trace(go.Scatter(x=list(k_range), y=wcss, mode='lines+markers', name='Inertia'), row=1, col=2)
    fig.add_vline(x=n_true_clusters, line_dash="dash", line_color="red",
                  annotation_text=f"True k = {n_true_clusters}", row=1, col=2)

    # Plot 3: Silhouette Score Plot
    fig.add_trace(go.Scatter(x=list(k_range), y=silhouette_scores, mode='lines+markers', name='Silhouette Score'), row=2, col=2)
    fig.add_vline(x=optimal_k, line_dash="dash", line_color="green",
                  annotation_text=f"Optimal k = {optimal_k}", row=2, col=2)

    fig.update_layout(height=800, title_text='<b>Unsupervised Clustering & Diagnostics Dashboard</b>', title_x=0.5, showlegend=False)
    fig.update_xaxes(title_text='Process Parameter 1', row=1, col=1); fig.update_yaxes(title_text='Process Parameter 2', row=1, col=1)
    fig.update_xaxes(title_text='Number of Clusters (k)', row=1, col=2); fig.update_yaxes(title_text='Inertia (WCSS)', row=1, col=2)
    fig.update_xaxes(title_text='Number of Clusters (k)', row=2, col=2); fig.update_yaxes(title_text='Mean Silhouette Score', row=2, col=2)
                             
    return fig, final_silhouette, optimal_k
    
# ==============================================================================
# HELPER & PLOTTING FUNCTION (Predictive QC) - SME ENHANCED
# ==============================================================================
@st.cache_data
def plot_classification_models(boundary_radius=12):
    """
    Generates an enhanced, more realistic classification dashboard, including
    probabilistic decision boundaries and a comparative ROC curve plot.
    """
    np.random.seed(1)
    n_points = 200
    # SME Enhancement: Frame data with a more realistic scenario
    purity = np.random.uniform(95, 105, n_points)
    bioactivity = np.random.uniform(80, 120, n_points)
    
    # The decision boundary is a circle centered at (100, 100).
    distance_from_center_sq = (purity - 100)**2 + (bioactivity - 100)**2
    prob_of_failure = 1 / (1 + np.exp(distance_from_center_sq - boundary_radius))
    y = np.random.binomial(1, prob_of_failure) # 1 = Fail (red), 0 = Pass (blue)
    
    X = np.column_stack((purity, bioactivity))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # --- Re-fit models and calculate dynamic KPIs ---
    lr = LogisticRegression().fit(X_train, y_train)
    lr_probs = lr.predict_proba(X_test)[:, 1]
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_test)[:, 1]

    # Calculate ROC curves
    fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_probs)
    auc_lr = auc(fpr_lr, tpr_lr)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
    auc_rf = auc(fpr_rf, tpr_rf)

    # --- Plotting Dashboard ---
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(f"<b>Logistic Regression</b>", f"<b>Random Forest</b>", "<b>Performance: ROC Curves</b>"),
        column_widths=[0.4, 0.4, 0.2]
    )

    # Create meshgrid for decision boundary
    xx, yy = np.meshgrid(np.linspace(95, 105, 100), np.linspace(80, 120, 100))
    
    # Plot Logistic Regression with probability contour
    Z_lr = lr.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
    contour_lr = go.Contour(x=xx[0], y=yy[:,0], z=Z_lr, colorscale='RdBu', showscale=False,
                            opacity=0.5, name='LR Prob.',
                            contours=dict(start=0, end=1, size=0.1))
    # Highlight the 50% decision boundary
    contour_lr_boundary = go.Contour(x=xx[0], y=yy[:,0], z=Z_lr, showscale=False,
                                     contours_coloring='lines', line_width=3,
                                     contours=dict(start=0.5, end=0.5, size=0))
    fig.add_trace(contour_lr, row=1, col=1)
    fig.add_trace(contour_lr_boundary, row=1, col=1)
    fig.add_trace(go.Scatter(x=X[:,0], y=X[:,1], mode='markers',
                             marker=dict(color=y, colorscale=[[0, 'blue'], [1, 'red']],
                                         line=dict(width=1, color='black'))), row=1, col=1)

    # Plot Random Forest with probability contour
    Z_rf = rf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
    contour_rf = go.Contour(x=xx[0], y=yy[:,0], z=Z_rf, colorscale='RdBu', showscale=False,
                            opacity=0.5, name='RF Prob.',
                            contours=dict(start=0, end=1, size=0.1))
    contour_rf_boundary = go.Contour(x=xx[0], y=yy[:,0], z=Z_rf, showscale=False,
                                     contours_coloring='lines', line_width=3,
                                     contours=dict(start=0.5, end=0.5, size=0))
    fig.add_trace(contour_rf, row=1, col=2)
    fig.add_trace(contour_rf_boundary, row=1, col=2)
    fig.add_trace(go.Scatter(x=X[:,0], y=X[:,1], mode='markers',
                             marker=dict(color=y, colorscale=[[0, 'blue'], [1, 'red']],
                                         line=dict(width=1, color='black'))), row=1, col=2)

    # Plot ROC Curves
    fig.add_trace(go.Scatter(x=fpr_lr, y=tpr_lr, mode='lines', name=f'Logistic Reg. (AUC={auc_lr:.3f})',
                             line=dict(color='royalblue', width=3)), row=1, col=3)
    fig.add_trace(go.Scatter(x=fpr_rf, y=tpr_rf, mode='lines', name=f'Random Forest (AUC={auc_rf:.3f})',
                             line=dict(color='darkorange', width=3)), row=1, col=3)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='No-Discrimination',
                             line=dict(color='grey', width=2, dash='dash')), row=1, col=3)

    fig.update_layout(title="<b>Predictive QC: Linear vs. Non-Linear Models</b>", showlegend=False, height=500)
    fig.update_xaxes(title_text="Peak Purity (HPLC)", row=1, col=1)
    fig.update_yaxes(title_text="Bioactivity (%)", row=1, col=1)
    fig.update_xaxes(title_text="Peak Purity (HPLC)", row=1, col=2)
    fig.update_yaxes(title_text="Bioactivity (%)", row=1, col=2)
    fig.update_xaxes(title_text="1 - Specificity", range=[-0.05, 1.05], row=1, col=3)
    fig.update_yaxes(title_text="Sensitivity", range=[-0.05, 1.05], row=1, col=3)
    
    return fig, auc_lr, auc_rf

def wilson_score_interval(p_hat, n, z=1.96):
    # This helper function remains the same, but it's good to keep it near the plotting function.
    if n == 0: return (0, 1)
    term1 = (p_hat + z**2 / (2 * n)); denom = 1 + z**2 / n; term2 = z * np.sqrt((p_hat * (1-p_hat)/n) + (z**2 / (4 * n**2))); return (term1 - term2) / denom, (term1 + term2) / denom
    
# ==============================================================================
# HELPER & PLOTTING FUNCTION (Anomaly Detection) - SME ENHANCED
# ==============================================================================
@st.cache_data
def plot_isolation_forest(contamination_rate=0.1):
    """
    Generates an enhanced, more realistic anomaly detection dashboard, including a
    visualization of an isolation tree and the distribution of anomaly scores.
    """
    np.random.seed(42)
    n_inliers = 200
    n_outliers = 15
    
    # --- Data Generation ---
    angle = np.random.uniform(0, 2 * np.pi, n_inliers)
    radius = np.random.normal(5, 0.5, n_inliers)
    X_inliers = np.array([radius * np.cos(angle), radius * np.sin(angle)]).T
    X_outliers = np.random.uniform(low=-10, high=10, size=(n_outliers, 2))
    X = np.concatenate([X_inliers, X_outliers], axis=0)
    ground_truth = np.concatenate([np.zeros(n_inliers), np.ones(n_outliers)])
    
    # --- Model Fitting ---
    clf = IsolationForest(contamination=contamination_rate, random_state=42, n_estimators=100)
    y_pred = clf.fit_predict(X)
    anomaly_scores = clf.decision_function(X) * -1
    
    df = pd.DataFrame(X, columns=['Process Parameter 1', 'Process Parameter 2'])
    df['Status'] = ['Anomaly' if p == -1 else 'Normal' for p in y_pred]
    df['Score'] = anomaly_scores
    df['GroundTruth'] = ['Outlier' if gt == 1 else 'Inlier' for gt in ground_truth]

    # --- Plotting Dashboard ---
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("<b>1. Anomaly Detection Results</b>",
                        "<b>2. Example Isolation Tree</b>",
                        "<b>3. Distribution of Anomaly Scores</b>"),
        specs=[[{"rowspan": 2}, {}], [None, {}]]
    )

    # Plot 1: Main Scatter Plot (Grid Position 1,1)
    fig.add_trace(px.scatter(df, x='Process Parameter 1', y='Process Parameter 2',
                             color='Status', color_discrete_map={'Normal': '#636EFA', 'Anomaly': '#EF553B'},
                             symbol='Status', symbol_map={'Normal': 'circle', 'Anomaly': 'x-thin-open'}
                            ).data[0], row=1, col=1)
    fig.add_trace(px.scatter(df, x='Process Parameter 1', y='Process Parameter 2',
                         color='Status', color_discrete_map={'Normal': '#636EFA', 'Anomaly': '#EF553B'},
                         symbol='Status', symbol_map={'Normal': 'circle', 'Anomaly': 'x-thin-open'}
                        ).data[1], row=1, col=1)

    # Plot 2: Visualize one of the trees (Grid Position 1,2)
    try:
        from sklearn.tree import export_graphviz
        import graphviz
        
        single_tree = clf.estimators_[0]
        dot_data = export_graphviz(single_tree, out_file=None,
                                   feature_names=['Param 1', 'Param 2'],
                                   filled=True, rounded=True,
                                   special_characters=True, max_depth=3)
        graph = graphviz.Source(dot_data)
        
        png_bytes = graph.pipe(format='png')
        img = Image.open(io.BytesIO(png_bytes))
        
        fig.add_layout_image(
            dict(
                source=img,
                xref="x2", yref="y2",
                x=0.5, y=0.5, sizex=1, sizey=1,
                xanchor="center", yanchor="middle",
                sizing="contain", layer="above"
            )
        )
    except (ImportError, FileNotFoundError, graphviz.backend.execute.ExecutableNotFound):
        fig.add_annotation(
            xref="x2 domain", yref="y2 domain", x=0.5, y=0.5,
            text="<b>Graphviz executable not found.</b><br>Install it to see the example tree.",
            showarrow=False, font=dict(size=14, color='red')
        )
    
    fig.update_xaxes(visible=False, showticklabels=False, range=[0, 1], row=1, col=2)
    fig.update_yaxes(visible=False, showticklabels=False, range=[0, 1], row=1, col=2)

    # --- THIS IS THE CORRECTED BLOCK ---
    # Plot 3: Score Distribution (Grid Position 2,2)
    # The `row` and `col` arguments have been changed from (2,1) to (2,2).
    fig.add_trace(px.histogram(df, x='Score', color='GroundTruth',
                               color_discrete_map={'Inlier': 'grey', 'Outlier': 'red'},
                               barmode='overlay', marginal='rug').data[0], row=2, col=2)
    fig.add_trace(px.histogram(df, x='Score', color='GroundTruth',
                           color_discrete_map={'Inlier': 'grey', 'Outlier': 'red'},
                           barmode='overlay', marginal='rug').data[1], row=2, col=2)
    
    score_threshold = np.percentile(anomaly_scores, 100 * (1-contamination_rate))
    fig.add_vline(x=score_threshold, line_dash="dash", line_color="black",
                  annotation_text="Decision Threshold", row=2, col=2)
    # --- END OF CORRECTION ---

    fig.update_layout(height=800, title_text='<b>Anomaly Detection Dashboard: Isolation Forest</b>', title_x=0.5,
                      showlegend=True, legend=dict(yanchor="top", y=1, xanchor="left", x=0.5))
    # Add axis titles for the new plot positions
    fig.update_xaxes(title_text="Anomaly Score", row=2, col=2)
    fig.update_yaxes(title_text="Count", row=2, col=2)

    return fig, (y_pred == -1).sum()
    
# ==============================================================================
# HELPER & PLOTTING FUNCTION (XAI/SHAP) - SME ENHANCED & PDP FIX
# ==============================================================================
@st.cache_data
def plot_xai_shap(case_to_explain="highest_risk", dependence_feature='Operator Experience (Months)'):
    """
    Trains an XGBoost model and generates a robust set of SHAP and PDP/ICE plots
    for comprehensive model explanation.
    """
    plt.style.use('default')
    
    # 1. Simulate Assay Data
    np.random.seed(42)
    n_runs = 200
    operator_experience = np.random.randint(1, 25, n_runs)
    cal_slope = np.random.normal(1.0, 0.05, n_runs) - (operator_experience * 0.0015)
    qc1_value = np.random.normal(50, 2, n_runs) - np.random.uniform(0, operator_experience / 8, n_runs)
    reagent_age_days = np.random.randint(5, 90, n_runs)
    instrument_id = np.random.choice(['Inst_A', 'Inst_B', 'Inst_C'], n_runs, p=[0.5, 0.3, 0.2])
    
    prob_failure = 1 / (1 + np.exp(-(-2.0 - 0.18 * operator_experience + (reagent_age_days / 20)**1.5
                                     - (cal_slope - 1.0) * 25 + (qc1_value < 48) * 0.5
                                     + (instrument_id == 'Inst_C') * 1.5)))
    run_failed = np.random.binomial(1, prob_failure)

    X_display = pd.DataFrame({
        'Operator Experience (Months)': operator_experience,
        'Reagent Age (Days)': reagent_age_days,
        'Calibrator Slope': cal_slope,
        'QC Level 1 Value': qc1_value,
        'Instrument ID': instrument_id
    })
    y = pd.Series(run_failed, name="Run Failed")
    
    X_encoded = pd.get_dummies(X_display, drop_first=True)
    X = X_encoded.astype(int)

    # Use XGBoost for better performance
    model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss').fit(X, y)
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    # Find the instance index for the local explanation
    failure_probabilities = model.predict_proba(X)[:, 1]
    if case_to_explain == "highest_risk":
        instance_index = np.argmax(failure_probabilities)
    elif case_to_explain == "lowest_risk":
        instance_index = np.argmin(failure_probabilities)
    else: # "most_ambiguous"
        instance_index = np.argmin(np.abs(failure_probabilities - 0.5))

    # 3. Generate Global Summary Plot (Beeswarm)
    fig_summary, ax_summary = plt.subplots()
    shap.summary_plot(shap_values, X, show=False)
    buf_summary = io.BytesIO()
    fig_summary.savefig(buf_summary, format='png', bbox_inches='tight')
    plt.close(fig_summary)
    buf_summary.seek(0)
    
    # 4. Generate Local Waterfall Plot
    fig_waterfall, ax_waterfall = plt.subplots()
    shap.waterfall_plot(shap_values[instance_index], show=False)
    buf_waterfall = io.BytesIO()
    fig_waterfall.savefig(buf_waterfall, format='png', bbox_inches='tight')
    plt.close(fig_waterfall)
    buf_waterfall.seek(0)
    
    # --- 5. SME ENHANCEMENT: Generate a robust PDP/ICE Plot instead of SHAP Dependence Plot ---
    fig_pdp, ax_pdp = plt.subplots(figsize=(8, 6))
    
    # The feature name might be different after one-hot encoding
    plot_feature = dependence_feature
    if dependence_feature == 'Instrument ID':
        inst_cols = [col for col in X.columns if 'Instrument ID_' in col]
        plot_feature = inst_cols[0] if inst_cols else 'Operator Experience (Months)'
    
    PartialDependenceDisplay.from_estimator(
        estimator=model,
        X=X,
        features=[plot_feature],
        kind="both",  # Show both PDP and ICE lines
        ice_lines_kw={"color": "lightblue", "alpha": 0.3, "linewidth": 0.5},
        pd_line_kw={"color": "red", "linewidth": 4, "linestyle": "--"},
        ax=ax_pdp
    )
    ax_pdp.set_title(f"Partial Dependence & ICE Plot for\n'{plot_feature}'", fontsize=14)
    ax_pdp.set_ylabel("Predicted Probability of Failure", fontsize=12)
    ax_pdp.set_xlabel(plot_feature, fontsize=12)
    
    buf_pdp = io.BytesIO()
    fig_pdp.savefig(buf_pdp, format='png', bbox_inches='tight')
    plt.close(fig_pdp)
    buf_pdp.seek(0)
    
    actual_outcome = "Failed" if y.iloc[instance_index] == 1 else "Passed"
    
    # Return the new PDP buffer instead of the old dependence buffer
    return buf_summary, buf_waterfall, buf_pdp, X_display.iloc[instance_index:instance_index+1], actual_outcome, instance_index
    
# ==============================================================================
# HELPER & PLOTTING FUNCTION (Advanced AI) - SME VISUAL OVERHAUL
# ==============================================================================
@st.cache_data
def plot_advanced_ai_concepts(concept, p1=0, p2=0, p3=0):
    """
    Generates professionally redesigned, interactive, and domain-specific Plotly figures
    for advanced AI topics in a V&V and Tech Transfer context.
    """
    fig = go.Figure()

    if concept == "Transformers":
        time = np.arange(14)
        vcd = 1 / (1 + np.exp(-0.8 * (time - 6))) * 15
        glucose = 5 - 0.4 * time + np.random.normal(0, 0.1, 14)
        do = 30 + 10 * np.sin(time / 2) + np.random.normal(0, 1, 14)
        
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                            specs=[[{"secondary_y": True}]] * 3)
        
        # Plot time series data
        fig.add_trace(go.Scatter(x=time, y=do, mode='lines', name='DO', line=dict(color='#636EFA')), row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=time, y=glucose, mode='lines', name='Glucose', line=dict(color='#00CC96')), row=2, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=time, y=vcd, mode='lines', name='VCD', line=dict(color='#FECB52')), row=3, col=1, secondary_y=False)

        # SME Enhancement: Visualize attention as a bar chart on a secondary axis
        attention_day = p1
        importance_profile = np.exp(-0.5 * (np.abs(time - attention_day)))
        importance_profile[6] = max(importance_profile[6], 1.5) # Peak growth is always important
        importance_profile /= importance_profile.sum()

        fig.add_trace(go.Bar(x=time, y=importance_profile, name='Attention', marker_color='#EF553B', opacity=0.6), row=1, col=1, secondary_y=True)
        fig.add_trace(go.Bar(x=time, y=importance_profile, showlegend=False, marker_color='#EF553B', opacity=0.6), row=2, col=1, secondary_y=True)
        fig.add_trace(go.Bar(x=time, y=importance_profile, showlegend=False, marker_color='#EF553B', opacity=0.6), row=3, col=1, secondary_y=True)
        
        fig.update_layout(title_text=f"<b>Transformer Attention: Predicting final titer from Day {attention_day}</b>",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig.update_xaxes(title_text="Day", row=3, col=1)
        fig.update_yaxes(title_text="DO (%)", row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Glucose (g/L)", row=2, col=1, secondary_y=False)
        fig.update_yaxes(title_text="VCD", row=3, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Attention Weight", showgrid=False, row=1, col=1, secondary_y=True)
        fig.update_yaxes(showticklabels=False, row=2, col=1, secondary_y=True)
        fig.update_yaxes(showticklabels=False, row=3, col=1, secondary_y=True)


    elif concept == "Graph Neural Networks (GNNs)":
        evidence_strength = p1
        nodes = {
            'Media Lot A1': {'pos': (1, 5)}, 'Buffer Lot B2': {'pos': (1, 3)},
            'Assay Kit C3': {'pos': (1, 1)}, 'Batch 101': {'pos': (3, 5)},
            'Batch 102': {'pos': (3, 3)}, 'Batch 103': {'pos': (3, 1)},
            'Final QC Test': {'pos': (5, 3)}
        }
        edges = [('Media Lot A1', 'Batch 101'), ('Media Lot A1', 'Batch 102'), ('Buffer Lot B2', 'Batch 101'),
                 ('Buffer Lot B2', 'Batch 102'), ('Buffer Lot B2', 'Batch 103'), ('Assay Kit C3', 'Batch 103'),
                 ('Batch 101', 'Final QC Test'), ('Batch 102', 'Final QC Test'), ('Batch 103', 'Final QC Test')]
        
        # Propagate "guilt" probability backwards
        node_probs = {name: 0.0 for name in nodes}
        node_probs['Final QC Test'] = 0.99
        node_probs['Batch 102'] = min(0.99, node_probs['Final QC Test'] * 0.8 * evidence_strength)
        node_probs['Media Lot A1'] = min(0.99, node_probs['Batch 102'] * 0.7 * evidence_strength)
        node_probs['Buffer Lot B2'] = min(0.99, node_probs['Batch 102'] * 0.9 * evidence_strength)

        for start, end in edges:
            fig.add_trace(go.Scatter(x=[nodes[start]['pos'][0], nodes[end]['pos'][0]],
                                     y=[nodes[start]['pos'][1], nodes[end]['pos'][1]],
                                     mode='lines', line=dict(color="lightgrey", width=2)))
        
        for name, props in nodes.items():
            prob = node_probs[name]
            color = f'rgb(255, {220 - prob*200}, {220 - prob*200})'
            fig.add_trace(go.Scatter(x=[props['pos'][0]], y=[props['pos'][1]], mode='markers+text',
                                     text=f"<b>{name.replace(' ', '<br>')}</b><br>P(Cause)={prob:.2f}", textposition="middle center",
                                     marker=dict(size=120, color=color, symbol='circle',
                                                 line=dict(width=3, color='black'))))
        fig.update_layout(title_text="<b>GNN: Root Cause Analysis via Lot Genealogy</b>")

    elif concept == "Reinforcement Learning (RL)":
        cost_of_waste = p1; process_variability = p2
        
        # Agent
        fig.add_shape(type="rect", x0=1, y0=3.5, x1=3, y1=5, fillcolor='#636EFA', line=dict(width=2))
        fig.add_annotation(x=2, y=4.25, text="<b>RL Agent</b><br>(Feed Policy)", showarrow=False, font_color="white")
        # Digital Twin
        fig.add_shape(type="rect", x0=5, y0=0.5, x1=11, y1=5.5, fillcolor='rgba(128,128,128,0.1)', line=dict(width=2, dash='dash'))
        fig.add_annotation(x=8, y=5.2, text="<b>Digital Twin (Bioreactor Sim)</b>", showarrow=False)
        # Real Process
        fig.add_shape(type="rect", x0=1, y0=1, x1=3, y1=2.5, fillcolor='#00CC96', line=dict(width=2))
        fig.add_annotation(x=2, y=1.75, text="<b>Real Process</b>", showarrow=False, font_color="white")
        
        # Arrows
        fig.add_annotation(x=4.8, y=4.25, ax=3.2, ay=4.25, text="<b>Action</b>", showarrow=True, arrowhead=2)
        fig.add_annotation(x=3.2, y=3.75, ax=4.8, ay=3.75, text="<b>State, Reward</b>", showarrow=True, arrowhead=2)
        fig.add_annotation(x=2, y=2.7, ax=2, ay=3.3, text="<b>Deploy<br>Optimal Policy</b>", showarrow=True, arrowhead=5, arrowcolor='#00CC96')

        time = np.linspace(5.5, 10.5, 100)
        target_vcd = 3.5
        aggressiveness = 1 / (1 + cost_of_waste * process_variability)
        vcd_profile = target_vcd - 1 * np.exp(-aggressiveness * (time-5.5)) + np.random.normal(0, process_variability*0.1, 100)
        
        fig.add_trace(go.Scatter(x=time, y=vcd_profile, mode='lines', line=dict(color='royalblue', width=3)))
        fig.add_hline(y=target_vcd, line=dict(color='black', dash='dot'), line_width=2)
        fig.update_yaxes(range=[2.2, 4.2], title_text="VCD")
        fig.update_xaxes(title_text="Days")
        fig.update_layout(title_text="<b>RL: Learning an Optimal Feeding Strategy</b>")

    elif concept == "Generative AI":
        creativity = p1
        np.random.seed(42)
        x_real = np.random.normal(98, 0.5, 4); y_real = np.random.normal(70, 2, 4)
        x_synth = np.random.normal(98, 0.5 + creativity, 100); y_synth = np.random.normal(70, 2 + creativity * 5, 100)
        
        fig.add_trace(go.Scatter(x=x_real, y=y_real, mode='markers', name='Real OOS Data',
                                 marker=dict(color='#EF553B', size=15, symbol='x-thin', line=dict(width=3))))
        fig.add_trace(go.Scatter(x=x_synth, y=y_synth, mode='markers', name='Synthetic OOS Data',
                                 marker=dict(color='#FECB52', size=8, symbol='circle', opacity=0.7)))
        
        fig.add_vrect(x0=95, x1=105, fillcolor='rgba(0,204,150,0.1)', line_width=0)
        fig.add_hrect(y0=80, y1=120, fillcolor='rgba(0,204,150,0.1)', line_width=0,
                      annotation_text="In-Spec Region", annotation_position="top left")
        
        fig.add_annotation(x=103, y=100, text="<b>Generative<br>Model</b>", showarrow=True, arrowhead=2,
                           ax=101, ay=85, font=dict(size=14, color='white'), bgcolor='#636EFA', borderpad=10)
        
        fig.update_layout(title_text="<b>Generative AI: Augmenting Rare Assay Failure Data</b>",
                          xaxis_title="Assay Parameter 1 (e.g., Purity)",
                          yaxis_title="Assay Parameter 2 (e.g., Potency)")

    # Standardize final layout
    fig.update_layout(height=500, showlegend=False, margin=dict(l=10, r=10, t=50, b=10))
    if concept in ["Graph Neural Networks (GNNs)"]:
        fig.update_layout(xaxis=dict(visible=False, showgrid=False, range=[0, 6]), 
                          yaxis=dict(visible=False, showgrid=False, range=[0, 6]))
    elif concept in ["Reinforcement Learning (RL)"]:
        fig.update_layout(xaxis=dict(visible=False, showgrid=False), 
                          yaxis=dict(visible=False, showgrid=False))
                      
    return fig
#================================================================================================================================================================================================
#=========================================================================NEW HYBRID METHODS ==================================================================================================
#===========================================================================================================================================================================================
@st.cache_data
def plot_mewma_xgboost(drift_magnitude=0.03, lambda_mewma=0.2):
    """
    Generates an enhanced, more realistic MEWMA dashboard, including autocorrelated data,
    a gradual drift failure mode, and a clearer waterfall plot for diagnostics.
    """
    np.random.seed(42)
    n_train, n_monitor = 100, 80
    n_total = n_train + n_monitor
    drift_start_point = n_train

    # --- 1. SME Enhancement: Simulate more realistic, autocorrelated process data ---
    mean_vec = np.array([7.0, 50.0, 100.0])
    cov_matrix = np.array([[0.04, 0.5, 0.8], [0.5, 16.0, 25.0], [0.8, 25.0, 64.0]])
    
    # Generate underlying random shocks
    innovations = np.random.multivariate_normal(np.zeros(3), cov_matrix, n_total)
    
    # Create autocorrelated noise (AR(1) process)
    phi = 0.6 # Autocorrelation coefficient
    noise = np.zeros_like(innovations)
    for i in range(1, n_total):
        noise[i, :] = phi * noise[i-1, :] + innovations[i, :]
        
    data = mean_vec + noise
    
    # SME Enhancement: Inject a subtle, GRADUAL drift in Temp and Pressure
    drift_duration = n_total - drift_start_point
    drift_vector = np.zeros_like(data)
    temp_drift = np.linspace(0, drift_magnitude * np.sqrt(cov_matrix[1,1]) * drift_duration, drift_duration)
    pressure_drift = np.linspace(0, drift_magnitude * np.sqrt(cov_matrix[2,2]) * drift_duration, drift_duration)
    drift_vector[drift_start_point:, 1] = temp_drift
    drift_vector[drift_start_point:, 2] = pressure_drift
    
    data += drift_vector
    df = pd.DataFrame(data, columns=['pH', 'Temperature', 'Pressure'])

    # 2. --- MEWMA Calculation (using Phase 1 data for covariance) ---
    train_cov = df.iloc[:n_train].cov().values
    S_inv = np.linalg.inv(train_cov)
    train_mean = df.iloc[:n_train].mean().values
    
    Z = np.zeros_like(data)
    t_squared_mewma = np.zeros(n_total)
    Z[0, :] = train_mean
    for i in range(1, n_total):
        Z[i, :] = (1 - lambda_mewma) * Z[i-1, :] + lambda_mewma * data[i, :]
        diff = Z[i, :] - train_mean
        t_squared_mewma[i] = diff.T @ S_inv @ diff

    # 3. --- Control Limit (Asymptotic) ---
    p = data.shape[1]
    ucl = (p * lambda_mewma / (2 - lambda_mewma)) * f.ppf(0.9973, p, 1000) # 3-sigma equivalent
    
    ooc_points_mask = t_squared_mewma[n_train:] > ucl
    first_ooc_index = np.argmax(ooc_points_mask) + n_train if np.any(ooc_points_mask) else None
    
    # 4. --- XGBoost Diagnostic Model ---
    buf_waterfall = None
    if first_ooc_index:
        y = np.array([0] * n_train + [1] * n_monitor)
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42).fit(df, y)
        explainer = shap.Explainer(model, df)
        shap_values = explainer(df)
        
        # SME Enhancement: Use a clearer Waterfall plot
        fig_waterfall, ax_waterfall = plt.subplots()
        shap.waterfall_plot(shap_values[first_ooc_index], show=False)
        buf_waterfall = io.BytesIO()
        fig_waterfall.savefig(buf_waterfall, format='png', bbox_inches='tight')
        plt.close(fig_waterfall)
        buf_waterfall.seek(0)

    # 5. --- Plotting Dashboard ---
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("<b>1. Raw Process Data (The 'Stealth' Drift)</b>",
                                        "<b>2. MEWMA Chart (The Detector)</b>"))

    # Plot 1: Raw Data
    for col in df.columns:
        fig.add_trace(go.Scatter(y=df[col], name=col, mode='lines'), row=1, col=1)
    
    # Plot 2: MEWMA Chart
    fig.add_trace(go.Scatter(y=t_squared_mewma, name='MEWMA T²', mode='lines+markers', line_color='#636EFA'), row=2, col=1)
    fig.add_hline(y=ucl, line=dict(color='red', dash='dash'), row=2, col=1)

    # Add annotations to both plots
    for r in [1, 2]:
        fig.add_vrect(x0=0, x1=n_train, fillcolor="rgba(0,204,150,0.1)", line_width=0,
                      annotation_text="Phase 1: Training", annotation_position="top left", row=r, col=1)
        fig.add_vrect(x0=n_train, x1=n_total, fillcolor="rgba(255,150,0,0.1)", line_width=0,
                      annotation_text="Phase 2: Monitoring", annotation_position="top right", row=r, col=1)
    if first_ooc_index:
        fig.add_vline(x=first_ooc_index, line=dict(color='red', width=2),
                      annotation_text="First Alarm", annotation_position='top', row='all', col=1)

    fig.update_layout(height=600, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="MEWMA T²", row=2, col=1)
    fig.update_xaxes(title_text="Observation Number", row=2, col=1)

    return fig, buf_waterfall, first_ooc_index

# ==============================================================================
# HELPER & PLOTTING FUNCTION (Method 2) - SME ENHANCED
# ==============================================================================
@st.cache_data
def plot_bocpd_ml_features(autocorr_shift=0.4, noise_increase=2.0):
    """
    Generates an enhanced, more realistic BOCPD dashboard, monitoring the residuals
    of an AR model and visualizing the run length probability.
    """
    np.random.seed(42)
    n_points = 200
    change_point = 100
    
    # --- 1. SME Enhancement: Simulate a process with a change in dynamics ---
    data = np.zeros(n_points)
    # Phase 1: High autocorrelation, low noise
    phi1 = 0.9
    for i in range(1, change_point):
        data[i] = phi1 * data[i-1] + np.random.normal(0, 1)
    # Phase 2: Lower autocorrelation, higher noise
    phi2 = phi1 - autocorr_shift
    for i in range(change_point, n_points):
        data[i] = phi2 * data[i-1] + np.random.normal(0, noise_increase)
        
    # --- 2. Create an "ML Feature": 1-step-ahead forecast error (residuals) ---
    # A simple AR(1) model is our "ML" feature extractor
    ml_feature = np.zeros(n_points)
    for i in range(1, n_points):
        prediction = phi1 * data[i-1] # Model assumes Phase 1 dynamics
        ml_feature[i] = data[i] - prediction
    
    # --- 3. BOCPD Algorithm ---
    hazard = 1 / 250.0 # Constant hazard rate
    # Assume a Gaussian model for the residuals
    mean0, var0 = 0, np.var(ml_feature[1:change_point])
    
    R = np.zeros((n_points + 1, n_points + 1))
    R[0, 0] = 1 # Initial state
    max_run_lengths = np.zeros(n_points)
    prob_rl_zero = np.zeros(n_points) # Probability that run length is 0
    
    for t in range(1, n_points + 1):
        x = ml_feature[t-1]
        pred_probs = stats.norm.pdf(x, loc=mean0, scale=np.sqrt(var0))
        
        # Growth probabilities
        R[1:t+1, t] = R[0:t, t-1] * pred_probs * (1 - hazard)
        # Change point probability
        R[0, t] = np.sum(R[:, t-1] * pred_probs * hazard)
        
        # Normalize
        R[:, t] /= np.sum(R[:, t])
        
        # Store metrics
        prob_rl_zero[t-1] = R[0, t]
        max_run_lengths[t-1] = np.argmax(R[:, t])

    # --- 4. Plotting Dashboard ---
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=("<b>1. Raw Process Data (Change is subtle)</b>",
                                        "<b>2. Model Residuals (ML Feature)</b>",
                                        "<b>3. BOCPD: Most Likely Run Length (MAP)</b>",
                                        "<b>4. BOCPD: Probability of a Changepoint</b>"))
    # Plot 1: Raw Data
    fig.add_trace(go.Scatter(y=data, mode='lines', name='Raw Data'), row=1, col=1)
    
    # Plot 2: ML Feature (Residuals)
    fig.add_trace(go.Scatter(y=ml_feature, mode='lines', name='Residuals', line_color='orange'), row=2, col=1)
    
    # Plot 3: Most Likely Run Length (MAP Estimate)
    fig.add_trace(go.Scatter(y=max_run_lengths, mode='lines', name='MAP Run Length', line_color='green'), row=3, col=1)
    
    # Plot 4: Probability of a Changepoint (Run Length = 0)
    fig.add_trace(go.Scatter(y=prob_rl_zero, mode='lines', name='P(Changepoint)', line_color='purple', fill='tozeroy'), row=4, col=1)
    
    # Add changepoint line to all plots
    fig.add_vline(x=change_point, line_dash='dash', line_color='red', row='all', col=1,
                  annotation_text="True Changepoint", annotation_position="top")

    fig.update_layout(height=800, title_text="<b>Bayesian Online Change Point Detection on Model Residuals</b>", showlegend=False)
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Residual", row=2, col=1)
    fig.update_yaxes(title_text="Run Length", row=3, col=1)
    fig.update_yaxes(title_text="Probability", row=4, col=1, range=[0, 1])
    fig.update_xaxes(title_text="Observation Number", row=4, col=1)
    
    return fig, prob_rl_zero[change_point]

@st.cache_data
def plot_kalman_nn_residual(measurement_noise=1.0, shock_magnitude=10.0, process_noise_q=0.01):
    """
    Generates an enhanced, more realistic Kalman Filter dashboard, simulating a non-linear
    process and visualizing the filter's uncertainty.
    """
    np.random.seed(123)
    n_points = 100
    
    # --- 1. SME Enhancement: Simulate a more realistic non-linear dynamic process ---
    # A damped sine wave, representing a process settling to a steady state.
    time = np.arange(n_points)
    true_state = 100 + 20 * np.exp(-time / 50) * np.sin(time / 5)
    
    # Add a sudden, unexpected shock (fault)
    shock_point = 70
    true_state[shock_point:] += shock_magnitude
    
    # Create noisy measurements
    measurements = true_state + np.random.normal(0, measurement_noise, n_points)
    
    # 2. --- Kalman Filter Implementation ---
    # The filter's internal model is a simple linear approximation of the true non-linear process
    # This mismatch is what a Neural Network would be used to correct in a real application.
    F = 1 # State transition matrix (assumes random walk)
    H = 1 # Measurement matrix
    Q = process_noise_q # Process noise (model uncertainty) - NOW A SLIDER
    R = measurement_noise**2 # Measurement noise (known from sensor)
    
    x_est = np.zeros(n_points) # Estimated state
    p_est = np.zeros(n_points) # Estimated error covariance
    residuals = np.zeros(n_points)
    
    x_est[0] = measurements[0]
    p_est[0] = 1.0
    
    for k in range(1, n_points):
        # Predict
        x_pred = F * x_est[k-1]
        p_pred = F * p_est[k-1] * F + Q
        
        # Update
        residual = measurements[k] - H * x_pred
        S = H * p_pred * H + R
        K = p_pred * H / S # Kalman Gain
        
        x_est[k] = x_pred + K * residual
        p_est[k] = (1 - K * H) * p_pred
        residuals[k] = residual

    # 3. --- Control Chart on Residuals ---
    limit_data = residuals[:shock_point-1]
    res_std = np.std(limit_data)
    ucl, lcl = 3 * res_std, -3 * res_std
    
    ooc_indices = np.where((residuals > ucl) | (residuals < lcl))[0]
    first_ooc = ooc_indices[0] if len(ooc_indices) > 0 else None
    
    # --- 4. Plotting Dashboard ---
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("<b>1. State Estimation: Tracking a Noisy Process</b>",
                                        "<b>2. Fault Detection: Control Chart on Residuals</b>"))

    # Plot 1: State Estimation
    fig.add_trace(go.Scatter(x=time, y=measurements, mode='markers', name='Noisy Measurements',
                             marker=dict(color='grey', opacity=0.7)), row=1, col=1)
    fig.add_trace(go.Scatter(x=time, y=true_state, mode='lines', name='True Hidden State',
                             line=dict(dash='dash', color='black', width=3)), row=1, col=1)
    # SME Enhancement: Show filter's uncertainty bounds
    upper_bound = x_est + np.sqrt(p_est)
    lower_bound = x_est - np.sqrt(p_est)
    fig.add_trace(go.Scatter(x=time, y=upper_bound, mode='lines', line=dict(width=0), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=time, y=lower_bound, mode='lines', line=dict(width=0), fill='tonexty',
                             fillcolor='rgba(255,0,0,0.2)', name='Kalman Uncertainty (±1σ)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=time, y=x_est, mode='lines', name='Kalman Estimate',
                             line=dict(color='red', width=3)), row=1, col=1)

    # Plot 2: Residuals Control Chart
    fig.add_trace(go.Scatter(x=time, y=residuals, mode='lines+markers', name='Residuals',
                             line=dict(color='#636EFA')), row=2, col=1)
    if first_ooc:
        fig.add_trace(go.Scatter(x=[first_ooc], y=[residuals[first_ooc]], mode='markers', name='Alarm',
                                 marker=dict(color='red', size=12, symbol='x')), row=2, col=1)
    fig.add_hline(y=ucl, line_color='red', row=2, col=1)
    fig.add_hline(y=lcl, line_color='red', row=2, col=1)
    fig.add_hline(y=0, line_color='black', line_dash='dash', row=2, col=1)
    
    fig.add_vline(x=shock_point, line_dash='dot', line_color='red',
                  annotation_text='Process Shock', annotation_position='top', row='all', col=1)
    
    fig.update_layout(height=800, title_text="<b>Kalman Filter for State Estimation & Fault Detection</b>",
                      legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.01))
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Residual (Surprise)", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    
    return fig, first_ooc

# ==============================================================================
# HELPER & PLOTTING FUNCTION (Method 4) - SME ENHANCED & CORRECTED
# ==============================================================================

@st.cache_data
def plot_rl_tuning(cost_false_alarm=1.0, cost_delay_unit=5.0, shift_size=1.0):
    """
    Generates an enhanced, more realistic RL dashboard for the economic design of an EWMA chart,
    optimizing over both lambda and control limit width (L).
    Returns TWO separate figures: one for the 3D plot, one for the 2D plots.
    """
    # 1. --- Create a grid of parameters to search ---
    lambdas = np.linspace(0.05, 0.5, 20)
    Ls = np.linspace(2.5, 3.5, 20) # Control limit widths
    
    # Pre-calculate Performance (ARL) over the grid
    arl0_grid = np.zeros((len(Ls), len(lambdas)))
    arl1_grid = np.zeros((len(Ls), len(lambdas)))
    
    for i, L in enumerate(Ls):
        for j, lam in enumerate(lambdas):
            delta = shift_size * np.sqrt(lam / (2 - lam))
            L_eff = L
            
            arl0_grid[i, j] = (np.exp(-2*L_eff*delta) * (1 + 2*L_eff*delta) + np.exp(2*L_eff*delta) * (1 - 2*L_eff*delta)) / (2*delta**2) if delta != 0 else np.inf
            if arl0_grid[i,j] == np.inf or arl0_grid[i,j] < 1: arl0_grid[i, j] = 1e9

            arl1_grid[i, j] = (np.exp(-2*(L_eff-delta)*delta) + 2*(L_eff-delta)*delta - 1) / (2*delta**2) if delta != 0 else np.inf
            if arl1_grid[i, j] <= 0 or arl1_grid[i, j] == np.inf: arl1_grid[i, j] = 1

    # 2. --- Calculate Economic Cost and find optimum ---
    total_cost_grid = (cost_false_alarm / arl0_grid) + (cost_delay_unit * arl1_grid)
    min_idx = np.unravel_index(np.argmin(total_cost_grid), total_cost_grid.shape)
    optimal_L = Ls[min_idx[0]]
    optimal_lambda = lambdas[min_idx[1]]
    min_cost = total_cost_grid[min_idx]
    
    # --- 3. Create Figure 1: The 3D Surface Plot ---
    fig_3d = go.Figure()
    fig_3d.add_trace(go.Surface(x=lambdas, y=Ls, z=total_cost_grid, colorscale='Viridis', showscale=False))
    fig_3d.add_trace(go.Scatter3d(x=[optimal_lambda], y=[optimal_L], z=[min_cost], mode='markers',
                                 marker=dict(color='red', size=8, symbol='x'), name="Optimal Point"))
    fig_3d.update_layout(
        title="<b>1. Economic Cost Surface</b>",
        scene=dict(xaxis_title='Lambda (λ)', yaxis_title='Limit Width (L)', zaxis_title='Total Cost'),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # --- 4. Create Figure 2: The 2D Diagnostic Plots ---
    fig_2d = make_subplots(
        rows=2, cols=2,
        subplot_titles=("<b>2. ARL₀ (Time to False Alarm)</b>",
                        "<b>3. ARL₁ (Time to Detect)</b>",
                        "<b>4. Optimal EWMA Chart</b>"),
        vertical_spacing=0.2, horizontal_spacing=0.15
    )
    
    # ARL0 Contour
    fig_2d.add_trace(go.Contour(x=lambdas, y=Ls, z=np.log10(arl0_grid), colorscale='Blues',
                               contours=dict(showlabels=True), name="log10(ARL₀)"), row=1, col=1)
    fig_2d.add_trace(go.Scatter(x=[optimal_lambda], y=[optimal_L], mode='markers',
                               marker=dict(color='red', size=12, symbol='x')), row=1, col=1)
    
    # ARL1 Contour
    fig_2d.add_trace(go.Contour(x=lambdas, y=Ls, z=arl1_grid, colorscale='Reds',
                               contours=dict(showlabels=True), name="ARL₁"), row=1, col=2)
    fig_2d.add_trace(go.Scatter(x=[optimal_lambda], y=[optimal_L], mode='markers',
                               marker=dict(color='black', size=12, symbol='x')), row=1, col=2)

    # Optimal EWMA chart
    np.random.seed(42)
    data = np.random.normal(0, 1, 50)
    data[25:] += shift_size
    ewma_opt = pd.Series(data).ewm(alpha=optimal_lambda, adjust=False).mean().values
    ucl = optimal_L * np.sqrt(optimal_lambda / (2 - optimal_lambda))
    
    fig_2d.add_trace(go.Scatter(y=ewma_opt, mode='lines+markers', name='Optimal EWMA'), row=2, col=1)
    fig_2d.add_hline(y=ucl, line_color='red', row=2, col=1)
    fig_2d.add_hline(y=-ucl, line_color='red', row=2, col=1)
    fig_2d.add_vline(x=25, line_dash='dash', line_color='red', row=2, col=1)
    
    fig_2d.update_layout(height=600, showlegend=False)
    fig_2d.update_xaxes(title_text='Lambda (λ)', row=1, col=1); fig_2d.update_yaxes(title_text='Limit Width (L)', row=1, col=1)
    fig_2d.update_xaxes(title_text='Lambda (λ)', row=1, col=2); fig_2d.update_yaxes(title_text='Limit Width (L)', row=1, col=2)
    fig_2d.update_xaxes(title_text='Observation Number', row=2, col=1); fig_2d.update_yaxes(title_text='EWMA Value', row=2, col=1)
    
    # Clear the unused subplot
    fig_2d.update_xaxes(visible=False, showticklabels=False, row=2, col=2)
    fig_2d.update_yaxes(visible=False, showticklabels=False, row=2, col=2)
    
    return fig_3d, fig_2d, optimal_lambda, optimal_L, min_cost

# ==============================================================================
# HELPER & PLOTTING FUNCTION (Method 5) - SME ENHANCED
# ==============================================================================
@st.cache_data
def plot_tcn_cusum(drift_magnitude=0.05, daily_cycle_strength=1.0):
    """
    Generates an enhanced, more realistic TCN-CUSUM dashboard with bioprocess-like data,
    receptive field visualization, and residual diagnostics.
    """
    np.random.seed(42)
    n_points = 200
    time = np.arange(n_points)
    
    # --- 1. SME Enhancement: Simulate more realistic bioprocess data ---
    # Logistic growth curve (S-shape)
    growth = 100 / (1 + np.exp(-0.05 * (time - 100)))
    # Diurnal (daily) cycles
    daily_cycle = daily_cycle_strength * np.sin(time * 2 * np.pi / 24) # 24-hour cycle
    # Gradual process drift
    drift = np.linspace(0, drift_magnitude * n_points, n_points)
    # Measurement noise
    noise = np.random.normal(0, 0.5, n_points)
    
    data = growth + daily_cycle + drift + noise
    
    # --- 2. Simulate a TCN Forecast ---
    # A real TCN is complex. We simulate its key property: it learns the predictable
    # non-linear components (growth and cycles), but is blind to the unexpected linear drift.
    tcn_forecast = growth + daily_cycle
    
    # --- 3. Calculate Residuals and Apply CUSUM ---
    residuals = data - tcn_forecast
    
    target = np.mean(residuals[:50]) # Target is the mean of the initial stable residuals
    k = 0.5 * np.std(residuals[:50]) # Slack parameter
    h = 5 * np.std(residuals[:50]) # Control limit
    
    sh, sl = np.zeros(n_points), np.zeros(n_points)
    for i in range(1, n_points):
        sh[i] = max(0, sh[i-1] + (residuals[i] - target) - k)
        sl[i] = max(0, sl[i-1] + (target - residuals[i]) - k)
    
    ooc_points = np.where(sh > h)[0]
    first_ooc = ooc_points[0] if len(ooc_points) > 0 else None
    
    # --- 4. Plotting Dashboard ---
    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.7, 0.3],
        row_heights=[0.6, 0.4],
        subplot_titles=("<b>1. TCN Forecast on Bioprocess Data</b>", "<b>3. Residual Diagnostics</b>",
                        "<b>2. CUSUM Chart on Residuals (Drift Detector)</b>"),
        vertical_spacing=0.15
    )
    
    # Plot 1: Main Forecast
    fig.add_trace(go.Scatter(x=time, y=data, mode='lines', name='Actual Data', line=dict(color='black')), row=1, col=1)
    fig.add_trace(go.Scatter(x=time, y=tcn_forecast, mode='lines', name='TCN Forecast', line=dict(dash='dash', color='red')), row=1, col=1)
    # SME Enhancement: Visualize the TCN's receptive field
    receptive_field_size = 32
    fig.add_vrect(x0=150 - receptive_field_size, x1=150, fillcolor="rgba(255,150,0,0.2)", line_width=0,
                  annotation_text="TCN Receptive Field", annotation_position="bottom left", row=1, col=1)

    # Plot 2: CUSUM on Residuals
    fig.add_trace(go.Scatter(x=time, y=sh, mode='lines', name='CUSUM High', line_color='purple'), row=2, col=1)
    fig.add_hline(y=h, line_color='red', row=2, col=1)
    if first_ooc:
        fig.add_trace(go.Scatter(x=[first_ooc], y=[sh[first_ooc]], mode='markers', name='Alarm',
                                 marker=dict(color='red', size=12, symbol='x')), row=2, col=1)

    # Plot 3: Residual Diagnostics
    # Histogram
    fig.add_trace(go.Histogram(x=residuals, name='Residuals Hist', histnorm='probability density', marker_color='grey'), row=1, col=2)
    # Q-Q Plot
    qq_data = stats.probplot(residuals, dist="norm")
    fig.add_trace(go.Scatter(x=qq_data[0][0], y=qq_data[0][1], mode='markers', name='QQ Points'), row=2, col=2)
    fig.add_trace(go.Scatter(x=qq_data[0][0], y=qq_data[1][0] * qq_data[0][0] + qq_data[1][1], mode='lines', name='QQ Fit', line_color='red'), row=2, col=2)
    
    fig.update_layout(height=800, title_text="<b>TCN-CUSUM: Hybrid Model for Complex Drift Detection</b>", showlegend=False)
    fig.update_yaxes(title_text="Biomass Conc.", row=1, col=1)
    fig.update_xaxes(title_text="Time (Hours)", row=2, col=1)
    fig.update_yaxes(title_text="CUSUM Value", row=2, col=1)
    fig.update_xaxes(title_text="Residual Value", row=1, col=2)
    fig.update_yaxes(title_text="Density", row=1, col=2)
    fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=2)
    fig.update_yaxes(title_text="Sample Quantiles", row=2, col=2)

    return fig, first_ooc

# ==============================================================================
# HELPER & PLOTTING FUNCTION (Method 6) - CORRECTED
# ==============================================================================
@st.cache_data
def plot_lstm_autoencoder_monitoring(drift_rate=0.02, spike_magnitude=2.0):
    """
    Generates an enhanced, more realistic LSTM Autoencoder dashboard, simulating
    multivariate data and visualizing the model's reconstruction performance.
    """
    np.random.seed(42)
    n_points = 250
    time = np.arange(n_points)
    
    # --- 1. SME Enhancement: Simulate realistic multivariate bioprocess data ---
    # Normal process has a slight correlation and oscillation
    temp = 70 + 5 * np.sin(time / 20) + np.random.normal(0, 0.5, n_points)
    ph = 7.0 - 0.2 * np.sin(time / 20 + np.pi/2) + np.random.normal(0, 0.05, n_points)
    df = pd.DataFrame({'Temperature': temp, 'pH': ph})
    
    # --- 2. Simulate the Reconstruction and Error ---
    # A real LSTM autoencoder is complex. We simulate its key property: it reconstructs
    # normal data well, but struggles with anomalies.
    df_recon = df.copy()
    recon_error = np.random.chisquare(df=2, size=n_points) * 0.1 # Base reconstruction error

    # Inject a gradual drift anomaly (e.g., sensor drift in both signals)
    drift_start = 100
    drift_duration = n_points - drift_start
    drift = np.linspace(0, drift_rate * drift_duration, drift_duration)
    # Add drift to the real data (the model won't know how to reconstruct this)
    df['Temperature'][drift_start:] += drift
    df['pH'][drift_start:] -= drift * 0.01
    recon_error[drift_start:] += np.linspace(0, 1.5, drift_duration)**2 # Error accelerates

    # Inject a sudden spike anomaly (e.g., process shock)
    spike_point = 200
    df['Temperature'][spike_point] += spike_magnitude * 2
    df['pH'][spike_point] += spike_magnitude * 0.1
    recon_error[spike_point] += spike_magnitude**2

    # --- 3. Apply Hybrid Monitoring to the Reconstruction Error ---
    # EWMA for drift detection
    lambda_ewma = 0.1
    ewma = pd.Series(recon_error).ewm(alpha=lambda_ewma, adjust=False).mean().values
    ewma_mean = np.mean(recon_error[:drift_start])
    ewma_std = np.std(recon_error[:drift_start])
    ewma_ucl = ewma_mean + 3 * ewma_std * np.sqrt(lambda_ewma / (2 - lambda_ewma))
    ewma_ooc_mask = ewma > ewma_ucl
    first_ewma_ooc = np.argmax(ewma_ooc_mask) if np.any(ewma_ooc_mask) else None

    # BOCPD for spike detection
    hazard = 1/500.0
    mean0, var0 = np.mean(recon_error[:drift_start]), np.var(recon_error[:drift_start])
    R = np.zeros((n_points + 1, n_points + 1)); R[0, 0] = 1
    prob_rl_zero = np.zeros(n_points)
    for t in range(1, n_points + 1):
        x = recon_error[t-1]
        pred_probs = stats.norm.pdf(x, loc=mean0, scale=np.sqrt(var0)) # Simplified model
        R[1:t+1, t] = R[0:t, t-1] * pred_probs * (1 - hazard)
        R[0, t] = np.sum(R[:, t-1] * pred_probs * hazard)
        if np.sum(R[:, t]) > 0: R[:, t] /= np.sum(R[:, t])
        prob_rl_zero[t-1] = R[0, t]
    
    # Find spike detection point
    bocpd_ooc_mask = prob_rl_zero > 0.5 # Threshold for spike detection
    first_bocpd_ooc = np.argmax(bocpd_ooc_mask) if np.any(bocpd_ooc_mask) else None
    
    # --- 4. Plotting Dashboard ---
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=("<b>1. Multivariate Process Data & AI Reconstruction</b>",
                                        "<b>2. AI Reconstruction Error (Process Health Score)</b>",
                                        "<b>3. Hybrid Monitoring Charts on Health Score</b>"))

    # Plot 1: Raw Data and Reconstruction
    for col, color in zip(df.columns, ['red', 'blue']):
        fig.add_trace(go.Scatter(x=time, y=df[col], name=f'Actual {col}', mode='lines', line=dict(color=color)), row=1, col=1)
        fig.add_trace(go.Scatter(x=time, y=df_recon[col], name=f'Reconstructed {col}', mode='lines',
                                 line=dict(color=color, dash='dash')), row=1, col=1)
    
    # Plot 2: Reconstruction Error
    fig.add_trace(go.Scatter(x=time, y=recon_error, mode='lines', name='Recon. Error', line_color='black'), row=2, col=1)

    # Plot 3: Hybrid Monitoring
    fig.add_trace(go.Scatter(x=time, y=ewma, mode='lines', name='EWMA (for Drift)', line_color='orange'), row=3, col=1)
    fig.add_hline(y=ewma_ucl, line_color='orange', line_dash='dash', row=3, col=1)
    fig.add_trace(go.Scatter(x=time, y=prob_rl_zero, mode='lines', name='BOCPD (for Spikes)', line_color='purple', yaxis='y2'), row=3, col=1)

    # Add dynamic annotations for alarms
    if first_ewma_ooc:
        fig.add_vline(x=first_ewma_ooc, line=dict(color='orange', width=2),
                      annotation_text="EWMA Drift Alarm", annotation_position='top', row='all', col=1)
    if first_bocpd_ooc:
        fig.add_vline(x=first_bocpd_ooc, line=dict(color='purple', width=2, dash='dot'),
                      annotation_text="BOCPD Spike Alarm", annotation_position='bottom', row='all', col=1)

    fig.update_layout(
        height=800, title_text="<b>LSTM Autoencoder with Hybrid Monitoring System</b>",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis3=dict(overlaying='y', side='right', title='P(Change)', range=[0,1]), # Secondary axis for BOCPD
        yaxis_title="Value",
        yaxis2_title="Recon. Error",
        yaxis3_title="EWMA",
        xaxis3_title="Time (Hours)"
    )
    
    return fig, first_ewma_ooc, first_bocpd_ooc
# =================================================================================================================================================================================================
# ALL UI RENDERING FUNCTIONS
# ==================================================================================================================================================================================================

def render_introduction_content():
    """Renders the complete, all-in-one introduction and framework dashboard."""
    st.title("🛠️ Biotech V&V Analytics Toolkit")
    st.markdown("### An Interactive Guide to Assay Validation, Tech Transfer, and Lifecycle Management")
# --- NEW, CLEAN & STACKED HEADER SECTION ---
    # We use a single markdown block to control the layout precisely.
    st.markdown(
        """
        <p style='color: grey; margin-bottom: 0;'>Developed by<br><b>Jose Bautista, MSc, LSSBB, PMP</b></p>
        
        <div style='color: grey; margin-top: 20px;'>
        <b>Contact Information</b><br>
        📧 jbautistads@gmail.com<br>
        🔗 linkedin.com/in/josebautista
        </div>
        """,
        unsafe_allow_html=True
    )
    st.divider()
    # --- END OF NEW HEADER SECTION ---
    # --- END OF NEW HEADER SECTION ---
    st.markdown("Welcome! This toolkit is a collection of interactive modules designed to explore the statistical and machine learning methods that form the backbone of a robust V&V, technology transfer, and process monitoring plan.")
    st.info("#### 👈 Select a tool from the sidebar to explore an interactive module.")
    
    st.header("📖 The Scientist's/Engineer's Journey: A Four-Act Story")
    st.markdown("""The journey from a novel idea to a robust, routine process can be viewed as a four-act story. A successful project is not just about executing tests; it begins with rigorous planning and strategy. The toolkit is structured to follow that complete narrative.""")
    
    # --- UPDATED TO A 4-COLUMN LAYOUT ---
    act0, act1, act2, act3 = st.columns(4)
    with act0:
        st.subheader("Act 0: Planning & Strategy")
        st.markdown("Before a single experiment is run, a successful project is defined. This is the act of creating the project's 'North Star'—defining the goals, assessing the risks, and creating the master plan for validation.")
    with act1: 
        st.subheader("Act I: Characterization")
        st.markdown("With a plan in place, we build a deep, data-driven understanding of the process. This act is about discovering the fundamental capabilities, limitations, and sensitivities of the new method or process.")
    with act2: 
        st.subheader("Act II: Qualification & Transfer")
        st.markdown("Here, the method or process faces its crucible. It must prove its performance in a new environment—a new lab, a new scale, a new team. This is about demonstrating stability, equivalence, and capability.")
    with act3: 
        st.subheader("Act III: Lifecycle Management")
        st.markdown("Once live, the journey isn't over. This final act is about continuous guardianship: monitoring process health, detecting subtle drifts, and using advanced analytics to predict and prevent future failures.")
    
    st.divider()

    st.header("🚀 The V&V Model: A Strategic Framework")
    st.markdown("The **Verification & Validation (V&V) Model**, shown below, provides a structured, widely accepted framework for ensuring a system meets its intended purpose, from initial requirements to final deployment.")
    fig_v_model = plot_v_model()
    st.plotly_chart(fig_v_model, use_container_width=True)

    st.markdown("The table below provides a side-by-side comparison of typical documents and activities for each stage of the V-Model across different biotech contexts.")
    summary_df = create_v_model_summary_table()
    fig_table = create_styled_v_model_table(summary_df)
    st.plotly_chart(fig_table, use_container_width=True)
    
    st.divider()
    
    st.header("📈 Project Workflow")
    st.markdown("This timeline organizes the entire toolkit by its application in a typical project lifecycle. Tools are grouped by the project phase where they provide the most value.")
    st.plotly_chart(plot_act_grouped_timeline(), use_container_width=True)

    st.header("⏳ A Chronological View of V&V Analytics")
    st.markdown("This timeline organizes the same tools purely by their year of invention, showing the evolution of statistical and machine learning thought over the last century.")
    st.plotly_chart(plot_chronological_timeline(), use_container_width=True)

    st.header("🗺️ Conceptual Map of Tools")
    st.markdown("This map illustrates the relationships between the foundational concepts and the specific tools available in this application. Use it to navigate how different methods connect to broader analytical strategies.")
    st.plotly_chart(create_toolkit_conceptual_map(), use_container_width=True)


# ==============================================================================
# UI RENDERING FUNCTIONS (ALL DEFINED BEFORE MAIN APP LOGIC)
# ==============================================================================

#===================================================================================================== ACT 0 Render=================================================================================================================
#===================================================================================================================================================================================================================================
def render_tpp_cqa_cascade():
    """Renders the comprehensive, interactive module for TPP & CQA Cascade, including a full QbD introduction."""
    st.markdown("""
    #### Purpose & Application: The "Golden Thread" of QbD
    **Purpose:** To be the **"North Star" of the entire project.** This tool visualizes the "golden thread" of Quality by Design (QbD). It starts with the high-level patient or business needs (the **Target Product Profile**), translates them into measurable product requirements (the **Critical Quality Attributes**), and finally links those to the specific process parameters and material attributes that must be controlled.
    
    **Strategic Application:** This cascade is the first and most important document in a modern, science- and risk-based validation program. It provides a clear, traceable line of sight from the needs of the patient all the way down to the specific dial a process engineer needs to turn. This is the essence of **ICH Q8**.
    """)
    
    st.info("""
    **Interactive Demo:** You are the Head of Product Development.
    1.  Select a **Project Type** to see its unique quality cascade.
    2.  Use the **TPP Target Sliders** in the sidebar to define your project's ambition. Notice how increasing a target highlights the specific CQAs (in yellow) that are critical to achieving that goal.
    """)
    
    project_type = st.selectbox(
        "Select a Project Type to visualize its Quality Cascade:", 
        ["Monoclonal Antibody", "IVD Kit", "Pharma Process (Small Molecule)", "Instrument Qualification", "Computer System Validation"]
    )
    
    target1_val, target2_val = None, None
    target1_tag, target2_tag = "", ""

    with st.sidebar:
        st.subheader("TPP Target Controls")
        if project_type == "Monoclonal Antibody":
            target1_val = st.slider("Desired Efficacy Target (%)", 80, 120, 95, 1, help="Why > 100%? Potency is often measured *relative* to a Reference Standard (defined as 100%). A process improvement could yield a more potent drug.")
            target1_tag = "Efficacy"
            target2_val = st.slider("Required Shelf-Life (Months)", 12, 36, 24, 1)
            target2_tag = "Shelf-Life"
        elif project_type == "IVD Kit":
            target1_val = st.slider("Desired Clinical Sensitivity (%)", 90, 100, 98, 1)
            target1_tag = "Efficacy"
            target2_val = st.slider("Required Shelf-Life (Months)", 6, 24, 18, 1)
            target2_tag = "Shelf-Life"
        elif project_type == "Pharma Process (Small Molecule)":
            target1_val = st.slider("Target Process Yield (%)", 75, 95, 85, 1, help="Higher yield is a key business driver.")
            target1_tag = "Yield"
            target2_val = st.slider("Target Purity Level (%)", 99.0, 99.9, 99.5, 0.1, format="%.1f", help="Higher purity is critical for patient safety.")
            target2_tag = "Purity"
        elif project_type == "Instrument Qualification":
            target1_val = st.slider("Target Throughput (Plates/hr)", 10, 100, 50, 5, help="Higher throughput is a key performance requirement.")
            target1_tag = "Throughput"
            target2_val = st.slider("Target Reliability (Uptime %)", 95.0, 99.9, 99.0, 0.1, format="%.1f", help="Higher uptime is critical for operational efficiency.")
            target2_tag = "Reliability"
        elif project_type == "Computer System Validation":
            target1_val = st.slider("Target Performance (Report Time in sec)", 1, 10, 5, 1, help="Faster performance is a key user requirement. Lower is better.")
            target1_tag = "Performance"
            target1_val = 11 - target1_val # Invert so higher on slider is better
            target2_val = st.checkbox("Full 21 CFR Part 11 Compliance Required", value=True)
            target2_tag = "Compliance"

    fig = plot_tpp_cqa_cascade(project_type, target1_val, target2_val, target1_tag, target2_tag)
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    st.subheader("Deeper Dive into Quality by Design (QbD)")
    
    tabs = st.tabs(["💡 Key Insights", "📖 Introduction to QbD", "📋 QbD Glossary", "✅ The Golden Rule", "📖 Theory & History", "🏛️ Regulatory & Compliance"])
    
    with tabs[0]:
        st.markdown("""
        **Reading the Cascade:**
        - **TPP (Left Node):** This is the contract with the patient and the business. It defines *what* the product, process, or system must do.
        - **CQAs (Middle Nodes):** These are the measurable properties the *product* must possess or performance attributes the *system* must have to fulfill the TPP. This is the translation of the "what" into the "how."
        - **CPPs/CMAs (Right Nodes):** These are the specific, controllable "knobs" on the *process* or *design* that have a direct impact on the CQAs. This is where science and engineering meet the quality target.

        **The Strategic Insight: From 'What' to 'Why' to 'How'**
        The Sankey diagram doesn't just show a list of items; it shows a **traceable chain of logic**. As you adjust the TPP sliders, the highlighted CQAs (yellow) show the immediate impact of a strategic business decision on technical requirements. This cascade is the core of a science- and risk-based approach because it forces the team to formally document the scientific evidence that links a specific process parameter to a critical product attribute, which in turn ensures patient safety and efficacy.
        """)
    
    with tabs[1]:
        st.markdown("""
        #### The QbD Paradigm Shift
        Quality by Design (QbD) represents a fundamental shift in the philosophy of manufacturing and validation, championed by global regulators.

        **The Traditional Approach (Quality by Testing):**
        - The process is a "black box." A fixed recipe is followed, and quality is ensured by performing extensive testing on the *final product*.
        - **Analogy:** Baking a cake by rigidly following an old recipe without understanding *why* it works. If the cake comes out badly, the only option is to throw it away (reject the batch) and try again. Quality is "inspected in" at the end.

        **The Modern Approach (Quality by Design):**
        - The process is a "glass box." We use scientific investigation (like DOE) and risk management (like FMEA) to gain a deep, predictive understanding of how raw materials (CMAs) and process parameters (CPPs) impact the final product quality (CQAs).
        - **Analogy:** Being a master baker who understands the chemistry of baking. You know exactly how adjusting the oven temperature (a CPP) will affect the cake's moistness (a CQA). You design a robust recipe and process, and you monitor the critical parameters in real-time. You *know* the cake will be good every time, with minimal final testing required. Quality is "designed in" from the beginning.
        
        The TPP/CQA/CPP cascade is the central tool that documents this deep understanding.
        """)
        
    with tabs[2]:
        st.markdown("""
        ##### The Language of Quality by Design
        - **Target Product Profile (TPP):**
          - **Definition (ICH Q8):** A prospective summary of the quality characteristics of a drug product that ideally will be achieved to ensure the desired quality, taking into account safety and efficacy.
          - **SME Translation:** What does this product need to do for the patient? What is our contract with them?
          - **Example:** "A sterile, injectable liquid that safely and effectively treats rheumatoid arthritis with a 24-month shelf life."

        - **Critical Quality Attribute (CQA):**
          - **Definition (ICH Q8):** A physical, chemical, biological, or microbiological property that should be within an appropriate limit to ensure the desired product quality.
          - **SME Translation:** What measurable property must the *product* have to meet the TPP?
          - **Example:** "Purity must be > 99%" (to ensure safety and efficacy).

        - **Critical Material Attribute (CMA):**
          - **Definition (ICH Q8):** A property of an input material that should be within an appropriate limit to ensure the desired quality of the output product.
          - **SME Translation:** What property of a *raw material* must be controlled?
          - **Example:** "The pH of the cell culture media must be between 6.9 and 7.3."

        - **Critical Process Parameter (CPP):**
          - **Definition (ICH Q8):** A process parameter whose variability has an impact on a CQA and therefore should be monitored or controlled.
          - **SME Translation:** What "knob" on the *process* must be controlled?
          - **Example:** "The bioreactor temperature must be maintained at 37.0 ± 0.5 °C."

        - **Design Space:**
          - **Definition (ICH Q8):** The multidimensional combination of input variables (e.g., CMAs) and process parameters (e.g., CPPs) that have been demonstrated to provide assurance of quality.
          - **SME Translation:** The "safe operating zone" or the proven "recipe for success."

        - **Control Strategy:**
          - **Definition (ICH Q10):** A planned set of controls, derived from product and process understanding, that assures process performance and product quality.
          - **SME Translation:** The complete police force for your process: the combination of raw material specifications (for CMAs), in-process controls (for CPPs), and final product tests (for CQAs) that guarantee quality.
        """)
        
    with tabs[3]:
        st.error("""🔴 **THE INCORRECT APPROACH: The "Test, Fail, and Repeat" Cycle**
A team develops a process based on a few successful runs. When a batch fails during validation, a lengthy investigation begins with no clear starting point. The process is a 'black box,' and troubleshooting is reactive guesswork.
- **The Flaw:** There is no documented, scientific understanding of the process. The team doesn't know *which* parameters are critical, so they can't effectively control them or troubleshoot them.""")
        st.success("""🟢 **THE GOLDEN RULE: Begin with the End in Mind, and Document the Links**
A compliant and robust QbD approach is a disciplined, multi-stage process.
1.  **Define the Goal (TPP):** First, formally document the "contract" with the patient.
2.  **Identify What Matters (CQAs):** Translate the TPP into a set of measurable, scientific targets for the product.
3.  **Link the Chain of Knowledge (FMEA & DOE):** Use risk assessment and designed experiments to discover and prove the links between the process/materials (CPPs/CMAs) and the product quality (CQAs).
4.  **Establish the Control Strategy:** Implement controls on the critical parameters and attributes you identified to ensure that every batch meets the CQAs and, by extension, the TPP. This cascade is the documented proof of this entire logical chain.""")

    with tabs[4]:
        st.markdown("""
        #### Historical Context: From Juran's Trilogy to ICH
        **The Problem:** For much of the 20th century, the pharmaceutical industry operated on a "quality by testing" paradigm. Processes were developed, locked down, and then extensively tested at the end to prove they worked. This was inefficient, costly, and led to a poor understanding of *why* processes sometimes failed, making continuous improvement difficult.
        
        **The 'Aha!' Moment:** The core ideas of QbD were articulated by the legendary quality pioneer **Joseph M. Juran** in his "Quality Trilogy" (Quality Planning, Quality Control, Quality Improvement). He argued forcefully that quality must be *planned* and *designed* into a product from the very beginning, not inspected in at the end. While these ideas were adopted by other high-tech industries like automotive and electronics, the pharmaceutical industry was slower to change due to its rigid regulatory structure.
            
        **The Impact (The ICH Revolution):** In the early 2000s, the FDA and other global regulators recognized that the traditional approach was stifling innovation and hindering process improvement. They launched the **"Pharmaceutical cGMPs for the 21st Century"** initiative to encourage a modern, science- and risk-based approach. This culminated in the landmark **ICH Q8(R2) Guideline on Pharmaceutical Development** in 2009. This guideline formally adopted Juran's philosophy, introducing the concepts of the **Target Product Profile (TPP)** and **Critical Quality Attributes (CQA)** to the industry. It marked a major philosophical shift away from a prescriptive, "cookbook" approach to a flexible, understanding-based framework, with the TPP/CQA cascade as its central pillar.
        """)
        
    with tabs[5]:
        st.markdown("""
        This entire framework is defined and championed by the International Council for Harmonisation (ICH) and adopted by global regulators like the FDA.
        - **ICH Q8(R2) - Pharmaceutical Development:** This is the primary guideline. It explicitly defines the **Target Product Profile (TPP)**, **Critical Quality Attributes (CQA)**, and **Critical Process Parameters (CPP)** as the foundational elements of QbD. It introduces the concepts of the **Design Space** and **Control Strategy**.
        - **ICH Q9 - Quality Risk Management:** The process of identifying which attributes and parameters are "critical" (i.e., identifying CQAs and CPPs) is a formal risk assessment activity that must be documented according to the principles of ICH Q9. The FMEA tool in this app is a direct implementation of this.
        - **FDA Guidance on Process Validation (2011):** The entire lifecycle approach is built on this foundation. **Stage 1 (Process Design)** is the activity of translating the CQAs into a robust manufacturing process by identifying and controlling the CPPs and CMAs.
        - **GAMP 5:** For instruments and software, the TPP is analogous to the **User Requirement Specification (URS)**, and the CQAs are analogous to the high-level **Functional Specifications (FS)**. This cascade provides the direct link between user needs and system design.
        """)
        
#=====================================================================2. ANALYTICAL TARGET PROFILE (ATP) BUILDER ====================================================
def render_atp_builder():
    """Renders the comprehensive, interactive module for building a Target Profile."""
    st.markdown("""
    #### Purpose & Application: The Project's "Contract"
    **Purpose:** To serve as the **"Design Specification" or "Contract" for a new product, process, or system.** Before significant work begins, the Target Profile formally documents the performance characteristics that *must* be achieved for the project to be considered a success.
    
    **Strategic Application:** This is the formal translation of high-level user needs (from the TPP or URS) into a concrete set of statistical and performance-based acceptance criteria. It is the objective scorecard against which all development and validation activities are measured, preventing "goalpost moving" and ensuring the final result is truly fit-for-purpose.
    """)
    
    st.info("""
    **Interactive Demo:** You are the Validation or Project Lead.
    1.  Select the **Project Type** you are planning.
    2.  Use the **Performance Requirement Sliders** in the sidebar to define the "contract" for this project.
    3.  Toggle on the "Simulate Results" to see how a hypothetical final result (green) performs against your targets.
    """)

    col1, col2 = st.columns([0.4, 0.6])
    
    with col1:
        st.subheader("Target Profile Requirements")
        project_type = st.selectbox("Select Project Type", [
            "Pharma Assay (HPLC)", "IVD Kit (ELISA)", "Instrument Qualification", "Software System (LIMS)", "Pharma Process (MAb)"
        ])
        
        atp_values = []
        achieved_values = None
        show_results = False
        
        with st.sidebar:
            st.subheader(f"Controls for {project_type}")
            if project_type == "Pharma Assay (HPLC)":
                atp_values.append(st.slider("Accuracy (%Rec)", 95.0, 102.0, 98.0, 0.5, help="Target: 100%. How close must the average measurement be to the true value?"))
                atp_values.append(st.slider("Precision (%CV)", 0.5, 5.0, 2.0, 0.1, help="How much random variability is acceptable? Lower is better."))
                atp_values.append(st.slider("Linearity (R²)", 0.9900, 1.0000, 0.9990, 0.0001, format="%.4f", help="How well does signal correlate with concentration? Higher is better."))
                atp_values.append(st.slider("Range (Turn-down)", 10, 100, 50, 5, help="Ratio of highest to lowest quantifiable point."))
                atp_values.append(st.slider("Sensitivity (LOD)", 1, 50, 20, 1, help="Qualitative score for required Limit of Detection. Higher score = more sensitive."))
                show_results = st.toggle("Simulate Validation Results", value=True, key="hplc_results")
                if show_results: achieved_values = [99.5, 1.5, 0.9995, 80, 30]
            
            elif project_type == "IVD Kit (ELISA)":
                atp_values.append(st.slider("Clinical Sensitivity (%)", 90.0, 100.0, 98.0, 0.5, help="Ability to correctly identify true positives."))
                atp_values.append(st.slider("Clinical Specificity (%)", 90.0, 100.0, 99.0, 0.5, help="Ability to correctly identify true negatives."))
                atp_values.append(st.slider("Precision (%CV)", 10.0, 20.0, 15.0, 1.0, help="Assay repeatability. Lower is better for consistent results."))
                atp_values.append(st.slider("Robustness Score", 1, 10, 7, 1, help="Qualitative score for performance across different lots, users, and sites."))
                atp_values.append(st.slider("Shelf-Life (Months)", 6, 24, 18, 1, help="Required stability of the kit at recommended storage."))
                show_results = st.toggle("Simulate Validation Results", value=True, key="ivd_results")
                if show_results: achieved_values = [99.0, 99.5, 12.0, 9, 24]

            elif project_type == "Instrument Qualification":
                atp_values.append(st.slider("Accuracy (Max Bias %)", 0.1, 5.0, 1.0, 0.1, help="Maximum acceptable systematic error. Lower is better."))
                atp_values.append(st.slider("Precision (Max %CV)", 0.5, 5.0, 1.5, 0.1, help="Maximum acceptable random error. Lower is better."))
                atp_values.append(st.slider("Throughput (Samples/hr)", 10, 200, 100, 10, help="Required sample processing speed to meet business needs."))
                atp_values.append(st.slider("Uptime (%)", 95.0, 99.9, 99.0, 0.1, format="%.1f", help="Required operational reliability."))
                atp_values.append(st.slider("Footprint (m²)", 1.0, 5.0, 2.0, 0.5, help="Maximum allowable lab space. A key logistical constraint. Lower is better."))
                show_results = st.toggle("Simulate Qualification Results", value=True, key="inst_results")
                if show_results: achieved_values = [0.8, 1.2, 120, 99.5, 1.8]

            elif project_type == "Software System (LIMS)":
                atp_values.append(st.slider("Reliability (Uptime %)", 99.0, 99.999, 99.9, 0.001, format="%.3f", help="Percentage of time the system must be available."))
                atp_values.append(st.slider("Performance (Query Time sec)", 0.5, 10.0, 2.0, 0.5, help="Maximum time for a key database query to return. Critical for user experience. Lower is better."))
                atp_values.append(st.slider("Security (Compliance Score)", 1, 10, 8, 1, help="Qualitative score for meeting all 21 CFR Part 11 requirements (e.g., audit trails, e-signatures)."))
                atp_values.append(st.slider("Usability (User Sat. Score)", 1, 10, 7, 1, help="Score from User Acceptance Testing (UAT), indicating how intuitive the system is."))
                atp_values.append(st.slider("Scalability (Concurrent Users)", 50, 5000, 500, 50, help="Maximum number of users the system must support simultaneously without performance degradation."))
                show_results = st.toggle("Simulate Validation Results", value=True, key="soft_results")
                if show_results: achieved_values = [99.99, 1.5, 10, 8, 1000]

            elif project_type == "Pharma Process (MAb)":
                atp_values.append(st.slider("Yield (g/L)", 1.0, 10.0, 5.0, 0.5, help="Grams of product per liter of bioreactor volume. A key economic driver."))
                atp_values.append(st.slider("Purity (%)", 98.0, 99.9, 99.5, 0.1, help="Final product purity via SEC-HPLC."))
                atp_values.append(st.slider("Consistency (Inter-batch Cpk)", 1, 10, 8, 1, help="Qualitative score for process predictability and low variability. Higher score means a more consistent process."))
                atp_values.append(st.slider("Robustness (PAR Size Score)", 1, 10, 6, 1, help="Qualitative score for the size of the proven acceptable range."))
                atp_values.append(st.slider("Cycle Time (Days)", 10, 20, 14, 1, help="Time from start to finish for a single batch. A key operational efficiency metric. Lower is better."))
                show_results = st.toggle("Simulate PPQ Results", value=True, key="proc_results")
                if show_results: achieved_values = [6.5, 99.7, 9, 8, 13]

    with col2:
        st.subheader("Target Profile Visualization")
        fig = plot_atp_radar_chart(project_type, atp_values, achieved_values)
        st.plotly_chart(fig, use_container_width=True)
        if show_results:
            st.success("The validation results (green) meet or exceed the target profile (blue) on all criteria.")

    st.divider()
    st.subheader("Deeper Dive")
    tabs = st.tabs(["💡 Key Insights", "📋 Glossary", "✅ The Golden Rule", "📖 Theory & History", "🏛️ Regulatory & Compliance"])
    with tabs[0]:
        st.markdown("""
        **Interpreting the Radar Chart:**
        - The chart provides an immediate, holistic view of the project's required performance characteristics. The **blue polygon is your 'contract' (the Target Profile)**. The **green polygon is the 'deliverable' (the final validated performance)**.
        - **Success Criteria:** A successful validation project is one where the green "Achieved" polygon **fully encompasses** the blue "Target" polygon.
        - **Identifying Trade-offs:** This visualization makes strategic trade-offs clear. For an instrument, you might see that achieving maximum **Throughput** could require accepting a slightly larger lab **Footprint**. The ATP forces this to be a conscious, documented decision.
        - **Avoiding "Gold-Plating":** This visualization helps teams determine if their requirements are reasonable. If the blue polygon is vastly larger than what is necessary for the intended use, it signals that the project may be "gold-plating" the requirements, leading to excessive development time and cost.
        """)
    with tabs[1]:
        st.markdown("""
        ##### Glossary of Performance Attributes
        - **Accuracy (%Rec / Bias):** The closeness of the measured average to the true or accepted reference value. Measures **systematic error**.
        - **Precision (%CV):** The closeness of agreement among a series of measurements from the same sample. Measures **random error**.
        - **Linearity (R²):** The ability to elicit test results that are directly proportional to the concentration of the analyte.
        - **Range (Turn-down):** The interval between the upper and lower concentration of analyte for which the procedure has been demonstrated to have a suitable level of precision, accuracy, and linearity.
        - **Sensitivity (LOD):** The lowest amount of analyte in a sample which can be detected but not necessarily quantitated as an exact value.
        - **Clinical Sensitivity/Specificity:** Metrics for diagnostic tests measuring the rates of true positives and true negatives, respectively.
        - **Robustness:** A measure of a procedure's capacity to remain unaffected by small, but deliberate variations in method parameters.
        - **Throughput:** An operational metric, often measured in samples, plates, or units per unit of time.
        - **Reliability (Uptime):** The percentage of time a system is available for normal operation.
        - **Performance (Query Time):** A software metric for the time it takes to complete a specific computational task.
        - **Security:** For software, the ability to meet regulatory requirements for data integrity, such as those in 21 CFR Part 11.
        """)
        
    with tabs[2]:
        st.error("""🔴 **THE INCORRECT APPROACH: "We'll Know It When We See It"**
A development team starts working on a new assay with a vague goal like "make a good potency assay." They spend months optimizing, and then present the final data to the QC and Regulatory teams, who then inform them that the precision or range is not sufficient for its intended use in a routine environment.
- **The Flaw:** The project lacked a pre-defined definition of success. This leads to wasted work, internal friction, and significant project delays when the method is transferred.""")
        st.success("""🟢 **THE GOLDEN RULE: The Target Profile is a Negotiated Contract**
The ATP is not just a scientific document; it's a formal **Service Level Agreement (SLA)** between all key stakeholders (Analytical Development, QC, Manufacturing, Regulatory, Quality) that is established *before* significant development work begins.
1.  **Define "Fit for Purpose" First:** All parties must negotiate and agree on the specific, numerical criteria in the Target Profile.
2.  **Develop to the Target:** The development team uses the profile as their explicit set of engineering goals.
3.  **Validate Against the Target:** The final validation protocol uses the profile's criteria as the formal, pre-approved acceptance criteria.
This ensures alignment from start to finish and guarantees the final deliverable is fit for its intended purpose.""")
        
    with tabs[3]:
        st.markdown("""
        #### Historical Context: From Checklist to Lifecycle
        **The Problem:** For decades, analytical method validation was treated as a one-time, checklist activity performed at the end of development. This often resulted in methods that were technically valid but not practically robust or well-suited for the harsh realities of routine use in a 24/7 QC environment. The common "over-the-wall" transfer from an R&D lab to a QC lab was a frequent source of project delays and failures.
        
        **The 'Aha!' Moment:** The **Quality by Design (QbD)** movement, championed by thought leaders like Janet Woodcock at the FDA in the early 2000s, proposed a new paradigm. This was formalized in the **"Pharmaceutical cGMPs for the 21st Century"** initiative. The core idea was to apply a proactive, lifecycle approach to all aspects of manufacturing, including the analytical methods themselves.
            
        **The Impact:** The **Analytical Target Profile (ATP)** emerged from this philosophy as a best practice. It was championed by the FDA and scientific bodies like the AAPS in the 2010s. The ATP is the direct application of QbD to method development. It parallels the **Target Product Profile (TPP)**, but instead of defining the goals for a drug product, it defines the goals for the *measurement system* used to test that product. This represents a mature, proactive, and lifecycle-based approach to ensuring analytical method quality.
        """)
        
    with tabs[4]:
        st.markdown("""
        The Target Profile is a modern best-practice that directly supports several key regulatory guidelines by providing a clear, a priori definition of what will be validated.
        - **ICH Q8(R2), Q14:** The ATP is the starting point for applying QbD principles to analytical methods.
        - **FDA Process Validation Guidance:** The TPP for a process defines the goals for **Stage 1 (Process Design)**.
        - **GAMP 5:** For instruments and software, the Target Profile is a direct translation of the **User Requirement Specification (URS)** into a set of verifiable performance criteria for OQ and PQ.
        - **USP Chapter <1220> - The Analytical Procedure Lifecycle:** This new chapter champions a holistic, lifecycle approach to method management. The ATP is the foundational element of **Stage 1 (Procedure Design)**, where the requirements for the method are formally defined.
        """)
#============================================================================== 3. QUALITY RISK MANAGEMENT (FMEA) ========================================================
def render_qrm_suite():
    """Renders the comprehensive, interactive module for the Quality Risk Management (QRM) Suite."""
    st.markdown("""
    This workbench provides a suite of formal, structured tools for **Quality Risk Management (QRM)**. These tools are used to systematically identify, analyze, evaluate, and control potential risks. The output of a risk assessment is the direct, auditable justification for the entire validation strategy, focusing resources where they matter most.
    """)
    
    st.info("""
    **How to Use:**
    1.  Select a **Project Type** and a **Risk Management Tool**.
    2.  The dashboard will load a realistic, contextual example.
    3.  Review the main plot and interact with the **"Workbench Gadgets"** below. For FMEA, use the sliders in the sidebar to see the charts update in real-time. Use the `(?)` for details on each control.
    """)

    col1, col2 = st.columns(2)
    with col1:
        project_type = st.selectbox(
            "Select a Scenario / Project Type:", 
            ["Pharma Process (MAb)", "IVD Assay (ELISA)", "Instrument Qualification (HPLC)", "Software System (LIMS)"]
        )
    with col2:
        tool_choice = st.selectbox(
            "Select a Quality Risk Management Tool:", 
            ["Preliminary Hazard Analysis (PHA)", "Failure Mode and Effects Analysis (FMEA)", "Fault Tree Analysis (FTA)", "Event Tree Analysis (ETA)"]
        )
    
    st.header(f"Dashboard: {tool_choice} for {project_type}")
    st.divider()

    # --- Master Data Dictionary for ALL Scenarios ---
    # This remains unchanged, but the FMEA data now assumes a 1-10 scale.
    qrm_templates = {
        "PHA": {
            "Pharma Process (MAb)": pd.DataFrame([
                {'Hazard': 'Mycoplasma Contamination in Bioreactor', 'Severity': 5, 'Likelihood': 1}, 
                {'Hazard': 'Incorrect Chromatography Buffer Leads to Impurity', 'Severity': 4, 'Likelihood': 3},
                {'Hazard': 'Filter Integrity Failure During Sterile Filtration', 'Severity': 5, 'Likelihood': 2}
            ]),
            "IVD Assay (ELISA)": pd.DataFrame([
                {'Hazard': 'Reagent Cross-Contamination gives False Positive', 'Severity': 4, 'Likelihood': 3}, 
                {'Hazard': 'Incorrect Sample Dilution gives False Negative', 'Severity': 5, 'Likelihood': 2},
                {'Hazard': 'Instrument Read Error leads to Misdiagnosis', 'Severity': 4, 'Likelihood': 4}
            ]),
            "Instrument Qualification (HPLC)": pd.DataFrame([
                {'Hazard': 'Chemical Spill from Solvent Leak', 'Severity': 3, 'Likelihood': 2}, 
                {'Hazard': 'Loss of Data due to Power Failure (No UPS)', 'Severity': 4, 'Likelihood': 1},
                {'Hazard': 'Incorrect Peak Integration leads to OOS', 'Severity': 5, 'Likelihood': 3}
            ]),
            "Software System (LIMS)": pd.DataFrame([
                {'Hazard': 'Unauthorized Access to Patient Data', 'Severity': 5, 'Likelihood': 2}, 
                {'Hazard': 'Data Corruption During Archival', 'Severity': 4, 'Likelihood': 3},
                {'Hazard': 'System Crash During Sample Login', 'Severity': 3, 'Likelihood': 4}
            ]),
        },
        "FMEA": {
            "Pharma Process (MAb)": pd.DataFrame({
                'Failure Mode': ['Contamination in Bioreactor', 'Incorrect Chromatography Buffer', 'Filter Integrity Failure'],
                'S': [10, 8, 10], 'O_Initial': [3, 5, 2], 'D_Initial': [4, 2, 3]
            }),
            "IVD Assay (ELISA)": pd.DataFrame({
                'Failure Mode': ['Degraded Capture Antibody', 'Operator Pipetting Error', 'Incorrect Incubation Time'],
                'S': [9, 7, 8], 'O_Initial': [4, 6, 5], 'D_Initial': [5, 3, 2]
            }),
            "Instrument Qualification (HPLC)": pd.DataFrame({
                'Failure Mode': ['Pump Seal Failure (Leak)', 'Detector Lamp Degradation', 'Autosampler Needle Clog'],
                'S': [6, 8, 7], 'O_Initial': [3, 4, 7], 'D_Initial': [4, 2, 3]
            }),
            "Software System (LIMS)": pd.DataFrame({
                'Failure Mode': ['Data Corruption on Database Write', 'Incorrect Calculation Logic', 'Server Downtime (No Failover)'],
                'S': [10, 10, 7], 'O_Initial': [2, 3, 4], 'D_Initial': [6, 8, 3]
            })
        },
        "FTA": {
            "Pharma Process (MAb)": {'title': 'Batch Contamination', 'nodes': {'Top':{'label':'Batch Contaminated','type':'top','shape':'square','pos':(.5,.9)}, 'OR1':{'label':'OR','type':'gate','shape':'circle','pos':(.5,.7)}, 'Filt':{'label':'Filter Failure','type':'basic','shape':'square','pos':(.25,.5),'prob':0.001}, 'Op':{'label':'Operator Error','type':'basic','shape':'square','pos':(.75,.5),'prob':0.005}}, 'links': [('OR1','Top'),('Filt','OR1'),('Op','OR1')]},
            "IVD Assay (ELISA)": {'title': 'False Negative Result', 'nodes': {'Top':{'label':'False Negative','type':'top','shape':'square','pos':(.5,.9)}, 'OR1':{'label':'OR','type':'gate','shape':'circle','pos':(.5,.7)}, 'Reag':{'label':'Inactive Reagent','type':'basic','shape':'square','pos':(.25,.5),'prob':0.02}, 'Samp':{'label':'Sample Degraded','type':'basic','shape':'square','pos':(.75,.5),'prob':0.01}}, 'links': [('OR1','Top'),('Reag','OR1'),('Samp','OR1')]},
            "Instrument Qualification (HPLC)": {'title': 'System Fails Suitability', 'nodes': {'Top':{'label':'System Fails Suitability','type':'top','shape':'square','pos':(.5,.9)}, 'AND1':{'label':'AND','type':'gate','shape':'circle','pos':(.5,.7)}, 'Peak':{'label':'Poor Peak Shape','type':'basic','shape':'square','pos':(.25,.5),'prob':0.1}, 'Press':{'label':'Unstable Pressure','type':'basic','shape':'square','pos':(.75,.5),'prob':0.05}}, 'links': [('AND1','Top'),('Peak','AND1'),('Press','AND1')]},
            "Software System (LIMS)": {'title': 'Data Integrity Loss', 'nodes': {'Top':{'label':'Data Integrity Loss','type':'top','shape':'square','pos':(.5,.9)}, 'OR1':{'label':'OR','type':'gate','shape':'circle','pos':(.5,.7)}, 'Bug':{'label':'Software Bug','type':'basic','shape':'square','pos':(.25,.5),'prob':0.001}, 'Access':{'label':'Unauthorized Edit','type':'basic','shape':'square','pos':(.75,.5),'prob':0.0005}}, 'links': [('OR1','Top'),('Bug','OR1'),('Access','OR1')]},
        },
        "ETA": {
            "Pharma Process (MAb)": {'title': 'Power Outage during Filling', 'nodes': {'IE':{'label':'Power Outage','pos':(.05,.5)}, 'UPS':{'label':'UPS Backup','pos':(.3,.5),'prob_success':0.99}, 'Gen':{'label':'Generator Starts','pos':(.6,.7),'prob_success':0.95}}, 'paths': [{'x':[.05,.3,.6,.9],'y':[.5,.7,.8,.8],'color':'green'},{'x':[.6,.9],'y':[.7,.6,.6],'color':'orange', 'dash':'dot'},{'x':[.3,.9],'y':[.5,.3,.3],'color':'red'}], 'outcomes': {'Safe Recovery':{'pos':(1,.8),'prob':0.9405,'color':'green'}, 'Partial Loss':{'pos':(1,.6),'prob':0.0495,'color':'orange'}, 'Batch Loss':{'pos':(1,.3),'prob':0.01,'color':'red'}}},
            "IVD Assay (ELISA)": {'title': 'Operator Error (Wrong Reagent)', 'nodes': {'IE':{'label':'Wrong Reagent','pos':(.05,.5)}, 'BC':{'label':'Barcode Check','pos':(.3,.5),'prob_success':0.999}, 'QC':{'label':'QC Check Fails','pos':(.6,.7),'prob_success':0.98}}, 'paths': [{'x':[.05,.3,.9],'y':[.5,.3,.3],'color':'red'},{'x':[.3,.6,.9],'y':[.5,.7,.8,.8],'color':'green', 'dash':'dot'},{'x':[.6,.9],'y':[.7,.6,.6],'color':'orange'}], 'outcomes': {'Run Aborted (Pre)':{'pos':(1,.8),'prob':0.97902,'color':'green'}, 'Run Aborted (Post)':{'pos':(1,.6),'prob':0.02,'color':'orange'}, 'Erroneous Result':{'pos':(1,.3),'prob':0.00098,'color':'red'}}},
            "Instrument Qualification (HPLC)": {'title': 'Column Pressure Spike', 'nodes': {'IE':{'label':'Pressure Spike','pos':(.05,.5)}, 'Limit':{'label':'Pressure Limit','pos':(.3,.5),'prob_success':0.9}, 'Shut':{'label':'Auto-Shutdown','pos':(.6,.7),'prob_success':0.99}}, 'paths': [{'x':[.05,.3,.6,.9],'y':[.5,.7,.8,.8],'color':'green'},{'x':[.6,.9],'y':[.7,.6,.6],'color':'orange', 'dash':'dot'},{'x':[.3,.9],'y':[.5,.3,.3],'color':'red'}], 'outcomes': {'Safe Shutdown':{'pos':(1,.8),'prob':0.891,'color':'green'}, 'Minor Damage':{'pos':(1,.6),'prob':0.009,'color':'orange'}, 'System Damage':{'pos':(1,.3),'prob':0.1,'color':'red'}}},
            "Software System (LIMS)": {'title': 'Network Loss to DB', 'nodes': {'IE':{'label':'Network Loss','pos':(.05,.5)}, 'Retry':{'label':'Retry Logic','pos':(.3,.5),'prob_success':0.8}, 'Cache':{'label':'Local Cache','pos':(.6,.7),'prob_success':0.95}}, 'paths': [{'x':[.05,.3,.6,.9],'y':[.5,.7,.8,.8],'color':'green'},{'x':[.6,.9],'y':[.7,.6,.6],'color':'orange', 'dash':'dot'},{'x':[.3,.9],'y':[.5,.3,.3],'color':'red'}], 'outcomes': {'Seamless Recovery':{'pos':(1,.8),'prob':0.76,'color':'green'}, 'Delayed Write':{'pos':(1,.6),'prob':0.04,'color':'orange'}, 'Data Lost':{'pos':(1,.3),'prob':0.2,'color':'red'}}},
        }
    }

    # --- Tool-Specific UI Rendering ---
    if tool_choice == "Preliminary Hazard Analysis (PHA)":
        pha_data = qrm_templates["PHA"][project_type]
        fig = plot_pha_matrix(pha_data, project_type)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(pha_data, use_container_width=True)

        st.subheader("FHA Workbench Gadgets")
        # Gadgets remain the same...
        c1, c2 = st.columns(2)
        with c1:
            c1.text_input("System/Process Selector", value=project_type, help="Select the overall system, subsystem, or process being analyzed (e.g., 'Blood Analyzer XC-500', 'User Authentication Module', 'Aseptic Filling Process').")
            c1.text_input("Function Input", value="Provide accurate results", help="Define a primary operation the system is intended to perform in verb-noun format (e.g., 'Dispense Reagent', 'Encrypt User Data', 'Mix Ingredients').")
            c1.selectbox("Failure Mode", ["Loss of function", "Incorrect function", "Unintended function", "Delayed function"], help="Select the category describing how the function could fail. This provides a structured way to brainstorm potential failures.")
        with c2:
            c2.text_area("Hazardous Situation Description", value="Instrument displays wrong result to user.", height=100, help="Describe the specific scenario that could expose a person or the environment to harm (e.g., 'Operator is exposed to uncontained bio-specimen', 'Software displays wrong patient data').")
            c2.text_area("Potential Harm", value="Misdiagnosis leading to improper treatment.", height=100, help="Describe the ultimate damage or injury that could result from the hazardous situation (e.g., 'Infection with bloodborne pathogen', 'Misdiagnosis leading to improper treatment', 'Data loss').")
        c1, c2, c3 = st.columns([2,1,1])
        c1.slider("Severity Scale", 1, 5, 4, format="%d-Catastrophic", help="Rate the worst-case potential harm that could credibly occur. This rating is independent of probability.")
        c2.button("Create FTA from Hazard", help="Escalate this high-level hazard into a more detailed investigation. 'Create FTA' starts a top-down analysis of the causes.")
        c3.button("Create FMEA from Hazard", help="Escalate this high-level hazard into a more detailed investigation. 'Create FMEA' starts a bottom-up analysis of the components involved.")
        
    elif tool_choice == "Failure Mode and Effects Analysis (FMEA)":
        # --- FIX: Added full interactivity for FMEA ---
        fmea_data = qrm_templates["FMEA"][project_type].copy() # Use copy to allow modification

        # Create interactive sliders in the sidebar
        with st.sidebar:
            st.subheader("FMEA Interactive Scoring")
            st.markdown("Adjust the initial risk scores to see the dashboard update in real-time.")
            for i, mode in enumerate(fmea_data['Failure Mode']):
                st.markdown(f"--- \n **Mode {i+1}:** *{mode}*")
                # Update the DataFrame directly with the slider values
                fmea_data.loc[i, 'O_Initial'] = st.slider(f"Occurrence (O)", 1, 10, int(fmea_data.loc[i, 'O_Initial']), key=f"o_{i}")
                fmea_data.loc[i, 'D_Initial'] = st.slider(f"Detection (D)", 1, 10, int(fmea_data.loc[i, 'D_Initial']), key=f"d_{i}")

        # Simulate mitigation actions based on the *interactive* initial RPN
        fmea_data['O_Final'] = fmea_data['O_Initial']
        fmea_data['D_Final'] = fmea_data['D_Initial']
        initial_rpn = fmea_data['S'] * fmea_data['O_Initial'] * fmea_data['D_Initial']
        
        # Apply mitigation if RPN is over the action threshold
        action_threshold = 100
        for i, rpn in enumerate(initial_rpn):
            if rpn >= action_threshold:
                # Simulate a risk reduction action
                fmea_data.loc[i, 'O_Final'] = max(1, fmea_data.loc[i, 'O_Initial'] - 3)
                fmea_data.loc[i, 'D_Final'] = max(1, fmea_data.loc[i, 'D_Initial'] - 2)

        # Plot the dashboard with the now-modified DataFrame
        fig_matrix, fig_pareto = plot_fmea_dashboard(fmea_data, project_type)
        
        st.plotly_chart(fig_matrix, use_container_width=True)
        st.plotly_chart(fig_pareto, use_container_width=True)

        st.subheader("FMEA Worksheet Gadgets (Example for one Failure Mode)")
        # Gadgets are updated to a 1-10 scale
        c1, c2 = st.columns(2)
        c1.text_input("Item / Function", value="Chromatography Column", help="The specific component, process step, or software function being analyzed.")
        c2.text_input("Potential Failure Mode", value="Column Overloading", help="How could this item fail to perform its intended function? (e.g., 'Cracked', 'Leaking', 'Returns null value', 'Incorrectly calculated', 'Degraded').")
        st.text_area("Potential Effect(s) of Failure", value="Poor separation of product and impurities, leading to out-of-specification (OOS) drug substance.", help="What is the consequence if this failure occurs? Describe the impact on the system, user, or patient.")
        
        c1, c2, c3 = st.columns(3)
        s_val = c1.slider("Severity (S)", 1, 10, 8, help="How severe is the *effect* of the failure? (1 = No effect, 10 = Catastrophic failure with potential for death or serious injury).")
        o_val = c2.slider("Occurrence (O)", 1, 10, 5, help="How likely is the *cause* to occur? (1 = Extremely unlikely, 10 = Inevitable or very high frequency).")
        d_val = c3.slider("Detection (D)", 1, 10, 2, help="How well can the *current controls* detect the failure mode or its cause before it reaches the customer? (1 = Certain to be detected, 10 = Cannot be detected).")
        
        c1, c2 = st.columns(2)
        c1.text_area("Potential Cause(s)", value="Incorrect protein concentration calculation prior to loading.", help="What is the root cause or mechanism that triggers the failure mode? (e.g., 'Material fatigue', 'Incorrect algorithm', 'Operator error', 'Power surge').")
        c2.text_area("Current Controls", value="SOP for protein quantification; manual calculation verification.", help="What existing design features, tests, or procedures are in place to prevent the cause or detect the failure mode?")
        
        rpn = s_val * o_val * d_val
        st.metric("Risk Priority Number (RPN)", f"{rpn} (S x O x D)", help="A calculated value (Severity × Occurrence × Detection) used to rank and prioritize risks. High RPNs should be addressed first.")
        st.text_area("Recommended Actions", value="Implement automated loading calculation in MES. Add post-column impurity sensor.", help="What specific actions will be taken to reduce the Severity, Occurrence, or improve the Detection of this risk? Assign a responsible person and a due date.")
        # --- END OF FMEA FIX ---

    elif tool_choice == "Fault Tree Analysis (FTA)":
        fta_data = qrm_templates["FTA"][project_type]
        # Logic to calculate probability remains the same
        if 'AND1' in fta_data['nodes'] and 'AND' in fta_data['nodes']['AND1']['label']:
            prob_top = fta_data['nodes']['Peak']['prob'] * fta_data['nodes']['Press']['prob']
        else:
            p1 = fta_data['nodes'][list(fta_data['nodes'].keys())[2]]['prob']
            p2 = fta_data['nodes'][list(fta_data['nodes'].keys())[3]]['prob']
            prob_top = 1 - (1 - p1) * (1 - p2)
        fta_data['nodes']['Top']['prob'] = prob_top
        
        fig = plot_fta_diagram(fta_data, project_type)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("FTA Constructor Gadgets")
        # Gadgets remain the same...
        c1, c2, c3 = st.columns(3)
        c1.text_input("Top Event Definition", value=fta_data['title'], help="Define the specific, undesirable system-level failure you want to analyze. This is the starting point at the top of the fault tree (e.g., 'Inaccurate Result Displayed', 'System Overheats').")
        c2.button("Add Logic Gate (OR, AND...)", help="Insert a logic gate to define the relationship between events. **OR:** any input event causes the output. **AND:** all input events must occur to cause the output.")
        c3.button("Add Basic Event", help="Add a root cause event that cannot be broken down further. This is a terminal point in the tree (e.g., 'Resistor Fails', 'Tire Punctures', 'Power Outage'). Represented by a circle.")
        c1, c2, c3 = st.columns(3)
        c2.button("Add Intermediate Event", help="Add a sub-system failure that is caused by other events below it. This is a non-terminal event in the tree (e.g., 'Power Supply Fails', 'Cooling System Fails'). Represented by a rectangle.")
        c3.button("Calculate Minimal Cut Sets", help="Click to identify the smallest combinations of basic events that will cause the Top Event to occur. This pinpoints the system's most critical vulnerabilities.")
        c1.text_input("Assign Event Probability", value="0.005", help="Assign a probability or failure rate to a Basic Event. This enables quantitative analysis of the Top Event's overall probability.")

    elif tool_choice == "Event Tree Analysis (ETA)":
        eta_data = qrm_templates["ETA"][project_type]
        fig = plot_eta_diagram(eta_data, project_type)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("ETA Modeler Gadgets")
        # Gadgets remain the same...
        c1, c2 = st.columns(2)
        c1.text_input("Initiating Event", value=eta_data['title'], help="Define the single failure or error that starts the sequence. This is the root of your event tree (e.g., 'Power Surge', 'Contaminated Reagent', 'User enters invalid command').")
        c2.button("Add Safety Function / Barrier", help="Introduce a system, action, or feature designed to mitigate the event. Each barrier creates a new branching point in the tree (Success path / Failure path).")
        
        c1, c2, c3 = st.columns(3)
        c1.text_input("Success/Failure Probability", value="0.99 / 0.01", help="For each safety barrier, enter the probability of it succeeding or failing. The sum of success and failure for a single barrier must equal 1.0.")
        c2.text_input("Outcome Node", value="System Safe Shutdown", help="Define the final state at the end of a branch (e.g., 'System Safe Shutdown', 'Minor Data Corruption', 'Catastrophic Failure', 'False Negative Result').")
        c3.text_input("Path Probability Calculator", value="0.9405", disabled=True, help="Automatically calculates the total probability of reaching a specific outcome by multiplying the probabilities of all events along that path.")

    # --- Educational Tabs (remain unchanged) ---
    st.divider()
    st.subheader("A Deeper Dive into QRM Strategy")
    tabs = st.tabs(["💡 Key Insights", "📋 Glossary", "✅ The Golden Rule", "📖 Theory & History", "🏛️ Regulatory & Compliance"])
    with tabs[0]:
        st.markdown("""
        **A validation leader must choose the right risk tool for the right question.** While all these tools analyze risk, they do so from fundamentally different perspectives. Using the wrong tool for your situation can be inefficient at best and dangerously misleading at worst.
    
        ---
        ##### A Medical Analogy for Risk Tools
        Think of your process or system as a patient, and you are the lead physician.
    
        - **PHA is the Annual Physical:** Broad, high-level, and performed early to identify the "big rock" hazards.
        - **FMEA is the Full Body System Review:** Detailed, systematic, and **bottom-up**. It asks, "What are all the ways each part could fail?"
        - **FTA is the Specialist's Diagnosis:** Forensic, focused, and **top-down (deductive)**. It asks, "For this critical symptom to occur, what must have gone wrong?"
        - **ETA is the Emergency Room Triage:** Forward-looking, sequential, and **bottom-up (inductive)**. It asks, "Now that this trauma has occurred, what are the possible outcomes?"
        
        ---
        ##### Understanding the Analytical Directions
        - **Bottom-Up (Inductive):** Start with individual **causes** and reason forward to their potential **effects**. (FMEA & ETA).
        - **Top-Down (Deductive):** Start with a known **effect** (the failure) and reason backward to deduce the potential **causes**. (FTA).
        """)
    with tabs[1]:
        st.markdown("""
        ##### Glossary of QRM Terms
        - **Risk:** The combination of the probability of occurrence of harm and the severity of that harm.
        - **Hazard:** A potential source of harm.
        - **PHA (Preliminary Hazard Analysis):** A tool to identify and prioritize hazards early in the design process.
        - **FMEA (Failure Mode and Effects Analysis):** A systematic, bottom-up method for identifying potential failure modes, their causes, and their effects.
        - **FTA (Fault Tree Analysis):** A top-down, deductive failure analysis where an undesired state is analyzed using boolean logic to combine a series of lower-level events.
        - **ETA (Event Tree Analysis):** A bottom-up, inductive logical model that explores the responses of a system to a particular initiating event.
        - **RPN (Risk Priority Number):** The product of Severity, Occurrence, and Detection scores from an FMEA, used to prioritize risks.
        """)
    with tabs[2]:
        st.success("""🟢 **THE GOLDEN RULE:** Risk Assessment is Not a One-Time Event. Quality Risk Management is a continuous lifecycle activity. An initial PHA should inform a more detailed FMEA. The FMEA identifies critical failure modes that might warrant a deep-dive FTA. The effectiveness of the controls identified in the FMEA and FTA should be monitored throughout the product lifecycle, and the risk assessments should be updated whenever new information (e.g., from a deviation or a process change) becomes available.""")
    with tabs[3]:
        st.markdown("""
        #### Historical Context: Forged in Fire, Ice, and the Void
        These advanced risk management tools were not born in sterile laboratories but were forged in the crucibles of the 20th century's most demanding engineering challenges, where failure carried the ultimate price. They represent a migration of hard-won knowledge from high-hazard industries directly into the heart of pharmaceutical quality systems.

        ---
        
        ##### FMEA: From Munitions to the Moon (and Back to Earth)
        - **The Origin (1949):** The FMEA methodology was first formalized in a U.S. military procedure, **MIL-P-1629**. In the tense aftermath of WWII, the goal was brutally practical: to proactively analyze potential failures in complex munitions and weapon systems *before* they were deployed, preventing catastrophic failures on the battlefield.
        - **The Ultimate High-Stakes Project (1960s):** FMEA became the bedrock of **NASA's reliability engineering for the Apollo program**. For a mission where millions of components had to function perfectly, FMEA was the exhaustive, bottom-up tool used to systematically assess every single bolt, switch, and circuit, asking "How can this fail, and what happens if it does?" The consequences of missing a failure mode were famously highlighted decades later by physicist Richard Feynman in his investigation of the Space Shuttle *Challenger* disaster, where he noted that a proper FMEA of the O-rings could have identified the critical temperature-related failure mode that led to the tragedy.
        - **The Automotive Awakening (1970s):** The Ford Motor Company, reeling from the infamous Pinto fuel tank crisis and facing intense competition, championed the use of FMEA to proactively design safety and reliability into their vehicles, shifting the industry from a reactive, "test and fix" mentality to a proactive, design-centric one.

        ---

        ##### FTA: The Logic of Armageddon
        - **The Cold War Problem (1961):** At Bell Laboratories, engineers working with the U.S. Air Force on the **Minuteman I Intercontinental Ballistic Missile (ICBM)** system faced a terrifying new question: "What is the probability of an *accidental* missile launch?" A simple FMEA was insufficient. FMEA is excellent for single-point failures, but the accidental launch required a complex *combination* of events.
        - **The Insight (H.A. Watson):** Engineers led by H.A. Watson developed Fault Tree Analysis as a top-down, deductive logic tool. They started with the catastrophic top event ("Accidental Launch") and used Boolean logic gates (AND, OR) to map all the lower-level equipment failures and human errors that could lead to it. This allowed them to identify the most critical vulnerabilities—the "minimal cut sets"—in a system where the stakes were literally global annihilation.

        ---

        ##### ETA: The Domino Effect of a Meltdown
        - **The Nuclear Age (1970s):** While FTA looked backward from a failure, engineers in the burgeoning nuclear power and chemical industries needed to look *forward*. They needed to answer a different kind of question: "A primary cooling pipe has just ruptured. **What happens next?**"
        - **The Consequence Map:** Event Tree Analysis was developed to model this sequential, cascading failure. It starts with an "initiating event" and then maps out the success or failure of each subsequent safety system (e.g., "Did the backup pump start? Did the containment doors seal?"). This creates a branching map of all possible outcomes and their probabilities. ETA was a core component of the landmark **WASH-1400 Reactor Safety Study** (1975), which, in a sense, predicted the types of failure sequences that would occur at the **Three Mile Island accident** just four years later, solidifying its importance in plant safety.

        ---
        
        ##### The Great Migration: From Engineering to Pharma
        For decades, the pharmaceutical industry relied primarily on intensive "quality by testing," often reacting to failures after they occurred. However, a series of high-profile manufacturing issues in the late 1990s and early 2000s, which led to costly recalls and drug shortages, prompted a major philosophical shift. Led by visionaries like Dr. Janet Woodcock, the FDA launched its **"Pharmaceutical cGMPs for the 21st Century"** initiative, calling for a modern, proactive, science- and risk-based approach to quality.

        This culminated in the **ICH Q9 guideline in 2005**, which formally "imported" this entire suite of battle-tested engineering tools into the pharmaceutical quality system. It was a landmark moment, signaling that the proactive, systems-thinking mindset forged in the aerospace, defense, and nuclear industries was now the regulatory expectation for ensuring patient safety, forever changing the landscape of validation and process control.
        """)
    with tabs[4]:
        st.markdown("""
        This suite of tools is the direct implementation of the principles outlined in **ICH Q9 - Quality Risk Management**.
        - **ICH Q9:** This guideline provides a framework for risk management and lists PHA, FMEA, FTA, and ETA as examples of appropriate tools.
        - **FDA Process Validation Guidance:** The guidance requires a risk-based approach. These tools provide the documented evidence for how risks were identified and controlled.
        - **ISO 14971 (Application of risk management to medical devices):** This is the international standard for risk management for medical devices, and it requires the use of a systematic process like FMEA or PHA.
        """)

#=============================================================== DfX MODULE ===========================================================================================================

def render_dfx_dashboard():
    """Renders the comprehensive, interactive module for Design for Excellence (DfX)."""
    st.markdown("""
    #### Purpose & Application: Designing for the Entire Lifecycle
    **Purpose:** To demonstrate the strategic and economic impact of **Design for Excellence (DfX)**, a proactive engineering philosophy that focuses on optimizing a product's design for its entire lifecycle, from manufacturing to disposal.
    
    **Strategic Application:** DfX is the practical implementation of "shifting left"—addressing downstream problems (like manufacturing costs, test time, or reliability) during the earliest stages of design, where changes are exponentially cheaper. For a validation leader, promoting DfX principles is a key strategy for ensuring that new products are not only effective but also robust, scalable, and profitable.
    """)
    
    st.info("""
    **Interactive Demo:** You are the Head of Engineering or Validation.
    1.  Select the **Project Type** you are leading.
    2.  Use the **DfX Effort Sliders** in the sidebar to allocate engineering resources to different design philosophies.
    3.  Observe the impact in real-time on the **KPI Dashboard**, including the critical **Risk-Adjusted Cost**.
    """)
    
    profiles = {
        "Pharma Assay (ELISA)": {
            'categories': ['Robustness', 'Run Time (hrs)', 'Reagent Cost (RCU)', 'Precision (%CV)', 'Ease of Use'], 'baseline': [5, 4.0, 25.0, 18.0, 5], 'direction': [1, -1, -1, -1, 1], 'reliability_idx': 0,
            'impact': {'mfg': [0.1, -0.1, -0.2, 0, 0.1], 'quality': [0.5, -0.05, 0, -0.6, 0.2], 'sustainability': [0, 0, -0.3, 0, 0], 'ux': [0.1, -0.2, 0, 0, 0.7]}
        },
        "Instrument (Liquid Handler)": {
            'categories': ['Throughput<br>(plates/hr)', 'Uptime (%)', 'Footprint (m²)', 'Service Cost<br>(RCU/yr)', 'Precision (%CV)'], 'baseline': [20, 95.0, 2.5, 5000, 5.0], 'direction': [1, 1, -1, -1, -1], 'reliability_idx': 1,
            'impact': {'mfg': [0.2, 0.1, -0.2, -0.1, 0], 'quality': [0.1, 0.8, 0, -0.2, -0.6], 'sustainability': [0, 0.1, -0.1, -0.4, 0], 'ux': [0, 0.2, 0, -0.6, 0]}
        },
        "Software (LIMS)": {
            'categories': ['Performance<br>(Query Time s)', 'Scalability<br>(Users)', 'Reliability<br>(Uptime %)', 'Compliance<br>Score', 'Dev Cost (RCU)'], 'baseline': [8.0, 100, 99.5, 6, 500], 'direction': [-1, 1, 1, 1, -1], 'reliability_idx': 2,
            'impact': {'mfg': [-0.1, 0.2, 0.2, 0, -0.4], 'quality': [-0.2, 0.1, 0.7, 0.8, 0.2], 'sustainability': [0, 0.5, 0.1, 0, -0.1], 'ux': [-0.4, 0.2, 0, 0.5, 0.1]}
        },
        "Pharma Process (MAb)": {
            'categories': ['Yield (g/L)', 'Cycle Time<br>(days)', 'COGS (RCU/g)', 'Purity (%)', 'Robustness<br>(PAR Size)'], 'baseline': [3.0, 18, 100, 98.5, 5], 'direction': [1, -1, -1, 1, 1], 'reliability_idx': 4,
            'impact': {'mfg': [0.3, -0.2, -0.4, 0.1, 0.2], 'quality': [0.1, 0, -0.1, 0.6, 0.8], 'sustainability': [0.05, -0.1, -0.2, 0, 0.1], 'ux': [0, 0, 0, 0, 0]}
        }
    }
    
    project_type = st.selectbox("Select a Project Type to Simulate DfX Impact:", list(profiles.keys()))
    selected_profile = profiles[project_type]

    with st.sidebar:
        st.subheader("DfX Effort Allocation")
        mfg_effort = st.slider("Manufacturing & Assembly Effort", 0, 10, 5, 1, help="DFM/DFA: Focus on part count reduction, standard components, and automation.")
        quality_effort = st.slider("Quality & Reliability Effort", 0, 10, 5, 1, help="DFR/DFT/DFQ: Focus on reliability, robust performance, and fast, accurate QC testing.")
        sustainability_effort = st.slider("Sustainability & Supply Chain Effort", 0, 10, 5, 1, help="DFE/DFS: Focus on standard/recyclable materials, energy use, and easy disassembly.")
        ux_effort = st.slider("Service & User Experience Effort", 0, 10, 5, 1, help="DFS/DFUX: Focus on ease of use, service, and maintenance.")

    fig_radar, fig_cost, kpis, categories, base_costs, optimized_costs = plot_dfx_dashboard(
        project_type, mfg_effort, quality_effort, sustainability_effort, ux_effort
    )

    st.header("Project KPI Dashboard")
    reliability_idx = selected_profile['reliability_idx']
    base_reliability = kpis['baseline'][reliability_idx]
    opt_reliability = kpis['optimized'][reliability_idx]
    
    base_total_cost = sum(base_costs)
    opt_total_cost = sum(optimized_costs)

    base_risk_premium = 1 + (10 - base_reliability) * 0.05 if "Score" in categories[reliability_idx] else 1 + (100 - base_reliability) * 0.02
    opt_risk_premium = 1 + (10 - opt_reliability) * 0.05 if "Score" in categories[reliability_idx] else 1 + (100 - opt_reliability) * 0.02

    base_risk_adjusted_cost = base_total_cost * base_risk_premium
    opt_risk_adjusted_cost = opt_total_cost * opt_risk_premium

    col1, col2 = st.columns(2)
    col1.metric("Total Cost (RCU)", f"{opt_total_cost:,.0f}", f"{opt_total_cost - base_total_cost:,.0f}")
    col2.metric("Risk-Adjusted Total Cost (RCU)", f"{opt_risk_adjusted_cost:,.0f}", f"{opt_risk_adjusted_cost - base_risk_adjusted_cost:,.0f}",
                help="Total Cost including a 'risk premium'. A less reliable or robust design is penalized more heavily, reflecting the long-term cost of potential failures.")
    
    st.markdown("##### Performance Profile")
    kpi_cols = st.columns(len(kpis['baseline']))
    for i, col in enumerate(kpi_cols):
        base_val = kpis['baseline'][i]
        opt_val = kpis['optimized'][i]
        delta = opt_val - base_val
        col.metric(categories[i].replace('<br>', ' '), f"{opt_val:.1f}", f"{delta:.1f}")

    st.header("Design Comparison Visualizations")
    col_radar, col_cost = st.columns(2)
    with col_radar:
        st.plotly_chart(fig_radar, use_container_width=True)
    with col_cost:
        st.plotly_chart(fig_cost, use_container_width=True)
    
    st.divider()
    st.subheader("Deeper Dive")
    tabs = st.tabs(["💡 Key Insights", "📋 Glossary", "✅ The Golden Rule", "📖 Theory & History", "🏛️ Regulatory & Compliance"])
    with tabs[0]:
        st.markdown("""
        **Interpreting the Dashboard:**
        - **KPI Dashboard:** This is your executive summary. It quantifies the return on investment for your DfX efforts in terms of performance and cost. The **Risk-Adjusted Cost** is the most important metric, as it represents the "true" cost of a design by penalizing lower quality and reliability.
        - **Performance Profile (Radar Chart):** This visualizes the multi-dimensional impact of your design choices. The goal is to create an "Optimized" profile (green) that meets or exceeds all performance targets.
        - **Cost Structure (Pie Charts):** This shows *how* you achieved cost savings. The units are **Relative Cost Units (RCU)**, focusing on proportions. The total cost is displayed in the center. A strong DFM/DFA effort will dramatically reduce the proportion of cost attributed to 'Manufacturing/Labor'.
        
        **The Strategic Insight:** The most profitable design is not always the one with the lowest initial cost. The **Risk-Adjusted Cost** demonstrates that investing in reliability and quality (DFR/DFQ) can be a financially sound decision by reducing the long-term costs associated with failures, service, and warranty claims.
        """)
    with tabs[1]:
        st.markdown("""
        ##### Glossary of DfX Principles
        - **DfX (Design for Excellence):** An engineering philosophy that integrates lifecycle concerns into the earliest design stages.
        - **DFM (Design for Manufacturability):** Designing parts for ease of manufacturing to reduce cost and improve yield.
        - **DFA (Design for Assembly):** Designing a product to be easy to assemble, primarily by reducing part count.
        - **DFT (Design for Test):** Designing a product to make QC testing fast, automated, and reliable.
        - **DFR (Design for Reliability):** Designing a product to be robust and have a long, predictable lifespan.
        - **DFQ (Design for Quality):** Integrating quality assurance principles from concept through production, often using statistical tools like DFSS.
        - **DFSS (Design for Six Sigma):** A data-driven approach to design that aims to create products and processes that are defect-free from the start.
        - **DFE (Design for Environment):** Designing to reduce environmental footprint, including material selection and end-of-life recycling.
        - **DFS (Design for Serviceability):** Designing a product to be easy to maintain and repair.
        - **DFUX (Design for User Experience):** Optimizing the product for usability and satisfaction by incorporating human factors and ergonomic principles.
        - **RCU (Relative Cost Unit):** An abstract unit of cost used for strategic planning and comparison when precise dollar amounts are unknown or variable. It focuses on the proportions and relative magnitudes of different cost drivers.
        """)
    with tabs[2]:
        st.error("""🔴 **THE INCORRECT APPROACH: "Over-the-Wall" Engineering**
The R&D team designs a product in isolation, focusing only on functionality. They then "throw the design over the wall" to the manufacturing and quality teams, who discover that it is impossibly difficult to build, assemble, or test reliably at scale.
- **The Flaw:** This sequential process creates massive rework, delays, and friction between departments. Problems that would have taken minutes to fix on a CAD model now require weeks of expensive re-tooling and re-validation.""")
        st.success("""🟢 **THE GOLDEN RULE: The Cost of a Design Change is Exponential**
The core principle of DfX is **concurrent engineering**, where design, manufacturing, quality, and other downstream teams work together as a cross-functional unit from the very beginning of the project.
1.  **Acknowledge the Cost Curve:** A design change on the whiteboard is free. A change after tooling is made costs thousands. A change after the product is in the field costs millions in recalls and reputational damage.
2.  **Shift Left:** The goal is to pull as many manufacturing, assembly, and testing considerations as far "left" into the early design phase as possible.
3.  **Use DfX as a Formal Checklist:** Proactively review designs against a formal DfX checklist at every Stage-Gate or Design Review to ensure the product is not just functional, but manufacturable, testable, and profitable.""")
        
    with tabs[3]:
        st.markdown("""
        #### Historical Context: The Birth of Concurrent Engineering
        The principles of DfX emerged from the intense manufacturing competition of the 1970s and 80s. While concepts like designing for ease of manufacturing existed for decades, they were formalized and popularized by several key forces:
        - **Japanese Manufacturing:** Companies like Toyota and Sony pioneered concepts of lean manufacturing and concurrent engineering, where design was not an isolated activity but a team sport involving all departments to eliminate "muda" (waste).
        - **Boothroyd & Dewhurst (DFA):** In the late 1970s, Geoffrey Boothroyd and Peter Dewhurst at the University of Massachusetts Amherst developed the first systematic, quantitative methodology for **Design for Assembly (DFA)**. Their work provided a structured way to analyze a design, estimate its assembly time, and identify opportunities for part reduction, transforming DFA from a vague idea into a rigorous engineering discipline.
        - **General Electric's "Bulls-eye":** In the 1980s, GE championed DFM, famously using a "bulls-eye" diagram to illustrate how the majority of a product's cost is locked in during the earliest design stages.
        
        The success of these methods led to the proliferation of the "DfX" acronym, extending the core philosophy to all aspects of a product's lifecycle.
        """)
        
    with tabs[4]:
        st.markdown("""
        DfX is the practical engineering methodology used to fulfill the requirements of formal **Design Controls**.
        - **FDA 21 CFR 820.30 (Design Controls):** This regulation for medical devices is the primary driver for DfX in the life sciences. The DfX process is how you fulfill the requirements for:
            - **Design Inputs:** Proactively considering manufacturing, assembly, and testing requirements from the very start.
            - **Design Review:** DfX checklists and scorecards are a key part of formal, documented design reviews.
            - **Design Verification & Validation:** Ensuring the design outputs meet the design inputs.
            - **Design Transfer:** A product designed with DfX principles has a much smoother and more successful design transfer into manufacturing.
        - **ICH Q8(R2) - Pharmaceutical Development:** The principles of QbD—understanding how product design and process parameters affect quality—are perfectly aligned with DfX.
        - **ISO 13485 (Medical Devices):** This international standard for quality management systems requires a structured design and development process, which is effectively implemented through DfX principles.
        """)
#========================================================================================= 5. VALIDATION MASTER PLAN (VMP) BUILDER =====================================================================
def render_vmp_builder():
    """Renders the comprehensive, interactive module for the Validation Master Plan Builder."""
    st.markdown("""
    #### Purpose & Application: The Project's Master Plan
    **Purpose:** To act as the **interactive project manager and compliance checklist** for any validation activity. The Validation Master Plan (VMP) is the highest-level strategic document outlining the scope, responsibilities, and methodologies for the entire validation effort.
    
    **Strategic Application:** This tool connects the strategic "why" (from the TPP and FMEA) to the tactical "how" (the specific analytical tools). It provides a clear roadmap for the project, showing which tools from this toolkit are deployed at each phase of a validation project. This is essential for project planning, resource allocation, and demonstrating a structured approach to regulators.
    """)
    
    st.info("""
    **Interactive Demo:** You are the Validation Manager. Select a **Project Type** from the dropdown menu. The diagram below will dynamically update to show the standard validation workflow for that project, highlighting which tools from this application are used at each critical stage.
    """)

    project_type = st.selectbox(
        "Select a Validation Project Type to Plan:",
        ("Analytical Method Validation", "Instrument Qualification", "Pharma Process (PPQ)", "Software System (CSV)"),
        index=0,
        help="Choose the type of project you are planning. The diagram will update to show the standard validation workflow and the key analytical tools used at each stage."
    )

    fig = plot_vmp_flow(project_type)
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    st.subheader("Deeper Dive")
    tabs = st.tabs(["💡 Key Insights", "📋 Glossary", "✅ The Golden Rule", "📖 Theory & History", "🏛️ Regulatory & Compliance"])
    with tabs[0]:
        st.markdown(f"""
        **Connecting Strategy to Execution for: {project_type}**
        This tool demonstrates how all the other modules in the toolkit fit together to form a complete, compliant validation project.
        - The **Analytical Method Validation** workflow follows the classic V-Model, showing the direct link between defining performance specifications (like Linearity) during design and later qualifying them during PQ.
        - The **Instrument Qualification** workflow also follows the V-Model, starting with User Needs (captured in the ATP Builder) and culminating in PQ testing (using Gage R&R and SPC).
        - The **Pharma Process (PPQ)** workflow is a linear process, moving from planning (using FMEA and Sample Size) to execution and final analysis (using SPC and Capability). This represents Stage 2 of the FDA's Process Validation lifecycle.
        - The **Software System (CSV)** workflow shows how modern AI/ML tools can be integrated into the GAMP 5 V-Model. For example, **Explainable AI (XAI)** is a key activity during the "Build" phase to ensure the model is transparent.
        """)
    with tabs[1]:
        st.markdown("""
        ##### Glossary of VMP & V-Model Terms
        - **Validation Master Plan (VMP):** A high-level document that describes an organization's overall philosophy, intention, and approach to validation. It governs all validation activities at a site.
        - **Validation Protocol (VP):** A detailed, written plan stating how a specific validation will be conducted, including test parameters, product characteristics, and pre-approved acceptance criteria.
        - **User Requirement Specification (URS):** Defines what the user needs the system to do, from a business perspective.
        - **Functional Specification (FS):** Translates the URS into a detailed description of what the system *must do* (its functions).
        - **Design Specification (DS):** Describes *how* the system will be built to meet the FS.
        - **Installation Qualification (IQ):** Documented verification that the system is installed correctly according to specifications and vendor recommendations.
        - **Operational Qualification (OQ):** Documented verification that the system operates as intended throughout its specified operating ranges. It is often called "testing against the Functional Spec."
        - **Performance Qualification (PQ):** Documented verification that the system, as installed and operated, can consistently perform its intended function and meet pre-determined acceptance criteria under real-world conditions. It is often called "testing against the User Requirement Spec."
        """)
    with tabs[2]:
        st.error("""🔴 **THE INCORRECT APPROACH: "Validation on the Fly"**
A project team begins executing test scripts for a new instrument without a pre-approved protocol or a VMP. When a test fails, they change the acceptance criteria so it will pass.
- **The Flaw:** This is not validation; it is "testing into compliance." The acceptance criteria were not pre-defined, and the changes are undocumented. This entire activity would be invalidated during an audit.""")
        st.success("""🟢 **THE GOLDEN RULE:** If it isn't written down, it didn't happen. The VMP is the single source of truth that defines the scope, strategy, and acceptance criteria for a validation project *before it begins*. It is the most important document for preventing ad-hoc decision making and is the first thing an auditor will ask to see.""")
    with tabs[3]:
        st.markdown("""
        #### Historical Context: From Chaos to Control
        The concept of a formal, high-level planning document for validation grew out of the need to manage increasingly complex projects in the pharmaceutical industry in the 1980s and 90s. As systems became more integrated and regulations more stringent, the old model of simple, standalone equipment qualification was no longer sufficient.
        
        The idea was heavily influenced by project management principles and was codified by two major forces:
        1.  **Regulatory Expectation:** The FDA and other global bodies began to expect a more holistic, planned approach to validation, looking for a "master plan" that governed all qualification activities at a site.
        2.  **GAMP Guidelines:** The **GAMP (Good Automated Manufacturing Practice)** Forum, a consortium of pharmaceutical and engineering professionals, pioneered the V-Model approach for computer systems. A core part of their philosophy was the necessity of high-level planning documents, like the VMP, to manage the complexity of software validation.
        
        Regulators quickly adopted the expectation for a VMP across all types of validation to ensure projects were well-planned, structured, and auditable from the top down.
        """)
    with tabs[4]:
        st.markdown("""
        The VMP is a key document for demonstrating a compliant and well-managed quality system.
        - **EU Annex 15: Qualification and Validation:** This is one of the most explicit regulations. It states that "A Validation Master Plan (VMP) or equivalent document should be established... providing an overview of the company’s validation strategy."
        - **FDA 21 CFR 211.100 (Written procedures; deviations):** Requires that "there shall be written procedures for production and process control..." The VMP is the highest-level document in this procedural hierarchy, governing all subordinate validation protocols and reports.
        - **GAMP 5:** The VMP is a foundational document in the GAMP 5 framework, outlining the plan for validating all GxP computerized systems.
        - **PIC/S Guide to GMP (PE 009-16):** This influential international guideline, adopted by dozens of regulatory agencies worldwide, also describes the expectation for a VMP to manage the overall validation effort.
        """)

def render_rtm_builder():
    """Renders the comprehensive, interactive module for the Requirements Traceability Matrix."""
    st.markdown("""
    #### Purpose & Application: The Auditor's Golden Thread
    **Purpose:** To be the **"Auditor's Golden Thread."** The RTM is a master document that links every high-level requirement (like a URS or CQA) to the specific design elements and, most critically, to the specific Test Case(s) that prove it was met, even across different validation streams.
    
    **Strategic Application:** The RTM provides irrefutable, traceable proof that **everything that was asked for was built, and everything that was tested was required.** It is the ultimate tool for managing and ensuring the completeness of complex, multi-faceted projects like a full technology transfer.
    """)
    
    st.info("""
    **Interactive Demo:** You are the Head of Engineering or Validation.
    1.  Select the **Project Type** you are leading.
    2.  Use the **DfX Effort Sliders** in the sidebar to allocate engineering resources to different design philosophies.
    3.  Observe the impact in real-time on the **KPI Dashboard** and the **Performance Profile** radar chart.
    """)
    
    project_type = st.selectbox(
        "Select a Project Type to Simulate DfX Impact:",
        ["Pharma Process (MAb)", "Pharma Assay (ELISA)", "Instrument (Liquid Handler)", "Software (LIMS)"]
    )

    with st.sidebar:
        st.subheader("DfX Effort Allocation")
        mfg_effort = st.slider("Manufacturing & Assembly Effort (DFM/DFA)", 0, 10, 5, 1, help="Focus on part count reduction, using standard components, and designing for automated assembly.")
        quality_effort = st.slider("Quality & Reliability Effort (DFR/DFT)", 0, 10, 5, 1, help="Focus on building in reliability, designing for robust performance, and adding features to make QC testing faster and more accurate.")
        sustainability_effort = st.slider("Sustainability & Supply Chain Effort (DFE)", 0, 10, 5, 1, help="Focus on using standard/recyclable materials, reducing energy use, and designing for easy disassembly.")
        ux_effort = st.slider("Service & User Experience Effort (DFS/DFUX)", 0, 10, 5, 1, help="Focus on making the device easy to use, service, and maintain, reducing long-term operational costs and human error.")

    # --- THIS IS THE KEY FIX ---
    # Call the cached function with simple, hashable arguments.
    fig_radar, fig_cost, kpis, categories = plot_dfx_dashboard(
        project_type, mfg_effort, quality_effort, sustainability_effort, ux_effort
    )
    # --- END OF FIX ---

    st.header("Project KPI Dashboard")
    kpi_cols = st.columns(len(kpis['baseline']))
    for i, col in enumerate(kpi_cols):
        base_val = kpis['baseline'][i]
        opt_val = kpis['optimized'][i]
        delta = opt_val - base_val
        col.metric(categories[i].replace('<br>', ' '), f"{opt_val:.1f}", f"{delta:.1f}")

    st.header("Design Comparison Visualizations")
    col_radar, col_cost = st.columns(2)
    with col_radar:
        st.plotly_chart(fig_radar, use_container_width=True)
    with col_cost:
        st.plotly_chart(fig_cost, use_container_width=True)
    
    st.divider()
    st.subheader("Deeper Dive")
    tabs = st.tabs(["💡 Key Insights", "📋 Glossary", "✅ The Golden Rule", "📖 Theory & History", "🏛️ Regulatory & Compliance"])
    with tabs[0]:
        st.markdown("""
        **Reading the Integrated RTM:**
        - **Swimlanes:** The diagram is organized into vertical "swimlanes" representing the different, parallel validation projects that must all succeed.
        - **Intra-Stream Traceability:** The horizontal links within a single swimlane (e.g., `INST-URS` → `INST-OQ`) show the standard V-Model traceability for that specific system.
        - **Inter-Stream Dependencies (The Critical Links):** The diagonal links show the crucial dependencies *between* projects. For example, you cannot complete the `PROC-TEST` (PPQ) until the `ASSAY-VAL` (the QC test method) and the `INST-PQ` (the lab instrument) are both complete and validated.
        
        **The Strategic Insight:** This visualization reveals that a validation project is a **network of dependencies**. A delay or failure in one stream (like the Instrument Qualification) can have a direct, cascading impact on the critical path of the main process validation. The RTM is the master tool for managing these complex relationships.
        """)
    with tabs[1]:
        st.markdown("""
        ##### Glossary of RTM & V-Model Terms
        - **Requirement:** A condition or capability needed by a user to solve a problem or achieve an objective. Can be a URS, an ATP element, or a CQA.
        - **Specification:** A document that specifies, in a complete, precise, verifiable manner, the requirements, design, behavior, or other characteristics of a system. (e.g., FS, DS).
        - **Verification:** The process of evaluating a system to determine whether the products of a given development phase satisfy the conditions imposed at the start of that phase. (Are we building the system right?)
        - **Validation:** The process of evaluating a system during or at the end of the development process to determine whether it satisfies the user requirements. (Are we building the right system?)
        - **Traceability:** The ability to trace a requirement both forwards to its implementation and testing, and backwards to its origin, even across system boundaries.
        """)
    with tabs[2]:
        st.error("""🔴 **THE INCORRECT APPROACH: The "Siloed Validation" Fallacy**
The Instrument team validates their HPLC, the Assay team validates their purity method, and the Process team validates the manufacturing run, all as separate projects. They only discover at the end that the validated assay cannot achieve the required precision on the newly qualified instrument.
- **The Flaw:** The teams operated in silos, ignoring the critical interdependencies between their systems. The project lacked a holistic RTM to manage these cross-stream links.""")
        st.success("""🟢 **THE GOLDEN RULE: One Project, One Trace Matrix**
A complex project like a tech transfer should be governed by a single, integrated Validation Master Plan (VMP) and a single, integrated Requirements Traceability Matrix (RTM).
- The RTM must capture not only the vertical traceability within each system (the V-Model) but also the **horizontal traceability between systems.**
- This integrated RTM becomes the master checklist for the entire project, ensuring that dependencies are managed, no gaps exist, and the final integrated system is truly validated as a whole.""")
        
    with tabs[3]:
        st.markdown("""
        #### Historical Context: From Systems Engineering to Pharma
        The concept of a Requirements Traceability Matrix has its roots in **systems engineering and software development**. As systems became more complex in the 1970s and 80s, projects were often plagued by "scope creep" and mismatches between user expectations and the final product. The RTM was developed as a formal project management tool to ensure all requirements were tracked and verified.
        
        Its value was immediately recognized for regulated software, where proof of completeness is non-negotiable. The **GAMP (Good Automated Manufacturing Practice)** Forum, a consortium of pharmaceutical and engineering professionals, formally adopted traceability as a core principle. The RTM became the de facto standard method for achieving and documenting this traceability, making it a cornerstone of modern Computer System Validation (CSV) in the pharmaceutical and medical device industries.
        """)
        
    with tabs[4]:
        st.markdown("""
        The RTM is the primary document used to demonstrate compliance with a variety of regulations governing complex systems.
        - **ICH Q10 - Pharmaceutical Quality System:** This guideline emphasizes a holistic approach to quality management, including the management of outsourced activities and tech transfers. An integrated RTM is the key document for demonstrating control over these complex, multi-faceted projects.
        - **GAMP 5 - A Risk-Based Approach to Compliant GxP Computerized Systems:** The RTM is a foundational document in the GAMP 5 framework, providing the traceability that underpins the entire V-Model validation approach.
        - **FDA 21 CFR 820.30 (Design Controls):** For medical device software, the RTM is the key to demonstrating that all design inputs (user needs) have been met by the design outputs (the software) and that this has been verified through testing. It is a critical component of the Design History File (DHF).
        """)


#====================================================================================================================================================================================================================================
#=====================================================================================================ACT 0 RENDER END ==============================================================================================================
#====================================================================================================================================================================================================================================
# ======================================== 1. EXPLORATORY DATA ANALYSIS (EDA)  ===============================================================
def render_eda_dashboard():
    """Renders the comprehensive, interactive module for Exploratory Data Analysis."""
    st.markdown("""
    #### Purpose & Application: The Data Scientist's First Look
    **Purpose:** To perform a thorough **Exploratory Data Analysis (EDA)**. Before any formal modeling or hypothesis testing, EDA is essential to understand the data's structure, identify potential quality issues, and form initial hypotheses.
    
    **Strategic Application:** This is the most critical first step in any data-driven project. Skipping EDA is like a surgeon operating without looking at the patient's chart—it's professional malpractice. This tool automates the creation of a comprehensive EDA report, allowing a validation leader or scientist to quickly assess the quality of a new dataset and discover hidden relationships that warrant formal investigation.
    """)
    
    st.info("""
    **Interactive Demo:** You are a Data Scientist receiving a new dataset.
    1.  **Select a sample dataset** from the dropdown menu to simulate a real-world analysis scenario.
    2.  The dashboard automatically generates a full EDA report for that dataset.
    3.  Review the **Data Quality KPIs** to check for problems, then analyze the plots to understand the data's structure and relationships.
    """)

    # --- NEW: Sample Dataset Selector ---
    @st.cache_data
    def load_datasets():
        np.random.seed(42)
        pharma_data = pd.DataFrame({
            'Yield': np.random.normal(85, 5, 100),
            'Purity': 100 - np.random.beta(2, 20, 100) * 5,
            'pH': np.random.normal(7.1, 0.1, 100),
            'Temperature': np.random.normal(37, 0.5, 100),
            'Raw_Material_Lot': np.random.choice(['Lot A', 'Lot B', 'Lot C'], 100, p=[0.5, 0.3, 0.2])
        })
        pharma_data.loc[5:10, 'Purity'] = np.nan # Introduce missing data

        instrument_data = pd.DataFrame({
            'Dispense_Volume': np.random.normal(50, 0.5, 150),
            'Pressure': np.random.normal(10, 0.2, 150) + (np.random.normal(50, 0.5, 150) - 50) * 0.1,
            'Run_Time_sec': np.random.gamma(20, 2, 150),
            'Operator': np.random.choice(['Alice', 'Bob', 'Charlie'], 150)
        })
        instrument_data.loc[20, 'Pressure'] = 25 # Introduce an outlier
        
        return {
            "Pharma Manufacturing Process": pharma_data,
            "Instrument Performance Data": instrument_data
        }

    datasets = load_datasets()
    dataset_choice = st.selectbox("Select a Sample Dataset to Analyze:", list(datasets.keys()))
    df = datasets[dataset_choice]
    # --- END OF NEW SECTION ---

    st.header("Exploratory Data Analysis Report")
    st.dataframe(df.head())
    
    st.subheader("Data Quality KPIs")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    missing_values = df.isnull().sum().sum()
    col3.metric("Missing Values", f"{missing_values}", help=f"Total number of empty cells. Found in columns: {', '.join(df.columns[df.isnull().any()].tolist())}")
    col4.metric("Duplicate Rows", f"{df.duplicated().sum()}")
    
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    if numeric_cols:
        fig_heatmap, fig_pairplot, fig_hist = plot_eda_dashboard(df, numeric_cols, cat_cols)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        st.plotly_chart(fig_pairplot, use_container_width=True)
        st.plotly_chart(fig_hist, use_container_width=True)

    st.divider()
    st.subheader("Deeper Dive")
    tabs = st.tabs(["💡 Key Insights", "📋 Glossary", "✅ The Golden Rule", "📖 Theory & History", "🏛️ Regulatory & Compliance"])
    with tabs[0]:
        st.markdown("""
        **A Realistic Workflow & Interpretation:**
        1.  **Check Data Quality First:** The KPIs at the top are your first stop. A high number of **Missing Values** or **Duplicate Rows** signals a problem with data collection or integrity that must be fixed before any analysis can be trusted.
        2.  **Understand Distributions (Histograms):** Review the univariate plots. Are the data normally distributed, or are they skewed? Are there potential outliers? This informs which statistical tests will be appropriate.
        3.  **Find the Strongest Relationships (Heatmap):** The correlation heatmap is your guide to what matters most. Bright red (strong positive correlation) or bright blue (strong negative correlation) cells highlight the most powerful relationships in your process, which should be investigated further with formal tools like DOE or Regression.
        4.  **Visualize the Interactions (Pair Plot):** This is the most powerful plot. It shows every bivariate relationship in one graphic. Look for clear trends between variables. If you color by a categorical variable (like `Raw_Material_Lot`), you can often spot group differences, such as Lot A consistently producing higher yields.
        """)
    with tabs[1]:
        st.markdown("""
        ##### Glossary of EDA Terms
        - **EDA (Exploratory Data Analysis):** An approach to analyzing datasets to summarize their main characteristics, often with visual methods.
        - **Data Quality:** The state of data regarding its accuracy, completeness, consistency, and reliability.
        - **Univariate Analysis:** The analysis of a single variable at a time (e.g., a histogram).
        - **Bivariate Analysis:** The analysis of the relationship between two variables (e.g., a scatter plot).
        - **Correlation:** A statistical measure that expresses the extent to which two variables are linearly related (meaning they change together at a constant rate).
        - **Outlier:** A data point that differs significantly from other observations.
        - **Missing Values:** Data points for which no value is stored (often represented as `NaN`).
        """)
    with tabs[2]:
        st.error("""🔴 **THE INCORRECT APPROACH: "Garbage In, Gospel Out"**
An analyst receives a new dataset, immediately feeds it into a sophisticated machine learning model, and presents the model's predictions as truth.
- **The Flaw:** They never checked the data quality. The dataset was riddled with missing values and outliers, which the model interpreted as real patterns. The resulting predictions are statistically invalid and dangerously misleading. This is the definition of 'Garbage In, Garbage Out.'""")
        st.success("""🟢 **THE GOLDEN RULE: Trust, but Verify Your Data**
Before performing any formal statistical analysis or building any model, you must first get to know your data.
1.  **Inspect for Quality:** Always begin by checking for fundamental issues like missing values, duplicates, and obvious errors.
2.  **Visualize the Big Picture:** Use tools like correlation heatmaps and pair plots to understand the overall structure and key relationships in your data.
3.  **Formulate Hypotheses:** Use the insights from EDA to form specific, testable hypotheses. For example, "It appears that Purity is negatively correlated with Temperature. Let's design a formal DOE to confirm this causal link."
EDA is the step that turns raw data into actionable scientific inquiry.""")
        
    with tabs[3]:
        st.markdown("""
        #### Historical Context: The Father of EDA
        While data visualization has existed for centuries, the formal discipline of **Exploratory Data Analysis (EDA)** was single-handedly championed by the brilliant American mathematician **John Tukey** in the 1970s. Tukey, a contemporary of the great quality gurus, argued that traditional statistics had become too focused on "confirmatory" analysis (hypothesis testing) and had neglected the critical first step of "exploratory" analysis.
        
        He believed that analysts should act as "data detectives," using graphical methods to uncover the hidden stories in their data. He invented several of the core visualization tools we use today, including the **box plot** and the **stem-and-leaf plot**. His 1977 book, *Exploratory Data Analysis*, is a classic that liberated statisticians from rigid formalism and encouraged a more intuitive, interactive, and curiosity-driven approach to data. With modern tools like Python and Plotly, we can now automate the powerful, interactive visualizations that Tukey could only dream of.
        """)
        
    with tabs[4]:
        st.markdown("""
        While EDA is an exploratory activity, it is a critical prerequisite for many formal validation activities and is implicitly required by several regulations.
        - **FDA Guidance on Process Validation (Stage 1 - Process Design):** The guidance states that process knowledge and understanding should be built upon a foundation of "development studies." EDA is the first step in analyzing the data from these studies to build that foundational understanding.
        - **Data Integrity (ALCOA+):** A core principle of data integrity is that data must be **Complete** and **Accurate**. The Data Quality KPIs in this dashboard are a direct check on these principles. An EDA report is often a key part of the evidence package for a new dataset, demonstrating that the data has been reviewed for quality before being used in formal GxP analysis.
        - **ICH Q9 (Quality Risk Management):** EDA is a powerful tool for risk identification. Discovering a strong, unexpected correlation in your data during EDA can highlight a previously unknown process risk that needs to be formally assessed with a tool like FMEA.
        """)
        
# ======================================== 2. CONFIDENCE INTERVAL CONCEPT ===============================================================
def render_ci_concept():
    """Renders the interactive module for Confidence Intervals."""
    st.markdown("""
    #### Purpose & Application: The Foundation of Inference
    **Purpose:** To build a deep, intuitive understanding of the fundamental concept of a **frequentist confidence interval** and to correctly interpret what it does—and does not—tell us.
    
    **Strategic Application:** This concept is the bedrock of all statistical inference in a frequentist framework. A misunderstanding of CIs leads to flawed conclusions and poor decision-making. This interactive simulation directly impacts resource planning and risk assessment. It allows scientists and managers to explore the crucial trade-off between **sample size (cost)** and **statistical precision (certainty)**. It provides a visual, data-driven answer to the perpetual question: "How many samples do we *really* need to run to get a reliable result and an acceptably narrow confidence interval?"
    """)
    
    st.info("""
    **Interactive Demo:** Use the **Sample Size (n)** slider below to dynamically change the number of samples in each simulated experiment. Observe how increasing the sample size dramatically narrows both the theoretical Sampling Distribution (orange curve) and the simulated Confidence Intervals (blue/red lines), directly demonstrating the link between sample size and precision.
    """)

    st.sidebar.subheader("Confidence Interval Controls")
    n_slider = st.sidebar.slider("Select Sample Size (n) for Each Simulated Experiment:", 5, 100, 30, 5,
        help="Controls the number of data points in each of the 100 simulated experiments. Notice the dramatic effect on the precision of the results.")
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        fig1_ci, fig2_ci, capture_count, n_sims, avg_width = plot_ci_concept(n=n_slider)
        st.plotly_chart(fig1_ci, use_container_width=True)
        st.plotly_chart(fig2_ci, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "📋 Glossary", "✅ The Golden Rule", "📖 Theory & History", "🏛️ Regulatory & Compliance"])
        with tabs[0]:
            st.metric(label=f"📈 KPI: Average CI Width (Precision) at n={n_slider}", value=f"{avg_width:.2f} units", help="A smaller width indicates higher precision. This is inversely proportional to the square root of n.")
            st.metric(label="🎯 Empirical Coverage Rate", value=f"{(capture_count/n_sims):.1%}", help=f"The % of our {n_sims} simulated CIs that successfully 'captured' the true population mean. Should be close to 95%.")
            st.markdown("""
            - **Theoretical Universe (Top Plot):**
                - The wide, light blue curve is the **true population distribution**. In real life, we *never* see this.
                - The narrow, orange curve is the **sampling distribution of the mean**. Its narrowness, guaranteed by the **Central Limit Theorem**, makes inference possible.
            - **CI Simulation (Bottom Plot):** This shows the reality we live in. We only get *one* experiment and *one* confidence interval.
            - **The n-slider is key:** As you increase `n`, the orange curve gets narrower and the CIs in the bottom plot become dramatically shorter.
            - **Diminishing Returns:** The gain in precision from n=5 to n=20 is huge. The gain from n=80 to n=100 is much smaller. This illustrates that to double your precision (halve the CI width), you must **quadruple** your sample size.
            """)
        with tabs[1]:
            st.markdown("""
            ##### Glossary of Core Concepts
            - **Population:** The entire group that you want to draw conclusions about (e.g., all possible measurements from a process). In reality, the true population parameters (like the mean) are unknown.
            - **Sample:** A specific group of individuals that you will collect data from (e.g., the 30 measurements you took today).
            - **Sampling Distribution:** The theoretical probability distribution of a statistic (like the sample mean) obtained through a large number of samples drawn from a specific population.
            - **Standard Error:** The standard deviation of the sampling distribution. It measures the precision of the sample statistic as an estimate of the population parameter.
            - **Confidence Level:** The percentage of all possible samples that can be expected to include the true population parameter within the calculated interval (e.g., 95%). This is a property of the *procedure*, not a single interval.
            """)
        with tabs[2]:
            st.error("""
            🔴 **THE INCORRECT (Bayesian) INTERPRETATION:**
            *"Based on my sample, there is a 95% probability that the true mean is in this interval [X, Y]."*
            
            This is wrong because in the frequentist view, the true mean is a fixed, unknown constant. It is either in our specific interval or it is not. The probability is 1 or 0.
            """)
            st.success("""
            🟢 **THE CORRECT (Frequentist) INTERPRETATION:**
            *"We are 95% confident that the interval [X, Y] contains the true mean."*
            
            The full meaning is: *"This interval was constructed using a procedure that, when repeated many times, will produce intervals that capture the true mean 95% of the time."* Our confidence is in the **procedure**, not in any single outcome.
            """)
        with tabs[3]:
            st.markdown("""
            #### Historical Context: The Great Debate
            **The Problem:** In the early 20th century, the field of statistics was in turmoil. The giant of the field, **Sir Ronald A. Fisher**, had developed a concept called "fiducial inference" to create intervals, but it was complex and controversial. A new school of thought, led by **Jerzy Neyman** and **Egon Pearson**, was emerging, focused on a more rigorous, decision-theoretic framework. They needed a way to define an interval estimate that was objective, mathematically sound, and had a clear, long-run performance guarantee.

            **The 'Aha!' Moment:** In a landmark 1937 paper, Neyman introduced the **confidence interval**. His revolutionary idea was to shift the probabilistic statement away from the fixed, unknown parameter (which a frequentist believes has no probability distribution) and onto the **procedure used to create the interval**. He defined the "95% confidence" not as a statement about a single interval, but as a guarantee about the long-run success rate of the method used to generate it.
            
            **The Impact:** This clever reframing was a triumph of the Neyman-Pearson school. It provided a practical and logically consistent solution that was easy to compute and understand (even if often misinterpreted!). Fisher fiercely debated against it for the rest of his life, but Neyman's confidence interval won out, becoming the dominant and most widely taught paradigm for interval estimation in the world. It is the bedrock on which most of the statistical tests in this toolkit are built.
            """)
            st.markdown("#### Mathematical Basis")
            st.markdown("The general form for a two-sided confidence interval is:")
            st.latex(r"\text{Point Estimate} \pm (\text{Critical Value} \times \text{Standard Error})")
            st.markdown("""
            - **Point Estimate:** Our single best guess for the population parameter, calculated from the sample (e.g., the sample mean, `x̄`).
            - **Standard Error:** The standard deviation of the sampling distribution of the point estimate (e.g., `s/√n`). It measures the typical error we expect in our point estimate due to random sampling.
            - **Critical Value:** A multiplier determined by our desired confidence level and the underlying distribution. For a CI for the mean with an unknown population standard deviation, this is a t-score from the t-distribution with `n-1` degrees of freedom.
            For a 95% CI for the mean, the formula is:
            """)
            st.latex(r"\bar{x} \pm t_{(0.975, n-1)} \cdot \frac{s}{\sqrt{n}}")
        with tabs[4]:
            st.markdown("""
            While not a standalone requirement, the correct application and interpretation of confidence intervals are a **foundational statistical principle** that underpins compliance with numerous guidelines.
            - **ICH Q2(R1) - Validation of Analytical Procedures:** Used to establish confidence intervals for key parameters like the slope and intercept in linearity studies.
            - **FDA Process Validation Guidance:** Used to set confidence bounds on process parameters and quality attributes during Process Performance Qualification (PPQ).
            - **21 CFR Part 211:** Implicitly required for demonstrating statistical control and for the "appropriate statistical quality control criteria" mentioned in §211.165.
            """)
# ======================================== 3. CONFIDENCE INTERVALS FOR PROPORTIONS ===============================================================
def render_proportion_cis():
    """Renders the comprehensive, interactive module for comparing binomial confidence intervals."""
    st.markdown("""
    #### Purpose & Application: Choosing the Right Statistical Ruler for Pass/Fail Data
    **Purpose:** To compare and contrast different statistical methods for calculating a confidence interval for pass/fail (binomial) data. This tool demonstrates that the choice of statistical method is not trivial and can have a significant impact on the final conclusion, especially in common validation scenarios with high success rates and limited sample sizes.
    
    **Strategic Application:** This is a critical decision point when writing a validation protocol or a statistical analysis plan. When you must prove that a process meets a high reliability target (e.g., >99% success rate) based on a limited sample, the statistical interval you choose determines your ability to make that claim. Using an overly conservative interval (like Clopper-Pearson) may require a much larger sample size, increasing project costs, while using an unreliable one (like the classic Wald interval) can lead to a false sense of confidence and significant compliance risk.
    """)
    
    st.info("""
    **Interactive Demo:** You are the Validation Scientist.
    1.  Use the top two sliders to simulate a validation run with a specific number of samples and successes.
    2.  Pay close attention to the **"Failure Scenarios"**—what happens when you have **few samples** and **zero or one failures**. This is where the methods differ most dramatically.
    3.  Use the **Bayesian Prior** sliders to see how prior knowledge (e.g., from R&D) can be formally incorporated to produce a more informed interval.
    """)

    with st.sidebar:
        st.subheader("Confidence Interval Controls")
        st.markdown("**Experimental Results**")
        n_samples_slider = st.slider("Number of Validation Samples (n)", 10, 200, 50, 5, help="The total number of samples tested in your validation run. Note how interval widths shrink as you increase n.")
        n_failures_slider = st.slider("Number of Failures Observed", 0, n_samples_slider, 1, 1, help="The number of non-conforming or failing results. Scenarios with 0 or 1 failures are common and where the choice of CI method is most critical.")
        n_successes = n_samples_slider - n_failures_slider
        
        st.markdown("**Bayesian Prior Belief**")
        st.write("Simulate prior knowledge (e.g., from R&D studies).")
        prior_successes = st.slider("Prior Successes (α)", 0, 100, 10, 1, help="The number of successes in your 'imaginary' prior data. A higher number represents a stronger prior belief in a high success rate.")
        prior_failures = st.slider("Prior Failures (β)", 0, 100, 1, 1, help="The number of failures in your 'imaginary' prior data. Even a small number here can make the model more conservative.")

    fig, metrics = plot_proportion_cis(n_samples_slider, n_successes, prior_successes, prior_failures)
    
    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Key Interval Results")
        st.metric("Observed Success Rate", f"{n_successes/n_samples_slider if n_samples_slider > 0 else 0:.2%}")
        st.markdown(f"**Wilson Score CI:** `[{metrics['Wilson Score'][0]:.3f}, {metrics['Wilson Score'][1]:.3f}]`")
        st.markdown(f"**Clopper-Pearson (Exact) CI:** `[{metrics['Clopper–Pearson (Exact)'][0]:.3f}, {metrics['Clopper–Pearson (Exact)'][1]:.3f}]`")
        st.markdown(f"**Custom Bayesian CI:** `[{metrics['Bayesian with Custom Prior'][0]:.3f}, {metrics['Bayesian with Custom Prior'][1]:.3f}]`")

    st.divider()
    st.subheader("Deeper Dive")
    tabs = st.tabs(["💡 Key Insights", "📋 Glossary", "✅ The Golden Rule", "📖 Theory & History", "🏛️ Regulatory & Compliance"])
    with tabs[0]:
        st.markdown("""
        **Interpreting the Comparison:**
        - **The Wald Interval's Failure:** Set the "Number of Failures" to 0. Notice that the Wald interval (red) collapses to a width of zero. This is a nonsensical result that falsely claims perfect certainty from a finite sample. This is why it is blacklisted in modern statistical practice.
        - **Conservatism vs. Accuracy:** The **Clopper-Pearson** interval is often the widest. It is guaranteed to meet the 95% confidence level, but this guarantee makes it conservative (less powerful). The **Wilson Score** interval is slightly narrower and has better average performance, making it a common "best practice" choice for frequentist analysis.
        - **The Power of Priors:** Adjust the **Bayesian Prior** sliders. If you have a strong prior belief in a high success rate (e.g., 99 successes, 1 failure), notice how the "Bayesian with Custom Prior" interval is "pulled" towards that high rate, resulting in a higher lower bound than other methods. This can be a powerful way to reduce sample sizes, provided the prior is well-justified.
        
        **The Strategic Insight:** The choice of interval method directly impacts your ability to meet a pre-defined acceptance criterion. For a result of 49/50 successes (98%), the lower bound of the Wilson interval is 0.888. If your acceptance criterion is ">90% success," you fail. But for the same data, if you had a strong prior, the Bayesian lower bound might be >0.90, allowing you to pass.
        """)
    with tabs[1]:
        st.markdown("""
        ##### Glossary of CI Methods for Proportions
        - **Wald Interval:** The simplest method, based on the normal approximation. **Known to perform very poorly** with small `n` or extreme proportions and should be avoided in GxP settings.
        - **Wilson Score Interval:** A more complex method also based on the normal approximation, but it inverts the score test, giving it excellent performance across all conditions. Often the recommended default for frequentist analysis.
        - **Agresti–Coull Interval:** A simplified version of the Wilson interval that is easier to compute by hand (it adds 2 successes and 2 failures before calculating a Wald interval). Performs similarly to Wilson but is slightly more conservative.
        - **Clopper–Pearson (Exact) Interval:** A method based directly on the binomial distribution. It guarantees that the true coverage will be *at least* 95%, but is often too wide (conservative), making it harder to pass acceptance criteria.
        - **Jeffreys Interval (Bayesian):** A Bayesian credible interval using a non-informative prior (`Beta(0.5, 0.5)`). It has excellent frequentist properties and is a good choice when no prior knowledge is available.
        - **Bayesian Credible Interval:** An interval derived from the posterior distribution. It represents a range where there is a 95% probability that the true parameter lies. Its location and width are influenced by both the data and the chosen prior.
        - **Bootstrapped CI:** A computational method that simulates thousands of new datasets by resampling from the original data. It does not rely on statistical assumptions, but can be unstable with very small sample sizes.
        """)
    with tabs[2]:
        st.error("""🔴 **THE INCORRECT APPROACH: The "Textbook Default" Fallacy**
An analyst uses the simple Wald interval because it's the first one taught in many introductory statistics courses. When validating a process with a 100% success rate in 50 samples (50/50), the Wald interval is `[1.0, 1.0]`, leading them to claim with 95% confidence that the true success rate is exactly 100%.
- **The Flaw:** This is a statistically indefensible claim of absolute certainty from a finite sample. The Wilson Score interval for the same data is `[0.93, 1.0]`, which correctly communicates that the true rate could plausibly be as low as 93%.""")
        st.success("""🟢 **THE GOLDEN RULE: Match the Method to the Risk and Justify It**
The choice of confidence interval method is a risk-based decision that must be pre-specified and justified in the validation protocol.
1.  **For General Use (Frequentist):** The **Wilson Score interval** is the recommended default, providing the best balance of accuracy and interval width.
2.  **For Absolute Guarantee:** When you absolutely must guarantee that your confidence level is not underestimated (e.g., for a critical safety claim), the **Clopper-Pearson (Exact) interval** is the most conservative and defensible choice.
3.  **When Prior Data Exists:** When you have strong, justifiable prior knowledge (e.g., from extensive R&D data), a **Bayesian credible interval** is the most powerful and efficient approach, but the prior must be explicitly defined and justified in the protocol.
**Never use the Wald interval in a formal validation report.**""")
    with tabs[3]:
        st.markdown("""
        #### Historical Context: Correcting a Century-Old Problem
        The problem of estimating an interval for a proportion seems simple, but its history is complex. The standard **Wald interval**, based on the work of Abraham Wald in the 1930s, was easy to teach and compute, so it became the default method in textbooks for decades. However, its poor performance was well-known to statisticians. A famous 1998 paper by Brown, Cai, and DasGupta, titled "Interval Estimation for a Binomial Proportion," systematically exposed the severe flaws of the Wald interval to a wider audience, calling it "persistently chaotic."
        
        The irony is that the superior solutions were much older. The **Wilson Score interval** was developed by Edwin Bidwell Wilson in **1927**, and the **Clopper-Pearson interval** was developed in **1934**. For much of the 20th century, these more accurate but computationally intensive methods were overlooked in favor of the simpler Wald interval.
        
        The "rediscovery" of these superior methods in the 1990s, driven by increased computing power and influential papers like Brown et al.'s, led to a major shift in statistical practice. Today, modern statistical software and guidelines strongly advocate for the use of Wilson, Clopper-Pearson, or other improved methods, and the simple Wald interval is largely considered obsolete for serious analysis.
        """)
    with tabs[4]:
        st.markdown("""
        The calculation of a statistically valid confidence interval for a proportion is a fundamental requirement in many validation activities where the outcome is binary (pass/fail, concordant/discordant, etc.).
        - **FDA Process Validation Guidance (Stage 2 - PPQ):** When validating a process attribute that is pass/fail (e.g., visual inspection for cosmetic defects), a confidence interval on the pass rate is used to demonstrate that the process can consistently produce conforming product. Using a robust interval is critical for making a high-confidence claim.
        - **Analytical Method Validation (ICH Q2):** For qualitative assays (e.g., a limit test), validation requires demonstrating a high rate of correct detections. For concordance studies comparing a new method to a reference, a confidence interval on the concordance rate is a key performance metric.
        - **21 CFR 820.250 (Statistical Techniques):** This regulation for medical devices explicitly requires that "Where appropriate, each manufacturer shall establish and maintain procedures for identifying valid statistical techniques..." Using a robust interval like the Wilson Score instead of the flawed Wald interval is a direct fulfillment of this requirement.
        """)
# ==================================================================================== 4. CORE VALIDATION PARAMETERS ===============================================================
def render_core_validation_params():
    """Renders the INTERACTIVE module for core validation parameters."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To formally establish the fundamental performance characteristics of an analytical method as required by global regulatory guidelines like ICH Q2(R1). This module deconstructs the "big three" pillars of method validation:
    - **🎯 Accuracy (Bias):** How close are your measurements to the *real* value?
    - **🏹 Precision (Random Error):** How consistent are your measurements with each other?
    - **🔬 Specificity (Selectivity):** Can your method find the target analyte in a crowded room, ignoring all the imposters?

    **Strategic Application:** These parameters are the non-negotiable pillars of any formal assay validation report. A weakness in any of these three areas is a critical deficiency that can lead to rejected submissions or flawed R&D conclusions. **Use the sliders in the sidebar to simulate different error types and see their impact on the plots.**
    """)
    
    st.info("""
    **Interactive Demo:** Now, when you navigate to the "Core Validation Parameters" tool, you will see a new set of dedicated sliders below. Changing these sliders will instantly update the three plots, allowing you to build a powerful, hands-on intuition for these critical concepts.
    """)
    
    # --- Sidebar controls for this specific module ---
    with st.sidebar:
        st.subheader("Core Validation Controls")
        bias_slider = st.slider(
            "🎯 Systematic Bias (%)", 
            min_value=-10.0, max_value=10.0, value=1.5, step=0.5,
            help="Simulates a constant positive or negative bias in the accuracy study. Watch the box plots shift."
    )
        repeat_cv_slider = st.slider(
            "🏹 Repeatability %CV", 
            min_value=0.5, max_value=10.0, value=1.5, step=0.5,
            help="Simulates the best-case random error (intra-assay precision). Watch the 'Repeatability' violin width."
    )
    # Ensure intermediate precision is always worse than or equal to repeatability
        intermed_cv_slider = st.slider(
            "🏹 Intermediate Precision %CV", 
            min_value=repeat_cv_slider, max_value=20.0, value=max(2.5, repeat_cv_slider), step=0.5,
            help="Simulates real-world random error (inter-assay). A large gap from repeatability indicates poor robustness."
    )
        interference_slider = st.slider(
            "🔬 Interference Effect (%)", 
            min_value=-20.0, max_value=20.0, value=8.0, step=1.0,
            help="Simulates an interferent that falsely increases (+) or decreases (-) the analyte signal."
    )
    
    # Generate plots using the slider values
    fig1, fig2, fig3 = plot_core_validation_params(
        bias_pct=bias_slider, 
        repeat_cv=repeat_cv_slider, 
        intermed_cv=intermed_cv_slider, 
        interference_effect=interference_slider
    )
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)
        st.plotly_chart(fig3, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "📋 Glossary", "✅ The Golden Rule", "📖 Theory & History", "🏛️ Regulatory & Compliance"])
        
        with tabs[0]:
            st.info("Play with the sliders in the sidebar to see how different sources of error affect the results!")
            st.markdown("""
            - **Accuracy Plot:** As you increase the **Systematic Bias** slider, watch the center of the box plots drift away from the black 'True Value' lines. This visually demonstrates what bias looks like.
            
            - **Precision Plot:** The **%CV sliders** control the width (spread) of the violin plots. Notice that Intermediate Precision must always be equal to or worse (wider) than Repeatability. A large gap between the two signals that the method is not robust to day-to-day changes.
            
            - **Specificity Plot:** The **Interference Effect** slider directly moves the 'Analyte + Interferent' box plot. A perfect assay would have this slider at 0%, making the two boxes identical. A large effect, positive or negative, indicates a failed specificity study.

            **The Core Strategic Insight:** This simulation shows that validation is a process of hunting for and quantifying different types of error. Accuracy is about finding *bias*, Precision is about characterizing *random error*, and Specificity is about eliminating *interference error*.
            """)
            
        with tabs[1]:
            st.markdown("""
            ##### Glossary of Validation Parameters
            - **Accuracy (Bias):** The closeness of the average test result to the true value. It measures **systematic error**. High accuracy means low bias.
            - **Precision (%CV):** The closeness of agreement (degree of scatter) between a series of measurements. It measures **random error**. High precision means low random error.
            - **Repeatability (Intra-assay precision):** Precision under the same operating conditions over a short interval of time (e.g., one analyst, one instrument, one day).
            - **Intermediate Precision (Inter-assay precision):** Precision within the same laboratory, but under different conditions (e.g., different analysts, different days, different instruments).
            - **Specificity:** The ability to assess the analyte unequivocally in the presence of components which may be expected to be present (e.g., impurities, degradants, matrix components).
            - **Interference:** A type of error caused by a substance in the sample matrix that falsely alters the assay's response to the target analyte.
            """)

        with tabs[2]:
            st.error("""
            🔴 **THE INCORRECT APPROACH: "Validation Theater"**
            The goal of validation is to get the protocol to pass by any means necessary.
            
            - *"My precision looks bad, so I'll have my most experienced 'super-analyst' run the experiment to guarantee a low %CV."*
            - *"The method failed accuracy at the low concentration. I'll just change the reportable range to exclude that level."*
            - *"I'll only test for interference from things I know won't be a problem and ignore the complex sample matrix."*
            
            This approach treats validation as a bureaucratic hurdle. It produces a method that is fragile, unreliable in the real world, and a major compliance risk.
            """)
            st.success("""
            🟢 **THE GOLDEN RULE: Rigorously Prove "Fitness for Purpose"**
            The goal of validation is to **honestly and rigorously challenge the method to prove it is robust and reliable for its specific, intended analytical application.**
            
            - This means deliberately including variability (different analysts, days, instruments) to prove the method can handle it.
            - It means understanding and documenting *why* a method fails at a certain level, not just hiding the failure.
            - It means demonstrating specificity in the actual, messy matrix the method will be used for.
            
            This approach builds a truly robust method that generates trustworthy data, ensuring product quality and patient safety.
            """)

        with tabs[3]:
            st.markdown("""
            #### Historical Context & Origin
            Before the 1990s, a pharmaceutical company wishing to market a new drug globally had to prepare different, massive submission packages for each region (USA, Europe, Japan), each with slightly different technical requirements for method validation. This created enormous, costly, and scientifically redundant work.
            
            In 1990, the **International Council for Harmonisation (ICH)** was formed, bringing together regulators and industry to create a single set of harmonized guidelines. The **ICH Q2(R1) guideline, "Validation of Analytical Procedures,"** is the direct result. It is the global "bible" for this topic, and the parameters of Accuracy, Precision, and Specificity form its core. Adhering to ICH Q2(R1) ensures your data is acceptable to major regulators worldwide.
            
            #### Mathematical Basis
            The validation report is a statistical argument built on quantitative metrics.
            """)
            st.markdown("**Accuracy is measured by Percent Recovery:**")
            st.latex(r"\% \text{Recovery} = \frac{\text{Mean Experimental Value}}{\text{Known True Value}} \times 100\%")
            
            st.markdown("**Precision is measured by Percent Coefficient of Variation (%CV):**")
            st.latex(r"\% \text{CV} = \frac{\text{Standard Deviation (SD)}}{\text{Mean}} \times 100\%")
            
            st.markdown("""
            **Specificity is often assessed via Hypothesis Testing:** A Student's t-test compares the means of the "Analyte Only" and "Analyte + Interferent" groups. The null hypothesis ($H_0$) is that the means are equal. A high p-value (e.g., > 0.05) means we fail to reject $H_0$, providing evidence that the interferent has no significant effect.
            """)
        with tabs[4]:
            st.markdown("""
            The concepts of Accuracy, Precision, and Specificity are the absolute core of analytical method validation as defined by global regulators.
            - **ICH Q2(R1) - Validation of Analytical Procedures:** This is the primary global guideline that explicitly defines these parameters and provides methodologies for their assessment.
            - **FDA Guidance for Industry - Analytical Procedures and Methods Validation:** The FDA's specific guidance, which is harmonized with ICH Q2(R1).
            - **USP General Chapter <1225> - Validation of Compendial Procedures:** Provides detailed requirements for validation within the United States Pharmacopeia framework.
            """)
#====================================================== 5. LOD & LOQ =========================================================================
def render_lod_loq():
    """Renders the INTERACTIVE module for Limit of Detection & Quantitation."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To formally establish the absolute lower performance boundaries of a quantitative assay. It determines the lowest analyte concentration an assay can reliably **detect (LOD)** and the lowest concentration it can reliably and accurately **quantify (LOQ)**.
    
    **Strategic Application:** This is a mission-critical parameter for any assay used to measure trace components, such as impurities in a drug product or biomarkers for early-stage disease diagnosis. **Use the sliders in the sidebar to simulate how assay sensitivity and noise impact the final LOD and LOQ.**
    """)
    
    st.info("""
    **Interactive Demo:** Now, when you select the "LOD & LOQ" tool, a new set of dedicated sliders will appear below. You can dynamically change the assay's slope and noise to see in real-time how these fundamental characteristics drive the final LOD and LOQ results.
    """)
    
    # --- Sidebar controls for this specific module ---
    st.subheader("LOD & LOQ Controls")
    slope_slider = st.slider(
        "📈 Assay Sensitivity (Slope)", 
        min_value=0.005, max_value=0.1, value=0.02, step=0.005, format="%.3f",
        help="How much the signal increases per unit of concentration. A steeper slope (higher sensitivity) is better."
    )
    noise_slider = st.slider(
        "🔇 Baseline Noise (SD)", 
        min_value=0.001, max_value=0.05, value=0.01, step=0.001, format="%.3f",
        help="The inherent random noise of the assay at zero concentration. A lower noise floor is better."
    )
    
    # Generate plots using the slider values
    fig, lod_val, loq_val = plot_lod_loq(slope=slope_slider, baseline_sd=noise_slider)
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "📋 Glossary", "✅ Acceptance Criteria", "📖 Theory & History", "🏛️ Regulatory & Compliance"])
        with tabs[0]:
            st.metric(label="📈 KPI: Limit of Quantitation (LOQ)", value=f"{loq_val:.2f} units", help="The lowest concentration you can report with confidence in the numerical value.")
            st.metric(label="💡 Metric: Limit of Detection (LOD)", value=f"{lod_val:.2f} units", help="The lowest concentration you can reliably claim is 'present'.")
            st.info("Play with the sliders in the sidebar to see how assay parameters affect the results!")
            st.markdown("""
            - **Increase `Assay Sensitivity (Slope)`:** As the slope gets steeper, the LOD and LOQ values get **lower (better)**. A highly sensitive assay needs very little analyte to produce a strong signal that can overcome the noise.
            - **Increase `Baseline Noise (SD)`:** As the noise floor of the assay increases, the LOD and LOQ values get **higher (worse)**. It becomes much harder to distinguish a true low-level signal from random background fluctuations.

            **The Core Strategic Insight:** The sensitivity of an assay is a direct battle between its **signal-generating power (Slope)** and its **inherent noise (SD)**. The LOD and LOQ are simply the statistical formalization of this signal-to-noise ratio.
            """)
        with tabs[1]:
            st.markdown("""
            ##### Glossary of Sensitivity Terms
            - **Limit of Blank (LOB):** The highest measurement result that is likely to be observed for a blank sample. It defines the "noise floor" of the assay.
            - **Limit of Detection (LOD):** The lowest concentration of analyte that can be reliably detected above the LOB, but not necessarily quantified with acceptable precision and accuracy.
            - **Limit of Quantitation (LOQ):** The lowest concentration of analyte that can be reliably quantified with a pre-defined level of precision and accuracy. This is the lower boundary of the assay's reportable range.
            - **Slope (Sensitivity):** In a calibration curve, the slope represents the change in analytical signal per unit change in analyte concentration. A steeper slope generally indicates a more sensitive assay.
            """)
        with tabs[2]:
            st.markdown("- The primary, non-negotiable criterion is that the experimentally determined **LOQ must be ≤ the lowest concentration that the assay is required to measure** for its specific application (e.g., a release specification for an impurity).")
            st.markdown("- For a concentration to be formally declared the LOQ, it must be experimentally confirmed. This typically involves analyzing 5-6 independent samples at the claimed LOQ concentration and demonstrating that they meet pre-defined criteria for precision and accuracy (e.g., **%CV < 20% and %Recovery between 80-120%** for a bioassay).")
            st.warning("""
            **The LOB, LOD, and LOQ Hierarchy: A Critical Distinction**
            A full characterization involves three distinct limits:
            - **Limit of Blank (LOB):** The highest measurement expected from a blank sample.
            - **Limit of Detection (LOD):** The lowest concentration whose signal is statistically distinguishable from the LOB.
            - **Limit of Quantitation (LOQ):** The lowest concentration meeting precision/accuracy requirements, which is almost always higher than the LOD.
            """)
        with tabs[3]:
            st.markdown("""
            #### Historical Context & Origin
            The need to define analytical sensitivity is old, but definitions were inconsistent until the **International Council for Harmonisation (ICH)** guideline **ICH Q2(R1)** harmonized the methodologies. This work was heavily influenced by the statistical framework established by **Lloyd Currie at NIST** in his 1968 paper, which established the clear, hypothesis-testing basis for the modern LOB/LOD/LOQ hierarchy.

            #### Mathematical Basis
            This method is built on the relationship between the assay's signal, its sensitivity (Slope, S), and its noise (standard deviation, σ).
            """)
            st.latex(r"LOD \approx \frac{3.3 \times \sigma}{S}")
            st.latex(r"LOQ \approx \frac{10 \times \sigma}{S}")
            st.markdown("The factor of 10 for LOQ is the standard convention that typically yields a precision of roughly 10% CV for a well-behaved assay.")
        with tabs[4]:
            st.markdown("""
            The determination of detection and quantitation limits is a mandatory part of validating quantitative assays for impurities or trace components.
            - **ICH Q2(R1) - Validation of Analytical Procedures:** Explicitly lists "Quantitation Limit" and "Detection Limit" as key validation characteristics and provides the statistical methodologies (e.g., based on signal-to-noise or standard deviation of the response and the slope).
            - **USP General Chapter <1225>:** Mirrors the requirements of ICH Q2(R1) for the validation of analytical procedures.
            """)
#====================================================== 6.LINEARITY & RANGE =========================================================================
def render_linearity():
    """Renders the INTERACTIVE module for Linearity analysis."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To verify that an assay's response is directly and predictably proportional to the known concentration of the analyte across its entire intended operational range.
    
    **Strategic Application:** This is a cornerstone of quantitative assay validation. A method exhibiting non-linearity might be accurate at a central control point but dangerously inaccurate at the specification limits. **Use the sliders in the sidebar to simulate different types of non-linear behavior and error.**
    """)
    
    st.info("""
    **Interactive Demo:** Use the sliders at the bottom of the sidebar to simulate different error types. The **Residual Plot** is your most important diagnostic tool! A "funnel" shape indicates proportional error, and a "U" shape indicates curvature. When you see a funnel, try switching to the WLS model to see how it can provide a better fit.
    """)
    
    # --- Sidebar controls for this specific module ---
    with st.sidebar:
        st.subheader("Linearity Controls")
        curvature_slider = st.slider("🧬 Curvature Effect", -5.0, 5.0, -1.0, 0.5,
            help="Simulates non-linearity. A negative value creates saturation at high concentrations. Zero is perfectly linear.")
        random_error_slider = st.slider("🎲 Random Error (Constant SD)", 0.1, 5.0, 1.0, 0.1,
            help="The baseline random noise of the assay, constant across all concentrations.")
        proportional_error_slider = st.slider("📈 Proportional Error (% of Conc.)", 0.0, 5.0, 2.0, 0.25,
            help="Error that increases with concentration. This creates a 'funnel' or 'megaphone' shape in the residual plot.")
        
        # --- NEW TOGGLE SWITCH ADDED HERE ---
        st.markdown("---")
        st.markdown("**Regression Model**")
        wls_toggle = st.toggle("Use Weighted Least Squares (WLS)", value=False,
            help="Activate WLS to give less weight to high-concentration points. Ideal for correcting the 'funnel' shape caused by proportional error.")
    
    # Generate plots using the slider values and the new toggle value
    fig, model = plot_linearity(
        curvature=curvature_slider,
        random_error=random_error_slider,
        proportional_error=proportional_error_slider,
        use_wls=wls_toggle # Pass the toggle state to the plotting function
    )
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights",  "📋 Glossary", "✅ Acceptance Criteria", "📖 Theory & History", "🏛️ Regulatory & Compliance"])
        with tabs[0]:
            st.metric(label="📈 KPI: R-squared (R²)", value=f"{model.rsquared:.4f}", help="Indicates the proportion of variance explained by the model. Note how a high R² can hide clear non-linearity!")
            st.metric(label="💡 Metric: Slope", value=f"{model.params['Nominal']:.3f}", help="Ideal = 1.0.")
            st.metric(label="💡 Metric: Y-Intercept", value=f"{model.params['const']:.2f}", help="Ideal = 0.0.")
            
            st.markdown("""
            - **The Residual Plot is Key:** This is the most sensitive diagnostic tool.
                - Add **Curvature**: Notice the classic "U-shape" or "inverted U-shape" that appears. This is a dead giveaway that your straight-line model is wrong.
                - Add **Proportional Error**: Watch the residuals form a "funnel" shape (heteroscedasticity). This means OLS is no longer valid. **Activate the WLS toggle in the sidebar** to see how a weighted model correctly handles this error structure.
            
            **The Core Strategic Insight:** A high R-squared is **not sufficient** to prove linearity. You must visually inspect the residual plot for hidden patterns. The residual plot tells the true story of your model's fit.
            """)
        with tabs[1]:
            st.markdown("""
            ##### Glossary of Linearity Terms
            - **Linearity:** The ability of an analytical procedure to obtain test results which are directly proportional to the concentration of analyte in the sample.
            - **Range:** The interval between the upper and lower concentration of analyte for which the procedure has demonstrated a suitable level of precision, accuracy, and linearity.
            - **Residuals:** The vertical distance between an observed data point and the fitted regression line. Analyzing patterns in residuals is the best way to diagnose non-linearity.
            - **R-squared (R²):** A statistical measure of how close the data are to the fitted regression line. While a high R² is necessary, it is not sufficient to prove linearity.
            - **Weighted Least Squares (WLS):** A regression method used when the variance of the errors is not constant (heteroscedasticity). It gives less weight to less precise (typically high-concentration) data points.
            """)
        with tabs[2]:
            st.markdown("These criteria are defined in the validation protocol and must be met to declare the method linear.")
            st.markdown("- **R-squared (R²):** Typically > **0.995**, but for high-precision methods (e.g., HPLC), > **0.999** is often required.")
            st.markdown("- **Slope & Intercept:** The 95% confidence intervals for the slope and intercept should contain 1.0 and 0, respectively.")
            st.markdown("- **Residuals:** There should be no obvious pattern or trend. A formal **Lack-of-Fit test** can be used for objective proof (requires true replicates at each level).")
            st.markdown("- **Recovery:** The percent recovery at each concentration level must fall within a pre-defined range (e.g., 80% to 120% for bioassays).")

        with tabs[3]:
            st.markdown("""
            #### Historical Context & Origin
            The mathematical engine is **Ordinary Least Squares (OLS) Regression**, developed independently by **Adrien-Marie Legendre (1805)** and **Carl Friedrich Gauss (1809)**. The genius of OLS is that it finds the one line that **minimizes the sum of the squared vertical distances (the "residuals")** between the data points and the line.
            
            However, OLS relies on a key assumption: that the variance of the errors is constant at all levels of X (homoscedasticity). In many biological and chemical assays, this is not true; variability often increases with concentration (heteroscedasticity). **Weighted Least Squares (WLS)** is the classical solution to this problem, where each point is weighted by the inverse of its variance, giving more influence to the more precise, low-concentration points.
            """)
            st.markdown("#### Mathematical Basis")
            st.markdown("The goal is to fit a simple linear model:")
            st.latex("y = \\beta_0 + \\beta_1 x + \\epsilon")
            st.markdown("""
            - **OLS** finds the `β` values that minimize: `Σ(yᵢ - ŷᵢ)²`
            - **WLS** finds the `β` values that minimize: `Σwᵢ(yᵢ - ŷᵢ)²`, where `wᵢ` is the weight for the i-th observation, typically `1/σ²ᵢ`.
            """)
        with tabs[4]:
            st.markdown("""
            Linearity is a fundamental characteristic required for all quantitative analytical methods.
            - **ICH Q2(R1) - Validation of Analytical Procedures:** Mandates the evaluation of Linearity and Range for quantitative tests. It specifies that a linear relationship should be evaluated across the range of the analytical procedure.
            - **FDA Guidance for Industry:** Recommends a minimum of five concentration levels to establish linearity and emphasizes the importance of visual inspection of the data and analysis of residuals.
            - **USP General Chapter <1225>:** Requires the statistical evaluation of linearity, including the calculation of the correlation coefficient, y-intercept, and slope of the regression line.
            """)
#====================================================== 7. NON-LINEAR REGRESSION (4PL/5PL) =========================================================================
def render_4pl_regression():
    """Renders the INTERACTIVE module for 4-Parameter Logistic (4PL) regression."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To accurately model the characteristic sigmoidal (S-shaped) dose-response relationship found in most immunoassays (e.g., ELISA) and biological assays.
    
    **Strategic Application:** This is the workhorse model for potency assays and immunoassays. The 4PL model allows for the accurate calculation of critical assay parameters like the EC50. **Use the sliders in the sidebar to control the "true" shape of the curve and see how the curve-fitting algorithm performs.**
    """)
    
    st.info("""
    **Interactive Demo:** Build your own "true" 4PL curves using the sliders. The **Residual Plot** is your key diagnostic: if you see a "funnel" shape (due to `Proportional Noise`), activate the **Weighted Fit (IRLS)** toggle to see how a more advanced algorithm can produce a better, more reliable fit.
    """)
    
    # --- Sidebar controls for this specific module ---
    with st.sidebar:
        st.subheader("4PL Curve Controls (True Values)")
        d_slider = st.slider("🅾️ Lower Asymptote (d)", 0.0, 0.5, 0.05, 0.01,
            help="The minimum signal response of the assay, typically the background signal at zero concentration. This defines the 'floor' of the curve.")
        a_slider = st.slider("🅰️ Upper Asymptote (a)", 1.0, 3.0, 1.5, 0.1,
            help="The maximum signal response of the assay at infinite concentration. This represents the saturation point or the 'ceiling' of the curve.")
        c_slider = st.slider("🎯 Potency / EC50 (c)", 1.0, 100.0, 10.0, 1.0,
            help="The concentration that produces a response halfway between the lower (d) and upper (a) asymptotes. A lower EC50 value indicates a more potent substance.")
        b_slider = st.slider("🅱️ Hill Slope (b)", 0.5, 5.0, 1.2, 0.1,
            help="Controls the steepness of the curve at its midpoint (the EC50). A steeper slope (higher value) often indicates a more sensitive assay in its dynamic range.")
        
        st.markdown("---")
        st.subheader("Noise Model Controls")
        noise_sd_slider = st.slider("🎲 Constant Noise (SD)", 0.0, 0.2, 0.05, 0.01,
            help="The baseline random noise, constant across all concentrations.")
        proportional_noise_slider = st.slider("📈 Proportional Noise (%)", 0.0, 5.0, 1.0, 0.5,
            help="Noise that increases with the signal. Creates a 'funnel' shape in the residuals (heteroscedasticity).")

        st.markdown("---")
        st.subheader("Fit Model Controls")
        irls_toggle = st.toggle("Use Weighted Fit (IRLS)", value=True,
            help="Activate Iteratively Reweighted Least Squares (IRLS). This is the correct model to use when proportional noise is present.")
    
    # Generate plots using the slider values
    fig, params, perr = plot_4pl_regression(
        a_true=a_slider, b_true=b_slider, c_true=c_slider, d_true=d_slider,
        noise_sd=noise_sd_slider, proportional_noise=proportional_noise_slider,
        use_irls=irls_toggle
    )
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        a_fit, b_fit, c_fit, d_fit = params
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History", "🏛️ Regulatory & Compliance"])
        
        with tabs[0]:
            # Display fitted parameters with their standard errors
            st.metric(label="🅰️ Fitted Upper Asymptote (a)", value=f"{a_fit:.3f} ± {perr[0]:.3f}")
            st.metric(label="🅱️ Fitted Hill Slope (b)", value=f"{b_fit:.3f} ± {perr[1]:.3f}")
            st.metric(label="🎯 Fitted EC50 (c)", value=f"{c_fit:.3f} ± {perr[2]:.2f} units")
            st.metric(label="🅾️ Fitted Lower Asymptote (d)", value=f"{d_fit:.3f} ± {perr[3]:.3f}")
            
            st.markdown("""
            **The Core Strategic Insight:** The 4PL curve is a complete picture of your assay's performance. The **residuals plot is your most important diagnostic tool**. A random scatter around zero means your model is a good fit. Any pattern (like a curve or funnel) indicates a problem with the model or the data weighting.
            """)
        with tabs[1]:
            st.markdown("""
            ##### Glossary of Bioassay Terms
            - **4PL (Four-Parameter Logistic) Model:** A type of non-linear regression model used to describe sigmoidal (S-shaped) dose-response curves.
            - **Upper Asymptote (a):** The maximum response of the assay at infinite concentration (the "ceiling").
            - **Lower Asymptote (d):** The response of the assay at zero concentration (the "floor" or background).
            - **EC50 / IC50 (c):** The concentration that gives a response halfway between the lower and upper asymptotes. It is the primary measure of a substance's **potency**.
            - **Hill Slope (b):** A parameter that describes the steepness of the curve at its midpoint (the EC50).
            - **Potency:** A measure of drug activity expressed in terms of the amount required to produce an effect of a given intensity. A lower EC50 means higher potency.
            """)
        with tabs[2]:
            st.error("""🔴 **THE INCORRECT APPROACH: "Force the Fit"**
- *"My R-squared is 0.999, so the fit must be perfect."* (R-squared is easily inflated and can hide significant lack of fit).
- *"The model doesn't fit a point well. I'll delete the outlier."* (Data manipulation without statistical justification).
- *"My residuals look like a funnel, but I'll ignore it and use standard least squares."* (This leads to incorrect parameter estimates and confidence intervals).""")
            st.success("""🟢 **THE GOLDEN RULE: Model the Biology, Weight the Variance**
- **Embrace the 'S' Shape:** Use a non-linear model for non-linear biological data. The 4PL is standard for a reason.
- **Weight Your Points:** Bioassay data is almost always heteroscedastic (non-constant variance). Use a weighted regression (like IRLS) to get the most accurate and reliable parameter estimates.
- **Inspect the Residuals:** The residuals must be visually random. Any pattern indicates your model is not correctly capturing the data's behavior.""")

        with tabs[3]:
            st.markdown("""
            #### Historical Context: Modeling Dose-Response
            **The Problem:** In the early 20th century, pharmacologists and biologists needed a mathematical way to describe the relationship between the dose of a substance and its biological response. This relationship was rarely linear; it typically showed a sigmoidal (S-shaped) curve, with a floor, a steep middle section, and a ceiling (saturation).

            **The 'Aha!' Moment:** The first major step was the **Hill Equation**, developed by physiologist Archibald Hill in 1910 to describe oxygen binding to hemoglobin. Later, A.J. Clark and others adapted these ideas into dose-response models. The 4-Parameter Logistic (4PL) model emerged as a flexible, robust, and empirically successful model for this type of data.

            **The Impact:** The proliferation of immunoassays like RIA and ELISA in the 1970s and 80s made the 4PL model a workhorse of the biotech industry. The development of accessible non-linear regression software, like that based on the **Levenberg-Marquardt algorithm**, made fitting these models routine. Today, it remains the standard model for most potency and immunogenicity assays.
            """)
            st.markdown("#### Mathematical Basis")
            st.markdown("The 4-Parameter Logistic function is a sigmoidal curve defined by:")
            st.latex(r"y = d + \frac{a - d}{1 + (\frac{x}{c})^b}")
            st.markdown("""
            - **`a`**: The upper asymptote (response at infinite concentration).
            - **`b`**: The Hill slope (steepness of the curve at the midpoint).
            - **`c`**: The EC50 or IC50 (concentration at 50% of the maximal response). This is often the primary measure of potency.
            - **`d`**: The lower asymptote (response at zero concentration).
            Since this equation is non-linear in its parameters, it cannot be solved directly with linear algebra. It must be fit using an iterative numerical optimization algorithm (like Levenberg-Marquardt) that finds the parameter values `(a,b,c,d)` that minimize the sum of squared errors between the data and the fitted curve.
            """)
        with tabs[4]:
            st.markdown("""
            While the 4PL model itself is a mathematical tool, its use is governed by guidelines on the validation of bioassays, where such non-linear responses are common.
            - **USP General Chapters <111>, <1032>, <1033>:** These chapters provide extensive guidance on the design and statistical analysis of biological assays. They discuss the importance of using an appropriate non-linear model to fit dose-response curves and assess parallelism.
            - **FDA Guidance on Bioanalytical Method Validation:** Stresses the need to characterize the full concentration-response relationship and use appropriate regression models (including weighted regression for heteroscedastic data).
            """)
#=======================================================================8. GAGE R&R / VCA   ========================================================================================
def render_gage_rr():
    """Renders the INTERACTIVE module for Gage R&R."""
    st.markdown("""
    #### Purpose & Application: The Voice of the Measurement
    **Purpose:** To rigorously quantify the inherent variability (error) of a measurement system itself. It answers the fundamental question: **"Is my measurement system a precision instrument, or a random number generator?"**
    
    **Strategic Application:** This is a non-negotiable gateway in any technology transfer or process validation. An unreliable measurement system creates a "fog of uncertainty," leading to two costly errors:
    -   **Type I Error (Producer's Risk):** Rejecting good product because of measurement error.
    -   **Type II Error (Consumer's Risk):** Accepting bad product because the measurement system couldn't detect the defect.
    A Gage R&R study provides the objective evidence that your measurement system is fit for purpose.
    """)
    
    st.info("""
    **Interactive Demo:** You are the Validation Lead. Use the sliders in the sidebar to simulate different sources of variation. Your goal is to achieve a **% Gage R&R < 10%**.
    - **`Part-to-Part Variation`**: The "true" variation you want to measure. A high value makes it *easier* to pass the Gage R&R.
    - **`Repeatability`**: The instrument's own noise. This is a primary driver of Gage R&R failure.
    - **`Operator Variation`**: Inconsistency between people. This is the other major driver of failure.
    - **`Operator-Part Interaction`**: A subtle effect where an operator's bias changes depending on the part being measured.
    """)
    
    with st.sidebar:
        st.subheader("Gage R&R Controls")
        part_sd_slider = st.slider("🏭 Part-to-Part Variation (SD)", 1.0, 10.0, 5.0, 0.5,
            help="The 'true' variation of the product. A well-designed study uses parts that span a wide range, making this value high.")
        repeat_sd_slider = st.slider("🔬 Repeatability / Instrument Noise (SD)", 0.1, 5.0, 1.5, 0.1,
            help="The inherent 'noise' of the instrument/assay. High values represent an imprecise measurement device.")
        operator_sd_slider = st.slider("👤 Operator-to-Operator Variation (SD)", 0.0, 5.0, 0.75, 0.25,
            help="The systematic bias between operators. High values represent poor training or inconsistent technique.")
        # --- NEW SLIDER ADDED HERE ---
        interaction_sd_slider = st.slider("🔄 Operator-Part Interaction (SD)", 0.0, 2.0, 0.5, 0.1,
            help="Simulates inconsistency where operators measure certain parts differently (e.g., struggling with smaller parts). This causes the operator mean lines to be non-parallel.")

    # Call the updated plot function with all four parameters
    fig, pct_rr, ndc = plot_gage_rr(
        part_sd=part_sd_slider, 
        repeatability_sd=repeat_sd_slider, 
        operator_sd=operator_sd_slider,
        interaction_sd=interaction_sd_slider
    )
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights",  "📋 Glossary", "✅ Acceptance Criteria", "📖 Theory & History", "🏛️ Regulatory & Compliance"])
        with tabs[0]:
            st.metric(label="📈 KPI: % Gage R&R", value=f"{pct_rr:.1f}%", delta="Lower is better", delta_color="inverse")
            st.metric(label="📊 KPI: Number of Distinct Categories (ndc)", value=f"{ndc}", help="How many distinct groups of parts the system can reliably distinguish. Must be ≥ 5.")

            st.markdown("""
            **Reading the Plots:**
            - **Main Plot (Left):** Now shows parts sorted by size. The colored lines represent each operator's average measurement for each part. If these lines are not parallel, it's a sign of **interaction**.
            - **Operator Plot (Top-Right):** Visualizes the overall bias between operators.
            - **Verdict (Bottom-Right):** The final bar chart. The colored bar (% Gage R&R) shows how much of the total observed variation is just measurement noise.

            **Core Insight:** A low % Gage R&R is achieved when measurement error is small *relative to* the true process variation. You can improve your Gage R&R by either reducing measurement error OR by testing it on parts that have a wider, more representative range of true variation.
            """)
        with tabs[1]:
            st.markdown("""
            ##### Glossary of MSA Terms
            - **Measurement System Analysis (MSA):** A formal statistical study to evaluate the total variation present in a measurement system.
            - **Gage R&R:** The combined estimate of a measurement system's Repeatability and Reproducibility. It quantifies the inherent variability of the measurement process itself.
            - **Repeatability (Equipment Variation):** The variation observed when the same operator measures the same part multiple times with the same device. It represents the inherent "noise" of the instrument.
            - **Reproducibility (Appraiser Variation):** The variation observed when different operators measure the same part using the same device. It represents the inconsistency between people.
            - **% Gage R&R:** The percentage of the total observed process variation that is consumed by measurement system error.
            - **Number of Distinct Categories (ndc):** An index that represents the number of distinct groups of parts the measurement system can reliably distinguish. A value ≥ 5 is considered acceptable.
            """)
        with tabs[2]:
            st.markdown("Acceptance criteria are derived from the **AIAG's Measurement Systems Analysis (MSA)** manual, the global standard.")
            st.markdown("- **< 10% Gage R&R:** The system is **acceptable**.")
            st.markdown("- **10% - 30% Gage R&R:** The system is **conditionally acceptable**, may be approved based on importance of application and cost. ")
            st.markdown("- **> 30% Gage R&R:** The system is **unacceptable** and must be improved.")
            st.error("""
            **The Part Selection Catastrophe**: The most common way to fail a Gage R&R is not bad math, but bad study design. If you select parts that are all very similar (low Part-to-Part variation), you are mathematically guaranteed to get a high % Gage R&R, even with a perfect instrument. **You must select parts that represent the full range of expected process variation.**
            """)
            
        with tabs[3]:
            st.markdown("""
            #### Historical Context: The Crisis that Forged a Standard
            **The Problem:** In the 1970s and 80s, the American automotive industry was in crisis, facing intense competition from Japanese manufacturers who had mastered statistical quality control. A major source of defects and waste was inconsistent measurement. A part might pass inspection at a supplier's factory but fail at the assembly plant simply because the two locations' measurement systems ("gages") didn't agree. There was no standardized way to qualify a measurement system.

            **The 'Aha!' Moment:** The "Big Three" US automakers—Ford, GM, and Chrysler—realized they couldn't solve this problem alone. They formed the **Automotive Industry Action Group (AIAG)** to create common quality standards for their entire supply chain. One of their most impactful creations was the **Measurement Systems Analysis (MSA)** manual, first published in 1990.
            
            **The Impact:** The MSA manual didn't invent Gage R&R, but it codified it into a simple, repeatable procedure that became the global standard. The critical evolution it championed was the move from older, less reliable methods to the **ANOVA (Analysis of Variance) method** as the preferred approach. The ANOVA method, pioneered by **Sir Ronald A. Fisher**, is statistically superior because it can correctly partition all sources of variation, including the crucial **interaction effect** between operators and parts (e.g., if one operator struggles to measure small parts specifically). This rigorous approach became the benchmark for quality-driven industries worldwide, from aerospace to pharmaceuticals.
            """)
            st.markdown("#### Mathematical Basis")
            st.markdown("The core idea is to partition the total observed variation into its components. The fundamental equation is:")
            st.latex(r"\sigma^2_{\text{Total}} = \sigma^2_{\text{Process}} + \sigma^2_{\text{Measurement System}}")
            st.markdown("The measurement system variation is further broken down:")
            st.latex(r"\sigma^2_{\text{Measurement System}} = \underbrace{\sigma^2_{\text{Repeatability}}}_\text{Equipment Variation} + \underbrace{\sigma^2_{\text{Reproducibility}}}_\text{Appraiser Variation}")
            st.markdown("ANOVA achieves this by partitioning the **Sum of Squares (SS)**:")
            st.latex(r"SS_{\text{Total}} = SS_{\text{Part}} + SS_{\text{Operator}} + SS_{\text{Interaction}} + SS_{\text{Error}}")
            st.markdown("These SS values are converted to Mean Squares (MS), and from the MS values, we can estimate the variance components (`σ²`). For example:")
            st.latex(r"\hat{\sigma}^2_{\text{Repeatability}} = MS_{\text{Error}}")
            st.latex(r"\hat{\sigma}^2_{\text{Operator}} = \frac{MS_{\text{Operator}} - MS_{\text{Interaction}}}{n \cdot r}")
            st.markdown("The final KPI is the **% Gage R&R**, which is the percentage of the total variation that is consumed by the measurement system:")
            st.latex(r"\% \text{Gage R\&R} = \frac{\hat{\sigma}_{\text{Gage R\&R}}}{\hat{\sigma}_{\text{Total}}} \times 100")
        with tabs[4]:
            st.markdown("""
            Gage R&R is the standard methodology for Measurement Systems Analysis (MSA), a critical component of ensuring data integrity and process control.
            - **AIAG MSA Manual:** While from the automotive industry, this is considered the global "gold standard" reference for Gage R&R methodology and acceptance criteria.
            - **FDA Process Validation Guidance:** Stage 1 (Process Design) and Stage 2 (Process Qualification) require an understanding of all sources of variability, including measurement error. A Gage R&R is the formal proof that a measurement system is suitable for its intended use.
            - **21 CFR 211.160(b):** Requires that "laboratory controls shall include the establishment of scientifically sound and appropriate... standards, and test procedures... to assure that components... and drug products conform to appropriate standards of identity, strength, quality, and purity." A qualified measurement system is a prerequisite.
            """)
#=============================================================== 9. ATTRIBUTE AGREEMENT ANALYSIS ===================================================
def render_attribute_agreement():
    """Renders the comprehensive, interactive module for Attribute Agreement Analysis."""
    st.markdown("""
    #### Purpose & Application: Validating Human Judgment
    **Purpose:** To validate your **human measurement systems**. It answers the critical question: "Can our inspectors consistently and accurately distinguish good product from bad?" This is the counterpart to Gage R&R, but for subjective, pass/fail, or categorical assessments.
    
    **Strategic Application:** This analysis is essential for validating any process that relies on human visual inspection or go/no-go gauges. A failed study indicates that inspectors are either missing true defects (a risk to the patient/customer) or rejecting good product (a risk to the business), and that retraining or improved inspection aids are required.
    """)
    
    st.info("""
    **Interactive Demo:** Use the sidebar controls to create a challenging inspection scenario with different inspector archetypes.
    - The **Effectiveness Plot** (right) is your main diagnostic, showing the two critical types of error for each inspector.
    - The **Kappa Matrix** (bottom) shows who agrees with whom. Low values between two inspectors indicate they are not aligned on their decision criteria.
    """)

    with st.sidebar:
        st.subheader("Attribute Agreement Controls")
        st.markdown("**Study Design**")
        n_parts_slider = st.slider("Number of Parts in Study", 20, 100, 50, 5, help="The total number of unique parts (both good and bad) that will be assessed.")
        prevalence_slider = st.slider("True Defect Rate in Parts (%)", 10, 50, 20, 5, help="The percentage of parts in the study that are known to be defective. A good study has a high prevalence of defects to properly challenge the inspectors.")
        st.markdown("**Inspector Archetypes**")
        skilled_acc_slider = st.slider("Skilled Inspector Accuracy (%)", 85, 100, 98, 1, help="The base accuracy of your best, most experienced inspector.")
        uncertain_acc_slider = st.slider("Uncertain Inspector Accuracy (%)", 70, 100, 90, 1, help="The base accuracy of an inspector who struggles with borderline cases. Their performance will degrade on ambiguous parts.")
        biased_acc_slider = st.slider("Biased Inspector Accuracy (%)", 70, 100, 92, 1, help="The base accuracy of an inspector who is generally good but has a specific bias.")
        bias_strength_slider = st.slider("Biased Inspector 'Safe Play' Bias", 0.5, 1.0, 0.8, 0.05, help="When the Biased Inspector is unsure about a GOOD part, what is the probability they will fail it to be safe? 0.5 is no bias; 1.0 is maximum bias.")

    fig_eff, fig_kappa, kappa, df_eff = plot_attribute_agreement(
        n_parts=n_parts_slider, n_replicates=3,
        prevalence=prevalence_slider/100.0, 
        skilled_accuracy=skilled_acc_slider/100.0,
        uncertain_accuracy=uncertain_acc_slider/100.0,
        biased_accuracy=biased_acc_slider/100.0,
        bias_strength=bias_strength_slider
    )

    st.header("Results Dashboard")
    col1, col2 = st.columns([0.4, 0.6])
    with col1:
        st.subheader("Overall Study Metrics")
        st.metric("Fleiss' Kappa (Overall Agreement)", f"{kappa:.3f}", help="Measures agreement between all inspectors, corrected for chance. >0.7 is considered substantial agreement.")
        st.markdown("##### Individual Inspector Performance")
        st.dataframe(df_eff.style.format({"Miss Rate": "{:.2%}", "False Alarm Rate": "{:.2%}", "Accuracy": "{:.2%}"}), use_container_width=True)
        
    with col2:
        st.plotly_chart(fig_eff, use_container_width=True)
    
    st.plotly_chart(fig_kappa, use_container_width=True)

    st.divider()
    st.subheader("Deeper Dive")
    tabs = st.tabs(["💡 Key Insights", "📋 Glossary", "✅ The Golden Rule", "📖 Theory & History", "🏛️ Regulatory & Compliance"])
    
    with tabs[0]:
        st.markdown("""
        **A Realistic Workflow & Interpretation:**
        1.  **Check Overall Agreement (Fleiss' Kappa):** The first KPI tells you if the inspection team, as a whole, is consistent. A low Kappa (<0.7) signals a systemic problem with the procedure or training.
        2.  **Diagnose Individual Performance (Effectiveness Plot):** This plot is the main tool for root cause analysis.
            - **Inspector A (Skilled)** should be in the bottom-left "Ideal Zone."
            - **Inspector B (Uncertain)** will drift away from the ideal zone as you increase defect prevalence, because there are more borderline parts to confuse them. This signals a need for better training or clearer defect standards.
            - **Inspector C (Biased)** will drift to the right (high False Alarm Rate). This shows they are incorrectly failing good product, indicating they are either misinterpreting a standard or are being overly cautious.
        3.  **Find Disagreements (Kappa Matrix):** This heatmap shows *who* disagrees with *whom*. A low Kappa value between Inspector B and C, for example, would be expected from the simulation. This tells you exactly which two inspectors need to sit down together with the defect library to align their criteria.
        """)
    with tabs[1]:
        st.markdown("""
        ##### Glossary of Agreement Terms
        - **Attribute Data:** Data that is categorical or discrete, such as pass/fail, good/bad, or a defect classification.
        - **Miss Rate (False Negative):** The proportion of known defective parts that an inspector incorrectly classified as good. This represents **consumer's risk**.
        - **False Alarm Rate (False Positive):** The proportion of known good parts that an inspector incorrectly classified as defective. This represents **producer's risk**.
        - **Inter-Rater Reliability:** The degree of agreement among different inspectors (raters).
        - **Cohen's Kappa (κ):** A statistic that measures inter-rater agreement for categorical items, while taking into account the possibility of the agreement occurring by chance.
        - **Fleiss' Kappa:** An adaptation of Cohen's Kappa for measuring agreement between a fixed number of raters (more than two).
        """)
    with tabs[2]:
        st.error("""🔴 **THE INCORRECT APPROACH: The "Percent Agreement" Trap**
An analyst simply calculates that all inspectors agreed with the standard 95% of the time and declares the system valid.
- **The Flaw:** If the study only contains 2% true defects, an inspector could pass *every single part* and still achieve 98% agreement! Simple percent agreement is dangerously misleading with imbalanced data.""")
        st.success("""🟢 **THE GOLDEN RULE: Use Kappa for Consistency, and Effectiveness for Risk**
A robust analysis separates two key questions that must be answered.
1.  **Are the inspectors CONSISTENT? (Precision)** This is about whether the inspectors agree with **each other**. The **Kappa Matrix** is the best tool for this, as it corrects for chance agreement and pinpoints specific disagreements.
2.  **Are the inspectors ACCURATE? (Bias/Error)** This is about whether the inspectors agree with the **truth** (the gold standard). The **Effectiveness Report** is the best tool for this, as it separates the two types of business and patient risk: Miss Rate (Consumer's Risk) and False Alarm Rate (Producer's Risk).""")

    with tabs[3]:
        st.markdown("""
        #### Historical Context: Beyond Simple Percentages
        **The Problem:** For decades, researchers in social sciences and medicine struggled to quantify the reliability of subjective judgments. Simple "percent agreement" was the common method, but it had a fatal flaw: it didn't account for agreement that could happen purely by chance. Two doctors who both diagnose 90% of patients with "common cold" will have high agreement, but their skill might be no better than a coin flip if the true rate is 90%.

        **The 'Aha!' Moment:** In 1960, the psychologist **Jacob Cohen** developed **Cohen's Kappa (κ)**, a statistic that brilliantly solved this problem. Kappa measures the *observed* agreement and then subtracts the *agreement expected by chance*, creating a much more robust measure of true inter-rater reliability. This concept was later extended by **Joseph L. Fleiss** in 1971 to handle cases with more than two raters, resulting in **Fleiss' Kappa**.
            
        **The Impact:** Kappa statistics became the gold standard for measuring agreement in fields from psychology to clinical diagnostics. The automotive industry, in its quest for quality, recognized that a human inspector is a "measurement system." They incorporated these advanced statistical techniques into their **Measurement Systems Analysis (MSA)** manual, which is now considered the global standard, codifying Attribute Agreement Analysis as an essential tool for any industry relying on human inspection.
        """)
        
    with tabs[4]:
        st.markdown("""
        This analysis is a key part of **Measurement Systems Analysis (MSA)**, which is a fundamental expectation of a robust quality system.
        - **FDA Process Validation Guidance & 21 CFR 820 (QSR):** Both require that all measurement systems used for process control and product release be validated and fit for purpose. This explicitly includes human inspection systems. A documented Attribute Agreement Analysis is the objective evidence that this requirement has been met.
        - **ICH Q9 (Quality Risk Management):** A poorly performing inspection system is a major quality risk. This analysis quantifies that risk (e.g., the Miss Rate is a direct measure of patient/consumer risk) and provides the data to justify mitigation, such as retraining or implementing automated inspection.
        - **Regulatory Audits:** A lack of qualification for visual inspection processes is a common finding during regulatory audits. Having a robust Attribute Agreement study in your validation package demonstrates a mature and compliant quality system.
        """)
#=============================================================== 10. COMPREHENSIVE DIAGNOSTIC VALIDATION ===================================================
def render_diagnostic_validation_suite():
    """Renders the comprehensive, interactive module for diagnostic test validation."""
    st.markdown("""
    #### Purpose & Application: The Definitive Diagnostic Scorecard
    **Purpose:** To provide a single, comprehensive dashboard that unites all key statistical metrics used to validate a diagnostic test. This tool moves beyond individual metrics to show how they are all interconnected and derived from the foundational **Confusion Matrix**.
    
    **Strategic Application:** This is an essential tool for R&D, clinical validation, and regulatory affairs when developing an In Vitro Diagnostic (IVD). It allows teams to simulate how a test's performance characteristics and the target population's disease rate (prevalence) will impact its real-world clinical utility.
    """)
    
    st.info("""
    **Interactive Demo:** Use the three main sliders in the sidebar to simulate any diagnostic scenario.
    - **Sensitivity & Specificity:** Control the *intrinsic quality* of your assay.
    - **Prevalence:** Control the *population* in which the test is used. Observe the powerful "Prevalence Effect" in Plot 3, and see how the predictive values in the KPI panel change.
    """)

    with st.sidebar:
        st.subheader("Diagnostic Test Controls")
        sensitivity_slider = st.slider("🎯 Sensitivity (True Positive Rate)", 0.80, 1.00, 0.98, 0.005, format="%.3f",
            help="The intrinsic ability of the test to correctly identify true positives. A value of 0.98 means it will correctly detect 98 out of every 100 diseased individuals.")
        specificity_slider = st.slider("🛡️ Specificity (True Negative Rate)", 0.80, 1.00, 0.95, 0.005, format="%.3f",
            help="The intrinsic ability of the test to correctly identify true negatives. A value of 0.95 means it will correctly clear 95 out of every 100 healthy individuals.")
        prevalence_slider = st.slider("📈 Disease Prevalence (%)", 0.1, 50.0, 5.0, 0.1, format="%.1f%%",
            help="The percentage of the target population that actually has the disease. This is a property of the population, not the test itself, but it has a massive impact on the test's real-world predictive power.")

    fig_cm, fig_roc, fig_pv, metrics, other_concepts = plot_diagnostic_dashboard(
        sensitivity=sensitivity_slider,
        specificity=specificity_slider,
        prevalence=prevalence_slider/100.0
    )
    
    st.header("Diagnostic Performance Dashboard")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_cm, use_container_width=True)
    with col2:
        st.plotly_chart(fig_roc, use_container_width=True)
    
    st.plotly_chart(fig_pv, use_container_width=True)
    
    st.subheader("Comprehensive Metrics Panel")
    with st.expander("Click to view all 24 calculated and conceptual metrics", expanded=True):
        st.markdown("##### Foundational Metrics (The Building Blocks)")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("True Positives (TP)", metrics["True Positive (TP)"], help="Correctly identified as diseased.")
        c2.metric("True Negatives (TN)", metrics["True Negative (TN)"], help="Correctly identified as healthy.")
        c3.metric("False Positives (FP)", metrics["False Positive (FP)"], help="Healthy individuals incorrectly identified as diseased.")
        c4.metric("False Negatives (FN)", metrics["False Negative (FN)"], help="Diseased individuals incorrectly identified as healthy.")
        c5.metric("Prevalence", f"{metrics['Prevalence']:.1%}", help="The proportion of the population that truly has the disease.")

        st.markdown("##### Core Rates & Error Types (Intrinsic Test Quality)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sensitivity (TPR / Power)", f"{metrics['Sensitivity (TPR / Power)']:.2%}", help="TPR = TP / (TP + FN). The ability to detect the disease when present. Also called Recall or Power.")
        c2.metric("Specificity (TNR)", f"{metrics['Specificity (TNR)']:.2%}", help="TNR = TN / (TN + FP). The ability to correctly clear healthy individuals.")
        c3.metric("False Positive Rate (α)", f"{metrics['False Positive Rate (α)']:.2%}", help="FPR = FP / (TN + FP) = 1 - Specificity. The probability of a false alarm. Also called Type I Error.")
        c4.metric("False Negative Rate (β)", f"{metrics['False Negative Rate (β)']:.2%}", help="FNR = FN / (TP + FN) = 1 - Sensitivity. The probability of missing a true case. Also called Type II Error.")

        st.markdown("##### Predictive Values (Real-World Performance in this Population)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("PPV (Precision)", f"{metrics['PPV (Precision)']:.2%}", help="PPV = TP / (TP + FP). If a patient tests positive, what is the probability they are truly diseased?")
        c2.metric("NPV", f"{metrics['NPV']:.2%}", help="NPV = TN / (TN + FN). If a patient tests negative, what is the probability they are truly healthy?")
        c3.metric("False Discovery Rate (FDR)", f"{metrics['False Discovery Rate (FDR)']:.2%}", help="FDR = FP / (TP + FP) = 1 - PPV. The proportion of positive results that are false positives.")
        c4.metric("False Omission Rate (FOR)", f"{metrics['False Omission Rate (FOR)']:.2%}", help="FOR = FN / (TN + FN) = 1 - NPV. The proportion of negative results that are false negatives.")

        st.markdown("##### Overall Performance & Agreement Scores (Single-Number Summaries)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{metrics['Accuracy']:.2%}", help="ACC = (TP + TN) / Total. Overall correctness. Can be misleading with imbalanced data.")
        c2.metric("F1 Score", f"{metrics['F1 Score']:.3f}", help="The harmonic mean of Precision (PPV) and Recall (Sensitivity). Best when you need a balance between them.")
        c3.metric("MCC", f"{metrics['Matthews Correlation Coefficient (MCC)']:.3f}", help="A robust correlation coefficient between -1 and +1. +1 is a perfect prediction, 0 is random, -1 is perfectly wrong. Excellent for imbalanced data.")
        c4.metric("Cohen's Kappa (κ)", f"{metrics['Cohen’s Kappa (κ)']:.3f}", help="Measures agreement between the prediction and reality, corrected for agreement that could happen by chance. Similar to MCC.")

        st.markdown("##### Advanced & Model-Based Metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("AUC", f"{metrics['Area Under Curve (AUC)']:.3f}", help="Area Under the ROC Curve. The probability a random diseased individual has a higher score than a random healthy one. A measure of overall test separability.")
        # --- THIS IS THE DEFINITIVE FIX ---
        # Get the value from the dictionary first
        youdens_value = metrics["Youden's Index (J)"]
        # Then use the variable in the f-string, which is always safe
        c2.metric("Youden's Index (J)", f"{youdens_value:.3f}", help="J = Sensitivity + Specificity - 1. The maximum vertical distance between the ROC curve and the diagonal line. Finds a single 'optimal' cutoff.")
        # --- END OF DEFINITIVE FIX ---
        c3.metric("Positive LR (+)", f"{metrics['Positive Likelihood Ratio (LR+)']:.2f}", help="LR+ = Sens / (1 - Spec). How much more likely a positive test is to be seen in a diseased person vs. a healthy one. >10 is considered strong evidence.")
        c4.metric("Log-Loss", f"{metrics['Log-Loss (Cross-Entropy)']:.3f}", help="A measure of a probabilistic model's accuracy. It heavily penalizes being confidently wrong. Lower is better.")
        
        st.markdown("---")
        st.markdown("##### Conceptual Terms")
        for term, definition in other_concepts.items():
            st.markdown(f"**{term}:** {definition}")

    st.divider()
    st.subheader("Deeper Dive")
    tabs = st.tabs(["💡 Key Insights", "📋 Glossary", "✅ The Golden Rule", "📖 Theory & History", "🏛️ Regulatory & Compliance"])
    with tabs[0]:
        st.markdown("""
        **The Prevalence Effect: The Most Important Insight**
        The most critical takeaway from this dashboard is the relationship between a test's intrinsic quality and its real-world performance, visualized in **Plot 3**.
        - **Sensitivity & Specificity** are properties of the *test itself*. They do not change based on who you test.
        - **Positive Predictive Value (PPV)** and **Negative Predictive Value (NPV)** are properties of the *test result in a specific population*. They are highly dependent on **Prevalence**.
        
        **Try this:** Set Sensitivity and Specificity to high values (e.g., 99%). Now, in the sidebar, move the **Prevalence** slider from 50% down to 1%. Watch how the PPV curve (red line in Plot 3) collapses at low prevalence. This demonstrates the **false positive paradox**: even a highly accurate test can produce a majority of false positives when used to screen a low-prevalence population.
        
        **Advanced Metrics Explained:**
        - **Likelihood Ratios (LR+ / LR-):** How much does a positive/negative result increase/decrease the odds of having the disease? They are powerful because, unlike PPV/NPV, they are independent of prevalence.
        - **MCC & Kappa:** These are advanced accuracy metrics that are robust to imbalanced data (unlike simple Accuracy). A score of +1 is perfect, 0 is random, and -1 is perfectly wrong. **MCC** is generally considered one of the most robust and informative single-number scores for a classifier.
        """)
    with tabs[1]:
        st.markdown("""
        ##### Glossary of Diagnostic Metrics
        - **Sensitivity (TPR):** The ability of the test to correctly identify individuals who *have* the disease. `TPR = TP / (TP + FN)`
        - **Specificity (TNR):** The ability of the test to correctly identify individuals who do *not* have the disease. `TNR = TN / (TN + FP)`
        - **Prevalence:** The proportion of individuals in a population who have the disease at a specific time.
        - **PPV (Positive Predictive Value):** If a patient tests positive, the probability that they actually have the disease. Highly dependent on prevalence.
        - **NPV (Negative Predictive Value):** If a patient tests negative, the probability that they are actually healthy. Also dependent on prevalence.
        - **Likelihood Ratio (LR+):** How much a positive test result increases the odds of having the disease. `LR+ = Sensitivity / (1 - Specificity)`. Independent of prevalence.
        - **AUC (Area Under Curve):** A single metric (0.5 to 1.0) summarizing the overall diagnostic power of a test across all possible cutoffs.
        - **MCC (Matthews Correlation Coefficient):** A balanced measure of a test's quality, ranging from -1 (perfectly wrong) to +1 (perfectly right), with 0 being random. Considered very robust for imbalanced data.
        """)
    with tabs[2]:
        st.error("""🔴 **THE INCORRECT APPROACH: "Accuracy is Everything"**
An analyst reports that their new test has "95% accuracy" and declares it a success.
- **The Flaw:** Accuracy is a simple but often misleading metric, especially with imbalanced data. A test for a rare disease (1% prevalence) can achieve 99% accuracy by simply calling every single patient "Healthy." It has perfect specificity but zero sensitivity, making it clinically useless.""")
        st.success("""🟢 **THE GOLDEN RULE: Validate for the Intended Use**
A robust diagnostic validation always considers the clinical context.
1.  **Define the Intended Use:** Is this a screening test for the general population (low prevalence) or a confirmatory test for symptomatic patients (high prevalence)?
2.  **Choose the Right Metrics:** For a screening test, high **Sensitivity** and **NPV** are critical (you must not miss a case). For a confirmatory test before a risky procedure, high **Specificity** and **PPV** are paramount (you must not treat a healthy person).
3.  **Evaluate Holistically:** Use advanced, prevalence-independent metrics like **Likelihood Ratios**, **MCC**, or **AUC** to get a complete picture of test performance, but always report the PPV/NPV expected for the target population.
        """)

    with tabs[3]:
        st.markdown("""
        #### Historical Context: A Multi-Disciplinary Synthesis
        The metrics on this dashboard were not developed at once, but represent a synthesis of ideas from epidemiology, statistics, computer science, and psychology over the 20th century.
        - **Epidemiology (1940s-50s):** The concepts of **Sensitivity, Specificity, and Prevalence** were formalized to understand the performance of disease screening programs.
        - **Signal Detection Theory (1950s):** The **ROC Curve** was developed to distinguish radar signals from noise, and was later adopted by psychologists. **Youden's Index** was proposed in 1950 as a simple way to summarize a test's performance in a single number.
        - **Bayesian Statistics (1763, revived 1990s):** **Bayes' Theorem** is the mathematical engine that links prevalence (prior probability) to **PPV/NPV** (posterior probability). **Likelihood Ratios** are a direct application of this theorem.
        - **Psychology (1960s):** **Cohen's Kappa** was developed by Jacob Cohen to measure inter-rater reliability, correcting for the probability that two raters might agree simply by chance.
        - **Machine Learning (1990s-2000s):** As classifiers became common, metrics were needed to handle imbalanced datasets. The **F1 Score** emerged from information retrieval, while the **Matthews Correlation Coefficient (MCC)** was developed in bioinformatics (1980). **Log-Loss** became a standard for evaluating probabilistic forecasts.
        """)

    with tabs[4]:
        st.markdown("""
        Demonstrating the performance of a diagnostic test is a primary focus of global medical device and IVD regulations.
        - **FDA 21 CFR 820.30 (Design Controls for Medical Devices):** Requires **design validation** to ensure devices conform to defined user needs and intended uses. For an IVD, this is proven through clinical performance studies that establish the metrics on this dashboard.
        - **EU IVDR (In Vitro Diagnostic Regulation 2017/746):** Requires a comprehensive **Performance Evaluation Report (PER)**. This report must contain detailed data and analysis of the test's **analytical performance** (e.g., precision) and **clinical performance** (sensitivity, specificity, PPV, NPV, LR+). All of these metrics are explicitly mentioned.
        - **CLSI Guidelines (Clinical & Laboratory Standards Institute):** Documents like **EP12-A2** provide detailed protocols for user-based evaluation of qualitative test performance, including the calculation of sensitivity, specificity, and predictive values.
        - **GAMP 5:** If the test's result is generated by software (e.g., an algorithm that analyzes an image), that software must be validated (CSV) to ensure its calculations are correct and reliable.
        """)

#=============================================================== 11. ROC CURVE ANALYSIS ===================================================
def render_roc_curve():
    """Renders the INTERACTIVE module for Receiver Operating Characteristic (ROC) curve analysis."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To solve **The Diagnostician's Dilemma**: a test must correctly identify patients with a disease (high **Sensitivity**) while also correctly clearing healthy patients (high **Specificity**). The ROC curve visualizes this trade-off.
    
    **Strategic Application:** This is the global standard for validating diagnostic tests. The Area Under the Curve (AUC) provides a single metric of a test's diagnostic power, while the choice of cutoff determines its real-world performance.
    """)
    
    st.info("""
    **Interactive Demo:** Use all three sliders to see how assay quality and decision-making interact.
    - **`Separation` & `Overlap`** control the fundamental quality of the assay.
    - **`Decision Cutoff`**: Move this slider to see the shaded TP/FP/TN/FN areas change in the top plot. Simultaneously, watch the black 'X' move along the ROC curve in the bottom plot, and see the real-time impact on all the performance metrics.
    """)
    
    # --- Sidebar controls for this specific module ---
    with st.sidebar:
        st.subheader("ROC Curve Controls")
        separation_slider = st.slider(
            "📈 Separation (Diseased Mean)", 
            min_value=50.0, max_value=80.0, value=65.0, step=1.0,
            help="Controls the distance between the Healthy and Diseased populations. More separation = better test."
        )
        overlap_slider = st.slider(
            "🌫️ Overlap (Population SD)", 
            min_value=5.0, max_value=20.0, value=10.0, step=0.5,
            help="Controls the 'noise' or spread of the populations. More overlap (a higher SD) = worse test."
        )
        # --- NEW SLIDER ADDED HERE ---
        cutoff_slider = st.slider(
            "🔪 Decision Cutoff",
            min_value=30, max_value=80, value=55, step=1,
            help="The threshold for calling a sample 'Positive'. Move this to trace the ROC curve and see the trade-offs."
        )

    # Generate plots using the slider values
    fig, auc_value, sensitivity, specificity, ppv, npv = plot_roc_curve(
        diseased_mean=separation_slider, 
        population_sd=overlap_slider,
        cutoff=cutoff_slider
    )
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "📋 Glossary", "✅ The Golden Rule", "📖 Theory & History", "🏛️ Regulatory & Compliance"])
        
        with tabs[0]:
            st.metric(label="📈 Overall KPI: Area Under Curve (AUC)", value=f"{auc_value:.3f}",
                      help="The overall diagnostic power of the test. 0.5 is useless, 1.0 is perfect.")
            st.markdown(f"--- \n ##### Performance at Cutoff = {cutoff_slider}")
            st.metric(label="🎯 Sensitivity (True Positive Rate)", value=f"{sensitivity:.2%}",
                      help="Of all the truly diseased patients, what percentage did we correctly identify?")
            st.metric(label="🛡️ Specificity (True Negative Rate)", value=f"{specificity:.2%}",
                      help="Of all the truly healthy patients, what percentage did we correctly clear?")
            st.metric(label="✅ Positive Predictive Value (PPV)", value=f"{ppv:.2%}",
                      help="If a patient tests positive, what is the probability they actually have the disease?")
            st.metric(label="❌ Negative Predictive Value (NPV)", value=f"{npv:.2%}",
                      help="If a patient tests negative, what is the probability they are actually healthy?")
            
            st.markdown("""
            **The Core Insight:** The AUC tells you how good your *assay* is. The four metrics on the right tell you how good your *decision* is at a specific cutoff. A great assay can still lead to poor outcomes if the wrong cutoff is chosen for the clinical context.
            """)
        with tabs[1]:
            st.markdown("""
            ##### Glossary of ROC Terms
            - **ROC Curve:** A plot of the True Positive Rate (Sensitivity) against the False Positive Rate (1 - Specificity) for all possible cutoff values of a diagnostic test.
            - **AUC (Area Under the Curve):** A single metric (0.5 to 1.0) summarizing the overall diagnostic power of a test. 0.5 is random chance; 1.0 is a perfect test.
            - **Cutoff / Threshold:** The specific test value used to make a decision (e.g., classify a patient as "diseased" or "healthy").
            - **Sensitivity (TPR):** The ability of the test to correctly identify individuals who *have* the disease.
            - **Specificity (TNR):** The ability of the test to correctly identify individuals who do *not* have the disease.
            - **Youden's Index (J):** A statistic that captures the performance of a diagnostic test. `J = Sensitivity + Specificity - 1`. The cutoff that maximizes J is the point on the ROC curve furthest from the random chance line.
            """)
        with tabs[2]:
            st.error("""🔴 **THE INCORRECT APPROACH: "Worship the AUC" & "Hug the Corner"**
- *"My AUC is 0.95, so we're done."* (The *chosen cutoff* might still be terrible for the clinical need).
- *"I'll just pick the cutoff closest to the top-left corner."* (This balances errors equally, which is rarely desired).""")
            st.success("""🟢 **THE GOLDEN RULE: The Best Cutoff Depends on the Consequence of Being Wrong**
Ask: **"What is worse? A false positive or a false negative?"**
- **For deadly disease screening:** You must catch every possible case. Prioritize **maximum Sensitivity**.
- **For confirming a diagnosis for a risky surgery:** You must be certain the patient has the disease. Prioritize **maximum Specificity** to avoid unnecessary procedures.""")

        with tabs[3]:
            st.markdown("""
            #### Historical Context: From Radar Blips to Medical Labs
            **The Problem:** During World War II, engineers were developing radar to detect enemy aircraft. They faced a classic signal-detection problem: how do you set the sensitivity of the receiver? If it's too sensitive, it will pick up random noise (birds, atmospheric clutter) as enemy planes (a **false alarm**). If it's not sensitive enough, it will miss real enemy planes (a **missed hit**).

            **The 'Aha!' Moment:** Engineers needed a way to visualize this trade-off. They developed the **Receiver Operating Characteristic (ROC)** curve. It plotted the probability of a "hit" (True Positive Rate) against the probability of a "false alarm" (False Positive Rate) for every possible sensitivity setting of the receiver. This allowed them to quantify the performance of different radar systems and choose the optimal operating point.
            
            **The Impact:** After the war, the technique was adapted by psychologists for signal detection in perception experiments. In the 1970s, it was introduced to medicine, where it became the undisputed gold standard for evaluating the performance of diagnostic tests, providing a clear, graphical language to communicate the critical trade-off between sensitivity and specificity.
            """)
            st.markdown("#### Mathematical Basis")
            st.markdown("The curve plots **Sensitivity (Y-axis)** versus **1 - Specificity (X-axis)**.")
            st.latex(r"\text{Sensitivity} = \frac{TP}{TP + FN} \quad , \quad \text{Specificity} = \frac{TN}{TN + FP}")
            st.markdown("Each point on the curve represents the (Sensitivity, 1-Specificity) pair for a specific cutoff value. The **Area Under the Curve (AUC)** has a powerful probabilistic interpretation: it is the probability that a randomly chosen 'Diseased' subject will have a higher test score than a randomly chosen 'Healthy' subject.")
        with tabs[4]:
            st.markdown("""
            ROC analysis is the global standard for demonstrating the clinical performance of In Vitro Diagnostics (IVDs) and medical devices.
            - **FDA 21 CFR 820 (Quality System Regulation):** The design validation section (§820.30(g)) requires objective evidence that the device conforms to user needs and intended uses. For a diagnostic, this evidence is typically clinical sensitivity and specificity, which are summarized by ROC analysis.
            - **EU IVDR (In Vitro Diagnostic Regulation):** The European regulation requires a Performance Evaluation Report (PER) that includes data on clinical sensitivity, specificity, and the rationale for the chosen cutoff value.
            - **ISO 13485:2016:** The international quality management standard for medical devices, which aligns with the principles of design validation found in 21 CFR 820.
            """)
            
#====================================================================================== 12. ASSAY ROBUSTNESS (DOE)  =================================================================================================   
def render_assay_robustness_doe():
    """Renders the comprehensive, interactive module for Assay Robustness (DOE/RSM)."""
    st.markdown("""
    #### Purpose & Application: Process Cartography - The GPS for Optimization
    **Purpose:** To create a detailed topographical map of your process landscape. This analysis moves beyond simple robustness checks to full **process optimization**, using **Response Surface Methodology (RSM)** to model curvature and find the true "peak of the mountain."
    
    **Strategic Application:** This is the statistical engine for Quality by Design (QbD) and process characterization. By developing a predictive model, you can:
    - **Find Optimal Conditions:** Identify the exact settings that maximize yield, efficacy, or any other Critical Quality Attribute (CQA).
    - **Define a Design Space:** Create a multi-dimensional "safe operating zone" where the process is guaranteed to produce acceptable results.
    - **Minimize Variability:** Find a "robust plateau" on the response surface where performance is not only high, but also insensitive to small variations.
    """)
    
    st.info("""
    **Interactive Demo:** You are the process expert. Use the sliders in the sidebar to define the "true" physics of a virtual assay. The plots will show how a DOE/RSM experiment can uncover this underlying response surface, allowing you to find the optimal operating conditions.
    """)
    
    with st.sidebar:
        st.subheader("DOE / RSM Controls (True Effects)")
        st.markdown("**Linear & Interaction Effects**")
        ph_slider = st.slider("🧬 pH Main Effect", -10.0, 10.0, 2.0, 1.0)
        temp_slider = st.slider("🌡️ Temperature Main Effect", -10.0, 10.0, 5.0, 1.0)
        interaction_slider = st.slider("🔄 pH x Temp Interaction Effect", -10.0, 10.0, 0.0, 1.0)
        
        st.markdown("**Curvature (Quadratic) Effects**")
        ph_quad_slider = st.slider("🧬 pH Curvature", -10.0, 10.0, -5.0, 1.0, help="A negative value creates a 'hill' (peak). A positive value creates a 'bowl' (valley).")
        temp_quad_slider = st.slider("🌡️ Temperature Curvature", -10.0, 10.0, -5.0, 1.0)

        st.markdown("**Experimental Noise**")
        noise_slider = st.slider("🎲 Random Noise (SD)", 0.1, 5.0, 1.0, 0.1)
    
    fig_contour, fig_3d, fig_pareto, anova_summary, opt_ph, opt_temp, max_resp = plot_doe_robustness(
        ph_effect=ph_slider, temp_effect=temp_slider, interaction_effect=interaction_slider,
        ph_quad_effect=ph_quad_slider, temp_quad_effect=temp_quad_slider, noise_sd=noise_slider
    )
    
    st.header("Results Dashboard")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Statistical Analysis")
        st.markdown("The Pareto plot identifies the vital few factors, while the ANOVA table provides the statistical proof.")
        
        tab1, tab2 = st.tabs(["Pareto Plot of Effects", "ANOVA Table"])
        with tab1:
            st.plotly_chart(fig_pareto, use_container_width=True)
        with tab2:
            st.dataframe(anova_summary.style.format({'p-value': '{:.4f}'}).applymap(
                lambda p: 'background-color: #C8E6C9' if p < 0.05 else '', subset=['p-value']),
                use_container_width=True)
            
    with col2:
        st.subheader("Predicted Optimum")
        st.markdown("Based on the model, these are the settings predicted to maximize the response.")
        m1, m2, m3 = st.columns(3)
        m1.metric("Optimal pH", f"{opt_ph:.2f}")
        m2.metric("Optimal Temp (°C)", f"{opt_temp:.2f}")
        m3.metric("Max Response", f"{max_resp:.1f}")
        
        st.markdown("""
        **Interpreting the Visuals:**
        - **Contour Plot:** A 2D topographical map. The "bullseye" of concentric circles (if present) marks the optimal region. The gold star shows the model's predicted peak.
        - **3D Surface Plot:** A 3D view of the process landscape, helping to visualize the "mountain" you are trying to climb.
        """)

    st.divider()
    st.header("Response Surface Visualizations")
    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(fig_contour, use_container_width=True)
    with col4:
        st.plotly_chart(fig_3d, use_container_width=True)
        
    st.divider()
    st.subheader("Deeper Dive")
    
    # --- THIS IS THE LINE THAT WAS FIXED ---
    tabs_deep = st.tabs(["💡 Key Insights", "📋 Glossary", "✅ The Golden Rule", "📖 Theory & History", "🏛️ Regulatory & Compliance"])
    # --- END OF FIX ---
    
    with tabs_deep[0]:
        st.markdown("""
        - **Pareto Plot is Key:** This is your primary diagnostic. It instantly shows you which factors (linear, interaction, quadratic) are the main drivers of the process. Green bars are statistically significant (p < 0.05).
        - **Linear Effects:** A large linear effect (e.g., `Temp`) means that factor has a strong, consistent impact.
        - **Interaction Effects:** A significant interaction (`pH:Temp`) means the factors are not independent. The effect of pH is different at high vs. low temperatures.
        - **Curvature Effects:** Significant quadratic terms (`I(pH**2)`) are the key to optimization. A negative curvature effect (as simulated by default) proves you have found a "peak" or optimal zone. A positive effect would indicate a "valley."
        """)
    with tabs[1]:
        st.markdown("""
        ##### Glossary of DOE/RSM Terms
        - **DOE (Design of Experiments):** A systematic method to determine the relationship between factors affecting a process and the output of that process.
        - **Factor:** An input variable that is intentionally varied during an experiment (e.g., Temperature, pH).
        - **Response:** The output variable that is measured (e.g., Yield, Purity).
        - **Main Effect:** The effect of a single factor on the response.
        - **Interaction Effect:** Occurs when the effect of one factor on the response depends on the level of another factor.
        - **RSM (Response Surface Methodology):** A collection of statistical and mathematical techniques useful for developing, improving, and optimizing processes. It uses designs (like the CCD) that can estimate curvature.
        - **Quadratic Effect:** A term in the model that describes the curvature of the response surface. A significant quadratic effect is necessary to find a true optimum (a peak or valley).
        """)
    with tabs_deep[2]:
        st.error("""🔴 **THE INCORRECT APPROACH: One-Factor-at-a-Time (OFAT)**
Imagine trying to find the highest point on a mountain by only walking in straight lines, first due North-South, then due East-West. You will almost certainly end up on a ridge or a local hill, convinced it's the summit, while the true peak was just a few steps to the northeast.
**The Flaw:** This is what OFAT does. It is statistically inefficient and, more importantly, it is **guaranteed to miss the true optimum** if any interaction between the factors exists.""")
        st.success("""🟢 **THE GOLDEN RULE: Map the Entire Territory at Once (DOE/RSM)**
By testing factors in combination using a dedicated design (like a Central Composite Design), you send out scouts to explore the entire landscape simultaneously. This allows you to:
1.  **Be Highly Efficient:** Gain more information from fewer experimental runs.
2.  **Understand the Terrain:** Uncover critical interaction and curvature effects that describe the true shape of the process space.
3.  **Find the True Peak:** Develop a predictive mathematical model that acts as a GPS, guiding you directly to the optimal operating conditions.""")

    with tabs_deep[3]:
        st.markdown("""
        #### Historical Context: From Screening to Optimization
        **The Problem (The Genesis):** In the 1920s, **Sir Ronald A. Fisher** invented Design of Experiments to solve agricultural problems. His factorial designs were brilliant for *screening*—efficiently figuring out *which* factors (e.g., fertilizer type, seed variety) were important.
        **The New Problem (Optimization):** The post-war chemical industry boom created a new need: not just to know *which* factors mattered, but *how to find their optimal settings*. A simple factorial design, which only tests the corners of the design space, can't model curvature and therefore can't find a peak.
        **The 'Aha!' Moment (RSM):** In 1951, **George Box** and K.B. Wilson developed **Response Surface Methodology (RSM)** to solve this. They created efficient new designs, like the **Central Composite Design (CCD)** shown here, which cleverly adds "axial" (star) points and center points to a factorial design. 
        **The Impact:** These extra points allow for the fitting of a **quadratic model**, which is the key to modeling curvature and finding the "peak of the mountain." This moved DOE from simple screening to true, powerful optimization, becoming the statistical foundation of modern process development and Quality by Design (QbD).
        """)
        st.markdown("#### Mathematical Basis")
        st.markdown("RSM typically fits a second-order (quadratic) model to the experimental data. For two factors, the model is:")
        st.latex(r"Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + \beta_{11}X_1^2 + \beta_{22}X_2^2 + \beta_{12}X_1X_2 + \epsilon")
        st.markdown(r"""
        - `β₀`: The intercept or baseline response.
        - `β₁`, `β₂`: The **linear main effects** (the tilt of the surface).
        - `β₁₁`, `β₂₂`: The **quadratic effects** (the curvature or "hill/bowl" shape).
        - `β₁₂`: The **interaction effect** (the twist of the surface).
        - `ϵ`: The random experimental error.
        To get stable estimates of these coefficients, the analysis is performed on **coded variables**, where the high and low levels of each factor are scaled to be +1 and -1, respectively.
        """)
        
    with tabs_deep[4]:
        st.markdown("""
        DOE and RSM are core methodologies for fulfilling the principles of Quality by Design (QbD), which is strongly encouraged by regulators.
        - **ICH Q8(R2) - Pharmaceutical Development:** This guideline introduces the concept of the **Design Space**, which is defined as "the multidimensional combination and interaction of input variables... that has been demonstrated to provide assurance of quality." RSM is the primary statistical tool used to establish a Design Space.
        - **ICH Q2(R1) - Validation of Analytical Procedures:** Requires the assessment of **Robustness**, which is typically evaluated through a DOE by making small, deliberate variations in method parameters.
        - **FDA Guidance on Process Validation:** Emphasizes a lifecycle approach and process understanding, which are best achieved through the systematic study of process parameters using DOE.
        """)
#====================================================================================== 13. MIXTURE DESIGN FORMULATIONS  =================================================================================================   
def render_mixture_design():
    """Renders the comprehensive, interactive module for Mixture DOE."""
    st.markdown("""
    #### Purpose & Application: The Formulation Scientist's GPS
    **Purpose:** To act as a **Formulation Scientist's GPS**. While a standard DOE maps a process, a Mixture DOE is specifically designed to navigate the complex world of formulations where components must sum to 100%. It answers questions like: "What is the optimal blend of three excipients to maximize drug solubility?"
    
    **Strategic Application:** This is an essential tool for developing stable drug products, buffers, or cell culture media. The resulting ternary plot provides an intuitive map of all possible formulations, instantly highlighting the "sweet spot" of optimal performance, which can be filed as a regulatory Design Space.
    """)
    
    st.info("""
    **Interactive Demo:** Use the sidebar controls to define the "true" properties of your formulation components.
    - The **Model Effects Plot** is your primary statistical diagnostic, showing *why* the map has its shape.
    - The **Ternary Map** is your visual guide to the Design Space (the region bounded by the orange line).
    """)

    with st.sidebar:
        st.subheader("Mixture Design Controls")
        st.markdown("**Component Main Effects**")
        a_slider = st.slider("Component A Effect", 0, 100, 60, 5, help="The response value when the formulation is 100% Component A.")
        b_slider = st.slider("Component B Effect", 0, 100, 80, 5, help="The response value when the formulation is 100% Component B.")
        c_slider = st.slider("Component C Effect", 0, 100, 50, 5, help="The response value when the formulation is 100% Component C.")
        st.markdown("**Component Interactions (Synergy/Antagonism)**")
        ab_slider = st.slider("A x B Interaction", -50, 50, 20, 5, help="Positive values mean A and B work better together than expected (synergy). Negative values mean they interfere with each other (antagonism).")
        ac_slider = st.slider("A x C Interaction", -50, 50, 0, 5, help="Controls the synergy or antagonism between components A and C.")
        bc_slider = st.slider("B x C Interaction", -50, 50, -30, 5, help="Controls the synergy or antagonism between components B and C.")
        st.markdown("**Experimental & Quality Controls**")
        noise_slider = st.slider("Experimental Noise (SD)", 0.1, 10.0, 2.0, 0.5, help="The random measurement error in each experimental run.")
        threshold_slider = st.slider("Min. Acceptable Response", 20, 100, 75, 5, help="The quality target. Any formulation predicted to be above this value will be inside the orange Design Space boundary.")

    fig_effects, fig_ternary, model, opt_blend = plot_mixture_design(
        a_effect=a_slider, b_effect=b_slider, c_effect=c_slider,
        ab_interaction=ab_slider, ac_interaction=ac_slider, bc_interaction=bc_slider,
        noise_sd=noise_slider, response_threshold=threshold_slider
    )

    st.header("Results Dashboard")
    col1, col2 = st.columns([0.45, 0.55])
    with col1:
        st.subheader("Statistical Diagnostics")
        st.metric("Model Adj. R-squared", f"{model.rsquared_adj:.3f}", help="How well the model fits the data.")
        st.markdown("##### Predicted Optimal Blend")
        st.dataframe(opt_blend.apply(lambda x: f"{x:.1%}").to_frame(name='Proportion'))
        st.plotly_chart(fig_effects, use_container_width=True)
        
    with col2:
        st.plotly_chart(fig_ternary, use_container_width=True)
    
    st.divider()
    st.subheader("Deeper Dive")
    tabs = st.tabs(["💡 Key Insights", "📋 Glossary", "✅ The Golden Rule", "📖 Theory & History", "🏛️ Regulatory & Compliance"])
    
    with tabs[0]:
        st.markdown("""
        **A Realistic Workflow & Interpretation:**
        1.  **Start with the Model Effects Plot:** This is your primary statistical diagnostic. It explains *why* the ternary map has its shape.
            - **Main Effects (Blue):** The coefficients for `A`, `B`, and `C` represent the predicted response at the pure, 100% vertices of the triangle.
            - **Interactions (Red):** These are the most important terms. A large **positive** coefficient (like `A:B`) indicates **synergy**—the blend is better than a simple average of its parts. A large **negative** coefficient (like `B:C`) indicates **antagonism**—the components interfere with each other.
        2.  **Consult the Ternary Map:** This is your visual guide to the formulation space.
            - **Color Contours:** Show the predicted response. The "hottest" color indicates the region of optimal performance, driven by the synergistic interactions.
            - **Orange Boundary:** This is your **Design Space** or **Proven Acceptable Range (PAR)**—the set of all formulations predicted to meet your acceptance criteria.
            - **White Star:** The single "best" blend predicted by the model.
        """)
    with tabs[1]:
        st.markdown("""
        ##### Glossary of Mixture Terms
        - **Mixture Design:** A special class of DOE for experiments with ingredients or components of a mixture as the factors, where the response depends on the proportions of the ingredients, not their absolute amounts.
        - **Constraint:** The mathematical requirement that the sum of the proportions of all components must equal a constant (usually 1 or 100%).
        - **Ternary Plot:** A triangular plot used to visualize the relationship between three components of a mixture and a response variable.
        - **Simplex:** The geometric space that defines the experimental region for a mixture. For three components, the simplex is a triangle.
        - **Synergy:** A positive interaction effect where the combined effect of two or more components is greater than the sum of their individual effects.
        - **Antagonism:** A negative interaction effect where the combined effect is less than the sum of their individual effects.
        """)
    with tabs[2]:
        st.error("""🔴 **THE INCORRECT APPROACH: Using a Standard DOE**
        An analyst tries to use a standard factorial or response surface design to optimize a formulation.
        - **The Flaw:** Standard DOEs treat factors as independent variables that can be changed freely. In a formulation, they are not independent; increasing one component *must* decrease another. This violates the core mathematical assumptions of a standard DOE, leading to incorrect models and nonsensical predictions.""")
        st.success("""🟢 **THE GOLDEN RULE: Use a Mixture Design for Mixture Problems**
        The experimental design must match the physical reality of the problem.
        1.  **Identify the Constraint:** If your factors are ingredients or components that must sum to a constant (e.g., 100%), you have a mixture problem.
        2.  **Choose the Right Design:** Use a specialized experimental design, like a **Simplex-Lattice** or **Simplex-Centroid** design, which efficiently places points at the vertices, edges, and center of the formulation space.
        3.  **Use the Right Model:** Analyze the results with a model designed for mixtures, like the **Scheffé polynomial**, which correctly handles the mathematical constraints.""")

    with tabs[3]:
        st.markdown("""
        #### Historical Context: Solving the Chemist's Dilemma
        **The Problem:** For the first half of the 20th century, optimizing formulations was more art than science. Chemists and food scientists relied on intuition and laborious one-factor-at-a-time experiments. The powerful Design of Experiments (DOE) tools developed by Fisher and Box were of little help, as they couldn't handle the fundamental constraint that `A + B + C = 100%`.

        **The 'Aha!' Moment:** In a landmark 1958 paper, the statistician **Henry Scheffé** solved this problem. He recognized that the experimental space was not a cube (like in a standard DOE) but a **simplex** (a triangle for 3 components, a tetrahedron for 4, etc.). He then derived a new class of polynomial models, now known as **Scheffé polynomials**, specifically for this geometry. These models cleverly omit the intercept and rearrange terms to perfectly suit the mixture constraint.
        
        **The Impact:** Scheffé's work gave scientists a systematic, statistically rigorous, and highly efficient methodology to optimize blends and formulations. It transformed formulation development from guesswork into a predictable science and is now a cornerstone of product development in industries ranging from pharmaceuticals and food science to petrochemicals and materials science.
        """)

    with tabs[4]:
        st.markdown("""
        Mixture DOE is a specialized tool for establishing a **Design Space** for formulation parameters (material attributes), a core concept of **Quality by Design (QbD)**.
        - **ICH Q8(R2) - Pharmaceutical Development:** This guideline is the primary driver for this type of work. The region inside the orange boundary on the map is a direct visualization of a formulation **Design Space**. Filing this with a regulatory agency provides significant manufacturing flexibility, as movement within this space is not considered a change.
        - **ICH Q11 - Development and Manufacture of Drug Substances:** The principles of QbD, including the use of DOE to understand the relationship between material attributes and quality, apply equally to drug substances.
        - **FDA Process Validation Guidance:** Emphasizes a lifecycle approach and deep process/product understanding. A mixture DOE provides this deep understanding for the formulation itself and can be used to justify the **Bill of Materials (BOM)** and the acceptable ranges for each component.
        """)

#====================================================================================== 14. PROCESS OPTIMIZATION: FROM DOE TO AI  =================================================================================================   
def render_process_optimization_suite():
    """Renders the comprehensive, interactive module for the full optimization workflow."""
    st.markdown("""
    #### Purpose & Application: The Complete Optimization Workflow
    **Purpose:** To demonstrate the end-to-end modern workflow for process optimization, from initial characterization with **DOE/RSM**, to deeper analysis with **Machine Learning**.
    
    **Strategic Application:** This dashboard integrates two powerful techniques to tell a complete story.
    1.  **DOE/RSM:** Provides a simple, causal, and interpretable model that serves as the basis for a regulatory Design Space filing.
    2.  **ML & PDP:** Uncovers more complex, non-linear relationships from the data, providing a more accurate "digital twin" of the process for deeper understanding and internal optimization.
    """)
    
    st.info("""
    **Interactive Demo:** Use the sidebar controls to define the "true" physics of your process. The dashboard follows a real-world workflow:
    1.  Check the **Statistical Results** to see which factors drive the process.
    2.  Compare the **Process Maps**: the simple, smooth RSM map vs. the complex, more accurate ML map.
    """)
    
    with st.sidebar:
        st.subheader("Process Simulation Controls")
        st.markdown("**True Process Effects**")
        temp_slider = st.slider("🌡️ Temperature Main Effect", -10.0, 10.0, 2.0, 1.0)
        ph_slider = st.slider("🧬 pH Main Effect", -10.0, 10.0, 1.0, 1.0)
        interaction_slider = st.slider("🔄 pH x Temp Interaction Effect", -10.0, 10.0, -3.0, 1.0)
        temp_quad_slider = st.slider("🌡️ Temperature Curvature", -10.0, 0.0, -5.0, 1.0)
        ph_quad_slider = st.slider("🧬 pH Curvature", -10.0, 0.0, -8.0, 1.0)
        asymmetry_slider = st.slider("⛰️ Process Asymmetry", -5.0, 5.0, -3.0, 0.5)
        noise_slider = st.slider("🎲 Experimental Noise (SD)", 0.1, 5.0, 1.0, 0.1)
        st.markdown("---")
        st.markdown("**Quality Requirement**")
        yield_threshold_slider = st.slider("Acceptable Yield Threshold (%)", 85, 99, 95, 1)

    fig_pareto, fig_rsm_3d, fig_rsm_2d, pdp_buffer, anova_table, opt_ph, opt_temp, max_resp = plot_doe_optimization_suite(
        ph_effect=ph_slider, temp_effect=temp_slider, interaction_effect=interaction_slider,
        ph_quad_effect=ph_quad_slider, temp_quad_effect=temp_quad_slider,
        asymmetry_effect=asymmetry_slider, noise_sd=noise_slider,
        yield_threshold=yield_threshold_slider
    )

    st.header("Statistical Results & Predicted Optimum")
    col1, col2 = st.columns([0.55, 0.45])
    with col1:
        st.plotly_chart(fig_pareto, use_container_width=True)
    with col2:
        st.subheader("Predicted Optimum & KPIs (from RSM Model)")
        m1, m2, m3 = st.columns(3)
        m1.metric("Optimal pH", f"{opt_ph:.2f}")
        m2.metric("Optimal Temp (°C)", f"{opt_temp:.2f}")
        m3.metric("Max Predicted Yield", f"{max_resp:.1f}%")
        st.markdown("---")
        st.markdown("##### Statistical Model Summary (ANOVA)")
        st.dataframe(anova_table.style.format({'p-value': '{:.4f}'}).applymap(
            lambda p: 'background-color: #C8E6C9' if p < 0.05 else '', subset=['p-value']),
            use_container_width=True, height=240)

    st.header("Process Visualizations: RSM vs. Machine Learning")
    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(fig_rsm_3d, use_container_width=True)
        st.plotly_chart(fig_rsm_2d, use_container_width=True)
    with col4:
        st.image(pdp_buffer)
        st.markdown("""
        **Comparing the Maps:**
        - The **RSM plots (left)** show a smooth, symmetrical, and easily interpretable model of the process. This is ideal for regulatory submissions.
        - The **ML plot (right)** reveals the more complex "ground truth" learned from the data, including the asymmetric cliff created by the simulation. This is superior for deep process understanding and internal optimization.
        """)

    st.divider()
    st.subheader("Deeper Dive")
    tabs = st.tabs(["💡 Key Insights", "📋 Glossary", "✅ The Golden Rule", "📖 Theory & History", "🏛️ Regulatory & Compliance"])
    with tabs[0]:
        st.markdown("""
        **A Realistic Workflow:**
        1.  **Start with the Pareto Plot:** This is your primary diagnostic from the simple RSM model. It tells you which factors are the main causal drivers of the process.
        2.  **Build the Regulatory Map (RSM):** The 2D topographic map (bottom-left) is your official process map. The green shaded area is your validated **Design Space (PAR)**, and the white-dashed box is your **NOR**.
        3.  **Build the High-Fidelity Map (ML):** The 2D heatmap (right) is your more accurate "digital twin." Notice how it captures the sharp "cliff" on the low-pH side that the smoother RSM model can only approximate. This map is better for finding the true optimum and understanding process risks.
        
        **Core Insight:** The simple RSM model is for **validation and communication**. The complex ML model is for **accuracy and deep understanding**. A mature QbD program uses both in tandem. The workflow demonstrates a powerful modern paradigm: **Use DOE to build a simple, causal foundation. Then, use ML on larger historical datasets to refine that understanding and create a high-fidelity "digital twin" of your process that can be used for automated optimization.**
        """)
    with tabs[1]:
        st.markdown("""
        ##### Glossary of Optimization Terms
        - **RSM (Response Surface Methodology):** A statistical technique for modeling and optimizing a response based on a set of input variables, typically using a quadratic model derived from a planned DOE.
        - **Gradient Boosting (e.g., XGBoost):** A powerful machine learning algorithm that builds a predictive model as an ensemble of many simple "weak" decision trees. It is known for high accuracy and the ability to capture complex, non-linear relationships.
        - **PDP (Partial Dependence Plot):** A visualization that shows the marginal effect of one or two features on the predicted outcome of a machine learning model. It helps to understand the model's behavior.
        - **Gradient Descent:** An iterative optimization algorithm used to find the minimum of a function. In this context (maximization), it follows the positive gradient ("steepest ascent") to "climb the hill" of the predicted response surface to find the optimum.
        """)
    with tabs[2]:
        st.error("""🔴 **THE INCORRECT APPROACH: The "ML is Magic" Fallacy**
An analyst takes a messy historical dataset, trains a complex ML model, and uses Gradient Descent to find a "perfect" optimum without understanding the underlying causality.
- **The Flaw:** The historical data may contain confounding. The model might learn that "high yield is correlated with Operator Bob," but this is not an actionable insight. Trying to optimize for "more Bob" is nonsensical. The model has learned a correlation, not a causal lever.""")
        st.success("""🟢 **THE GOLDEN RULE: DOE for Causality, ML for Complexity**
A robust optimization strategy uses the best of both worlds.
1.  **Establish Causality with DOE:** First, use a planned DOE to prove which parameters are true causal levers for the process. This grounds your model in scientific reality.
2.  **Capture Complexity with ML:** Once you have a large historical dataset, use a more powerful ML model to capture the complex, non-linear interactions your simple RSM model might miss.
3.  **Optimize on Causal Levers:** Finally, use optimization algorithms like Gradient Descent on your ML model, but only allow it to optimize the parameters you have already proven to be causal. This ensures your final "optimum" is both accurate and actionable.""")

    with tabs[3]:
        st.markdown("""
        #### Historical Context: A Convergence of Titans
        This dashboard represents the convergence of three separate, powerful intellectual traditions that developed over nearly a century.
        1.  **Statistical Experimentation (DOE/RSM, 1920s-1950s):** Pioneered by **Sir Ronald A. Fisher** and later perfected for optimization by **George Box**. The focus was on extracting maximum causal information from a minimum number of planned, physical experiments. This was the era of "small data" and deep statistical theory.
        2.  **Machine Learning (Gradient Boosting, 1990s-2000s):** With the rise of computing power, researchers like Jerome Friedman developed algorithms like Gradient Boosting Machines (GBM). The focus shifted from simple, interpretable models to highly accurate, complex algorithms that could learn intricate patterns from large "found" datasets. **XGBoost** (2014) is a highly optimized, modern implementation of these ideas.
        3.  **Mathematical Optimization (Gradient Descent, 1847):** The core idea of "climbing the hill" by following the steepest path was first proposed by Augustin-Louis Cauchy. For over a century, it was a mathematical curiosity. The AI revolution of the 2010s turned it into the engine of modern deep learning, as it is the algorithm used to train virtually every neural network.
        
        **The Modern Synthesis:** Today, we can combine these three titans. We use the principles of Fisher and Box to design smart experiments, the predictive power of Friedman's algorithms to build an accurate map, and the efficiency of Cauchy's optimization to find the peak.
        """)
        
    with tabs[4]:
        st.markdown("""
        This integrated workflow is a direct implementation of the most advanced principles of **Quality by Design (QbD)** and **Process Analytical Technology (PAT)**.
        - **ICH Q8(R2) - Pharmaceutical Development:** The DOE/RSM portion is the standard method for establishing a **Design Space**. The ML and Gradient Descent portions represent an advanced method for achieving a deeper **Process Understanding** and identifying an optimal **Control Strategy**.
        - **FDA Guidance for Industry - PAT:** Using an ML model as a "digital twin" for your process and running optimization algorithms on it is a core concept of PAT. It allows for process control and optimization to be done proactively and based on a deep, data-driven model.
        - **FDA AI/ML Action Plan & GMLP:** Deploying an ML model for process optimization in a GxP environment requires a robust validation lifecycle. This includes justifying the model choice, validating its predictive accuracy, and using explainability tools like PDP plots to ensure its reasoning is scientifically sound. The entire workflow shown here would be a key part of the validation package for an AI-driven control strategy.
        """)
        st.markdown("""
        This tool directly implements the core principles of the **ICH Q8(R2) Pharmaceutical Development** guideline, which is the foundational document for Quality by Design (QbD).
        
        ---
        ##### Key ICH Q8(R2) Definitions:
        - **Design Space (DSp):** "The multidimensional combination and interaction of input variables (e.g., material attributes) and process parameters that have been demonstrated to provide assurance of quality."
          - **Your Action:** The area inside the orange boundary on the 2D plot is your experimentally derived Design Space.

        - **Proven Acceptable Range (PAR):** "A characterized range of a process parameter for which operation within this range, while keeping other parameters constant, will result in producing a material meeting relevant quality criteria."
          - **Your Action:** In practice, the Design Space and PAR are often used interchangeably. The PAR represents the validated "edges of failure" for your process.
        
        - **Normal Operating Range (NOR):** "The range of a process parameter that is typically used during routine manufacturing."
          - **Your Action:** The NOR (green box) is a tighter range set well within the PAR. It provides a buffer and serves as the target for routine operations. An excursion within the PAR but outside the NOR triggers an investigation but is not necessarily a deviation.
          
        ---
        **The Regulatory Advantage:**
        The guideline explicitly states: **"Working within the design space is not considered as a change. Movement out of the design space is considered to be a change and would normally initiate a regulatory post-approval change process."** This provides enormous operational and regulatory flexibility, which is the primary business driver for adopting a QbD approach.
        """)

#====================================================================================== 15. SPLIT-PLOT DESIGNS =================================================================================================   
def render_split_plot():
    """Renders the module for Split-Plot Designs."""
    st.markdown("""
    #### Purpose & Application: The Efficient Experimenter
    **Purpose:** To design an efficient and statistically valid experiment when your study involves both **Hard-to-Change (HTC)** and **Easy-to-Change (ETC)** factors. This is a specialized form of Design of Experiments (DOE).
    
    **Strategic Application:** This design is a lifesaver during process characterization and tech transfer. A standard DOE might require you to change all factors randomly, which can be prohibitively expensive or time-consuming.
    - **Tech Transfer:** Validating a new `Media Lot` (HTC) is a major undertaking. However, once a run is started, testing different `Supplement Concentrations` (ETC) is easy.
    A split-plot design saves immense resources by minimizing the number of times you have to change the difficult factor.
    """)

    st.info("""
    **Interactive Demo:** Use the sliders to control the "true" underlying effects in the process.
    - **`Lot-to-Lot Variation`**: Creates a main effect for the 'Lot' factor. Watch the plots for Lot B shift down.
    - **`Interaction Effect`**: Makes the effect of the Supplement *depend* on the Lot. Watch the lines in the **Interaction Plot** become non-parallel, a classic sign of an interaction.
    """)

    with st.sidebar:
        st.subheader("Split-Plot Controls")
        variation_slider = st.slider(
            "Lot-to-Lot Variation (SD)",
            min_value=0.0, max_value=5.0, value=0.5, step=0.25,
            help="Controls the 'true' difference between the hard-to-change media lots. Higher values simulate a larger main effect."
        )
        interaction_slider = st.slider(
            "Lot x Supplement Interaction Effect",
            min_value=-2.0, max_value=2.0, value=0.0, step=0.2,
            help="Controls how much the supplement's effect changes between Lot A and Lot B. A non-zero value creates an interaction."
        )
    
    fig_main, fig_interaction, anova_table = plot_split_plot_doe(
        lot_variation_sd=variation_slider,
        interaction_effect=interaction_slider
    )
    
    # --- Redesigned Layout ---
    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        st.plotly_chart(fig_main, use_container_width=True)
        st.plotly_chart(fig_interaction, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "📋 Glossary", "✅ The Golden Rule", "📖 Theory & History", "🏛️ Regulatory & Compliance"])
        
        with tabs[0]:
            st.markdown("##### ANOVA Results")
            st.dataframe(anova_table.style.format({'p-value': '{:.4f}'}).applymap(
                lambda p: 'background-color: #C8E6C9' if p < 0.05 else '', subset=['p-value']),
                use_container_width=True)
            
            st.markdown("""
            **Reading the Plots & Table:**
            - **Main Plot:** Visualizes the raw data and means for each experimental condition.
            - **Interaction Plot:** This is your primary tool for diagnosing interactions. If the lines are not parallel, an interaction is likely present.
            - **ANOVA Table:** Provides the statistical proof. Look for p-values < 0.05 to identify significant effects.
                - `C(Lot)`: Tests the main effect of the media lot.
                - `C(Supplement)`: Tests the main effect of the supplement.
                - `C(Lot):C(Supplement)`: Tests the **interaction effect**. This is often the most important result.
            """)
        with tabs[1]:
            st.markdown("""
            ##### Glossary of Split-Plot Terms
            - **Split-Plot Design:** A type of DOE used when one or more factors are "hard-to-change" while others are "easy-to-change."
            - **Hard-to-Change (HTC) Factor:** A factor that is difficult, expensive, or time-consuming to change between experimental runs (e.g., bioreactor setup, media lot).
            - **Easy-to-Change (ETC) Factor:** A factor that can be easily changed between runs (e.g., supplement concentration, agitation speed).
            - **Whole Plot:** The experimental unit to which the levels of the HTC factor are applied.
            - **Sub-Plot:** The smaller experimental units within a whole plot, to which the levels of the ETC factor are applied.
            - **Restricted Randomization:** The key feature of a split-plot design. The levels of the HTC factor are not fully randomized, which must be accounted for in the statistical analysis.
            """)

        with tabs[2]:
            st.error("""🔴 **THE INCORRECT APPROACH: The "Pretend it's Standard" Fallacy**
            An analyst runs a split-plot experiment for convenience but analyzes it as if it were a standard, fully randomized DOE.
            - **The Flaw:** This is statistically invalid. A standard analysis assumes every run is independent, but in a split-plot, all the sub-plots within a whole plot are correlated. This error leads to incorrect p-values and a high risk of declaring an effect significant when it's just random noise.""")
            st.success("""🟢 **THE GOLDEN RULE: Design Dictates Analysis**
            The way you conduct your experiment dictates the only valid way to analyze it.
            1.  **Recognize the Constraint:** Identify if you have factors that are much harder, slower, or more expensive to change than others.
            2.  **Choose the Right Design:** If you do, a Split-Plot design is likely the most efficient and practical choice.
            3.  **Use the Right Model:** Analyze the results using a statistical model that correctly accounts for the two different error structures (the "whole plot error" for the HTC factor and the "sub-plot error" for the ETC factor). This is typically done with a mixed-model ANOVA.""")

        with tabs[3]:
            st.markdown("""
            #### Historical Context: The Fertile Fields of Rothamsted
            **The Problem:** Like much of modern statistics, the Split-Plot design was born from the practical challenges of agriculture. Its inventors, **Sir Ronald A. Fisher** and **Frank Yates**, were working at the Rothamsted Experimental Station in the 1920s and 30s, the oldest agricultural research institution in the world. They faced a logistical nightmare: they wanted to test different large-scale treatments (like irrigation methods) and small-scale treatments (like crop varieties) in the same experiment.
            
            **The 'Aha!' Moment:** They couldn't fully randomize everything. Changing the irrigation method (the **Hard-to-Change** factor) required digging large trenches and re-routing water, so it could only be done on large sections of a field, which they called **"whole plots."** However, within each irrigated whole plot, it was very easy to plant multiple different crop varieties (the **Easy-to-Change** factor) in smaller **"sub-plots."** This physical constraint of not being able to irrigate a tiny plot differently from its neighbor forced a new way of thinking.
            
            **The Impact:** Fisher and Yates developed the specific mathematical framework for the Split-Plot ANOVA to correctly analyze the data from this **restricted randomization**. They recognized that there were two different levels of experimental error: a larger error for comparing whole plots and a smaller error for comparing sub-plots within a whole plot. By correctly partitioning the variance, they created one of the most practical and widely used experimental designs ever conceived, saving researchers in countless fields immense time and resources.
            """)
            st.markdown("#### Mathematical Basis")
            st.markdown("The key to a split-plot analysis is recognizing it has two different error terms. The linear model for the design is often expressed as a mixed model:")
            st.latex(r"Y_{ijk} = \mu + \alpha_i + \gamma_{ik} + \beta_j + (\alpha\beta)_{ij} + \epsilon_{ijk}")
            st.markdown("""
            -   `μ`: Overall mean.
            -   `αᵢ`: Fixed effect of the `i`-th level of the **whole-plot factor A**.
            -   `γᵢₖ`: The random **whole-plot error**, ~ N(0, σ²_γ). This is the error term for testing factor A.
            -   `βⱼ`: Fixed effect of the `j`-th level of the **sub-plot factor B**.
            -   `(αβ)ᵢⱼ`: The interaction effect.
            -   `εᵢⱼₖ`: The random **sub-plot error**, ~ N(0, σ²_ε). This is the error term for testing factor B and the interaction.
            Because `σ²_γ` is typically larger than `σ²_ε`, the test for the hard-to-change factor (A) is less powerful than the test for the easy-to-change factor (B), which is the fundamental trade-off of this design.
            """)
        with tabs[4]:
            st.markdown("""
            As a specific type of Design of Experiments, Split-Plot designs are tools used to fulfill the broader regulatory expectations around process understanding and robustness.
            - **ICH Q8(R2) - Pharmaceutical Development:** The principles of efficient experimentation to gain process knowledge are central to QbD. A split-plot design is a practical tool for achieving this when certain factors are hard to change.
            - **FDA Guidance on Process Validation:** Encourages a scientific, risk-based approach to validation. Using an efficient design like a split-plot demonstrates statistical maturity and a commitment to resource optimization while still generating the required process knowledge.
            """)
#====================================================================================== 16. CAUSAL INFERENCE  =================================================================================================          
def render_causal_inference():
    """Renders the INTERACTIVE module for Causal Inference."""
    st.markdown("""
    #### Purpose & Application: Beyond "What" to "Why"
    **Purpose:** To move beyond mere correlation ("what") and ascend to the level of causation ("why"). While predictive models see shadows on a cave wall (associations), Causal Inference provides the tools to understand the true objects casting them (the underlying causal mechanisms).
    
    **Strategic Application:** This is the ultimate goal of root cause analysis and the foundation of intelligent intervention.
    - **💡 Effective CAPA:** A predictive model might say high sensor readings are *associated* with *low* purity. Causal Inference helps determine if the sensor readings *cause* low purity, or if both are driven by a third hidden variable (a "confounder") like calibration drift. This prevents wasting millions on fixing the wrong problem.
    - **🗺️ Process Cartography:** It allows for the creation of a **Directed Acyclic Graph (DAG)**, which is a formal causal map of your process, documenting scientific understanding and guiding future analysis.
    """)
    
    st.info("""
    **Interactive Demo:** Use the slider to control the **Confounding Strength** of the `Calibration Age`. 
    - At low strength, the naive correlation (orange) is close to the true effect (green).
    - As you increase the strength, watch the naive correlation become not just wrong, but **completely inverted**—a classic demonstration of **Simpson's Paradox**.
    """)
    
    with st.sidebar:
        st.subheader("Causal Inference Controls")
        confounding_slider = st.sidebar.slider(
            "🚨 Confounding Strength", 
            min_value=0.0, max_value=10.0, value=5.0, step=0.5,
            help="How strongly the 'Calibration Age' affects BOTH the Sensor Reading (drift) and the Purity (degradation)."
        )
    
    fig_dag, fig_scatter, naive_effect, adjusted_effect = plot_causal_inference(confounding_strength=confounding_slider)
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig_dag, use_container_width=True)
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "📋 Glossary", "✅ The Golden Rule", "📖 Theory & History", "🏛️ Regulatory & Compliance"])
        
        with tabs[0]:
            st.metric(label="Biased Estimate (Naive Correlation)", value=f"{naive_effect:.3f}",
                      help="The misleading conclusion you would draw by just plotting Purity vs. Sensor Reading.")
            st.metric(label="Unbiased Estimate (True Causal Effect)", value=f"{adjusted_effect:.3f}",
                      help="The true effect of the Sensor Reading on Purity after adjusting for the confounder.")

            st.markdown("""
            **The Paradox Explained:**
            - **The DAG (Top Plot):** This map shows our scientific belief. We believe a higher `Sensor Reading` *causes* higher `Purity`. However, `Calibration Age` is a **confounder**: it independently *increases* the Sensor Reading (drift) and *decreases* the Purity.
            - **The Scatter Plot (Bottom):** The naive orange line looks at all data together and concludes that higher sensor readings are associated with *lower* purity. This is **Simpson's Paradox**. The green lines show the truth: *within each calibration group (New or Old)*, the relationship is positive. The adjusted model correctly identifies this true, positive causal effect.
            """)
        with tabs[1]:
            st.markdown("""
            ##### Glossary of Causal Terms
            - **Causation vs. Correlation:** Correlation indicates that two variables move together, but does not imply that one causes the other. Causation means that a change in one variable directly produces a change in another.
            - **Confounding:** A situation where the relationship between two variables is distorted by a third, unobserved variable (the confounder) that is associated with both.
            - **DAG (Directed Acyclic Graph):** A visual map of causal assumptions. Nodes represent variables, and directed arrows represent assumed causal effects.
            - **Backdoor Path:** A non-causal path between two variables in a DAG that creates a spurious correlation. To find the true causal effect, all backdoor paths must be "blocked."
            - **Simpson's Paradox:** A statistical phenomenon where a trend appears in several different groups of data but disappears or reverses when these groups are combined. It is a classic example of confounding.
            """)
        with tabs[2]:
            st.error("""🔴 **THE INCORRECT APPROACH: The Correlation Trap**
            - An analyst observes that higher sensor readings are correlated with lower final purity. They recommend changing the process target to achieve lower sensor readings, believing this will improve purity.
            - **The Flaw:** This intervention would be a disaster. They are acting on a spurious correlation. The real cause of low purity is the old calibration. Their "fix" would actually make things worse by targeting the wrong variable.""")
            st.success("""🟢 **THE GOLDEN RULE: Draw the Map, Block the Backdoors**
            A robust causal analysis follows a disciplined process.
            1.  **Draw the Map (Build the DAG):** Collaborate with Subject Matter Experts to encode all domain knowledge and causal beliefs into a formal DAG.
            2.  **Identify the Backdoor Paths:** Use the DAG to identify all non-causal "backdoor" paths that create confounding. In our case, the path `Sensor Reading <- Calibration Age -> Purity` is a backdoor.
            3.  **Block the Backdoors:** Use the appropriate statistical technique (like multiple regression) to "adjust for" the confounding variable (`Calibration Age`), blocking the backdoor path and isolating the true causal effect.""")

        with tabs[3]:
            st.markdown("""
            #### Historical Context: The Causal Revolution
            **The Problem:** For most of the 20th century, mainstream statistics was deeply allergic to the language of causation. The mantra, famously drilled into every student, was **"correlation is not causation."** While true, this left a massive void: if correlation isn't the answer, what is? Statisticians were excellent at describing relationships but had no formal language to discuss *why* those relationships existed, leaving a critical gap between data and real-world action.
            
            **The 'Aha!' Moment:** The revolution was sparked by the computer scientist and philosopher **Judea Pearl** in the 1980s and 90s. His key insight was that the missing ingredient was **structure**. He argued that scientists carry causal models in their heads all the time, and that these models could be formally written down as graphs. He introduced the **Directed Acyclic Graph (DAG)** as the language for this structure. The arrows in a DAG are not mere correlations; they are bold claims about the direction of causal influence.
            
            **The Impact:** This was a paradigm shift. By making causal assumptions explicit in a DAG, Pearl developed a complete mathematical framework—including his famous **do-calculus**—to determine if a causal question *could* be answered from observational data, and if so, how. This "Causal Revolution" provided the first-ever rigorous, mathematical language to move from seeing (`P(Y|X)`) to doing (`P(Y|do(X))`), transforming fields from epidemiology to economics. For this work, Judea Pearl was awarded the Turing Award in 2011, the highest honor in computer science.
            """)
            st.markdown("#### Mathematical Basis")
            st.markdown("The core difference is between **Observation** and **Intervention**.")
            st.markdown("- **Observation (Correlation):** `P(Y | X = x)` asks, \"What is the expected Y for the subset of units that *we happened to observe* had a value of `x`?\" This is vulnerable to confounding.")
            st.markdown("- **Intervention (Causation):** `P(Y | do(X = x))` asks, \"What would Y be if we *intervened and forced every unit* to have a value of `x`?\" This is the true causal effect.")
            st.markdown("Pearl's **backdoor adjustment formula** shows how to calculate the intervention from observational data. To find the effect of `X` on `Y` with a set of confounders `Z`, we calculate:")
            st.latex(r"P(Y | do(X=x)) = \sum_z P(Y | X=x, Z=z) P(Z=z)")
            st.markdown("In simple terms, this means: for each level of the confounder `z`, find the relationship between `X` and `Y`, and then average those relationships across the distribution of `z`. This is precisely what a multiple regression model does when you include `Z` as a covariate.")
        with tabs[4]:
            st.markdown("""
            Causal inference is an advanced technique that provides a rigorous framework for Root Cause Analysis (RCA), a fundamental requirement of a compliant quality system.
            - **ICH Q10 - Pharmaceutical Quality System:** Mandates a system for Corrective and Preventive Actions (CAPA) that includes a thorough investigation to determine the root cause of deviations. Causal inference provides a formal language and toolset to move beyond simple correlation in these investigations.
            - **21 CFR 211.192 - Production Record Review:** Requires that any unexplained discrepancy or failure of a batch to meet its specifications "shall be thoroughly investigated."
            - **GAMP 5:** While focused on software, its principles of risk management and root cause analysis for deviations apply broadly.
            """)

##=========================================================================================================================================================================================================
##===============================================================================END ACT I UI Render ========================================================================================================================================
##=========================================================================================================================================================================================================

#========================================================================================= 1. SAMPLE SIZE FOR QUALIFICATION =====================================================================
def render_sample_size_calculator():
    """Renders the comprehensive, interactive module for calculating sample size for qualification."""
    st.markdown("""
    #### Purpose & Application: The Auditor's Question
    **Purpose:** To provide a statistically valid, audit-proof justification for a sampling plan. This tool answers the fundamental question asked in every process validation: **"How did you decide that testing 'n' samples was enough?"**
    
    **Strategic Application:** This is a critical step in writing any Validation Plan (VP) for a Performance Qualification (PQ) or for defining a lot acceptance sampling plan.
    - **Demonstrates Statistical Rigor:** Moves beyond arbitrary or "tribal knowledge" sample sizes (e.g., "we've always used n=30") to a defensible, risk-based approach.
    - **Optimizes Resources:** Using the correct statistical model (e.g., Hypergeometric for finite lots) can often reduce the required sample size compared to overly conservative methods, saving significant time and cost.
    """)
    
    st.info("""
    **Interactive Demo:** Use the controls in the sidebar to define your statistical requirements (Confidence & Reliability) and process type.
    - The **KPI** on the left shows the calculated sample size needed.
    - The **Plot** on the right visualizes the trade-off. Your specific requirement is marked by the green star. Notice how increasing reliability or confidence demands a larger sample size.
    """)
    
    with st.sidebar:
        st.subheader("Sample Size Controls")
        calc_method = st.radio(
            "Select the statistical model:",
            ["Binomial (Continuous Process / Large Lot)", "Hypergeometric (Finite Lot)"],
            help="Choose Binomial for ongoing processes or very large batches. Choose Hypergeometric for discrete, smaller batches to get a more accurate (and often smaller) sample size."
        )
        confidence_level = st.slider("Confidence Level (C)", 80.0, 99.9, 95.0, 0.1, format="%.1f%%")
        reliability = st.slider("Required Reliability (R)", 90.0, 99.9, 99.0, 0.1, format="%.1f%%")

        lot_size = None
        if "Hypergeometric" in calc_method:
            lot_size = st.number_input(
                "Enter the total Lot Size (M)", 
                min_value=10, max_value=100000, value=1000, step=10,
                help="The total number of units in the discrete batch you are sampling from."
            )
            
    # --- Calculation Logic (moved up for clarity) ---
    c = confidence_level / 100
    r = reliability / 100
    sample_size, model_used = "N/A", ""

    if "Binomial" in calc_method:
        model_used = "Binomial"
        sample_size = int(np.ceil(np.log(1 - c) / np.log(r))) if r < 1.0 else "Infinite"
    elif lot_size:
        model_used = "Hypergeometric"
        D = math.floor((1 - r) * lot_size)
        @st.cache_data
        def find_hypergeometric_n(M, D, C):
            if M <= D: return 1
            log_alpha = math.log(1 - C)
            for n in range(1, M - D + 2):
                if n > M - D: return M - D
                log_p0 = (math.lgamma(M-D+1) - math.lgamma(n+1) - math.lgamma(M-D-n+1)) - \
                         (math.lgamma(M+1) - math.lgamma(n+1) - math.lgamma(M-n+1))
                if log_p0 <= log_alpha: return n
            return M - D
        sample_size = find_hypergeometric_n(lot_size, D, c)

    # --- Dashboard Layout ---
    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        fig = plot_sample_size_curves(confidence_level, reliability, lot_size, calc_method, sample_size)
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "📋 Glossary", "✅ The Golden Rule", "📖 Theory & History", "🏛️ Regulatory & Compliance"])

        with tabs[0]:
            st.metric(
                label=f"Required Sample Size (n) via {model_used}",
                value=f"{sample_size} units",
                help="Minimum units to test with zero failures to meet your claim."
            )
            st.success(f"""
            **Actionable Conclusion:**
            
            To demonstrate with **{confidence_level:.1f}% confidence** that your process is at least **{reliability:.1f}% reliable**, you must test **{sample_size} units** and find **zero failures**.
            """)
            st.markdown("""
            **Reading the Plot:**
            - The curves show the **best possible reliability** you can claim for a given sample size (with zero failures).
            - As `n` increases, your statistical power increases, allowing you to claim higher reliability.
            - Notice the **Hypergeometric curve (orange dash)** is always above the Binomial curve. This shows that for a finite lot, you need a slightly smaller sample size to make the same statistical claim, as each good part you draw slightly increases the chance the next one is good.
            """)
        with tabs[1]:
            st.markdown("""
            ##### Glossary of Sampling Terms
            - **Acceptance Sampling:** A statistical method used to determine whether to accept or reject a production lot of material based on the inspection of a sample.
            - **Confidence (C):** The probability that the conclusion you draw from your sample is correct. A 95% confidence level means you are accepting a 5% risk of being wrong (drawing a "lucky" good sample from a bad lot).
            - **Reliability (R):** The proportion of conforming items in the lot or process. A reliability of 99% means the true defect rate is no more than 1%.
            - **Binomial Distribution:** A probability model used when sampling from a very large or continuous population (sampling with replacement). The probability of success is constant for each trial.
            - **Hypergeometric Distribution:** A probability model used when sampling from a small, finite population without replacement. The probability of success changes with each draw.
            """)
        with tabs[2]:
            st.error("""🔴 **THE INCORRECT APPROACH: The "Square Root of N Plus One" Fallacy**
An engineer is asked for a sampling plan and defaults to an arbitrary, non-statistical rule of thumb they learned years ago, like `n = sqrt(Lot Size) + 1`.
- **The Flaw:** This plan is completely disconnected from risk. It cannot answer the question: "What level of confidence and reliability does this plan provide?" It is indefensible during a regulatory audit.""")
            st.success("""🟢 **THE GOLDEN RULE: State Your Risk, Then Calculate Your Sample**
A compliant and statistically sound sampling plan is always derived from pre-defined risk criteria.
1.  **First, Define the Claim:** Before touching a calculator, stakeholders (Quality, Regulatory, Clinical) must agree on the required claim. "We need to be **95% confident** that the batch is **99% compliant**."
2.  **Then, Justify the Model:** Choose the correct statistical model for the situation (Binomial for a continuous process, Hypergeometric for a finite lot).
3.  **Finally, Calculate `n`:** The required sample size is the direct mathematical output of the first two steps. This creates a clear, traceable, and audit-proof justification for your validation plan.""")
            
        with tabs[3]:
            st.markdown("""
            #### Historical Context: From Gosset's Brewery to Modern Industry
            **The Problem:** At the turn of the 20th century, William Sealy Gosset, a chemist at the Guinness brewery in Dublin, faced a problem central to industrial quality control: how to make reliable conclusions from very small samples. The classical statistical theory of the time required large samples, but Gosset could only afford to take a few measurements from each batch of beer. This practical constraint forced a revolution in statistics.
            
            **The 'Aha!' Moment:** Gosset, writing under the pseudonym "Student," developed what we now know as the **Student's t-distribution** (1908). This was the first "small-sample" statistical method. It mathematically demonstrated that with very few data points, our uncertainty is much higher than previously thought. The convention of using `n=3` is a direct, practical consequence of this insight: it represents the smallest possible sample size where you can calculate a meaningful standard deviation and thus begin to characterize the uncertainty Gosset was trying to tame.
            
            **The Military-Industrial Complex:** Decades later, during World War II, the need for robust sampling exploded. Statisticians like Harold Dodge, Harry Romig, and Abraham Wald developed modern **acceptance sampling** for the military. Their work, codified in standards like **MIL-STD-105**, provided the rigorous mathematical framework (using Binomial and Hypergeometric distributions) to link sample size to specific, contractual quality levels (AQL).
            
            **The Modern Synthesis:** Today's practices blend both legacies. Gosset's "small-sample" thinking justifies using triplicates for preliminary repeatability checks, while the military's acceptance sampling framework provides the high-assurance models needed for final product release and process qualification.
            """)
        
        with tabs[4]:
            st.markdown("""
            A statistically justified sampling plan is a fundamental expectation in any GxP-regulated environment. It provides the **objective evidence** required to support validation claims and batch disposition decisions.
            
            ---
            ##### FDA - Process Validation & CFR
            - **Process Validation Guidance (2011):** This calculator is most directly applicable to **Stage 2: Process Performance Qualification (PPQ)**. The guidance states that the number of samples should be "sufficient to provide statistical confidence of quality both within a batch and between batches." This tool provides the statistical rationale for "sufficient."
            - **21 CFR 211.165(d):** Requires that "acceptance criteria for the sampling and testing conducted... shall be adequate to assure that batches of drug products meet **appropriate statistical quality control criteria**." This calculator is a method for defining such criteria.
            - **21 CFR 211.110(b):** Requires written procedures for in-process controls and tests, including "Control procedures shall include... scientifically sound and appropriate sampling plans."
            
            ---
            ##### Medical Devices - ISO 13485 & 21 CFR 820
            - **ISO 13485:2016 (Section 7.5.6):** This global standard for medical device quality management requires that process validation activities ensure the process can "consistently produce product which meets specifications."
            - **21 CFR 820.250 (Statistical Techniques):** States that "Where appropriate, each manufacturer shall establish and maintain procedures for identifying valid statistical techniques required for establishing, controlling, and verifying the acceptability of process capability and product characteristics." This calculator is a prime example of such a technique.
            
            ---
            ##### ICH Guidelines - A Global Perspective
            - **ICH Q9 (Quality Risk Management):** The selection of a confidence and reliability level is a direct input from the Quality Risk Management process. A high-risk product or process parameter would demand higher confidence and reliability, leading to a larger sample size. This tool provides the direct link between the risk assessment and the validation sampling plan.
            
            **The Golden Thread:** Across all regulations, the expectation is the same. The choice of a sample size cannot be arbitrary. It must be **pre-defined, justified, and linked to the level of quality and assurance** required for the product. This calculator provides that traceable, scientific justification.
            """)

    with st.expander("🔎 Special Topic: The Role of Triplicates (n=3) in Experimental Design"):
        st.markdown("""
        While the calculator above determines sample size for lot acceptance (a `go/no-go` decision), a common question in experimental design is: **"Why do we so often test in triplicates?"** While `n=3` is rarely sufficient for a full PQ, it is a common and justifiable choice for smaller-scale experiments for several reasons.
        
        #### Why triplicates are often used
        *   **Statistical estimate of variability:** With only one measurement, you have no way to know if the result is representative. With two, you can see if results differ, but you can’t estimate variance reliably. With three, you can calculate a mean, standard deviation, and %RSD, giving you a basic measure of repeatability.
        *   **Outlier detection:** If one result deviates significantly from the other two, you can identify possible anomalies due to instrument noise, handling error, or environmental changes.
        *   **Compliance with common scientific practice:** In many biological, chemical, and engineering fields, `n=3` is a de facto minimum for demonstrating reproducibility without making the experiment prohibitively costly or time-consuming.

        #### Mathematical Basis: The Power of `n-1`
        The ability to estimate variance from a sample is based on the concept of **degrees of freedom (df)**. The formula for sample standard deviation is:
        """)
        st.latex(r"s = \sqrt{\frac{\sum_{i=1}^{n}(x_i - \bar{x})^2}{n-1}}")
        st.markdown("""
        -   For `n=1`, the denominator is `1-1=0`. The standard deviation is undefined. You have zero degrees of freedom to estimate variability.
        -   For `n=2`, the denominator is `2-1=1`. You have one degree of freedom. You can calculate a standard deviation, but it's a very unstable "point estimate" of the true process variability.
        -   For `n=3`, the denominator is `3-1=2`. You have two degrees of freedom. This is the smallest sample size where the estimate of variability begins to gain some (though still limited) stability. It is the absolute minimum for a meaningful statistical characterization of repeatability.

        #### When triplicates alone are NOT sufficient
        For formal method validation (especially in regulated environments like FDA, ISO 17025, CLSI, ICH Q2), triplicates in a single run are almost never enough. Regulators expect a more comprehensive demonstration of robustness, usually including:
        *   Multiple runs across different days.
        *   Multiple analysts/operators.
        *   Multiple instruments (if applicable).
        *   Larger datasets for key performance parameters (e.g., accuracy, linearity, robustness).
        
        > ✅ **Bottom line:** Triplicates are the statistical minimum for assessing within-run repeatability. For formal validation, they are just one piece of the puzzle—not the whole picture.
        """)
    
    with st.expander("View Detailed Statistical Methodology for Lot Acceptance"):
        st.markdown(f"""
        #### Mathematical Basis
        This calculation is rooted in probability theory. The choice of model depends on the nature of the population being sampled.
        ---
        #### 1. Binomial Model (Large Lot / Continuous Process)
        **Assumption:** The population is effectively **infinite**. Each sample is independent, and the act of sampling does not change the underlying defect rate of the process.
        **Formula:** We solve for `n` in the inequality `R^n <= 1 - C`, which gives:
        """)
        st.latex(r''' n \ge \frac{\ln(1 - C)}{\ln(R)} ''')
        st.markdown("---")
        st.markdown("""
        ##### 2. Hypergeometric Model (Finite Lot)
        **Assumption:** Sampling is done **without replacement** from a discrete lot of a known, finite size `M`.
        **Formula:** We iterate to find the smallest integer `n` that satisfies:
        """)
        st.latex(r''' P(X=0) = \frac{\binom{M-D}{n}}{\binom{M}{n}} \le 1 - C ''')
        st.markdown("""
        Where `M` is Lot Size, `D` is max allowable defects (`floor((1-R) * M)`), and `n` is Sample Size.
        """)
#========================================================================== 2. ADVANCED STABILITY DESIGN ==================================
def render_stability_design():
    """Renders the comprehensive, interactive module for Stability Study Design."""
    st.markdown("""
    #### Purpose & Application: Strategic Cost-Savings in Stability
    **Purpose:** To demonstrate how to apply risk-based, statistically justified strategies (**Bracketing** and **Matrixing**) to reduce the cost and complexity of large-scale stability studies, as outlined in **ICH Q1D**.
    
    **Strategic Application:** For a product with many variations (e.g., multiple strengths and container types), testing every combination at every timepoint is prohibitively expensive. This tool provides a powerful way for a validation leader to design and justify a resource-saving validation strategy to management and regulators without compromising compliance or scientific rigor.
    """)
    
    st.info("""
    **Interactive Demo:** You are the Stability Program Manager.
    1.  Use the **Product Factors** selectors in the sidebar to define the complexity of your product.
    2.  Select a **Reduced Study Design** to see how it compares to the full study.
    3.  The **Heatmap Tables** visualize the number of stability pulls required for each design, and the KPIs quantify the cost savings.
    """)

    with st.sidebar:
        st.subheader("Stability Design Controls")
        strengths = st.multiselect("Select Product Strengths (mg)", [10, 25, 50, 100], default=[10, 25, 50, 100],
            help="The different dosage strengths of the product. The more strengths, the greater the benefit of a reduced design.")
        containers = st.multiselect("Select Container Types", ["Vial", "Pre-filled Syringe"], default=["Vial", "Pre-filled Syringe"],
            help="The different primary packaging configurations for the product.")
        design_type = st.radio("Select Reduced Study Design", ["Bracketing", "Matrixing"],
            help="**Bracketing:** Tests only the extremes (e.g., lowest and highest strengths). **Matrixing:** Tests a strategic subset of all combinations at specific timepoints.")

    if not strengths or not containers:
        st.warning("Please select at least one Strength and one Container type.")
        return

    fig_full, fig_reduced, pulls_saved, total_full, total_reduced = plot_stability_design_comparison(strengths, containers, design_type)

    st.header("Stability Study Design Comparison")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Pulls (Full Study)", f"{total_full}")
    col2.metric("Total Pulls (Reduced Study)", f"{total_reduced}")
    col3.metric("Samples Saved (%)", f"{pulls_saved} ({pulls_saved/total_full:.0%})")

    col_full, col_reduced = st.columns(2)
    with col_full:
        st.plotly_chart(fig_full, use_container_width=True)
    with col_reduced:
        st.plotly_chart(fig_reduced, use_container_width=True)
        
    st.divider()
    st.subheader("Deeper Dive")
    tabs = st.tabs(["💡 Key Insights", "📋 Glossary", "✅ The Golden Rule", "📖 Theory & History", "🏛️ Regulatory & Compliance"])
    with tabs[0]:
        st.markdown("""
        **Interpreting the Designs:**
        - **Full Study:** This is the baseline, testing every combination of strength and container at every timepoint. It is the most comprehensive but also the most expensive.
        - **Bracketing:** This design assumes that the stability of the intermediate strengths is represented by the stability at the extremes. It is a powerful cost-saver, but requires strong scientific justification. As you can see, all intermediate strength/container combinations are removed from the study.
        - **Matrixing:** This design assumes that the stability of the product is similar across all combinations. It reduces the testing burden by, for example, testing only half the samples at each timepoint in a checkerboard pattern. It provides more information about the entire product family than bracketing.
        
        **The Strategic Insight:** The choice between these designs is a risk-based decision. **Bracketing** is ideal when the primary stability risk is related to the extremes (e.g., highest strength has a new excipient, lowest strength is more susceptible to degradation). **Matrixing** is ideal when the process and formulation are very consistent across all combinations and you want to reduce the overall testing load.
        """)
    with tabs[1]:
        st.markdown("""
        ##### Glossary of Stability Design Terms
        - **Stability Study:** A formal study to determine the time period during which a drug product remains within its specifications under defined storage conditions.
        - **Full Design:** A study design in which samples for every combination of all design factors (e.g., strength, container size, lot) are tested at all timepoints.
        - **Reduced Design:** A study design in which samples for every factor combination are not all tested at all timepoints.
        - **Bracketing:** A reduced design in which only samples on the extremes of certain design factors (e.g., lowest and highest strengths) are tested at all timepoints.
        - **Matrixing:** A reduced design in which a selected subset of the total number of possible samples is tested at a specified timepoint. At a subsequent timepoint, a different subset of samples is tested.
        """)
    with tabs[2]:
        st.error("""🔴 **THE INCORRECT APPROACH: "Ad-Hoc Reduction"**
A team is running over budget and decides to arbitrarily skip testing some samples from their full stability protocol without pre-approval or a statistical justification.
- **The Flaw:** This invalidates the entire study. The missing data creates unexplainable gaps, and regulators will reject the study, forcing a costly repeat. It introduces unmanaged risk.""")
        st.success("""🟢 **THE GOLDEN RULE: Justify, Document, and Pre-Approve**
A reduced stability design is a powerful, compliant strategy *only* if it is handled with formal discipline.
1.  **Justify:** The choice of a reduced design (and the specific design chosen) must be justified based on scientific knowledge and a formal risk assessment (FMEA).
2.  **Document:** The justification and the exact bracketing or matrixing plan must be explicitly detailed in the Stability Protocol.
3.  **Pre-Approve:** The protocol must be reviewed and approved by Quality Assurance *before* the study begins.""")
        
    with tabs[3]:
        st.markdown("""
        #### Historical Context: The Cost of Complexity
        As the pharmaceutical industry grew in the latter half of the 20th century, the complexity of product portfolios exploded. A single drug product might be marketed in five different strengths, three different container sizes, and be manufactured in two different facilities. The traditional expectation of running a full, multi-year stability study on every single combination became an enormous and often scientifically redundant financial burden.
        
        Regulators and industry, working through the **International Council for Harmonisation (ICH)**, recognized this challenge. They sought to create a scientifically sound, risk-based framework that would allow companies to reduce their testing burden without compromising patient safety. This led to the development of the **ICH Q1D guideline**, which formally defined and sanctioned the use of Bracketing and Matrixing designs for formal stability studies.
        """)
        
    with tabs[4]:
        st.markdown("""
        The design of stability studies is governed by a specific set of harmonized international guidelines.
        - **ICH Q1D - Bracketing and Matrixing Designs for Stability Testing:** This is the primary global guideline that provides the rationale, definitions, and requirements for implementing and justifying reduced stability study designs.
        - **ICH Q1A(R2) - Stability Testing of New Drug Substances and Products:** This guideline defines the overall requirements for stability programs, and Q1D serves as a specific appendix to it.
        - **FDA Guidance for Industry - Q1D Bracketing and Matrixing:** The FDA's formal adoption and implementation of the ICH guideline, making it a regulatory expectation in the United States.
        """)

#======================================================================= 3. METHOD COMPARISON ========================================================================
def render_method_comparison():
    """Renders the INTERACTIVE module for Method Comparison."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To formally assess and quantify the degree of agreement and systemic bias between two different measurement methods intended to measure the same quantity.
    
    **Strategic Application:** This study is the "crucible" of method transfer, validation, or replacement. It answers the critical business and regulatory question: “Do these two methods produce the same result, for the same sample, within medically or technically acceptable limits?”
    """)
    
    st.info("""
    **Interactive Demo:** Use the sliders at the bottom of the sidebar to simulate different types of disagreement between a "Test" method and a "Reference" method. See in real-time how each diagnostic plot (Deming, Bland-Altman, %Bias) reveals a different aspect of the problem, helping you build a deep intuition for method comparison statistics.
    """)
    
    # --- Sidebar controls for this specific module ---
    st.sidebar.subheader("Method Comparison Controls")
    constant_bias_slider = st.sidebar.slider(
        "⚖️ Constant Bias", 
        min_value=-10.0, max_value=10.0, value=2.0, step=0.5,
        help="A fixed offset where the Test method reads consistently higher (+) or lower (-) than the Reference method across the entire range."
    )
    proportional_bias_slider = st.sidebar.slider(
        "📈 Proportional Bias (%)", 
        min_value=-10.0, max_value=10.0, value=3.0, step=0.5,
        help="A concentration-dependent error. A positive value means the Test method reads progressively higher than the Reference at high concentrations."
    )
    random_error_slider = st.sidebar.slider(
        "🎲 Random Error (SD)", 
        min_value=0.5, max_value=10.0, value=3.0, step=0.5,
        help="The imprecision or 'noise' of the methods. Higher error widens the Limits of Agreement on the Bland-Altman plot."
    )

    # Generate plots using the slider values
    fig, slope, intercept, bias, ua, la = plot_method_comparison(
        constant_bias=constant_bias_slider,
        proportional_bias=proportional_bias_slider,
        random_error_sd=random_error_slider
    )
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "📋 Glossary", "✅ Acceptance Criteria", "📖 Theory & History", "🏛️ Regulatory & Compliance"])
        
        with tabs[0]:
            st.metric(label="📈 Mean Bias (Bland-Altman)", value=f"{bias:.2f} units", help="The average systematic difference.")
            st.metric(label="💡 Deming Slope", value=f"{slope:.3f}", help="Ideal = 1.0. Measures proportional bias.")
            st.metric(label="💡 Deming Intercept", value=f"{intercept:.2f}", help="Ideal = 0.0. Measures constant bias.")
            
            st.info("Play with the sliders in the sidebar and observe the plots!")
            st.markdown("""
            - **Add `Constant Bias`:** The Deming line shifts up/down but stays parallel to the identity line. The Bland-Altman plot's mean bias line moves away from zero.
            - **Add `Proportional Bias`:** The Deming line *rotates* away from the identity line. The Bland-Altman and %Bias plots now show a clear trend, a major red flag.
            - **Increase `Random Error`:** The points scatter more widely. This has little effect on the average bias but dramatically **widens the Limits of Agreement**, making the methods less interchangeable.
            """)
        with tabs[1]:
            st.markdown("""
            ##### Glossary of Comparison Terms
            - **Agreement:** The degree to which two different methods produce the same result for the same sample. This is different from correlation.
            - **Constant Bias:** A systematic error where one method consistently reads higher or lower than the other by a fixed amount across the entire measurement range.
            - **Proportional Bias:** A systematic error where the difference between the two methods is concentration-dependent, typically increasing as the concentration increases.
            - **Passing-Bablok Regression:** A robust, non-parametric regression method that is insensitive to outliers and assumes error in both methods. It is superior to OLS for method comparison.
            - **Bland-Altman Plot:** A graphical method to plot the difference between two measurements against their average. It is the gold standard for visualizing agreement and identifying bias.
            - **Limits of Agreement (LoA):** On a Bland-Altman plot, the interval `[mean_diff ± 1.96 * std_diff]` within which 95% of future differences are expected to fall.
            """)
        with tabs[2]:
            st.markdown("Acceptance criteria must be pre-defined and clinically/technically justified.")
            st.markdown("- **Deming Regression:** The 95% confidence interval for the **slope must contain 1.0**, and the 95% CI for the **intercept must contain 0**.")
            st.markdown(f"- **Bland-Altman:** The primary criterion is that the **95% Limits of Agreement (`{la:.2f}` to `{ua:.2f}`) must be clinically or technically acceptable**.")
            st.error("""
            **The Correlation Catastrophe:** Never use the correlation coefficient (R²) to assess agreement. Two methods can be perfectly correlated (R²=1.0) but have a huge bias (e.g., one method always reads twice as high).
            """)

        with tabs[3]:
            # FIX: Restored the full, detailed content for this tab
            st.markdown("""
            #### Historical Context & Origin
            For decades, scientists committed a cardinal sin: using **Ordinary Least Squares (OLS) regression** and the **correlation coefficient (r)** to compare methods. This is flawed because OLS assumes the x-axis (reference method) is measured without error, an impossibility.
            
            - **Deming's Correction:** While known to statisticians, **W. Edwards Deming** championed this type of regression in the 1940s. It correctly assumes both methods have measurement error, providing an unbiased estimate of the true relationship. **Passing-Bablok regression** is a robust non-parametric alternative.
            
            - **The Bland-Altman Revolution:** A 1986 paper in *The Lancet* by **J. Martin Bland and Douglas G. Altman** ruthlessly exposed the misuse of correlation and proposed their brilliantly simple alternative. Instead of plotting Y vs. X, they plotted the **Difference (Y-X) vs. the Average ((Y+X)/2)**. This directly visualizes the magnitude and patterns of disagreement and is now the undisputed gold standard.
            
            #### Mathematical Basis
            **Deming Regression:** OLS minimizes the sum of squared vertical distances. Deming regression minimizes the sum of squared distances from the points to the line, weighted by the ratio of the error variances of the two methods.
            
            **Bland-Altman Plot:** This is a graphical analysis. The key metrics are the **mean difference (bias)**, $\bar{d}$, and the **standard deviation of the differences**, $s_d$. The 95% Limits of Agreement (LoA) are calculated assuming the differences are approximately normally distributed:
            """)
            st.latex(r"LoA = \bar{d} \pm 1.96 \cdot s_d")
            st.markdown("This interval provides a predictive range: we can be 95% confident that the difference between the two methods for a future sample will fall within these limits.")
        with tabs[4]:
            st.markdown("""
            Method comparison studies are essential for method transfer, validation of a new method against a standard, or bridging studies.
            - **ICH Q2(R1) - Validation of Analytical Procedures:** The principles of comparing methods fall under the assessment of **Accuracy** and **Intermediate Precision**.
            - **USP General Chapter <1224> - Transfer of Analytical Procedures:** This chapter is entirely dedicated to the process of qualifying a laboratory to use an analytical test procedure. It explicitly mentions "Comparative Testing" as a transfer option, for which Bland-Altman and Deming regression are the standard analysis tools.
            - **CLIA (Clinical Laboratory Improvement Amendments):** In the US, clinical labs are required to perform method comparison studies to validate new tests.
            """)

#===============================================================  4. EQUIVALENCE TESTING (TOST) ================================================
def render_tost():
    """Renders the INTERACTIVE module for Two One-Sided Tests (TOST) for equivalence."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To statistically prove that two methods or groups are **equivalent** within a predefined, practically insignificant margin. This flips the logic of standard hypothesis testing from trying to prove a difference to trying to prove a lack of meaningful difference.
    
    **Strategic Application:** This is the statistically rigorous way to handle comparisons where the goal is to prove similarity, not difference, such as in biosimilarity studies, analytical method transfers, or validating a new manufacturing site.
    """)
    
    st.info("""
    **Interactive Demo:** This new 3-plot dashboard tells a complete story.
    1.  See the raw sample data at the top.
    2.  Watch how that translates into the evidence about the *difference* in the middle plot.
    3.  See the final verdict at the bottom. The bar in Plot 3 is just a summary of the shaded area in Plot 2.
    """)
    
    with st.sidebar:
        st.subheader("TOST Controls")
        delta_slider = st.slider(
            "⚖️ Equivalence Margin (Δ)", 1.0, 15.0, 5.0, 0.5,
            help="The 'goalposts'. Defines the zone where differences are considered practically meaningless."
        )
        diff_slider = st.slider(
            "🎯 True Difference", -10.0, 10.0, 1.0, 0.5,
            help="The actual underlying difference between the two groups in the simulation."
        )
        sd_slider = st.slider(
            "🌫️ Standard Deviation (Variability)", 1.0, 15.0, 5.0, 0.5,
            help="The random noise in the data. Higher variability widens the CI, making equivalence harder to prove."
        )
        n_slider = st.slider(
            "🔬 Sample Size (n)", 10, 200, 50, 5,
            help="The number of samples per group. Higher sample size narrows the CI, increasing your power."
        )
    
    fig, p_tost, is_equivalent, ci_lower, ci_upper, mean_A, mean_B, diff_mean = plot_tost(
        delta=delta_slider,
        true_diff=diff_slider,
        std_dev=sd_slider,
        n_samples=n_slider
    )
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "📋 Glossary", "✅ The Golden Rule", "📖 Theory & History", "🏛️ Regulatory & Compliance"])
        
        with tabs[0]:
            status = "✅ EQUIVALENT" if is_equivalent else "❌ NOT EQUIVALENT"
            if is_equivalent:
                st.success(f"### Result: {status}")
            else:
                st.error(f"### Result: {status}")

            st.metric(label="p-value (TOST)", value=f"{p_tost:.4f}", help="If p < 0.05, we conclude equivalence.")
            st.metric(label="📊 Observed 90% CI for Difference", value=f"[{ci_lower:.2f}, {ci_upper:.2f}]")
            st.metric(label="📈 Observed Difference", value=f"{diff_mean:.2f}",
                      help="The difference between the two sample means (Mean B - Mean A).")
            st.metric(label="⚖️ Equivalence Margin", value=f"± {delta_slider:.1f} units")

            st.markdown("---")
            st.markdown("##### The 3-Plot Story: How the Plots Connect")
            st.markdown("""
            1.  **Plot 1 (The Samples):** Shows the raw data you collected. The vertical dashed lines mark the *mean* of each sample.
            2.  **Plot 2 (The Evidence):** This is the crucial link. It shows our statistical uncertainty about the true difference in means. The shaded area is the **90% Confidence Interval**.
            3.  **Plot 3 (The Verdict):** This is just a compact summary of Plot 2. The bar represents the exact same 90% Confidence Interval.

            **The conclusion of 'Equivalence' is reached when the entire shaded distribution in Plot 2 falls inside the light green 'Equivalence Zone'.**
            """)
        with tabs[1]:
            st.markdown("""
            ##### Glossary of Equivalence Terms
            - **Equivalence Testing:** A statistical procedure used to demonstrate that the difference between two groups or methods is smaller than a pre-specified, practically meaningless amount.
            - **TOST (Two One-Sided Tests):** The standard statistical method for performing an equivalence test. It involves testing two separate null hypotheses of "too different."
            - **Equivalence Margin (Δ):** A pre-defined range `[-Δ, +Δ]` within which two products or methods are considered to be practically equivalent. Setting this margin is a critical, risk-based decision.
            - **Confidence Interval Approach:** An equivalent method to TOST. If the 90% confidence interval for the difference between the two groups falls entirely within the equivalence margin, equivalence is demonstrated at the 5% significance level.
            """)
        with tabs[2]:
            st.error("""🔴 **THE INCORRECT APPROACH: The Fallacy of the Non-Significant P-Value**
- A scientist runs a standard t-test and gets a p-value of 0.25. They exclaim, *"Great, p > 0.05, so the methods are the same!"*
- **This is wrong.** All they have shown is a *failure to find evidence of a difference*. **Absence of evidence is not evidence of absence.** Their study may have been underpowered (too much noise or too few samples).""")
            st.success("""🟢 **THE GOLDEN RULE: Define 'Same Enough', Then Prove It**
The TOST procedure forces a more rigorous scientific approach.
1.  **First, Define the Margin:** Before collecting data, stakeholders must use scientific and clinical judgment to define the equivalence margin (`Δ`). This is the zone where a difference is considered practically meaningless.
2.  **Then, Prove You're Inside:** Conduct the experiment. The burden of proof is on you to show that your evidence (the 90% CI for the difference) is precise enough to fall entirely within that pre-defined margin.""")

        with tabs[3]:
            st.markdown("""
            #### Historical Context: The Rise of Generic Drugs
            **The Problem:** In the early 1980s, the pharmaceutical landscape was changing. The **1984 Hatch-Waxman Act** in the US created the modern pathway for generic drug approval. This created a new statistical challenge for regulators: how could a generic manufacturer *prove* that their drug was "the same as" the innovator's drug in terms of how it was absorbed by the body (bioequivalence)?

            **The 'Aha!' Moment:** A standard t-test was useless; failing to find a difference wasn't proof of no difference. The solution was championed by statisticians like **Donald J. Schuirmann** at the FDA. The **Two One-Sided Tests (TOST)** procedure, which had existed in the statistical literature, was identified as the perfect tool.
            
            **The Impact:** The TOST procedure became the statistical engine for bioequivalence studies worldwide. Instead of one null hypothesis of "no difference," it brilliantly frames the problem with two null hypotheses of "too different." To prove equivalence, you must reject both. This places the burden of proof squarely on the manufacturer to demonstrate similarity, a much higher and more appropriate standard for ensuring patient safety.
            """)
            st.markdown("#### Mathematical Basis")
            st.markdown("TOST brilliantly flips the null hypothesis. Instead of one null of \"no difference\" (`H₀: μ₁ - μ₂ = 0`), you have two null hypotheses of \"too different\":")
            st.latex(r"H_{01}: \mu_B - \mu_A \leq -\Delta \quad (\text{The difference is too low})")
            st.latex(r"H_{02}: \mu_B - \mu_A \geq +\Delta \quad (\text{The difference is too high})")
            st.markdown("""
            You must run two separate one-sided t-tests to reject **both** of these null hypotheses. The overall p-value for the TOST procedure is the larger of the two individual p-values. If this final p-value is less than your alpha (e.g., 0.05), you have statistically demonstrated equivalence within the margin `[-Δ, +Δ]`.
            
            A mathematically equivalent shortcut is to calculate the **90% confidence interval** for the difference. If this entire interval falls within `[-Δ, +Δ]`, you can conclude equivalence at the 5% significance level.
            """)
            
        with tabs[4]:
            st.markdown("""
            TOST is the required statistical method for demonstrating similarity or equivalence in various regulated contexts.
            - **FDA Guidance on Bioequivalence Studies:** TOST is the standard method for proving that the rate and extent of absorption of a generic drug are not significantly different from the reference listed drug.
            - **USP General Chapter <1224> - Transfer of Analytical Procedures:** Suggests the use of equivalence testing to formally demonstrate that a receiving laboratory can obtain comparable results to the transferring laboratory.
            - **Biosimilars (BPCIA Act):** The principles of equivalence testing are central to the analytical and clinical studies required to demonstrate biosimilarity to a reference biologic product.
            """)
#===============================================================  5. STATISTICAL EQUIVALENCE FOR PROCESS TRANSFER ================================================
def render_process_equivalence():
    """Renders the comprehensive, interactive module for Process Transfer Equivalence."""
    st.markdown("""
    #### Purpose & Application: Statistical Proof of Transfer Success
    **Purpose:** To provide **objective, statistical proof** that a manufacturing process transferred to a new site, scale, or equipment set performs equivalently to the original, validated process.
    
    **Strategic Application:** This is a high-level validation activity that goes beyond simply showing the new site is "in control." It formally proves that the new process is **statistically indistinguishable** from the original, providing powerful evidence for regulatory filings and ensuring consistent product quality across a global network. It is the final exam of a technology transfer.
    """)
    
    st.info("""
    **Interactive Demo:** You are the Head of Tech Transfer. Use the sidebar controls to simulate the performance of the new manufacturing site (Site B).
    - The dashboard tells a 3-part story: the **raw process comparison** (top), the **statistical evidence** about the difference (middle), and the **final verdict** (bottom).
    - **The Goal:** Achieve a "PASS" verdict by ensuring the entire evidence distribution in Plot 2 falls within the red equivalence margins.
    """)
    
    with st.sidebar:
        st.subheader("Process Equivalence Controls")
        st.markdown("**Baseline Process**")
        cpk_a_slider = st.slider("Original Site A Performance (Cpk)", 1.33, 2.5, 1.67, 0.01, help="The historical, validated process capability of the sending site. This is your benchmark.")
        st.markdown("**New Process Simulation**")
        mean_shift_slider = st.slider("Mean Shift at Site B", -2.0, 2.0, 0.5, 0.1, help="Simulates a systematic bias or shift in the process average at the new site. A key risk in tech transfer.")
        var_change_slider = st.slider("Variability Change Factor at Site B", 0.8, 1.5, 1.1, 0.05, help="Simulates a change in process precision. >1.0 means the new site is more variable (worse); <1.0 means it is less variable (better).")
        st.markdown("**Statistical Criteria**")
        n_samples_slider = st.slider("Samples per Site (n)", 30, 200, 50, 10, help="The number of samples taken during the PPQ runs at each site. More samples increase statistical power.")
        margin_slider = st.slider("Equivalence Margin for Cpk (±)", 0.1, 0.5, 0.2, 0.05, help="The 'goalposts'. How much can the new site's Cpk differ from the original and still be considered equivalent? This is a risk-based decision.")

    fig, is_equivalent, diff_cpk, cpk_a_sample, cpk_b_sample, ci_lower, ci_upper = plot_process_equivalence(
        cpk_site_a=cpk_a_slider, mean_shift=mean_shift_slider,
        var_change_factor=var_change_slider, n_samples=n_samples_slider,
        margin=margin_slider
    )
    
    st.header("Results Dashboard")
    col1, col2 = st.columns([0.65, 0.35])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Analysis & Interpretation")
        if is_equivalent:
            st.success("### Verdict: ✅ PASS - Processes are Equivalent")
        else:
            st.error("### Verdict: ❌ FAIL - Processes are NOT Equivalent")
        
        c1, c2 = st.columns(2)
        c1.metric("Site A Sample Cpk", f"{cpk_a_sample:.2f}")
        c2.metric("Site B Sample Cpk", f"{cpk_b_sample:.2f}", delta=f"{(diff_cpk):.2f} vs Site A")
        
        st.metric("90% CI for Cpk Difference", f"[{ci_lower:.2f}, {ci_upper:.2f}]", help="The range of plausible true differences between the sites' Cpk values, based on the sample data.")
        st.metric("Equivalence Margin", f"± {margin_slider}", help="The pre-defined goalposts for success.")
        
    st.divider()
    st.subheader("Deeper Dive")
    tabs = st.tabs(["💡 Key Insights", "📋 Glossary", "✅ The Golden Rule", "📖 Theory & History", "🏛️ Regulatory & Compliance"])
    
    with tabs[0]:
        st.markdown("""
        **The 3-Plot Story: From Data to Decision**
        1.  **Plot 1 (Process Comparison):** This shows what you see in the raw data from the validation runs. The smooth curves represent the "Voice of the Process" for each site relative to the specification limits (the "Voice of the Customer").
        2.  **Plot 2 (Statistical Evidence):** This is the crucial bridge. It shows the result of a bootstrap simulation—a histogram of all the likely "true" differences in Cpk between the sites. The shaded area is the 90% confidence interval, representing our statistical evidence.
        3.  **Plot 3 (The Verdict):** This is a simple summary of Plot 2. The colored bar is the same 90% confidence interval. **The test passes only if this entire bar is inside the equivalence zone defined by the red dashed lines.**
        
        **Core Insight:** A tech transfer doesn't just need to produce good product (high Cpk); it needs to produce product that is **statistically consistent** with the original site. This analysis provides the formal proof. Notice how a small `Mean Shift` or an increase in `Variability` at Site B can quickly lead to a failed equivalence test, even if Site B's Cpk is still above 1.33.
        """)
    with tabs[1]:
        st.markdown("""
        ##### Glossary of Transfer Terms
        - **Technology Transfer:** The process of transferring skills, knowledge, technologies, and methods of manufacturing among organizations to ensure that scientific and technological developments are accessible to a wider range of users.
        - **Process Performance Qualification (PPQ):** Stage 2 of the FDA Process Validation lifecycle, where the process design is evaluated to determine if it is capable of reproducible commercial manufacturing.
        - **Cpk (Process Capability Index):** A key performance indicator for a manufacturing process. A high Cpk (>1.33) indicates a capable process.
        - **Equivalence Testing:** A statistical procedure used to demonstrate that the difference in performance between two processes (e.g., the original and transferred site) is smaller than a pre-specified, practically meaningless amount.
        - **Bootstrap Simulation:** A computer-intensive statistical method that uses resampling of the original data to estimate the sampling distribution and confidence intervals of a statistic (like the difference in Cpk).
        """)
    with tabs[2]:
        st.error("""🔴 **THE INCORRECT APPROACH: The "Cpk is Cpk" Fallacy**
A manager reviews the Site B PPQ data, sees a Cpk of 1.40 (which is > 1.33), and declares the transfer a success, even though the original site's Cpk was 1.80.
- **The Flaw:** This significant drop in performance is ignored, introducing a new, hidden level of risk into the manufacturing network. The process is now less robust and more likely to fail in the future. They have proven capability, but not comparability.""")
        st.success("""🟢 **THE GOLDEN RULE: Pre-Define Equivalence, Then Prove It**
A robust tech transfer plan treats equivalence as a formal acceptance criterion.
1.  **Define the Margin:** Before the transfer, stakeholders must agree on the equivalence margin for a key performance metric (like Cpk). This is a risk-based decision: how much of a performance drop are we willing to accept?
2.  **Prove You're Inside:** Conduct the PPQ runs and perform the equivalence test. The burden of proof is on the receiving site to demonstrate that their process performance is statistically indistinguishable from the sending site, within the pre-defined margin.""")

    with tabs[3]:
        st.markdown("""
        #### Historical Context: A Modern Synthesis
        This tool represents a modern synthesis of two powerful statistical ideas that both came to prominence in the 1980s but in different industries:
        1.  **Process Capability (Cpk):** Popularized by the **Six Sigma** movement at Motorola, Cpk became the universal language for quantifying how well a process fits within its specification limits. It answered the question, "Is our process good enough?"
        2.  **Equivalence Testing (TOST):** Championed by the **FDA** for generic drug approvals, equivalence testing provided the rigorous framework for proving two things were "the same" within a practical margin. It answered the question, "Is Drug B the same as Drug A?"
        
        **The Impact:** In modern tech transfer and lifecycle management, these two ideas are fused. By applying the rigorous logic of equivalence testing to a key performance indicator like Cpk, we create a powerful, modern tool for validating process transfers, scale-up, and other post-approval changes. The use of computer-intensive **bootstrapping** to calculate the confidence interval for a complex metric like Cpk is a distinctly 21st-century statistical technique that makes this analysis possible.
        """)
        
    with tabs[4]:
        st.markdown("""
        This analysis is a best-practice implementation for several key regulatory activities that require demonstrating comparability.
        - **FDA Process Validation Guidance:** This tool is ideal for **Stage 2 (Process Qualification)** when transferring a process. It provides objective evidence that the receiving site has successfully reproduced the performance of the sending site.
        - **ICH Q5E - Comparability of Biotechnological/Biological Products:** While this guideline focuses on product quality attributes, its core principle is demonstrating comparability after a manufacturing process change. This statistical approach provides a quantitative framework for that demonstration.
        - **Technology Transfer (ICH Q10):** A robust tech transfer protocol should have pre-defined acceptance criteria. Proving statistical equivalence of process capability is a state-of-the-art criterion.
        - **SUPAC (Scale-Up and Post-Approval Changes):** When making a change to a validated process, this analysis can be used to prove that the change has not adversely impacted process performance.
        """)

#===============================================================  6. PROCESS STABILITY (SPC) ================================================
def render_spc_charts():
    """Renders the INTERACTIVE module for Statistical Process Control (SPC) charts."""
    st.markdown("""
    #### Purpose & Application: The Voice of the Process
    **Purpose:** To serve as an **EKG for your process**—a real-time heartbeat monitor that visualizes its stability. The goal is to distinguish between two fundamental types of variation:
    - **Common Cause Variation:** The natural, random "static" or "noise" inherent to a stable process. It's predictable.
    - **Special Cause Variation:** A signal that something has changed or gone wrong. It's unpredictable and requires investigation.
    
    **Strategic Application:** SPC is the bedrock of modern quality control. These charts provide an objective, data-driven answer to the critical question: "Is my process stable and behaving as expected?" They are used to prevent defects, reduce waste, and provide the evidence needed to justify (or reject) process changes.
    """)
    
    st.info("""
    **Interactive Demo:** Use the controls at the bottom of the sidebar to inject different types of "special cause" events into a simulated stable process. Observe how the I-MR, Xbar-R, and P-Charts each respond, helping you learn to recognize the visual signatures of common process problems.
    """)
    
    st.sidebar.subheader("SPC Scenario Controls")
    scenario = st.sidebar.radio(
        "Select a Process Scenario to Simulate:",
        ('Stable', 'Sudden Shift', 'Gradual Trend', 'Increased Variability'),
        captions=[
            "Process is behaving normally.",
            "e.g., A new raw material lot is introduced.",
            "e.g., An instrument is slowly drifting out of calibration.",
            "e.g., An operator becomes less consistent."
        ]
    )

    fig_imr, fig_xbar, fig_p = plot_spc_charts(scenario=scenario)
    
    st.subheader(f"Analysis & Interpretation: {scenario} Process")
    tabs = st.tabs(["💡 Key Insights", "📋 Glossary", "✅ The Golden Rule", "📖 Theory & History", "🏛️ Regulatory & Compliance"])

    with tabs[0]:
        st.info("💡 Each chart type is a different 'lead' on your EKG, designed for a specific kind of data. Use the expanders below to see how to read each one.")

        with st.expander("Indivduals & Moving Range (I-MR) Chart", expanded=True):
            st.plotly_chart(fig_imr, use_container_width=True)
            st.markdown("- **Interpretation:** The I-chart tracks the process center, while the MR-chart tracks short-term variability. **Both** must be stable. An out-of-control MR chart is a leading indicator of future problems.")

        with st.expander("X-bar & Range (X̄-R) Chart", expanded=True):
            st.plotly_chart(fig_xbar, use_container_width=True)
            st.markdown("- **Interpretation:** The X-bar chart tracks variation *between* subgroups and is extremely sensitive to small shifts. The R-chart tracks variation *within* subgroups, a measure of process consistency.")
        
        with st.expander("Proportion (P) Chart", expanded=True):
            st.plotly_chart(fig_p, use_container_width=True)
            st.markdown("- **Interpretation:** This chart tracks the proportion of defects. The control limits become tighter for larger batches, reflecting increased statistical certainty.")
        with tabs[1]:
            st.markdown("""
            ##### Glossary of SPC Terms
            - **SPC (Statistical Process Control):** A method of quality control which employs statistical methods to monitor and control a process.
            - **Control Chart:** A graph used to study how a process changes over time. Data are plotted in time order.
            - **Control Limits:** Horizontal lines on a control chart (typically at ±3σ) that represent the natural variation of a process. They are calculated from the process data itself.
            - **Common Cause Variation:** The natural, random variation inherent in a stable process. It is the "noise" of the system.
            - **Special Cause Variation:** Variation that is not inherent to the process and is caused by a specific, assignable event (e.g., a machine malfunction, a new operator). It is the "signal" that something has changed.
            - **In a State of Statistical Control:** A process from which all special causes of variation have been removed, leaving only common cause variation. Such a process is stable and predictable.
            """)
    with tabs[2]:
        st.error("""🔴 **THE INCORRECT APPROACH: "Process Tampering"**
        This is the single most destructive mistake in SPC. The operator sees any random fluctuation within the control limits and reacts as if it's a real problem.
        - *"This point is a little higher than the last one, I'll tweak the temperature down a bit."*
        Reacting to "common cause" noise as if it were a "special cause" signal actually **adds more variation** to the process, making it worse. This is like trying to correct the path of a car for every tiny bump in the road—you'll end up swerving all over the place.""")
        st.success("""🟢 **THE GOLDEN RULE: Know When to Act (and When Not To)**
        The control chart's signal dictates one of two paths:
        1.  **Process is IN-CONTROL (only common cause variation):**
            - **Your Action:** Leave the process alone! To improve, you must work on changing the fundamental system (e.g., better equipment, new materials).
        2.  **Process is OUT-OF-CONTROL (a special cause is present):**
            - **Your Action:** Stop! Investigate immediately. Find the specific, assignable "special cause" for that signal and eliminate it.""")

    with tabs[3]:
        st.markdown("""
        #### Historical Context: The Birth of Modern Quality
        **The Problem:** In the early 1920s, manufacturing at Western Electric for the Bell Telephone system was a chaotic affair. The challenge was immense: how could you ensure consistency across millions of components when you couldn't tell the difference between normal, random variation and a real production problem? Engineers were lost in a "fog" of data, constantly "tampering" with the process based on noise, often making things worse.

        **The 'Aha!' Moment:** A brilliant physicist at Bell Labs, **Dr. Walter A. Shewhart**, had a revolutionary insight. In a famous 1924 internal memo, he was the first to formally articulate the critical distinction between what he called **"chance cause"** (common cause) and **"assignable cause"** (special cause) variation. He realized that as long as a process only exhibited chance cause variation, it was stable, predictable, and in a "state of statistical control."
        
        **The Impact:** The control chart was the simple, graphical tool he invented to detect the exact moment an assignable cause entered the system. This single idea was the birth of modern Statistical Process Control and laid the foundation for the entire 20th-century quality revolution, influencing giants like W. Edwards Deming and the rise of Japanese manufacturing excellence.
        """)
        st.markdown("#### Mathematical Basis")
        st.markdown("The control limits on a Shewhart chart are famously set at the process average plus or minus three standard deviations of the statistic being plotted.")
        st.latex(r"\text{Control Limits} = \mu \pm 3\sigma_{\text{statistic}}")
        st.markdown("""
        - **Why 3-Sigma?** Shewhart chose this value for sound economic and statistical reasons. For a normally distributed process, 99.73% of all data points will naturally fall within these limits. This means there's only a 0.27% chance of a point falling outside the limits purely by chance (a false alarm). This makes the chart robust; when you get a signal, you can be very confident it's real. It strikes an optimal balance between being sensitive to real problems and not causing "fire drills" for false alarms.
        - **Estimating Sigma:** In practice, the true `σ` is unknown. For an I-MR chart, it is estimated from the average moving range (`MR-bar`) using a statistical constant `d₂`:
        """)
        st.latex(r"\hat{\sigma} = \frac{\overline{MR}}{d_2}")
    with tabs[4]:
        st.markdown("""
        SPC is the primary tool for Stage 3 of the process validation lifecycle, known as Continued or Ongoing Process Verification (CPV/OPV).
        - **FDA Process Validation Guidance (Stage 3):** Explicitly states that "an ongoing program to collect and analyze product and process data... must be established." SPC charts are the standard method for this real-time monitoring.
        - **ICH Q7 - Good Manufacturing Practice for APIs:** Section 2.5 on Quality Risk Management discusses the importance of monitoring and reviewing process performance.
        - **21 CFR 211.110(a):** Requires the establishment of control procedures "to monitor the output and to validate the performance of those manufacturing processes that may be responsible for causing variability."
        """)
#======================================================= 7. PROCESS CAPABILITY (CpK  ============================================================================
def render_capability():
    """Renders the interactive module for Process Capability (Cpk)."""
    st.markdown("""
    #### Purpose & Application: Voice of the Process vs. Voice of the Customer
    **Purpose:** To quantitatively determine if a process, once proven to be in a state of statistical control, is **capable** of consistently producing output that meets pre-defined specification limits (the "Voice of the Customer").
    
    **Strategic Application:** This is the ultimate verdict on process performance, often the final gate in a process validation or technology transfer. It directly answers the critical business question: "Is our process good enough to reliably meet customer or regulatory requirements?" 
    - A high Cpk provides objective evidence that the process is robust and delivers high quality.
    - A low Cpk is a clear signal that the process requires fundamental improvement.
    """)
    
    st.info("""
    **Interactive Demo:** Use the **Process Scenario** radio buttons below to simulate common real-world process states. Notice how the **Capability Verdict** is only valid when the top control chart shows a stable process. The bottom plot shows how the process distribution (blue line) fits within the specification limits (red lines).
    """)

    scenario = st.radio(
        "Select Process Scenario:",
        ('Ideal (High Cpk)', 'Shifted (Low Cpk)', 'Variable (Low Cpk)', 'Out of Control (Shift)', 'Out of Control (Trend)', 'Out of Control (Bimodal)'),
        horizontal=True
    )
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        fig, cpk_val = plot_capability(scenario)
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "📋 Glossary", "✅ Acceptance Criteria", "📖 Theory & History", "🏛️ Regulatory & Compliance"])
        with tabs[0]:
            st.metric(label="📈 KPI: Process Capability (Cpk)",
                      value=f"{cpk_val:.2f}" if not np.isnan(cpk_val) else "INVALID",
                      help="Measures how well the process fits within the spec limits, accounting for centering. Higher is better.")
            
            st.markdown("""
            **The Mantra: Stability First, Capability Second.**
            - The control chart (top plot) is a prerequisite. The Cpk metric is **statistically invalid** if the process is unstable, as an unstable process has no single, predictable "voice" to measure.
            - Notice how the **'Out of Control'** scenarios all produce an invalid result. You must fix the stability problem *before* you can assess capability.
            
            **The Key Insight: Control ≠ Capability.**
            - A process can be perfectly stable but still produce bad product. The **'Shifted'** and **'Variable'** scenarios are stable but have poor Cpk values for different reasons (poor accuracy vs. poor precision).
            
            **The Bimodal Case:**
            - The **'Bimodal'** scenario shows two distinct sub-processes running. This violates the normality assumption of Cpk and requires investigation to find and eliminate the source of the two populations.
            """)
        with tabs[1]:
            st.markdown("""
            ##### Glossary of Capability Terms
            - **Process Capability:** A measure of the ability of a process to produce output within specification limits.
            - **Specification Limits (LSL/USL):** The limits that define the acceptable range for a product's characteristic. They are determined by customer requirements or engineering design (the "Voice of the Customer").
            - **Control Limits:** The limits on a control chart that represent the natural variation of the process (the "Voice of the Process"). **They are completely unrelated to specification limits.**
            - **Cpk (Process Capability Index):** A statistical measure of process capability that accounts for how well the process is centered within the specification limits. It measures the distance from the process mean to the *nearest* specification limit.
            - **Cp (Process Potential Index):** A measure of process capability that does not account for centering. It only measures if the process is "narrow" enough to fit within the specifications.
            """)
        with tabs[2]:
            st.markdown("These are industry-standard benchmarks. For pharmaceuticals, a high Cpk in validation provides strong assurance of lifecycle performance.")
            st.markdown("- `Cpk < 1.00`: Process is **not capable**.")
            st.markdown("- `1.00 ≤ Cpk < 1.33`: Process is **marginally capable**.")
            st.markdown("- `Cpk ≥ 1.33`: Process is considered **capable** (a '4-sigma' quality level).")
            st.markdown("- `Cpk ≥ 1.67`: Process is considered **highly capable** (approaching 'Six Sigma').")

        with tabs[3]:
            st.markdown("""
            #### Historical Context: The Six Sigma Revolution
            **The Problem:** In the 1980s, the American electronics manufacturer Motorola was facing a quality crisis. Despite using traditional quality control methods, defect rates were too high to compete globally. They needed a new, more ambitious way to think about quality.

            **The 'Aha!' Moment:** An engineer named **Bill Smith**, with the backing of CEO Bob Galvin, championed a radical new idea. Instead of just being "in-spec," a process should be so good that the specification limits are at least **six standard deviations** away from the process mean. This "Six Sigma" concept was a quantum leap in quality thinking. The **Cpk index** became the simple, powerful metric to measure progress toward this goal. A Cpk of 2.0 was the statistical equivalent of achieving Six Sigma capability.

            **The Impact:** The Six Sigma initiative was a spectacular success, reportedly saving Motorola billions of dollars. It was later adopted and popularized by companies like General Electric under Jack Welch, becoming one of the most influential business management strategies of the late 20th century. Cpk moved from a niche statistical tool to a globally recognized KPI for process excellence.
            """)
            st.markdown("#### Mathematical Basis")
            st.markdown("Capability analysis is a direct comparison between the **\"Voice of the Customer\"** (the allowable spread, USL - LSL) and the **\"Voice of the Process\"** (the actual, natural spread, conventionally 6σ).")
            st.markdown("- **Cp (Potential Capability):** Measures if the process is narrow enough, ignoring centering.")
            st.latex(r"C_p = \frac{\text{USL} - \text{LSL}}{6\hat{\sigma}}")
            st.markdown("- **Cpk (Actual Capability):** The more important metric, as it accounts for process centering. It measures the distance from the process mean to the *nearest* specification limit, in units of 3-sigma.")
            st.latex(r"C_{pk} = \min \left( \frac{\text{USL} - \bar{x}}{3\hat{\sigma}}, \frac{\bar{x} - \text{LSL}}{3\hat{\sigma}} \right)")
        with tabs[4]:
            st.markdown("""
            Process capability analysis (Cpk) is the key metric used during Stage 2 of the validation lifecycle, Process Performance Qualification (PPQ).
            - **FDA Process Validation Guidance (Stage 2):** The goal of PPQ is to demonstrate that the process, operating under normal conditions, is capable of consistently producing conforming product. A high Cpk is the statistical evidence that this goal has been met.
            - **Global Harmonization Task Force (GHTF):** For medical devices, guidance on process validation similarly requires demonstrating that the process output consistently meets predetermined requirements.
            """)
#======================================================= 8. FIRST TIME YIELD & COST OF QUALITY  ============================================================================
def render_fty_coq():
    """Renders the comprehensive, interactive module for First Time Yield & Cost of Quality."""
    st.markdown("""
    #### Purpose & Application: The Business Case for Quality
    **Purpose:** To demonstrate the powerful financial and operational relationship between process performance (**First Time Yield**) and its business consequences (**Cost of Quality**). This tool moves beyond simple pass/fail metrics to provide a holistic view of process efficiency and the hidden costs of failure.
    
    **Strategic Application:** This dashboard is a critical communication tool for validation and engineering leaders to justify quality improvement projects to business leadership. It translates technical process metrics (like yield) into the language of the business (cost and risk), making the return on investment for validation and process improvement activities clear and tangible.
    """)
    
    st.info("""
    **Interactive Demo:** You are the Operations Director.
    1.  Select the **Project Type** to see a realistic multi-step process.
    2.  Use the **"Process Improvement Effort"** slider in the sidebar to simulate investing in better process controls, training, and validation.
    3.  Observe the impact across the four-plot dashboard, from root cause to financial outcome.
    """)

    project_type = st.selectbox(
        "Select a Project Type to Simulate:",
        ["Pharma Process (MAb)", "Analytical Assay (ELISA)", "Instrument Qualification", "Software System (CSV)"]
    )

    with st.sidebar:
        st.subheader("Improvement Effort Controls")
        improvement_effort = st.slider("Process Improvement Effort", 0, 10, 0, 1,
            help="Simulates the level of investment in process understanding and control (e.g., more validation, better training, improved equipment). Higher effort increases 'Good Quality' costs but dramatically reduces 'Poor Quality' costs and improves yield.")

    fig_pareto, fig_spc, fig_sankey, fig_iceberg, rty_base, rty_improved, base_coq, improved_coq = plot_fty_coq(project_type, improvement_effort)

    st.header("Process Performance & Cost Dashboard")
    total_coq_base = sum(base_coq.values())
    total_coq_improved = sum(improved_coq.values())
    
    rty_name = "Right First Time" if "Software" in project_type else "Rolled Throughput Yield (RTY)"
    col1, col2, col3 = st.columns(3)
    col1.metric(rty_name, f"{rty_improved:.1%}", f"{rty_improved - rty_base:.1%}")
    col2.metric("Total Cost of Quality (COQ)", f"{total_coq_improved:,.0f} RCU", f"{total_coq_improved - total_coq_base:,.0f} RCU")
    col3.metric("Return on Quality Investment", f"{(total_coq_base - total_coq_improved) / (improvement_effort*3500 + 1):.1f}x", help="Ratio of cost savings to the investment in prevention/appraisal.")

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.plotly_chart(fig_pareto, use_container_width=True)
        st.plotly_chart(fig_sankey, use_container_width=True)
    with col_p2:
        st.plotly_chart(fig_spc, use_container_width=True)
        st.plotly_chart(fig_iceberg, use_container_width=True)
    
    st.divider()
    st.subheader("Deeper Dive")
    tabs = st.tabs(["💡 Key Insights", "📋 Glossary", "✅ The Golden Rule", "📖 Theory & History", "🏛️ Regulatory & Compliance"])
    with tabs[0]:
        st.markdown("""
        **The 4-Plot Story: A Realistic Improvement Workflow**
        This dashboard tells a story from left to right, top to bottom, mirroring a real process improvement project.
        1.  **Where is the problem? (Pareto Chart):** This chart identifies the "vital few" steps causing the most scrap/rework. Your improvement efforts should always start here. Notice how the "Optimized" (green) bar is lowest for the step that was worst in the "Baseline" (grey).
        2.  **Why is it a problem? (SPC Chart):** This chart provides a statistical root cause for the failure at the worst step. A process with low yield is often unstable or off-center. As you apply improvement effort, this chart becomes more stable and centered, visually linking investment to improved process control.
        3.  **What is the overall impact? (Sankey Plot):** This shows the cumulative effect of all step yields on the final output (RTY). Improving the worst step has the biggest impact on widening the green "Final Output" flow.
        4.  **What is the financial consequence? (Iceberg Chart):** This translates the operational improvements into business terms. The investment in better process control (making the iceberg tip bigger) dramatically shrinks the hidden costs of failure (the much larger submerged part).
        """)
    with tabs[1]:
        st.markdown("""
        ##### Glossary of Quality Management Terms
        - **First Time Yield (FTY):** The percentage of units that pass a single process step without any defects or rework. Also known as First Pass Yield.
        - **Rolled Throughput Yield (RTY):** The probability that a unit will pass through all process steps without any defects. It is calculated by multiplying the FTY of all individual steps (`RTY = FTY₁ × FTY₂ × ... × FTYₙ`).
        - **Cost of Quality (COQ):** A methodology that quantifies the total cost of quality-related efforts and deficiencies.
        - **Prevention Costs:** Costs incurred to prevent defects from occurring in the first place (e.g., validation, training, FMEA).
        - **Appraisal Costs:** Costs incurred to detect defects (e.g., inspections, QC testing, audits).
        - **Internal Failure Costs:** Costs of defects found *before* the product is delivered to the customer (e.g., scrap, rework, investigation).
        - **External Failure Costs:** Costs of defects found *after* the product is delivered to the customer (e.g., recalls, warranty claims, lawsuits). These are the most damaging costs.
        """)
    with tabs[2]:
        st.error("""🔴 **THE INCORRECT APPROACH: "The Firefighting Mentality"**
A company under-invests in prevention and appraisal to minimize short-term costs. Their quality system is entirely reactive, consisting of a large QC department to "inspect quality in" at the end, and a large QA team to manage the constant deviations, rework, and scrap.
- **The Flaw:** They are paying the highest possible Cost of Quality. The massive, hidden costs of internal and external failures far outweigh the savings from skimping on prevention, and their low RTY creates unpredictable production schedules.""")
        st.success("""🟢 **THE GOLDEN RULE: Invest in Prevention, Not Failure**
The goal of a mature quality system is to strategically shift spending from failure costs to prevention and appraisal costs.
1.  **Measure Your Yield:** Calculate FTY for every step and the overall RTY to understand where your process is "leaking."
2.  **Quantify the Cost of Quality:** Use the COQ framework to translate yield losses and failures into a financial number that gets management's attention.
3.  **Justify Investment:** Use the RTY and COQ data to build a powerful business case for investing in process validation, better equipment, and more robust quality systems. This tool shows that such investments have a massive positive return.""")
        
    with tabs[3]:
        st.markdown("""
        #### Historical Context: The Quality Gurus
        The concepts of FTY and COQ were developed by the pioneers of the 20th-century quality management movement.
        - **Rolled Throughput Yield:** This concept is a cornerstone of **Six Sigma**, the quality improvement methodology famously developed at **Motorola in the 1980s**. RTY was a powerful metric for quantifying the cumulative effect of defects in a complex process and for measuring the impact of improvement projects.
        - **Cost of Quality:** The COQ framework was first described by **Armand V. Feigenbaum** in a 1956 Harvard Business Review article and was later popularized in his book *Total Quality Control*. He was the first to categorize costs into the four buckets (Prevention, Appraisal, Internal Failure, External Failure). Independently, **Joseph M. Juran** discussed the economics of quality in his *Quality Control Handbook* and emphasized the distinction between the "Cost of Good Quality" and the "Cost of Poor Quality."
        
        Together, these concepts provided the financial and operational language for the quality revolution, allowing engineers and scientists to frame quality not as an expense, but as a high-return investment.
        """)
        
    with tabs[4]:
        st.markdown("""
        FTY and COQ are not explicitly named in many regulations, but they are the underlying metrics and business drivers for the entire GxP quality system.
        - **ICH Q9 - Quality Risk Management:** The COQ framework is a powerful tool for quantifying the financial impact of risks identified in an FMEA. The potential for high "External Failure Costs" is a key driver for risk mitigation activities.
        - **ICH Q10 - Pharmaceutical Quality System:** This guideline emphasizes the importance of **continuous improvement** and **process performance monitoring**. RTY is a key metric for monitoring process performance, and reducing the COQ is a primary goal of a continuous improvement program.
        - **FDA Process Validation Guidance (Stage 3 - CPV):** An effective Continued Process Verification program should monitor metrics like FTY and RTY. A negative trend in these metrics would trigger an investigation and corrective action, demonstrating that the quality system is working as intended.
        """)
#======================================= 9. TOLERANCE INTERVALS  ============================================================================
def render_tolerance_intervals():
    """Renders the INTERACTIVE module for Tolerance Intervals."""
    st.markdown("""
    #### Purpose & Application: The Quality Engineer's Secret Weapon
    **Purpose:** To construct an interval that we can claim, with a specified level of confidence, contains a certain proportion of all individual values from a process.
    
    **Strategic Application:** This is often the most critical statistical interval in manufacturing. It directly answers the high-stakes question: **"Based on this sample, what is the range where we can expect almost all of our individual product units to fall?"**
    """)
    
    st.info("""
    **Interactive Demo:** Use the sliders below to explore the trade-offs in tolerance intervals. This simulation demonstrates how sample size and the desired quality guarantee (coverage) directly impact the calculated interval, which in turn affects process specifications and batch release decisions.
    """)
    
    # --- NEW: Sidebar controls for this specific module ---
    st.subheader("Tolerance Interval Controls")
    n_slider = st.slider(
        "🔬 Sample Size (n)", 
        min_value=10, max_value=200, value=30, step=10,
        help="The number of samples collected. More samples lead to a narrower, more reliable interval."
    )
    coverage_slider = st.select_slider(
        "🎯 Desired Population Coverage",
        options=[90.0, 95.0, 99.0, 99.9],
        value=99.0,
        help="The 'quality promise'. What percentage of all future parts do you want this interval to contain? A higher promise requires a wider interval."
    )

    # Generate plots using the slider values
    fig, ci, ti = plot_tolerance_intervals(n=n_slider, coverage_pct=coverage_slider)
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "📋 Glossary", "✅ The Golden Rule", "📖 Theory & History", "🏛️ Regulatory & Compliance"])
        
        with tabs[0]:
            st.metric(label="🎯 Desired Coverage", value=f"{coverage_slider:.1f}% of Population", help="The proportion of the entire process output we want our interval to contain.")
            st.metric(label="📏 Resulting Tolerance Interval", value=f"[{ti[0]:.1f}, {ti[1]:.1f}]", help="The final calculated range. Note how much wider it is than the CI.")
            
            st.info("Play with the sliders in the sidebar and observe the results!")
            st.markdown("""
            - **Increase `Sample Size (n)`:** As you collect more data, your estimates of the mean and standard deviation become more reliable. Notice how both the **Confidence Interval (orange)** and the **Tolerance Interval (green)** become **narrower**. This shows the direct link between sampling cost and statistical precision.
            - **Increase `Desired Population Coverage`:** As you increase the strength of your quality promise from 90% to 99.9%, the **Tolerance Interval becomes dramatically wider**. To be more certain of capturing a larger percentage of parts, you must widen your interval.
            """)
        with tabs[1]:
            st.markdown("""
            ##### Glossary of Statistical Intervals
            - **Confidence Interval (CI):** An interval estimate for a population **parameter** (like the mean). A 95% CI provides a plausible range for the *average* of the process.
            - **Tolerance Interval (TI):** An interval estimate for a specified **proportion of a population**. A 95%/99% TI provides a plausible range for where 99% of all *individual units* from the process will fall.
            - **Prediction Interval (PI):** An interval estimate for a **single future observation**. A 95% PI provides a plausible range for the *very next* data point.
            - **Coverage:** The proportion of the population that the tolerance interval is intended to contain (e.g., 99%).
            - **Confidence:** The probability that a given interval, constructed from a random sample, will actually contain the true value it is intended to estimate.
            """)
        with tabs[2]:
            st.error("""
            🔴 **THE INCORRECT APPROACH: The Confidence Interval Fallacy**
            - A manager sees that the 95% **Confidence Interval** for the mean is [99.9, 100.1] and their product specification is [95, 105]. They declare victory, believing all their product is in spec.
            - **The Flaw:** They've proven the *average* is in spec, but have made no claim about the *individuals*. If process variation is high, many parts could still be out of spec.
            """)
            st.success("""
            🟢 **THE GOLDEN RULE: Use the Right Interval for the Right Question**
            - **Question 1: "Where is my long-term process average located?"**
              - **Correct Tool:** ✅ **Confidence Interval**.
            - **Question 2: "Will the individual units I produce meet the customer's specification?"**
              - **Correct Tool:** ✅ **Tolerance Interval**.
              
            Never use a confidence interval to make a statement about where individual values are expected to fall.
            """)

        with tabs[3]:
            st.markdown("""
            #### Historical Context: The Surviving Bomber Problem
            The development of tolerance intervals is credited to the brilliant mathematician **Abraham Wald** during World War II. He is famous for the "surviving bombers" problem: when analyzing bullet holes on returning planes, the military wanted to reinforce the most-hit areas. Wald's revolutionary insight was that they should reinforce the areas with **no bullet holes**—because planes hit there never made it back.
            
            This ability to reason about an entire population from a limited sample is the same thinking behind the tolerance interval. Wald developed the statistical theory to allow engineers to make a reliable claim about **all** manufactured parts based on a **small sample**, a critical need for mass-producing interchangeable military hardware.
            
            #### Mathematical Basis
            """)
            st.latex(r"\text{TI} = \bar{x} \pm k \cdot s")
            st.markdown("""
            - **`k`**: The **k-factor** is the magic ingredient. It is a special value that depends on **three** inputs: the sample size (`n`), the desired population coverage (e.g., 99%), and the desired confidence level (e.g., 95%). This `k`-factor is mathematically constructed to account for the "double uncertainty" of not knowing the true mean *or* the true standard deviation.
            """)
        with tabs[4]:
            st.markdown("""
            Tolerance intervals are a statistically rigorous method for setting acceptance criteria and release specifications based on validation data.
            - **FDA Process Validation Guidance (Stage 2):** PPQ runs are used to demonstrate that the process can reliably produce product meeting its Critical Quality Attributes (CQAs). A tolerance interval calculated from PPQ data provides a high-confidence range where a large proportion of all future production will fall.
            - **USP General Chapter <1010> - Analytical Data:** Discusses various statistical intervals and their correct application, including tolerance intervals for making claims about a proportion of a population.
            """)

#======================================= 10. BAYESIAN INFERENCE  ============================================================================
def render_bayesian():
    """Renders the interactive module for Bayesian Inference."""
    st.markdown("""
    #### Purpose & Application: The Science of Belief Updating
    **Purpose:** To employ Bayesian inference to formally and quantitatively synthesize existing knowledge (a **Prior** belief) with new experimental data (the **Likelihood**) to arrive at an updated, more robust conclusion (the **Posterior** belief).
    
    **Strategic Application:** This is a paradigm-shifting tool for driving efficient, knowledge-based validation and decision-making. In a traditional (Frequentist) world, every study starts from a blank slate. In the Bayesian world, we can formally leverage what we already know.
    - **Accelerating Tech Transfer:** Use data from an R&D validation study to form a **strong, informative prior**. This allows the receiving QC lab to demonstrate success with a smaller confirmation study, saving time and resources.
    - **Answering Direct Business Questions:** Provides a natural framework to answer the question: "Given all the evidence, what is the probability that our true pass rate is above 90%?"
    """)
    st.info("""
    **Interactive Demo:** Use the controls in the sidebar to define your prior belief and the new experimental data. Observe how the final **Posterior (blue curve)** is always a weighted compromise between your initial **Prior (green curve)** and the new **Data (red diamond)**.
    """)
    
    with st.sidebar:
        st.subheader("Bayesian Inference Controls")
        prior_type_bayes = st.radio(
            "Select Prior Belief:",
            ("Strong R&D Prior", "No Prior (Uninformative)", "Skeptical/Regulatory Prior"),
            index=0,
            help="Your belief about the pass rate *before* seeing the new data. A 'Strong' prior has a large influence; an 'Uninformative' prior lets the data speak for itself."
        )
        # --- NEW INTERACTIVE SLIDERS ---
        st.markdown("---")
        st.markdown("**New QC Data**")
        n_qc_slider = st.slider("Number of QC Samples (n)", 1, 100, 20, 1,
            help="The total number of new QC samples run in the experiment.")
        k_qc_slider = st.slider("Number of Passes (k)", 0, n_qc_slider, 18, 1,
            help="Of the new samples run, how many passed?")

    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        fig, prior_mean, mle, posterior_mean, credible_interval, prob_gt_spec = plot_bayesian(
            prior_type=prior_type_bayes,
            n_qc=n_qc_slider,
            k_qc=k_qc_slider,
            spec_limit=0.90
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "📋 Glossary", "✅ Acceptance Criteria", "📖 Theory & History", "🏛️ Regulatory & Compliance"])
        with tabs[0]:
            st.metric(
                label="🎯 Primary KPI: Prob(Pass Rate > 90%)",
                value=f"{prob_gt_spec:.2%}",
                help="The posterior probability that the true pass rate meets the 90% specification. This is a direct risk statement."
            )
            st.metric(
                label="📈 Posterior Mean Pass Rate",
                value=f"{posterior_mean:.1%}",
                help="The final, data-informed belief; a weighted average of the prior and the data."
            )
            st.metric(
                label="📊 95% Credible Interval",
                value=f"[{credible_interval[0]:.1%}, {credible_interval[1]:.1%}]",
                help="We can be 95% certain that the true pass rate lies within this interval."
            )
            st.markdown("---")
            st.metric(label="Prior Mean Rate", value=f"{prior_mean:.1%}", help="The initial belief *before* seeing the new QC data.")
            st.metric(label="Data-only Estimate (k/n)", value=f"{mle:.1%}", help="The evidence from the new QC data alone (the frequentist result).")
        with tabs[1]:
            st.markdown("""
            ##### Glossary of Bayesian Terms
            - **Prior Distribution:** A probability distribution that represents one's belief about a parameter *before* seeing the new data.
            - **Likelihood:** The probability of observing the new data, given a specific value of the parameter. This is the information contained in the new experiment.
            - **Posterior Distribution:** The updated probability distribution for the parameter *after* combining the prior belief with the likelihood from the new data. `Posterior ∝ Likelihood × Prior`.
            - **Credible Interval:** The Bayesian equivalent of a confidence interval. A 95% credible interval `[X, Y]` means there is a 95% probability that the true value of the parameter lies within that range.
            - **Conjugate Prior:** A prior distribution that, when combined with a given likelihood, results in a posterior distribution from the same family (e.g., Beta prior for binomial likelihood results in a Beta posterior).
            """)
        with tabs[2]:
            st.markdown("- The acceptance criterion is framed in terms of the **posterior distribution** and is probabilistic.")
            st.markdown("- **Example Criterion 1 (Probability Statement):** 'There must be at least a 95% probability that the true pass rate is greater than 90%.'")
            st.markdown("- **Example Criterion 2 (Credible Interval):** 'The lower bound of the **95% Credible Interval** must be above the target of 90%.'")
            st.warning("**The Prior is Critical:** In a regulated setting, the prior must be transparent, justified by historical data, and pre-specified in the validation protocol. An unsubstantiated, overly optimistic prior is a major red flag.")
        with tabs[3]:
            st.markdown("""
            #### Historical Context: The 200-Year Winter
            **The Problem:** The underlying theorem was conceived by the Reverend **Thomas Bayes** in the 1740s and published posthumously. However, for nearly 200 years, Bayesian inference remained a philosophical curiosity, largely overshadowed by the Frequentist school of Neyman and Fisher. There were two huge barriers:
            1.  **The Philosophical Barrier:** The "subjective" nature of the prior was anathema to frequentists who sought pure objectivity.
            2.  **The Computational Barrier:** For most real-world problems, calculating the denominator in Bayes' theorem (the "evidence") was mathematically intractable.
            
            **The 'Aha!' Moment (MCMC):** The **"Bayesian Revolution"** began in the late 20th century, driven by the explosion of computing power and the development of powerful simulation algorithms like **Markov Chain Monte Carlo (MCMC)**. MCMC provided a clever workaround: instead of trying to calculate the posterior distribution directly, you could create a smart algorithm that *draws samples from it*.
            
            **The Impact:** MCMC made the previously impossible possible. Scientists could now approximate the posterior distribution for incredibly complex models, making Bayesian methods practical for the first time. This has led to a renaissance of Bayesian thinking in fields from astrophysics to pharmaceutical development.
            """)
            st.markdown("#### Mathematical Basis")
            st.markdown("Bayes' Theorem is elegantly simple:")
            st.latex(r"P(\theta|D) = \frac{P(D|\theta) \cdot P(\theta)}{P(D)}")
            st.markdown(r"In words: **Posterior = (Likelihood × Prior) / Evidence**")
            st.markdown(r"""
            - $P(\theta|D)$ (Posterior): The probability of our parameter $\theta$ (e.g., the true pass rate) given the new Data D.
            - $P(D|\theta)$ (Likelihood): The probability of observing our Data D, given a specific value of the parameter $\theta$.
            - $P(\theta)$ (Prior): Our initial belief about the distribution of the parameter $\theta$.
            
            For binomial data (pass/fail), the **Beta distribution** is a **conjugate prior**. This means if you start with a Beta prior and have a binomial likelihood, your posterior is also a simple, updated Beta distribution.
            """)
            st.latex(r"\text{Posterior} \sim \text{Beta}(\alpha_{prior} + k, \beta_{prior} + n - k)")
        with tabs[4]:
            st.markdown("""
            While less common than frequentist methods, Bayesian statistics are explicitly accepted and even encouraged by regulators in certain contexts, particularly where prior information is valuable.
            - **FDA Guidance on Adaptive Designs for Clinical Trials:** This guidance openly discusses and accepts the use of Bayesian methods for modifying trial designs based on accumulating data.
            - **FDA Guidance on Medical Device Decision Making:** The benefit-risk assessments for medical devices are often framed in a way that is highly compatible with Bayesian thinking, allowing for the formal incorporation of prior knowledge.
            - **ICH Q8, Q9, Q10:** The lifecycle and risk-based principles of these guidelines are well-aligned with the Bayesian paradigm of updating knowledge as more data becomes available.
            """)


##=======================================================================================================================================================================================================
##=================================================================== END ACT II UI Render ========================================================================================================================
##=======================================================================================================================================================================================================
#======================================= 1. PROCESS CONTROL PLAN BUILDER  ============================================================================
def render_control_plan_builder():
    """Renders the comprehensive, interactive module for a Process Control Plan."""
    st.markdown("""
    #### Purpose & Application: The Operational Playbook
    **Purpose:** To create the **Operational Playbook for Process Control.** A Control Plan is a formal, living document that links a process step to its critical parameters (CPPs), the measurement method, the specification, the sample size/frequency, the control chart used, and, most critically, the **Out-of-Control Action Plan (OCAP)**.
    
    **Strategic Application:** This tool bridges the gap between the statistical analysis performed by engineers and the on-the-floor reality of operators. It translates the outputs of your validation studies (like CPPs from a DOE and control limits from an SPC chart) into a single, clear, and actionable document that becomes the standard operating procedure for maintaining a process in a validated state.
    """)
    
    st.info("""
    **Interactive Demo:** You are the Process Steward for a drug product.
    1.  Choose a **Critical Process Parameter (CPP)** from a real manufacturing process.
    2.  Use the **Control Strategy** inputs in the sidebar to define how this CPP will be monitored.
    3.  Observe the **Simulated SPC Chart** in real-time. Notice how a higher sampling frequency leads to faster detection of the process shift.
    """)

    cpp_options = {
        "Granulation Moisture (%)": {'lsl': 2.0, 'usl': 5.0, 'method': 'NIR Spectroscopy'},
        "Tablet Hardness (kp)": {'lsl': 10, 'usl': 15, 'method': 'Hardness Tester'},
        "Bioreactor DO (%)": {'lsl': 30, 'usl': 60, 'method': 'DO Probe'}
    }
    cpp_choice = st.selectbox("Select a Critical Process Parameter (CPP) to Control:", list(cpp_options.keys()))
    cpp_data = {'name': cpp_choice, **cpp_options[cpp_choice]}
    
    with st.sidebar:
        st.subheader("Control Strategy Builder")
        sample_size = st.slider("Sample Size (n)", 1, 10, 3, 1, help="How many samples are taken at each measurement point? A larger n increases confidence but also cost.")
        frequency = st.select_slider("Sampling Frequency", ["Once per batch", "Once per hour", "Every 15 minutes"], value="Once per hour",
            help="How often is the CPP measured? More frequent sampling detects shifts faster but has higher operational and testing costs.")
        spc_tool = st.selectbox("SPC Chart to Use", ["I-MR Chart", "Xbar-R Chart", "EWMA Chart"],
            help="The specific statistical control chart that will be used to monitor this CPP. Choose EWMA for higher sensitivity to small shifts.")

    fig_table, fig_spc, fig_flowchart = plot_control_plan_dashboard(cpp_data, sample_size, frequency, spc_tool)
    
    st.header("Control Strategy Dashboard")
    st.plotly_chart(fig_table, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_spc, use_container_width=True)
    with col2:
        st.plotly_chart(fig_flowchart, use_container_width=True)
    
    st.divider()
    st.subheader("Deeper Dive")
    tabs = st.tabs(["💡 Key Insights", "📋 Glossary", "✅ The Golden Rule", "📖 Theory & History", "🏛️ Regulatory & Compliance"])
    with tabs[0]:
        st.markdown("""
        **Interpreting the Dashboard:**
        - **Control Plan Table:** This is the formal, auditable document. It contains all the critical information for controlling the process parameter in a single place. In a real GMP environment, this would be a controlled document that links directly to SOPs and batch records.
        - **Simulated SPC Chart:** This chart provides a crucial "what-if" analysis of your chosen control strategy. A small process shift is introduced at batch #15. The purple line shows the predicted detection time. **Try changing the 'Sampling Frequency' in the sidebar from 'Once per batch' to 'Every 15 minutes' and watch the purple line move to the left, showing how faster sampling leads to faster detection.**
        - **OCAP Flowchart:** This is the user-friendly guide for the operator on the manufacturing floor. It provides a simple, unambiguous, visual decision tree for exactly what to do when a process alarm occurs.
        
        **The Strategic Insight:** The Control Plan is a balancing act between **risk and cost**. More frequent sampling provides better process control and faster deviation detection, which reduces the risk of producing non-conforming material. However, it also increases laboratory costs and operational complexity. This dashboard allows you to simulate and visualize this trade-off to find the optimal strategy that provides the necessary level of quality assurance without overburdening the system.
        """)
    with tabs[1]:
        st.markdown("""
        ##### Glossary of Control Strategy Terms
        - **Control Plan:** A written description of the systems and processes required for controlling a process or product. It is a living document that is updated as the process matures.
        - **Control Strategy (ICH Q10):** A planned set of controls, derived from product and process understanding, that assures process performance and product quality. It encompasses raw material specifications, process parameter controls, facility conditions, and final product testing.
        - **OCAP (Out-of-Control Action Plan):** A flowchart or documented procedure that prescribes the sequence of actions an operator must take in response to an out-of-control signal from an SPC chart. It is a critical tool for ensuring consistent and compliant responses to deviations.
        - **CPP (Critical Process Parameter):** A process parameter whose variability has an impact on a CQA and therefore must be monitored and controlled to ensure the process produces the desired quality.
        - **In-Process Control (IPC):** Checks performed during production in order to monitor and, if necessary, to adjust the process to ensure that the product conforms to its specification.
        - **Specification:** A list of detailed requirements with which the products or materials used or obtained during manufacture have to conform.
        """)
    with tabs[2]:
        st.error("""🔴 **THE INCORRECT APPROACH: The "Trust the Engineers" Fallacy**
Engineers install a new SPC system that monitors 50 parameters in real-time. When a chart alarms, the operator is unsure which alarms are important and what to do, so they call the busy engineer, who may not be immediately available. By the time the engineer arrives, significant non-conforming material may have been produced.
- **The Flaw:** The system provides data but no **actionable information** for the operator. The lack of a clear Control Plan and OCAP creates confusion, delays, and significant compliance risk on the manufacturing floor.""")
        st.success("""🟢 **THE GOLDEN RULE: If It Can Alarm, It Must Have a Plan**
A compliant and effective control strategy requires that every single monitoring chart has a clear, pre-defined, and operator-friendly action plan.
1.  **Link Controls to Risks:** The parameters included in the Control Plan should be directly linked to the high-risk items identified in the **FMEA**. This provides a documented, risk-based rationale for your monitoring strategy.
2.  **Define the Full Strategy:** The Control Plan must specify *all* aspects of the control: the what (CPP), how (Measurement System), how much (Sample Size), how often (Frequency), and who (Operator).
3.  **Create a Simple OCAP:** The OCAP must be a simple, unambiguous flowchart that an operator can follow under pressure without needing to consult an engineer for routine alarms. This empowers operators and ensures consistent, rapid responses to process deviations.""")
        
    with tabs[3]:
        st.markdown("""
        #### Historical Context: From Shewhart's Chart to the Control Plan
        When **Walter Shewhart** invented the control chart in the 1920s, his primary focus was on the statistical tool for *detection*. The reaction to the signal was largely left to the judgment of the on-site engineer. This worked well in the simpler manufacturing environments of the time.
        
        As manufacturing processes became more complex and quality management systems more formalized, the need for a more structured response became clear. The **automotive industry**, through its development of the **Advanced Product Quality Planning (APQP)** framework in the late 1980s, formalized the **Control Plan** as a key deliverable for any new product. They recognized that a control chart was useless without a documented plan for how it would be used, by whom, and what actions would be taken in response to a signal.
        
        The **Out-of-Control Action Plan (OCAP)** evolved from this as a best practice, translating the formal table of the Control Plan into a simple, visual flowchart that is ideal for use by operators on the factory floor, especially in high-pressure situations. The combination of SPC, the Control Plan, and the OCAP creates a complete, closed-loop system for process control that is a hallmark of a mature quality system.
        """)
        
    with tabs[4]:
        st.markdown("""
        The Control Plan is a key document that demonstrates a state of control and is a primary focus of regulatory audits. It is the practical embodiment of your entire validation effort.
        - **ICH Q10 - Pharmaceutical Quality System:** The Control Plan is the operational embodiment of the **Control Strategy**, which is a central concept in ICH Q10. It is the documented proof that product and process understanding (from QbD and validation) has been successfully translated into effective, routine controls.
        - **FDA Process Validation Guidance (Stage 3 - CPV):** The guidance requires an "ongoing program to collect and analyze product and process data." The Control Plan is the document that formally defines this program, specifying what is monitored, how often, and what to do when a deviation is detected.
        - **21 CFR 211.110 (Sampling and testing of in-process materials and drug products):** This regulation requires written procedures for in-process controls, including "Control procedures shall be established to monitor the output and to validate the performance of those manufacturing processes that may be responsible for causing variability." The Control Plan is this established procedure.
        - **EU Annex 15: Qualification and Validation:** Emphasizes that processes should be monitored during routine production to assure that they remain in a state of control. The Control Plan is the primary document defining this monitoring.
        """)
#======================================= 2. RUN VALIDATION WESTGARD  ============================================================================
def render_multi_rule():
    """Renders the comprehensive, interactive module for Multi-Rule SPC (Westgard Rules)."""
    st.markdown("""
    #### Purpose & Application: The Statistical Detective
    **Purpose:** To serve as a high-sensitivity "security system" for your assay. Instead of one simple alarm, this system uses a combination of rules to detect specific types of problems, catching subtle shifts and drifts long before a catastrophic failure occurs. It dramatically increases the probability of detecting true errors while minimizing false alarms.
    
    **Strategic Application:** This is the global standard for run validation in regulated QC and clinical laboratories. While a basic control chart just looks for "big" errors, the multi-rule system acts as a **statistical detective**, using a toolkit of rules to diagnose different failure modes.
    """)
    
    st.info("""
    **Interactive Demo:** Use the **Process Scenario** radio buttons to simulate common assay failures. The chart will automatically detect and highlight any rule violations. Hover over a red diamond for a detailed diagnosis. Compare the chart to the **Power Functions** plot to understand *why* certain rules are better at catching different types of errors.
    """)
    
    st.sidebar.subheader("Westgard Scenario Controls")
    scenario = st.sidebar.radio(
        "Select a Process Scenario to Simulate:",
        options=('Stable', 'Large Random Error', 'Systematic Shift', 'Increased Imprecision'),
        captions=[
            "A normal, in-control run for reference.",
            "e.g., A single major blunder like a transcription error.",
            "e.g., A new reagent lot causes a persistent bias.",
            "e.g., A faulty pipette causes inconsistency."
        ]
    )
    fig, fig_power, violations = plot_westgard_scenario(scenario=scenario)
    
    # --- Redesigned Layout ---
    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "📋 Glossary", "✅ The Golden Rule", "📖 Theory & History", "🏛️ Regulatory & Compliance"])
        
        with tabs[0]:
            st.markdown("##### Detected Violations")
            if not violations:
                st.success("No violations detected. Process appears to be in control.")
            else:
                # Use a more compact way to display violations
                violation_summary = {}
                for point, ruleset in sorted(violations.items()):
                    for rule in ruleset.strip().split():
                        if rule not in violation_summary:
                            violation_summary[rule] = []
                        if point + 1 not in violation_summary[rule]:
                            violation_summary[rule].append(point + 1)

                for rule, points in violation_summary.items():
                    st.warning(f"**{rule}:** Violated at points {', '.join(map(str, points))}")

            st.markdown("---")
            st.markdown("##### Rule Power Functions")
            st.plotly_chart(fig_power, use_container_width=True)
            st.markdown("""
            **Reading the Power Plot:** This chart shows which rules are best for different problems.
            - The `1-3s` rule is powerful for **large shifts** (>3σ) but blind to small ones.
            - Rules like `4-1s` and `10-x` are weak for large shifts but much more powerful for detecting **small, persistent shifts** (<2σ).
            This is why a multi-rule system is essential.
            """)
        with tabs[1]:
                st.markdown("""
                ##### Glossary of Westgard Rules
                - **Systematic Error:** A consistent bias in the measurement process (e.g., a miscalibrated instrument). Detected by rules like `2-2s`, `4-1s`, `10-x`.
                - **Random Error:** Unpredictable, random fluctuations in the measurement process (e.g., pipetting variability). Detected by rules like `1-3s` and `R-4s`.
                - **1-3s Rule:** One control measurement exceeds the mean ± 3 standard deviations. Rejection rule, sensitive to large errors.
                - **2-2s Rule:** Two consecutive control measurements exceed the same mean ± 2 standard deviations. Rejection rule, sensitive to systematic error.
                - **R-4s Rule:** The range between two consecutive control measurements exceeds 4 standard deviations. Rejection rule, sensitive to random error.
                - **4-1s Rule:** Four consecutive control measurements exceed the same mean ± 1 standard deviation. Rejection rule, sensitive to small systematic shifts.
                - **10-x Rule:** Ten consecutive control measurements fall on the same side of the mean. Rejection rule, very sensitive to small systematic shifts.
                """)
        with tabs[2]:
            st.error("""🔴 **THE INCORRECT APPROACH: The "Re-run & Pray" Mentality**
This operator sees any alarm, immediately discards the run, and starts over without thinking.
- They don't use the specific rule (`2-2s` vs `R-4s`) to guide their troubleshooting.
- They might engage in "testing into compliance" by re-running a control until it passes, a serious compliance violation.""")
            st.success("""🟢 **THE GOLDEN RULE: The Rule is the First Clue**
The goal is to treat the specific rule violation as the starting point of a targeted investigation.
- **Think like a detective:** "The chart shows a `2-2s` violation. This suggests a systematic shift. I should check my calibrators and reagents first, not my pipetting technique."
- **Document Everything:** The investigation, the root cause, and the corrective action for each rule violation must be documented.""")

        with tabs[3]:
            st.markdown("""
            #### Historical Context: From the Factory Floor to the Hospital Bed
            **The Problem:** In the 1970s, clinical laboratories were becoming highly automated, but their quality control methods hadn't kept up. They were using Shewhart's simple `1-3s` rule, designed for manufacturing. However, in a clinical setting, the cost of a missed error (a misdiagnosis) is infinitely higher than the cost of a false alarm (re-running a control). The `1-3s` rule was not sensitive enough to catch the small but medically significant drifts that could occur with automated analyzers.

            **The 'Aha!' Moment:** **Dr. James O. Westgard**, a professor of clinical chemistry, recognized this dangerous gap. He realized that a single rule was a blunt instrument. Instead, he proposed using a *combination* of rules, like a series of increasingly fine filters. A "warning" rule (like `1-2s`) could trigger a check of more stringent rejection rules. 
            
            **The Impact:** In his 1981 paper, Westgard introduced his multi-rule system. It was a paradigm shift for clinical QC. It gave laboratorians a logical, flowchart-based system that dramatically increased the probability of detecting true errors while keeping the false alarm rate manageable. The "Westgard Rules" became the de facto global standard for run validation in medical labs, directly improving the quality of diagnostic data and patient safety worldwide.
            """)
            st.markdown("#### Mathematical Basis")
            st.markdown("The logic is built on the properties of the normal distribution and the probability of rare events. For a stable process:")
            st.markdown("- A point outside **±3σ** is a rare event. The probability of a single point falling outside these limits by chance is very low (p ≈ 0.0027). This makes the **1-3s** rule a high-confidence signal of a major error.")
            st.markdown("- A point outside **±2σ** is more common (p ≈ 0.0455). Seeing one is not a strong signal. However, the probability of seeing *two consecutive points* on the same side of the mean purely by chance is much, much lower:")
            st.latex(r"P(\text{2-2s}) \approx \left( \frac{0.0455}{2} \right)^2 \approx 0.0005")
            st.markdown("This makes the **2-2s** rule a powerful and specific detector of systematic shifts with a very low false alarm rate, even though the individual points themselves are not extreme.")
        with tabs[4]:
            st.markdown("""
            Westgard Rules are the de facto standard for routine QC run validation in clinical and diagnostic laboratories, and their principles are widely adopted in pharmaceutical QC.
            - **CLIA (Clinical Laboratory Improvement Amendments):** US federal regulations that require clinical laboratories to monitor the accuracy and precision of their testing. Westgard Rules provide a compliant framework for this.
            - **ISO 15189:** The international quality standard for medical laboratories, which requires robust internal quality control procedures.
            - **USP General Chapter <1010> - Analytical Data:** Discusses the treatment of analytical data and the principles of statistical control, for which multi-rule systems are a best practice.
            """)

#======================================= 3. SMALL DRIFT DETECTION  ============================================================================
def render_ewma_cusum():
    """Renders the comprehensive, interactive module for Small Shift Detection (EWMA/CUSUM)."""
    st.markdown("""
    #### Purpose & Application: The Process Sentinel
    **Purpose:** To deploy a high-sensitivity monitoring system designed to detect small, sustained shifts in a process mean that would be invisible to a standard Shewhart control chart (like an I-MR or X-bar chart). These charts have "memory," accumulating evidence from past data to find subtle signals.

    **Strategic Application:** This is an essential "second layer" of process monitoring for mature, stable processes where large, sudden failures are rare, but slow, gradual drifts are a significant risk.
    - **🔬 EWMA (The Sentinel):** A robust, general-purpose tool that smoothly weights past observations, excellent for detecting the onset of a gradual drift.
    - **🐕 CUSUM (The Bloodhound):** A specialized, high-power tool that is the fastest possible detector for a shift of a specific, pre-defined magnitude.
    """)
    
    st.info("""
    **Interactive Demo:** Select a **Failure Scenario** and a **Shift Size**. Observe how quickly each chart (marked with a red 'X') detects the problem after the "Process Change Begins" line. For small shifts and gradual drifts, notice how the I-Chart often fails to detect the issue at all.
    """)
    
    with st.sidebar:
        st.sidebar.subheader("Small Shift Detection Controls")
        scenario_slider = st.sidebar.radio(
            "Select Failure Scenario:",
            ('Sudden Shift', 'Gradual Drift'),
            captions=["An abrupt change in the process mean.", "A slow, creeping change over time."]
        )
        shift_size_slider = st.sidebar.slider(
            "Select Process Shift Size (in multiples of σ):",
            min_value=0.25, max_value=3.5, value=0.75, step=0.25,
            help="Controls the magnitude of the process shift. Small shifts are much harder for standard charts to detect."
        )

    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        fig, i_time, ewma_time, cusum_time = plot_ewma_cusum_comparison(
            shift_size=shift_size_slider,
            scenario=scenario_slider
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "📋 Glossary", "✅ The Golden Rule", "📖 Theory & History", "🏛️ Regulatory & Compliance"])

        with tabs[0]:
            st.markdown(f"##### Detection Performance for a **{shift_size_slider}σ** {scenario_slider}")
            st.metric(
                label="I-Chart: Time to Detect",
                value=f"{int(i_time)} points" if not np.isnan(i_time) else "Not Detected",
                help="Number of points after the shift began before the I-Chart alarmed."
            )
            st.metric(
                label="EWMA: Time to Detect",
                value=f"{int(ewma_time)} points" if not np.isnan(ewma_time) else "Not Detected",
                help="Number of points after the shift began before the EWMA chart alarmed."
            )
            st.metric(
                label="CUSUM: Time to Detect",
                value=f"{int(cusum_time)} points" if not np.isnan(cusum_time) else "Not Detected",
                help="Number of points after the shift began before the CUSUM chart alarmed."
            )

            st.markdown("""
            **The Core Insight:**
            Try simulating a small (`< 1.5σ`) **Gradual Drift**. The I-Chart is completely blind, giving a false sense of security. The EWMA and CUSUM charts, because they have memory, accumulate the small signals over time and reliably sound the alarm. This demonstrates why relying only on Shewhart charts creates a significant blind spot for modern, high-precision processes.
            """)
        with tabs[1]:
            st.markdown("""
            ##### Glossary of Small Shift Terms
            - **Shewhart Chart (e.g., I-Chart):** A "memoryless" control chart that evaluates each data point independently. It is excellent for detecting large shifts but insensitive to small, gradual drifts.
            - **EWMA (Exponentially Weighted Moving Average):** A "memory-based" chart that computes a weighted average of all past and current observations. The weights decay exponentially over time.
            - **CUSUM (Cumulative Sum):** A "memory-based" chart that plots the cumulative sum of deviations from a target. It is the fastest possible chart for detecting a shift of a specific, pre-defined magnitude.
            - **Lambda (λ):** The weighting parameter for an EWMA chart (0 < λ ≤ 1). A small λ gives the chart a long memory, making it sensitive to tiny shifts.
            - **ARL (Average Run Length):** The average number of points that will be plotted on a control chart before an out-of-control signal occurs.
            """)
        with tabs[2]:
            st.error("""🔴 **THE INCORRECT APPROACH: The "One-Chart-Fits-All Fallacy"**
A manager insists on using only I-MR charts for everything because they are easy to understand.
- They miss a slow 1-sigma drift for weeks, producing tons of near-spec material.
- When a batch finally fails, they are shocked and have no leading indicators to explain why. They have been flying blind.""")
            st.success("""🟢 **THE GOLDEN RULE: Layer Your Statistical Defenses**
The goal is to use a combination of charts to create a comprehensive security system.
- **Use Shewhart Charts (I-MR, X-bar) as your front-line "Beat Cops":** They are unmatched for detecting large, sudden special causes.
- **Use EWMA or CUSUM as your "Sentinels":** Deploy them alongside Shewhart charts to stand guard against the silent, creeping threats that the beat cops will miss.
This layered approach provides a complete picture of process stability.""")

        with tabs[3]:
            st.markdown(r"""
            #### Historical Context: The Second Generation of SPC
            **The Problem:** Dr. Walter Shewhart's control charts of the 1920s were a monumental success. However, they were designed like a **smoke detector**—brilliantly effective at detecting large, sudden events ("fires"), but intentionally insensitive to small, slow changes to avoid overreaction to random noise. By the 1950s, industries like chemistry and electronics required higher precision. The critical challenge was no longer just preventing large breakdowns, but detecting subtle, gradual drifts that could slowly degrade quality. A new kind of sensor was needed.

            **The 'Aha!' Moment (CUSUM - 1954):** The first breakthrough came from British statistician **E. S. Page**. Inspired by **sequential analysis** from WWII munitions testing, he realized that instead of looking at each data point in isolation, he could **accumulate the evidence** of small deviations over time. The Cumulative Sum (CUSUM) chart was born. It acts like a **bloodhound on a trail**, ignoring random noise by using a "slack" parameter `k`, but rapidly accumulating the signal once it detects a persistent scent in one direction.

            **The 'Aha!' Moment (EWMA - 1959):** Five years later, **S. W. Roberts** of Bell Labs proposed a more flexible alternative, inspired by **time series forecasting**. The Exponentially Weighted Moving Average (EWMA) chart acts like a **sentinel with a memory**. It gives the most weight to the most recent data point, a little less to the one before, and so on, with the influence of old data decaying exponentially. This creates a smooth, sensitive trend line that effectively filters out noise while quickly reacting to the beginning of a real drift.

            **The Impact:** These two inventions were not replacements for Shewhart's charts but essential complements. They gave engineers the sensitive, memory-based tools they needed to manage the increasingly precise and complex manufacturing processes of the late 20th century.
            """)
            st.markdown("#### Mathematical Basis")
            st.markdown("The elegance of these charts lies in their simple, recursive formulas.")
            st.markdown("- **EWMA (Exponentially Weighted Moving Average):**")
            st.latex(r"EWMA_t = \lambda \cdot Y_t + (1-\lambda) \cdot EWMA_{t-1}")
            st.markdown(r"""
            - **`λ` (lambda):** This is the **memory parameter** (0 < λ ≤ 1). A small `λ` (e.g., 0.1) creates a chart with a long memory, making it very sensitive to tiny, persistent shifts. A large `λ` (e.g., 0.4) creates a chart with a short memory, behaving more like a Shewhart chart.
            """)
            st.markdown("- **CUSUM (Cumulative Sum):**")
            st.latex(r"SH_t = \max(0, SH_{t-1} + (Y_t - T) - k)")
            st.markdown(r"""
            - This formula tracks upward shifts (`SH`).
            - **`T`**: The process target or historical mean.
            - **`k`**: The **"slack" or "allowance" parameter**, typically set to half the size of the shift you want to detect quickly (e.g., `k = 0.5σ`). This makes the CUSUM chart a highly targeted detector.
            """)
        with tabs[4]:
            st.markdown("""
            These advanced analytical methods are key enablers for modern, data-driven approaches to process monitoring and control, as encouraged by global regulators.
            - **FDA Guidance for Industry - PAT — A Framework for Innovative Pharmaceutical Development, Manufacturing, and Quality Assurance:** This tool directly supports the PAT initiative's goal of understanding and controlling manufacturing processes through timely measurements to ensure final product quality.
            - **FDA Process Validation Guidance (Stage 3 - Continued Process Verification):** These advanced methods provide a more powerful way to meet the CPV requirement of continuously monitoring the process to ensure it remains in a state of control.
            - **ICH Q8(R2), Q9, Q10 (QbD Trilogy):** The use of sophisticated models for deep process understanding, real-time monitoring, and risk management is the practical implementation of the principles outlined in these guidelines.
            - **21 CFR Part 11 / GAMP 5:** If the model is used to make GxP decisions (e.g., real-time release), the underlying software and model must be fully validated as a computerized system.
            """)
#======================================= 4. MULTIVARIATE SPC  ============================================================================
def render_multivariate_spc():
    """Renders the comprehensive, interactive module for Multivariate SPC."""
    st.markdown("""
    #### Purpose & Application: The Process Doctor
    **Purpose:** To monitor the **holistic state of statistical control** for a process with multiple, correlated parameters. Instead of using an array of univariate charts (like individual nurses reading single vital signs), Multivariate SPC (MSPC) acts as the **head physician**, integrating all information into a single, powerful diagnosis.
    
    **Strategic Application:** This is an essential methodology for modern **Process Analytical Technology (PAT)** and real-time process monitoring. In complex systems like bioreactors or chromatography, parameters are interdependent. A small, coordinated deviation—a "stealth shift"—can be invisible to individual charts but represents a significant excursion from the normal operating state. MSPC is designed to detect exactly these events.
    """)
    
    st.info("""
    **Interactive Demo:** You are the Process Engineer. Use the **Process Scenario** radio buttons in the sidebar to simulate different types of multivariate process failures. First, observe the **Process State Space** plot to see how the failure looks visually, then see which **Control Chart (T² or SPE)** detects the problem, and finally, check the **Contribution Plot** in the 'Key Insights' tab to diagnose the root cause.
    """)

    with st.sidebar:
        st.subheader("Multivariate SPC Controls")
        scenario = st.sidebar.radio(
            "Select a Process Scenario to Simulate:",
            ('Stable', 'Shift in Y Only', 'Correlation Break'),
            captions=["A normal, in-control process.", "A 'stealth shift' in one variable.", "An unprecedented event breaks the model."],
            help="Choose a scenario to see how T² and SPE charts are sensitive to different types of failures."
        )

    fig_scatter, fig_charts, fig_contrib, t2_ooc, spe_ooc, error_type_str = plot_multivariate_spc(scenario=scenario)
    
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig_charts, use_container_width=True)
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "📋 Glossary", "✅ The Golden Rule", "📖 Theory & History", "🔬 SME Analysis", "🏛️ Regulatory & Compliance"])
        
        with tabs[0]:
            st.markdown("""
            **Understanding the Process State Space Plot:**
            The top plot is your map of the process's **Normal Operating Region (NOR)**, defined by the two ellipses.
            -   **Green Ellipse (95% Confidence):** Represents the most common, expected region of operation.
            -   **Red Ellipse (99% Confidence):** This is the multivariate **control limit**. Any point falling outside this ellipse is a statistical signal of a special cause.
            
            The ellipses are stretched diagonally because the process parameters are **correlated**. The shape is determined by the **Mahalanobis distance**, which accounts for this correlation.

            ---
            **Analysis of the '{scenario}' Scenario:**
            """.format(scenario=scenario))

            if scenario == 'Stable':
                st.success("The process is stable. The monitoring points (black stars) fall within the confidence ellipses, and both the T² and SPE charts show only normal variation.")
            elif scenario == 'Shift in Y Only':
                st.warning("**Diagnosis: A 'Stealth Shift' has occurred.** The monitoring points have shifted upwards, falling outside the red ellipse. This is detected by the **T² chart**, which is sensitive to deviations from the process center.")
            elif scenario == 'Correlation Break':
                st.error("**Diagnosis: An Unprecedented Event.** The monitoring points have fallen off the established correlation line. This is detected by the **SPE chart**, which is sensitive to deviations *from* the process model itself.")
            
            if fig_contrib is not None:
                st.markdown("--- \n ##### Root Cause Diagnosis")
                st.plotly_chart(fig_contrib, use_container_width=True)
        
        with tabs[1]:
            st.markdown("""
            ##### Glossary of MSPC Terms
            - **MSPC (Multivariate SPC):** A method for monitoring the stability of a process with multiple, correlated variables simultaneously.
            - **Mahalanobis Distance:** A measure of the distance between a point and a distribution. Unlike Euclidean distance, it accounts for the correlation between variables, effectively measuring distance in "standard deviations" in a multidimensional space.
            - **Hotelling's T²:** A multivariate statistic that is the square of the Mahalanobis distance. It measures the distance of a point from the center of a data cloud, adjusted for correlation. It is sensitive to shifts *along* the correlation structure of the data.
            - **SPE (Squared Prediction Error):** Also known as DModX. A statistic that measures the distance of a point *from* the PCA model of the process. It is sensitive to new events or a breakdown in the correlation structure.
            - **PCA (Principal Component Analysis):** An unsupervised machine learning technique used to reduce the dimensionality of a dataset while preserving as much variance as possible. It is the engine for building the MSPC model.
            - **Contribution Plot:** A diagnostic plot used to identify which of the original process variables are responsible for a T² or SPE alarm.
            """)
            
        with tabs[2]:
            st.error("""🔴 **THE INCORRECT APPROACH: The "Army of Univariate Charts" Fallacy**
Using dozens of individual charts is doomed to fail due to alarm fatigue and its blindness to "stealth shifts." It's like having nurses for each vital sign, but no doctor to interpret the full patient picture. A process can be "out of control" even when every individual parameter is "in control" if their combination is abnormal.""")
            st.success("""🟢 **THE GOLDEN RULE: Detect with T²/SPE, Diagnose with Contributions**
A robust MSPC program is a two-stage process.
1.  **Stage 1: Detect.** Use the **T² and SPE charts** as your primary, holistic health monitors to answer the simple question: "Is something wrong with my process?"
2.  **Stage 2: Diagnose.** If either chart alarms, *then* you use **contribution plots** to drill down and identify which of the original process variables are responsible for the signal. This provides a clear, data-driven starting point for the root cause investigation.""")

        with tabs[3]:
            st.markdown("""
            #### Historical Context: The Crisis of Dimensionality
            In the 1930s, statistics was largely a univariate world. Tools like Student's t-test and Shewhart's control charts were brilliant for analyzing one variable at a time. But scientists and economists were facing increasingly complex problems with dozens of correlated measurements. 
            
            **The 'Aha!' Moment (Hotelling):** The creator of this powerful technique was **Harold Hotelling**, one of the giants of 20th-century mathematical statistics. His genius was in generalization. He recognized that the squared t-statistic, $t^2 = (\\bar{x} - \\mu)^2 / (s^2/n)$, was a measure of squared distance, normalized by variance. In a 1931 paper, he introduced the **Hotelling's T-squared statistic**, which replaced the univariate terms with their vector and matrix equivalents. It provided a single number that represented the "distance" of a point from the center of a multivariate distribution, elegantly solving the problem of testing multiple means at once while accounting for all their correlations. The T² statistic is fundamentally a measure of the **Mahalanobis distance**, which was earlier formulated by P. C. Mahalanobis in 1936.
            """)
            st.markdown("#### Mathematical Basis")
            st.markdown("- **T² (Hotelling's T-Squared):** A measure of the **Mahalanobis distance**. It calculates the squared distance of a point `x` from the center of the data `x̄`, but it first 'warps' the space by the inverse of the covariance matrix `S⁻¹` to account for correlations.")
            st.latex(r"T^2 = (\mathbf{x} - \mathbf{\bar{x}})' \mathbf{S}^{-1} (\mathbf{x} - \mathbf{\bar{x}})")
            st.markdown("- **SPE (Squared Prediction Error):** The sum of squared residuals after projecting a data point onto the principal component model of the process. For a new point **x**, it is the squared distance to the PCA model plane.")
            st.latex(r"SPE = || \mathbf{x} - \mathbf{P}\mathbf{P}'\mathbf{x} ||^2")

        with tabs[4]:
            st.markdown("""
            #### SME Analysis: From Raw Data to Actionable Intelligence
            As a Subject Matter Expert (SME), this tool isn't just a data science curiosity; it's a powerful diagnostic and risk-management engine.

            ##### How is this data gathered and what are the parameters?
            The data used by this model is a simplified version of what we collect during **late-stage development and tech transfer validation runs**. Key parameters like `Temperature` and `Pressure` are logged in a LIMS or ELN from instruments.

            ##### How do we interpret the plots and gain insights?
            The true power here is moving from "what happened" to "why it happened."
            -   **Process State Space Plot:** This is our primary visualization of the process's "healthy" state, defined by the ellipses. It immediately shows if a deviation is a simple shift or a more complex breakdown of the process model itself.
            -   **Contribution Plots:** When an alarm sounds, this is our **automated root cause investigation tool**. For a T² alarm, it points to the variable that has shifted. For an SPE alarm, it points to the variables whose relationship has broken down.

            ##### How would we implement this?
            1.  **Phase 1 (Model Building):** Use data from 20-30 successful, validated runs to build the PCA model and establish the control limits (T² and SPE).
            2.  **Phase 2 (Monitoring):** Deploy the charts for real-time monitoring as part of Continued Process Verification (CPV).
            3.  **Phase 3 (Automated Triage):** When an alarm triggers, automatically generate the contribution plot and attach it to an electronic alert sent to the process engineer. This dramatically accelerates the investigation.
            """)
            
        with tabs[5]:
            st.markdown("""
            These advanced analytical methods are key enablers for modern, data-driven approaches to process monitoring and control, as encouraged by global regulators.
            - **FDA Guidance for Industry - PAT — A Framework for Innovative Pharmaceutical Development, Manufacturing, and Quality Assurance:** This tool directly supports the PAT initiative's goal of understanding and controlling manufacturing processes through timely measurements to ensure final product quality.
            - **FDA Process Validation Guidance (Stage 3 - Continued Process Verification):** These advanced methods provide a more powerful way to meet the CPV requirement of continuously monitoring the process to ensure it remains in a state of control.
            - **ICH Q8(R2), Q9, Q10 (QbD Trilogy):** The use of sophisticated models for deep process understanding, real-time monitoring, and risk management is the practical implementation of the principles outlined in these guidelines.
            - **21 CFR Part 11 / GAMP 5:** If the model is used to make GxP decisions (e.g., real-time release), the underlying software and model must be fully validated as a Computerized System.
            """)
            
#====================================================================== 5. STABILITY ANALYSIS (SHELF LIFE) ACT III ============================================================================
def render_stability_analysis():
    """Renders the module for pharmaceutical stability analysis."""
    st.markdown("""
    #### Purpose & Application: The Expiration Date Contract
    **Purpose:** To fulfill a statistical contract with patients and regulators. This analysis determines the shelf-life for a drug product by proving, with high confidence, that a Critical Quality Attribute (CQA) like potency will remain within specification over time.
    
    **Strategic Application:** This is a mandatory, high-stakes analysis for any commercial pharmaceutical product, as required by the **ICH Q1E guideline**. It is the data-driven foundation of the expiration date printed on every vial and box. An incorrectly calculated shelf-life can lead to ineffective medicine, patient harm, and massive product recalls.
    """)
    
    st.info("""
    **Interactive Demo:** Use the sliders to simulate different stability profiles.
    - **`Batch-to-Batch Variation`**: The most critical parameter. At low values, batches are consistent and can be "pooled". At high values, the slopes differ, the **Poolability test fails (p < 0.25)**, and the pooled model is technically invalid.
    - **`Degradation Rate` & `Assay Variability`**: These control the overall stability and measurement noise, directly impacting the final shelf-life.
    """)
    
    with st.sidebar:
        st.sidebar.subheader("Stability Analysis Controls")
        degradation_slider = st.sidebar.slider("📉 Mean Degradation Rate (%/month)", -1.0, -0.1, -0.4, 0.05)
        noise_slider = st.sidebar.slider("🎲 Assay Variability (SD)", 0.2, 2.0, 0.5, 0.1)
        batch_var_slider = st.sidebar.slider("🏭 Batch-to-Batch Variation (SD)", 0.0, 0.5, 0.1, 0.05,
            help="Controls the variability in starting potency and degradation rate between batches. High values will cause the poolability test to fail.")

    fig, shelf_life, fitted_slope, poolability_p = plot_stability_analysis(
        degradation_rate=degradation_slider,
        noise_sd=noise_slider,
        batch_to_batch_sd=batch_var_slider
    )
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "📋 Glossary", "✅ The Golden Rule", "📖 Theory & History", "🏛️ Regulatory & Compliance"])
        
        with tabs[0]:
            st.metric(label="📈 Approved Shelf-Life (from Pooled Model)", value=f"{shelf_life}")
            st.metric(label="📊 Poolability Test (p-value for slopes)", value=f"{poolability_p:.3f}",
                      help="ICH Q1E suggests p > 0.25 to justify pooling batches. A low p-value indicates slopes are significantly different.")

            if poolability_p > 0.25:
                st.success("✅ p > 0.25: Batches are statistically similar. Pooling data is justified.")
            else:
                st.error("❌ p < 0.25: Batches show different degradation rates. A pooled shelf-life is not appropriate; the worst batch should be considered.")

            st.markdown("""
            **Reading the Plot:**
            - **Dashed Lines:** The individual trend line for each batch. If these lines are not parallel, the batches are degrading at different rates.
            - **Black Line:** The average trend across all batches (the pooled model).
            - **Red Dotted Line:** The conservative 95% lower confidence bound on the pooled mean. The shelf-life is where this line crosses the red specification limit.
            """)
        with tabs[1]:
            st.markdown("""
            ##### Glossary of Stability Terms
            - **Stability Study:** A formal study to determine the time period during which a drug product remains within its established specifications under defined storage conditions.
            - **Shelf-Life (Expiration Date):** The time period during which a drug product is expected to remain within the approved specification for use, if stored under defined conditions.
            - **ANCOVA (Analysis of Covariance):** A statistical test used to compare the slopes of regression lines between different groups (e.g., batches). It is the required test for determining if stability data can be pooled.
            - **Pooling:** The practice of combining data from multiple batches into a single dataset to estimate a common shelf-life. This is only permissible if the batches are statistically shown to be degrading at the same rate.
            - **Confidence Interval on the Mean:** The statistical basis for shelf-life. The shelf-life is the time point where the lower 95% confidence bound for the mean degradation trend line intersects the specification limit.
            """)
        with tabs[2]:
            st.error("""🔴 **THE INCORRECT APPROACH: The "Blind Pooling" Fallacy**
An analyst takes all stability data, throws it into one regression model, and calculates a shelf-life, without checking if the batches are behaving similarly.
- **The Flaw:** If one batch is a "fast degrader," its behavior will be masked by the better-performing batches. The pooled model will overestimate the shelf-life, creating a significant risk that some batches of product will fail specification long before their printed expiration date.""")
            st.success("""🟢 **THE GOLDEN RULE: Earn the Right to Pool**
The ICH Q1E guideline is built on a principle of statistical conservatism to protect patients.
1.  **First, Prove Poolability:** Perform a statistical test (ANCOVA) to check for a significant difference between batch slopes. The standard criterion is a **p-value > 0.25** for the interaction term.
2.  **If Poolable:** Combine the data and determine the shelf-life from the pooled model's confidence interval.
3.  **If NOT Poolable:** Analyze batches separately. The overall shelf-life must be based on the shortest shelf-life determined among all the batches.""")

        with tabs[3]:
            st.markdown(r"""
            #### Historical Context: The ICH Revolution
            **The Problem:** Prior to the 1990s, the requirements for stability testing could differ significantly between major markets like the USA, Europe, and Japan. This forced pharmaceutical companies to run slightly different, redundant, and costly stability programs for each region to gain global approval. The lack of a harmonized statistical approach meant that data might be interpreted differently by different agencies, creating regulatory uncertainty.
            
            **The 'Aha!' Moment:** The **International Council for Harmonisation (ICH)** was formed to end this inefficiency. A key working group was tasked with creating a single, scientifically sound standard for stability testing. This resulted in a series of guidelines, with **ICH Q1A** defining the required study conditions and **ICH Q1E ("Evaluation of Stability Data")** providing the definitive statistical methodology.
            
            **The Impact:** ICH Q1E, adopted in 2003, was a landmark guideline. It codified the use of regression analysis, formal statistical tests for pooling data across batches (ANCOVA), and the critical principle of using confidence intervals on the mean trend to determine shelf-life. It created a level playing field and a global gold standard, ensuring that the expiration date on a medicine means the same thing in New York, London, and Tokyo, and that it is backed by rigorous statistical evidence.
            """)
            st.markdown("#### Mathematical Basis")
            st.markdown("The core of the analysis is the **ANCOVA (Analysis of Covariance)** model, which tests if the slopes differ between batches:")
            st.latex(r"Y_{ij} = \beta_{0} + \alpha_{i} + \beta_{1}X_{ij} + (\alpha\beta)_{i}X_{ij} + \epsilon_{ij}")
            st.markdown("""
            -   `Yᵢⱼ`: The CQA for batch `i` at time `j`.
            -   `αᵢ`: The effect of batch `i` on the intercept.
            -   `β₁`: The common slope for all batches.
            -   `(αβ)ᵢ`: The **interaction term**, representing the *additional* slope for batch `i`.
            The poolability test is a hypothesis test on the interaction term:
            -   `H₀`: All `(αβ)ᵢ` are zero (all slopes are equal).
            -   `H₁`: At least one `(αβ)ᵢ` is not zero (at least one slope is different).
            If the p-value for this test is > 0.25, we fail to reject H₀ and proceed with a simpler, pooled model: `Y = β₀ + β₁X + ε`.
            """)
        with tabs[4]:
            st.markdown("""
            Stability analysis and shelf-life determination are governed by a specific set of harmonized international guidelines.
            - **ICH Q1E - Evaluation of Stability Data:** This is the primary global guideline that dictates the statistical methodology for analyzing stability data, including the use of regression analysis, confidence intervals, and the rules for pooling data from different batches.
            - **ICH Q1A(R2) - Stability Testing of New Drug Substances and Products:** Defines the study design, storage conditions, and testing frequency for stability studies.
            - **FDA Guidance for Industry - Q1E Evaluation of Stability Data:** The FDA's adoption and implementation of the ICH guideline.
            """)
#=================================================================== 6. RELIABILITY / SURVIVAL ANALYSIS ACT III ============================================================================
def render_survival_analysis():
    """Renders the module for Survival Analysis."""
    st.markdown("""
    #### Purpose & Application: The Statistician's Crystal Ball
    **Purpose:** To model "time-to-event" data and forecast the probability of survival over time. Its superpower is its unique ability to handle **censored data**-observations where the study ends before the event (e.g., failure or death) occurs. It allows us to use every last drop of information, even from the subjects who "survived" the study.
    
    **Strategic Application:** This is the core methodology for reliability engineering and is essential for predictive maintenance, risk analysis, and clinical research.
    - **Predictive Maintenance:** Instead of replacing parts on a fixed schedule, you can model their failure probability over time, moving from guesswork to a data-driven strategy.
    - **Clinical Trials:** The gold standard for analyzing endpoints like "time to disease progression" or "overall survival."
    """)

    st.info("""
    **Interactive Demo:** Use the sliders in the sidebar to simulate different reliability scenarios.
    - **`Group B Reliability`**: A higher value simulates a more reliable new component. Watch the red curve flatten and separate from the blue curve.
    - **`Censoring Rate`**: A higher rate simulates a shorter study. Notice the uncertainty (shaded confidence intervals) grows wider, and the "At Risk" numbers drop faster.
    """)

    with st.sidebar:
        st.sidebar.subheader("Survival Analysis Controls")
        lifetime_slider = st.sidebar.slider(
            "Group B Reliability (Lifetime Scale)",
            min_value=15, max_value=45, value=30, step=1,
            help="Controls the characteristic lifetime of the 'New Component' (Group B). A higher value means it's more reliable."
        )
        censor_slider = st.sidebar.slider(
            "Censoring Rate (%)",
            min_value=0, max_value=80, value=20, step=5,
            help="The percentage of items that are still 'surviving' when the study ends. Simulates shorter vs. longer studies."
        )
    
    fig, median_a, median_b, p_value = plot_survival_analysis(
        group_b_lifetime=lifetime_slider, 
        censor_rate=censor_slider/100.0
    )
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "📋 Glossary", "✅ The Golden Rule", "📖 Theory & History", "🏛️ Regulatory & Compliance"])
        
        with tabs[0]:
            st.metric(
                label="Log-Rank Test p-value", 
                value=f"{p_value:.3f}", 
                help="A p-value < 0.05 indicates a statistically significant difference between the survival curves."
            )
            st.metric(
                label="Median Survival (Group A)", 
                value=f"{median_a:.1f} Months" if not np.isnan(median_a) else "Not Reached",
                help="Time at which 50% of Group A have experienced the event."
            )
            st.metric(
                label="Median Survival (Group B)", 
                value=f"{median_b:.1f} Months" if not np.isnan(median_b) else "Not Reached",
                help="Time at which 50% of Group B have experienced the event."
            )

            st.markdown("""
            **Reading the Dashboard:**
            - **The Stepped Lines:** The Kaplan-Meier curves show the estimated probability of survival over time.
            - **Shaded Areas:** The 95% confidence intervals. They widen as fewer subjects are "at risk."
            - **Vertical Ticks:** Censored items (e.g., study ended).
            - **"At Risk" Table:** Shows how many subjects are still being followed at each time point. This provides crucial context for the reliability of the curve estimates.
            """)
        with tabs[1]:
            st.markdown("""
            ##### Glossary of Survival Terms
            - **Survival Analysis:** A branch of statistics for analyzing the expected duration of time until one or more events happen (e.g., failure of a component, death of a patient).
            - **Time-to-Event Data:** Data that consists of a time measurement until an event of interest occurs.
            - **Censoring:** A key feature of survival data where the event of interest has not occurred for some subjects by the end of the study. Censored data provides valuable information (e.g., "the lifetime is *at least* 24 months").
            - **Kaplan-Meier Estimator:** A non-parametric statistic used to estimate the survival function from lifetime data. It is the standard method for creating survival curves.
            - **Log-Rank Test:** A statistical test used to compare the survival distributions of two or more groups.
            - **Median Survival Time:** The time point at which 50% of the subjects in a group have experienced the event.
            """)
        with tabs[2]:
            st.error("""🔴 **THE INCORRECT APPROACH: The "Pessimist's Fallacy"**
This is a catastrophic but common error that leads to dangerously biased results.
- An analyst wants to know the average lifetime of a component. They take data from a one-year study, **throw away all the censored data** (the units that were still working at one year), and calculate the average time-to-failure for only the units that broke.
- **The Flaw:** This is a massive pessimistic bias. You have selected **only the weakest items** that failed early and completely ignored the strong, reliable items that were still going strong.""")
            st.success("""🟢 **THE GOLDEN RULE: Respect the Censored Data**
The core principle of survival analysis is that censored data is not missing data; it is valuable information.
- A tick on the curve at 24 months is not an unknown. It is a powerful piece of information: **The lifetime of this unit is at least 24 months.**
- The correct approach is to **always use a method specifically designed to handle censoring**, like the Kaplan-Meier estimator. This method correctly incorporates the information from both the "failures" and the "survivors" to produce an unbiased estimate of the true survival function.""")

        with tabs[3]:
            st.markdown(r"""
            #### Historical Context: The 1958 Revolution
            **The Problem:** In the mid-20th century, clinical research was booming, but a major statistical hurdle remained. How could you fairly compare two cancer treatments in a trial where, at the end of the study, many patients in both groups were still alive? Or some had moved away and were "lost to follow-up"? Simply comparing the percentage of deaths at the end was inefficient and biased. Researchers needed a way to use the information from every single patient, for the entire duration they were observed.

            **The 'Aha!' Moment:** This all changed in 1958 with a landmark paper in the *Journal of the American Statistical Association* by **Edward L. Kaplan** and **Paul Meier**. Their paper, "Nonparametric Estimation from Incomplete Observations," introduced the world to what we now universally call the **Kaplan-Meier estimator**.
            
            **The Impact:** It was a revolutionary breakthrough. They provided a simple, elegant, and statistically robust non-parametric method to estimate the true survival function, even with heavily censored data. This single technique unlocked a new era of research in medicine, enabling the rigorous analysis of clinical trials that is now standard practice. It also became a cornerstone of industrial reliability engineering, allowing for accurate lifetime predictions of components from studies that end before all components have failed.
            """)
            st.markdown("#### Mathematical Basis")
            st.markdown("The Kaplan-Meier estimate of the survival function `S(t)` is a product of conditional probabilities, calculated at each distinct event time `tᵢ`:")
            st.latex(r"S(t_i) = S(t_{i-1}) \times \left( 1 - \frac{d_i}{n_i} \right)")
            st.markdown(r"""
            - **`S(tᵢ)`** is the probability of surviving past time `tᵢ`.
            - **`nᵢ`** is the number of subjects "at risk" (i.e., still surviving and not yet censored) just before time `tᵢ`.
            - **`dᵢ`** is the number of events (e.g., failures) that occurred at time `tᵢ`.
            """)
            
            # --- THIS IS THE CORRECTED BLOCK ---
            st.markdown("The confidence interval for the survival probability is often calculated using **Greenwood's formula**, which estimates the variance of `S(t)`:")
            st.latex(r"\hat{Var}(S(t)) \approx S(t)^2 \sum_{t_i \leq t} \frac{d_i}{n_i(n_i - d_i)}")
        with tabs[4]:
            st.markdown("""
            Survival analysis is the standard methodology for time-to-event data in clinical trials and is also used for reliability engineering in medical devices.
            - **ICH E9 - Statistical Principles for Clinical Trials:** Discusses the appropriate analysis of time-to-event data, including the handling of censored data, for which Kaplan-Meier is the standard non-parametric method.
            - **FDA 21 CFR 820.30 (Design Controls):** For medical devices, design validation requires demonstrating reliability. Survival analysis is used to analyze data from reliability testing to predict the probability of failure over time.
            - **ICH Q1E:** The principles can also be applied to stability data to model the "time to Out-of-Specification (OOS)" event.
            """)
#======================================================================== 7. TIME SERIES ANALYSIS ACT III ============================================================================
def render_time_series_analysis():
    """Renders the module for Time Series analysis."""
    st.markdown("""
    #### Purpose & Application: The Watchmaker vs. The Smartwatch
    **Purpose:** To model and forecast time-dependent data by understanding its internal structure, such as trend, seasonality, and autocorrelation. This module compares two powerful philosophies for this task.
    
    **Strategic Application:** This is fundamental for demand forecasting, resource planning, and proactive process monitoring.
    - **⌚ ARIMA (The Classical Watchmaker):** A powerful "white-box" model. Like a master watchmaker, you must understand every gear, but you get a highly interpretable model that excels at short-term forecasting of stable processes.
    - **📱 Prophet (The Modern Smartwatch):** A modern, automated tool from Meta. It is designed to handle complex seasonalities, holidays, and changing trends with minimal user input, making it ideal for forecasting at scale.
    """)
    
    st.info("""
    **Interactive Demo:** Use the sliders to change the underlying structure of the time series data. 
    - **`Trend Changepoint`**: Introduce an abrupt change in the trend's slope. Notice how Prophet (red) adapts more easily than the rigid ARIMA model (green).
    - **`Random Noise`**: Observe how forecasting becomes more difficult and forecast intervals widen as the data gets noisier.
    - **Check the `ACF Plot`**: A good ARIMA model should have no significant bars outside the red lines, indicating the residuals are random noise.
    """)

    with st.sidebar:
        st.sidebar.subheader("Time Series Controls")
        trend_slider = st.sidebar.slider("📈 Trend Strength", 0, 50, 10, 5)
        noise_slider = st.sidebar.slider("🎲 Random Noise (SD)", 0.5, 10.0, 2.0, 0.5)
        # --- NEW SLIDER ADDED HERE ---
        changepoint_slider = st.sidebar.slider("🔄 Trend Changepoint Strength", -50.0, 50.0, 0.0, 5.0,
            help="Controls the magnitude of an abrupt change in the trend's slope halfway through the data. Prophet is designed to handle this well.")
    
    fig, mae_arima, mae_prophet = plot_time_series_analysis(
        trend_strength=trend_slider,
        noise_sd=noise_slider,
        changepoint_strength=changepoint_slider
    )
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "📋 Glossary", "✅ The Golden Rule", "📖 Theory & History", "🏛️ Regulatory & Compliance"])
        
        with tabs[0]:
            st.metric(label="⌚ ARIMA Forecast Error (MAE)", value=f"{mae_arima:.2f} units")
            st.metric(label="📱 Prophet Forecast Error (MAE)", value=f"{mae_prophet:.2f} units")

            st.markdown("""
            **Reading the Dashboard:**
            - **1. Main Forecast:** This plot shows the historical data and the two competing forecasts, including their 80% confidence intervals.
            - **2. ARIMA Residuals:** These are the one-step-ahead forecast errors from the ARIMA model. For a good model, this plot should look like random white noise with no discernible pattern.
            - **3. ARIMA Residuals ACF:** The Autocorrelation Function plot is the key diagnostic. It shows the correlation of the residuals with their own past values. **If any bars go outside the red significance lines, it means there is still predictable structure in the errors that the model has failed to capture.**

            **The Core Strategic Insight:** Prophet's main advantage is its automatic flexibility. Introduce a large **Trend Changepoint**, and watch Prophet's forecast adjust while the linear-trending ARIMA model fails to adapt, resulting in a much higher forecast error.
            """)
        with tabs[1]:
            st.markdown("""
            ##### Glossary of Time Series Terms
            - **Time Series:** A sequence of data points indexed in time order.
            - **Trend:** The long-term direction of the series (e.g., increasing, decreasing, or flat).
            - **Seasonality:** A distinct, repeating pattern in the data that occurs at regular intervals (e.g., daily, weekly, yearly).
            - **ARIMA (AutoRegressive Integrated Moving Average):** A class of statistical models for analyzing and forecasting time series data. It models the relationships between an observation and its own past values and past forecast errors.
            - **Prophet:** A forecasting procedure developed by Facebook. It is an additive model that fits non-linear trends with yearly, weekly, and daily seasonality, plus holiday effects.
            - **ACF (Autocorrelation Function):** A plot that shows the correlation of a time series with its own past values (lags). It is a key tool for diagnosing the fit of an ARIMA model.
            """)
        with tabs[2]:
            st.error("""🔴 **THE INCORRECT APPROACH: The "Blind Forecasting" Fallacy**
An analyst takes a column of data, feeds it into `model.fit()` and `model.predict()`, and presents the resulting line without checking the diagnostics.
- **The Flaw:** They've made no attempt to validate the model's assumptions. The residuals might show a clear pattern or the ACF plot might have significant lags, proving the model is wrong. This "black box" approach produces a forecast that is fragile and untrustworthy.""")
            st.success("""🟢 **THE GOLDEN RULE: Decompose, Validate, and Monitor**
A robust forecasting process is disciplined and applies regardless of the model you use.
1.  **Decompose and Understand:** Before modeling, visualize the data to understand its trend, seasonality, and any changepoints.
2.  **Fit and Diagnose:** After fitting a model, **always** analyze its residuals. The residuals must look like random noise. The ACF plot of the residuals is the statistical proof of this.
3.  **Validate on a Test Set:** The model's true performance is only revealed when it is tested on data it has never seen before.""")

        with tabs[3]:
            st.markdown("""
            #### Historical Context: Two Cultures of Forecasting
            **The Problem (The Classical Era):** Before the 1970s, forecasting was often an ad-hoc affair. There was no single, rigorous methodology that combined modeling, estimation, and validation into a coherent whole. 

            **The 'Aha!' Moment (ARIMA):** In their seminal 1970 book *Time Series Analysis: Forecasting and Control*, statisticians **George Box** and **Gwilym Jenkins** changed everything. They provided a comprehensive, step-by-step methodology for time series modeling. The **Box-Jenkins method**—a rigorous process of model identification (using ACF/PACF plots), parameter estimation, and diagnostic checking—became the undisputed gold standard for decades. The ARIMA model is the heart of this methodology, a testament to deep statistical theory.

            **The Problem (The Modern Era):** Fast forward to the 2010s. **Facebook** faced a new kind of challenge: thousands of internal analysts, not all of them statisticians, needed to generate high-quality forecasts for business metrics at scale. The manual, expert-driven Box-Jenkins method was too slow and complex for this environment.
            
            **The 'Aha!' Moment (Prophet):** In 2017, their Core Data Science team released **Prophet**. It was designed from the ground up for automation, performance, and intuitive tuning. Its key insight was to treat forecasting as a curve-fitting problem, making it robust to missing data and shifts in trend, and allowing analysts to easily incorporate domain knowledge like holidays. It sacrificed some of the statistical purity of ARIMA for massive gains in usability and scale.
            """)
            st.markdown("#### Mathematical Basis")
            st.markdown("- **ARIMA (AutoRegressive Integrated Moving Average):** A linear model that explains a series based on its own past.")
            st.latex(r"Y'_t = \sum_{i=1}^{p} \phi_i Y'_{t-i} + \sum_{j=1}^{q} \theta_j \epsilon_{t-j} + \epsilon_t")
            st.markdown("""
              - **AR (p):** The model uses the relationship between an observation `Y'` and its own `p` past values.
              - **I (d):** `Y'` is the series after being **d**ifferenced `d` times to make it stationary.
              - **MA (q):** The model uses the relationship between an observation and the residual errors `ε` from its `q` past forecasts.
            """)
            st.markdown("- **Prophet:** A decomposable additive model.")
            st.latex(r"y(t) = g(t) + s(t) + h(t) + \epsilon_t")
            st.markdown(r"""
            Where `g(t)` is a saturating growth trend with automatic changepoint detection, `s(t)` models complex seasonality using Fourier series, `h(t)` is for holidays, and `ε` is the error.
            """)
        with tabs[4]:
            st.markdown("""
            These advanced analytical methods are key enablers for modern, data-driven approaches to process monitoring and control, as encouraged by global regulators.
            - **FDA Guidance for Industry - PAT — A Framework for Innovative Pharmaceutical Development, Manufacturing, and Quality Assurance:** This tool directly supports the PAT initiative's goal of understanding and controlling manufacturing processes through timely measurements to ensure final product quality.
            - **FDA Process Validation Guidance (Stage 3 - Continued Process Verification):** These advanced methods provide a more powerful way to meet the CPV requirement of continuously monitoring the process to ensure it remains in a state of control.
            - **ICH Q8(R2), Q9, Q10 (QbD Trilogy):** The use of sophisticated models for deep process understanding, real-time monitoring, and risk management is the practical implementation of the principles outlined in these guidelines.
            - **21 CFR Part 11 / GAMP 5:** If the model is used to make GxP decisions (e.g., real-time release), the underlying software and model must be fully validated as a computerized system.
            """)
#========================================================================== 8. MULTIVARIATE ANALYSIS (MVA) ACT III============================================================================
def render_mva_pls():
    """Renders the module for Multivariate Analysis (PLS)."""
    st.markdown("""
    #### Purpose & Application: The Statistical Rosetta Stone
    **Purpose:** To act as a **statistical Rosetta Stone**, translating a massive, complex, and correlated set of input variables (X, e.g., an entire spectrum) into a simple, actionable output (Y, e.g., product concentration). **Partial Least Squares (PLS)** is the key that deciphers this code.
    
    **Strategic Application:** This is the statistical engine behind **Process Analytical Technology (PAT)** and modern chemometrics. It is specifically designed to solve the "curse of dimensionality" - problems where you have more input variables than samples and the inputs are highly correlated.
    - **Real-Time Spectroscopy:** Builds models that predict a chemical concentration from its NIR or Raman spectrum in real-time.
    - **"Golden Batch" Modeling:** PLS can learn the "fingerprint" of a perfect batch, modeling the complex relationship between hundreds of process parameters and final product quality.
    """)

    st.info("""
    **Interactive Demo:** Use the sliders to control the data quality. The dashboard shows the full modeling workflow:
    1.  **Raw Data:** See the spectral data, color-coded by the Y-value.
    2.  **Cross-Validation:** This plot is crucial! It shows how we choose the optimal number of latent variables (LVs) by finding the peak of the Q² (green) curve, avoiding overfitting.
    3.  **Performance:** Shows how well the final model's predictions match the actual values.
    4.  **Interpretation:** The VIP plot shows which variables (wavelengths) the model found most important.
    """)

    with st.sidebar:
        st.sidebar.subheader("Multivariate Analysis Controls")
        signal_slider = st.sidebar.slider("Signal Strength", 0.5, 5.0, 2.0, 0.5,
            help="Controls the strength of the true underlying relationship between the spectra (X) and the concentration (Y).")
        noise_slider = st.sidebar.slider("Noise Level (SD)", 0.1, 2.0, 0.2, 0.1,
            help="Controls the amount of random noise in the spectral measurements. Higher noise makes the signal harder to find.")
    
    fig, r2, q2, n_comp, rmsecv = plot_mva_pls(signal_strength=signal_slider, noise_sd=noise_slider)
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "📋 Glossary", "✅ The Golden Rule", "📖 Theory & History", "🏛️ Regulatory & Compliance"])
        
        with tabs[0]:
            st.metric(label="🎯 Model Q² (Predictive Power)", value=f"{q2:.3f}",
                      help="The cross-validated R². This is the most important measure of a model's predictive ability. Higher is better.")
            st.metric(label="📈 Model R² (Goodness of Fit)", value=f"{r2:.3f}",
                      help="How well the model fits the training data. A high R² with a low Q² is a sign of overfitting.")
            st.metric(label="🧬 Optimal Latent Variables (LVs)", value=f"{n_comp}",
                      help="The optimal number of hidden factors chosen via cross-validation.")
            st.metric(label="📉 RMSECV", value=f"{rmsecv:.2f} units",
                      help="Root Mean Squared Error of Cross-Validation. The typical prediction error in the original units of Y.")
        with tabs[1]:
            st.markdown("""
            ##### Glossary of MVA Terms
            - **MVA (Multivariate Analysis):** A set of statistical techniques used for analysis of data that contain more than one variable.
            - **PLS (Partial Least Squares) Regression:** A supervised regression technique that is a powerful alternative to standard linear regression, especially when input variables are numerous and highly correlated.
            - **Latent Variable (LV):** A hidden, underlying variable that is not directly measured but is inferred from the original variables. PLS works by finding LVs that both summarize the X-data and are highly correlated with the Y-data.
            - **Q² (Cross-Validated R²):** The most important metric for a PLS model. It measures the model's ability to *predict* new data, and is used to select the optimal number of LVs and prevent overfitting.
            - **VIP (Variable Importance in Projection) Score:** A measure of a variable's importance in the PLS model. Variables with a VIP score > 1 are generally considered important.
            """)
        with tabs[2]:
            st.error("""🔴 **THE INCORRECT APPROACH: The "Overfitting" Trap**
- An analyst keeps adding more and more Latent Variables (LVs) to their PLS model. They are thrilled to see the R-squared value (blue line in Plot 2) climb to 0.999.
- **The Flaw:** They've ignored the Q-squared value (green line), which has started to decrease. The model hasn't learned the true signal; it has simply memorized the noise in the training data. This model will fail when shown new data.""")
            st.success("""🟢 **THE GOLDEN RULE: The Best Model Predicts, It Doesn't Memorize**
A robust chemometric workflow is disciplined:
1.  **Use Cross-Validation:** Always use cross-validation to assess how the model will perform on future, unseen data.
2.  **Choose LVs based on Q²:** Select the number of latent variables that **maximizes the Q² (green line)**, not the R² (blue line). This is your best defense against overfitting.
3.  **Validate on an Independent Test Set:** The ultimate test of the model is its performance on a completely held-out set of samples that were not used in training or cross-validation.""")

        with tabs[3]:
            st.markdown("""
            #### Historical Context: The Father-Son Legacy
            **The Problem (The Social Sciences):** In the 1960s, social scientists and economists faced a major modeling challenge. They had complex systems with many correlated input variables and often a small number of observations. Standard multiple linear regression would fail spectacularly in these "data-rich but theory-poor" situations.

            **The 'Aha!' Moment (Herman Wold):** The brilliant Swedish statistician **Herman Wold** developed a novel solution. Instead of regressing Y on the X variables directly, he devised an iterative algorithm, **Partial Least Squares (PLS)**, that first extracts a small number of underlying "latent variables" from the X's that are maximally correlated with Y. This dimensionality reduction step elegantly solved the correlation and dimensionality problem.

            **The Impact (Svante Wold):** However, PLS's true potential was unlocked by Herman's son, **Svante Wold**, a chemist. In the late 1970s, Svante recognized that the problems his father was solving were mathematically identical to the challenges in **chemometrics**. Analytical instruments like spectrometers were producing huge, highly correlated datasets that traditional statistics couldn't handle. Svante Wold and his colleagues adapted and popularized PLS, turning it into the powerhouse of modern chemometrics and the statistical engine for the PAT revolution in the pharmaceutical industry.
            """)
            st.markdown("#### Mathematical Basis")
            st.markdown("PLS decomposes the input matrix `X` and output vector `y` into a set of latent variables (LVs), `T`, and associated loadings, `P` and `q`.")
            st.latex(r"X = T P^T + E")
            st.latex(r"y = T q^T + f")
            st.markdown("""
            The key is how the LVs (`T`) are found. Unlike PCA, which finds LVs that explain the most variance in `X` alone, PLS finds LVs that maximize the **covariance** between `X` and `y`. This means the LVs are constructed not just to summarize the inputs, but to be maximally useful for *predicting the output*. This makes PLS a supervised dimensionality reduction technique, which is why it is often more powerful than PCA followed by regression.
            """)
        with tabs[4]:
            st.markdown("""
            These advanced analytical methods are key enablers for modern, data-driven approaches to process monitoring and control, as encouraged by global regulators.
            - **FDA Guidance for Industry - PAT — A Framework for Innovative Pharmaceutical Development, Manufacturing, and Quality Assurance:** This tool directly supports the PAT initiative's goal of understanding and controlling manufacturing processes through timely measurements to ensure final product quality.
            - **FDA Process Validation Guidance (Stage 3 - Continued Process Verification):** These advanced methods provide a more powerful way to meet the CPV requirement of continuously monitoring the process to ensure it remains in a state of control.
            - **ICH Q8(R2), Q9, Q10 (QbD Trilogy):** The use of sophisticated models for deep process understanding, real-time monitoring, and risk management is the practical implementation of the principles outlined in these guidelines.
            - **21 CFR Part 11 / GAMP 5:** If the model is used to make GxP decisions (e.g., real-time release), the underlying software and model must be fully validated as a computerized system.
            """)
#========================================================================== 9. PREDICTIVE QC (CLASSIFICATION) ACT III============================================================================
def render_classification_models():
    """Renders the module for Predictive QC (Classification)."""
    st.markdown("""
    #### Purpose & Application: The AI Gatekeeper
    **Purpose:** To build an **AI Gatekeeper** that can inspect in-process data and predict, with high accuracy, whether a batch will ultimately pass or fail its final QC specifications. This moves quality control from a reactive, end-of-line activity to a proactive, predictive science.
    
    **Strategic Application:** This is the foundation of real-time release and "lights-out" manufacturing. By predicting outcomes early, we can:
    - **Prevent Failures:** Intervene in a batch that is trending towards failure.
    - **Optimize Resources:** Divert QC lab resources to focus on high-risk batches.
    - **Accelerate Release:** Provide statistical evidence for release based on in-process data.
    """)
    
    st.info("""
    **Interactive Demo:** Use the **Boundary Complexity** slider to change the true pass/fail relationship.
    - **High values (e.g., 20):** Creates a simple, almost linear boundary. Both models perform well, with similar AUCs.
    - **Low values (e.g., 8):** Creates a complex, non-linear "island" of failures. Watch the performance of the linear Logistic Regression model collapse, while the Random Forest's AUC remains high.
    """)

    with st.sidebar:
        st.sidebar.subheader("Predictive QC Controls")
        complexity_slider = st.sidebar.slider(
            "Boundary Complexity",
            min_value=4, max_value=25, value=12, step=1,
            help="Controls how non-linear the true pass/fail boundary is. Lower values create a more complex 'island' that is harder for linear models to solve."
        )
    
    fig, auc_lr, auc_rf = plot_classification_models(boundary_radius=complexity_slider)
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "📋 Glossary", "✅ The Golden Rule", "📖 Theory & History", "🏛️ Regulatory & Compliance"])
        
        with tabs[0]:
            st.metric(label="📈 Logistic Regression AUC", value=f"{auc_lr:.3f}",
                      help="Overall performance of the simpler, linear model.")
            st.metric(label="🚀 Random Forest AUC", value=f"{auc_rf:.3f}",
                      help="Overall performance of the more complex, non-linear model.")

            st.markdown("""
            **Reading the Dashboard:**
            - **Plots 1 & 2 (Decision Boundaries):** These plots show each model's *predicted probability of failure*. The dark line is the 50% threshold (the decision boundary).
            - **Logistic Regression** is forced to draw a straight line boundary.
            - **Random Forest** can draw a complex, circular boundary, correctly identifying the "island" of failures.
            - **Plot 3 (ROC Curves):** This is the ultimate scorecard. A curve closer to the top-left corner is better. The **Area Under the Curve (AUC)** provides a single number to quantify performance.

            **The Core Strategic Insight:** For complex biological processes, the relationship between parameters and quality is rarely linear. Modern ML models like Random Forest are often required to capture this complexity and build an effective AI Gatekeeper.
            """)
        with tabs[1]:
            st.markdown("""
            ##### Glossary of Classification Terms
            - **Classification:** A supervised machine learning task where the goal is to predict a categorical class label (e.g., "Pass" or "Fail").
            - **Logistic Regression:** A linear model used for binary classification. It models the probability of the default class.
            - **Random Forest:** A powerful ensemble learning method that operates by constructing a multitude of decision trees at training time. The final prediction is the mode of the classes output by individual trees.
            - **Decision Boundary:** The line or surface that separates the different classes in the feature space. A linear model can only create a straight-line boundary, while a non-linear model can create a complex, curved boundary.
            - **AUC (Area Under the Curve):** The primary metric for evaluating a classifier's performance. It represents the model's ability to distinguish between the positive and negative classes.
            """)
        with tabs[2]:
            st.error("""🔴 **THE INCORRECT APPROACH: The "Garbage In, Garbage Out" Fallacy**
An analyst takes all 500 available sensor tags, feeds them directly into a model, and trains it.
- **The Flaw:** With more input variables than batches, the model is likely to find spurious correlations and will fail to generalize to new data. The model hasn't been given any scientific context.""")
            st.success("""🟢 **THE GOLDEN RULE: Feature Engineering is the Secret Ingredient**
The success of a predictive model depends less on the algorithm and more on the quality of the inputs ("features").
1.  **Collaborate with SMEs:** Work with scientists to identify which process parameters are *scientifically likely* to be causal drivers of quality.
2.  **Engineer Smart Features:** Don't just use raw sensor values. Create more informative features, like the *slope* of a temperature profile or the *cumulative* feed volume.
3.  **Validate on Unseen Data:** The model's true performance is only revealed when it is tested on a hold-out set of batches it has never seen before.""")

        with tabs[3]:
            st.markdown("""
            #### Historical Context: The Two Cultures
            **The Problem:** For much of the 20th century, the world of statistical modeling was dominated by what statistician Leo Breiman called the **"Data Modeling Culture."** The goal was to use data to infer a simple, interpretable stochastic model (like linear or logistic regression) that could explain the relationship between inputs and outputs. The model's interpretability was paramount.

            **The 'Aha!' Moment:** The rise of computer science and machine learning in the latter half of the century gave rise to the **"Algorithmic Modeling Culture."** In this world, the internal mechanism of the model was treated as a black box. The primary goal was predictive accuracy, pure and simple. If a complex algorithm could get 99% accuracy, who cared how it worked?

            **The Impact:** This module showcases both cultures.
            - **Logistic Regression (Cox, 1958):** A masterpiece of the Data Modeling culture. It's a direct, interpretable generalization of linear regression for binary outcomes.
            - **Random Forest (Breiman, 2001):** A quintessential algorithm from the Algorithmic Modeling culture. It is an **ensemble method** that builds hundreds of individual decision trees and makes its final prediction based on a "majority vote." This "wisdom of the crowd" approach is highly accurate but inherently a black box.
            
            Today, the field of **Explainable AI (XAI)** is dedicated to bridging this gap, allowing us to use the power of algorithmic models while still understanding their reasoning.
            """)
            st.markdown("#### Mathematical Basis")
            st.markdown("- **Logistic Regression:** This model predicts the **log-odds** of the outcome as a linear function of the inputs, then uses the logistic (sigmoid) function to map this to a probability.")
            st.latex(r"\ln\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1X_1 + \dots + \beta_nX_n")
            st.latex(r"p = \frac{1}{1 + e^{-(\beta_0 + \beta_1X_1 + \dots)}}")
            st.markdown("- **Random Forest:** It is a collection of `N` individual decision tree models. For a new input `x`, the final prediction is the mode (most common vote) of all the individual tree predictions:")
            st.latex(r"\text{Prediction}(x) = \text{mode}\{ \text{Tree}_1(x), \text{Tree}_2(x), \dots, \text{Tree}_N(x) \}")
            st.markdown("Randomness is injected in two ways to ensure the trees are diverse: each tree is trained on a random bootstrap sample of the data, and at each split in a tree, only a random subset of features is considered.")
        with tabs[4]:
            st.markdown("""
            These advanced analytical methods are key enablers for modern, data-driven approaches to process monitoring and control, as encouraged by global regulators.
            - **FDA Guidance for Industry - PAT — A Framework for Innovative Pharmaceutical Development, Manufacturing, and Quality Assurance:** This tool directly supports the PAT initiative's goal of understanding and controlling manufacturing processes through timely measurements to ensure final product quality.
            - **FDA Process Validation Guidance (Stage 3 - Continued Process Verification):** These advanced methods provide a more powerful way to meet the CPV requirement of continuously monitoring the process to ensure it remains in a state of control.
            - **ICH Q8(R2), Q9, Q10 (QbD Trilogy):** The use of sophisticated models for deep process understanding, real-time monitoring, and risk management is the practical implementation of the principles outlined in these guidelines.
            - **21 CFR Part 11 / GAMP 5:** If the model is used to make GxP decisions (e.g., real-time release), the underlying software and model must be fully validated as a computerized system.
            """)

def render_xai_shap():
    """Renders the module for Explainable AI (XAI) using SHAP."""
    st.markdown("""
    #### Purpose & Application: The AI Root Cause Investigator
    **Purpose:** To deploy an **AI Investigator** that forces a complex "black box" model to confess exactly *why* it predicted a specific assay run would fail. **Explainable AI (XAI)** cracks open the black box to reveal the model's reasoning.
    
    **Strategic Application:** This is a crucial tool for validating and deploying predictive models in a regulated GxP environment. Instead of just getting a pass/fail prediction, you get a full root cause analysis for every run.
    - **Model Validation:** Confirm that the model is flagging runs for scientifically valid reasons (e.g., a low calibrator slope) and not due to spurious correlations.
    - **Proactive Troubleshooting:** If the model predicts a high risk of failure, SHAP immediately points to the most likely reasons, allowing for proactive intervention.
    """)

    st.info("""
    **Interactive Demo:** This dashboard shows a full XAI workflow. **The tool is computationally intensive, and depending on latency, it may take a minute to load.**
    1.  **Global Explanations:** See the model's overall strategy in the Beeswarm plot.
    2.  **Local Explanations:** Select a specific case to investigate and see its root cause analysis in the Waterfall plot.
    3.  **Feature Deep Dive:** Use the PDP/ICE Plot to explore how the model uses a single feature across all samples.
    """)

    with st.sidebar:
        st.sidebar.subheader("XAI Investigation Controls")
        case_choice = st.sidebar.radio(
            "Select Case for Local Explanation:",
            options=['highest_risk', 'lowest_risk', 'most_ambiguous'],
            format_func=lambda key: {
                'highest_risk': "Highest Failure Risk",
                'lowest_risk': "Lowest Failure Risk",
                'most_ambiguous': "Most Ambiguous (50/50)"
            }[key]
        )
        
        feature_names = ['Operator Experience (Months)', 'Reagent Age (Days)', 'Calibrator Slope', 'QC Level 1 Value', 'Instrument ID']
        dependence_feature_choice = st.sidebar.selectbox(
            "Select Feature for Deep Dive Plot:",
            options=feature_names
        )
    
    summary_buf, waterfall_buf, pdp_buf, instance_df, outcome, idx = plot_xai_shap(
        case_to_explain=case_choice,
        dependence_feature=dependence_feature_choice
    )
    
    # --- THIS IS THE CORRECTED TAB CREATION ---
    tabs = st.tabs(["Global Explanations", "Local Explanation", "Feature Deep Dive", "🔬 SME Analysis", "📋 Glossary", "🏛️ Regulatory & Compliance"])
    # --- END OF CORRECTION ---

    with tabs[0]:
        st.subheader("Global Feature Importance (The Model's General Strategy)")
        st.markdown("This **Beeswarm plot** shows the impact of every feature for every sample. Each dot is a single run. Red dots are high feature values, blue are low. A positive SHAP value increases the prediction of failure.")
        st.image(summary_buf)

    with tabs[1]:
        st.subheader(f"Local Root Cause Analysis for Run #{idx} ({case_choice.replace('_', ' ').title()})")
        st.markdown(f"**This run was selected for analysis. Actual Outcome: `{outcome}`**")
        st.dataframe(instance_df)
        st.markdown("The **Waterfall plot** below shows exactly how each feature contributed to this specific prediction, starting from the base rate and building to the final risk score.")
        st.image(waterfall_buf)

    with tabs[2]:
        st.subheader(f"Feature Deep Dive: '{dependence_feature_choice}'")
        st.markdown("""
        The **Partial Dependence Plot (PDP)** with **Individual Conditional Expectation (ICE)** lines shows how a feature affects predictions.
        - **The Red Dashed Line (PDP):** Shows the *average* effect on the predicted probability of failure as the feature value changes.
        - **The Light Blue Lines (ICE):** Each line represents a *single run*. This shows if the feature's effect is consistent across the dataset or if there are interactions.
        """)
        st.image(pdp_buf)

    with tabs[3]:
        st.markdown("""
        #### SME Analysis: From Raw Data to Actionable Intelligence
        As a Subject Matter Expert (SME) in process validation and tech transfer, this tool isn't just a data science curiosity; it's a powerful diagnostic and risk-management engine. Here’s how we would use this in a real-world GxP environment.

        ---
        
        ##### How is this data gathered and what are the parameters?
        The data used by this model is a simplified version of what we collect during **late-stage development, process characterization, and tech transfer validation runs**.
        -   **Data Gathering:** Every time an assay run is performed, we log key parameters in a Laboratory Information Management System (LIMS) or an Electronic Lab Notebook (ELN).
        -   **Parameters Considered:** `Operator Experience`, `Reagent Age`, `Calibrator Slope`, `QC Value`, and `Instrument ID` are all critical process parameters and material attributes that influence assay performance.

        ---

        ##### How do we interpret the plots and gain insights?
        The true power here is moving from "what happened" to "why it happened."
        -   **Global Plot (Beeswarm):** This is our first validation checkpoint for the model itself. If I saw that a scientifically irrelevant factor was the most important, I would reject the model. The fact that `Operator Experience` and `Calibrator Slope` are top drivers gives me confidence that the AI's "thinking" aligns with scientific reality.
        -   **Local Plot (Waterfall):** This is our **automated root cause investigation tool**. For the "Highest Risk" case, the plot instantly shows the root cause narrative: e.g., an inexperienced operator combined with old reagents created a high-risk situation.
        -   **Feature Deep Dive (PDP/ICE):** This helps us define our **Design Space or Normal Operating Range**. For example, by plotting `Reagent Age`, we can see the exact point where the red line (average risk) starts to sharply increase, allowing us to set an internal expiry date (e.g., "do not use reagents older than 60 days") based on data, not just a guess.
        
        ---

        ##### How would we implement this?
        1.  **Phase 1 (Silent Monitoring):** The model runs in the background, helping us spot trends in failure causes during process monitoring meetings.
        2.  **Phase 2 (Advisory Mode):** The system is integrated with the LIMS. It can generate advisories like: **"Warning: Reagent Lot XYZ is 85 days old. This significantly increases risk. Consider using a newer lot."**
        3.  **Phase 3 (Proactive Control / Real-Time Release):** A fully validated model's predictions can become part of the batch record. A run with a very low predicted risk and a favorable SHAP explanation could be eligible for **Real-Time Release Testing (RTRT)**, accelerating production timelines.
        """)
    with tabs[4]:
        st.markdown("""
        ##### Glossary of XAI Terms
        - **XAI (Explainable AI):** A field of AI dedicated to creating techniques that produce machine learning models that are understandable and trustworthy to human users.
        - **SHAP (SHapley Additive exPlanations):** A game theory-based approach to explain the output of any machine learning model. It connects optimal credit allocation with local explanations using the classic Shapley values from game theory.
        - **SHAP Value:** The average marginal contribution of a feature value to the prediction across all possible coalitions of features. It represents the impact of that feature on pushing the model's prediction away from the baseline.
        - **Global Explanation:** An explanation of the model's overall behavior (e.g., the Beeswarm plot, which shows which features are most important on average).
        - **Local Explanation:** An explanation for a single, specific prediction (e.g., the Waterfall plot, which shows how each feature contributed to the risk score for one specific batch).
        - **PDP (Partial Dependence Plot):** A plot that shows the marginal effect of a feature on the predicted outcome of a model.
        """)
    with tabs[5]:
        st.markdown("""
        Explainable AI (XAI) is a critical emerging field for the validation of AI/ML models in a regulated environment. It addresses the "black box" problem.
        - **FDA AI/ML Action Plan:** The FDA is actively developing its framework for regulating AI/ML-based software. A key principle is transparency, and XAI methods like SHAP provide the evidence that a model's reasoning is scientifically sound.
        - **GAMP 5 - A Risk-Based Approach to Compliant GxP Computerized Systems:** The principles of system validation require a thorough understanding and verification of the system's logic. For an AI model, XAI is essential for fulfilling the spirit of User and Functional Requirement Specifications (URS/FS).
        - **Good Machine Learning Practice (GMLP):** While not yet a formal regulation, this set of principles is emerging as a standard for developing robust and trustworthy ML models in healthcare, and explainability is a core pillar.
        """)

def render_clustering():
    """Renders the module for unsupervised clustering."""
    st.markdown("""
    #### Purpose & Application: The Data Archeologist
    **Purpose:** To act as a **data archeologist**, sifting through your process data to discover natural, hidden groupings or "regimes." Without any prior knowledge, it can uncover distinct "civilizations" within your data, answering the question: "Are all of my 'good' batches truly the same, or are there different ways to be good?"
    
    **Strategic Application:** This is a powerful exploratory tool for deep process understanding. It moves you from assumption to discovery.
    - **Process Regime Identification:** Can reveal that a process is secretly operating in different states (e.g., due to raw material lots, seasons, or operator techniques), even when all batches are passing specification.
    - **Root Cause Analysis:** If a failure occurs, clustering can help determine which "family" of normal operation the failed batch was most similar to, providing critical clues for the investigation.
    """)
    
    st.info("""
    **Interactive Demo:** Use the sliders to control the underlying data structure.
    - **`True Number of Clusters`**: Changes the "correct" answer for `k`. See if the diagnostic plots (Elbow and Silhouette) can find it.
    - **`Cluster Separation` & `Spread`**: Control how easy or hard the clustering task is. Well-separated, tight clusters are easy to find and will result in a high Silhouette Score.
    """)

    with st.sidebar:
        st.sidebar.subheader("Clustering Controls")
        n_clusters_slider = st.sidebar.slider("True Number of Clusters", 2, 8, 3, 1,
            help="Controls the actual number of distinct groups generated in the data.")
        separation_slider = st.sidebar.slider("Cluster Separation", 5, 25, 15, 1,
            help="Controls how far apart the centers of the data clusters are.")
        spread_slider = st.sidebar.slider("Cluster Spread (Noise)", 1.0, 10.0, 2.5, 0.5,
            help="Controls the standard deviation (spread) within each cluster. Higher spread means more overlap.")
    
    fig, silhouette_val, optimal_k = plot_clustering(
        separation=separation_slider,
        spread=spread_slider,
        n_true_clusters=n_clusters_slider
    )
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "📋 Glossary", "✅ The Golden Rule", "📖 Theory & History", "🏛️ Regulatory & Compliance"])
        
        with tabs[0]:
            st.metric(label="📈 Optimal 'k' Found by Model", value=f"{optimal_k}",
                      help="The number of clusters recommended by the Silhouette analysis.")
            st.metric(label="🗺️ Cluster Quality (Silhouette Score)", value=f"{silhouette_val:.3f}",
                      help="A measure of how distinct the final clusters are. Higher is better (max 1.0).")

            st.markdown("""
            **Reading the Dashboard:**
            - **1. Discovered Regimes:** The main plot shows how the K-Means algorithm (using the optimal `k`) has partitioned the data.
            - **2. Elbow Method:** A heuristic for finding `k`. The "elbow" (often near the true `k`) is the point of diminishing returns where adding more clusters doesn't significantly reduce the total within-cluster variance (Inertia).
            - **3. Silhouette Analysis:** A more robust method. The peak of this curve indicates the value of `k` that results in the most dense and well-separated clusters. This is often a more reliable guide than the Elbow Method.
            """)
        with tabs[1]:
            st.markdown("""
            ##### Glossary of Clustering Terms
            - **Unsupervised Learning:** A type of machine learning where the algorithm learns patterns from untagged data, without a pre-defined outcome to predict.
            - **Clustering:** The task of grouping a set of objects in such a way that objects in the same group (a cluster) are more similar to each other than to those in other groups.
            - **K-Means:** A popular clustering algorithm that aims to partition `n` observations into `k` clusters in which each observation belongs to the cluster with the nearest mean (cluster centroid).
            - **Inertia (WCSS):** The sum of squared distances of samples to their closest cluster center. The Elbow Method looks for the "elbow" in the plot of Inertia vs. `k`.
            - **Silhouette Score:** A metric used to evaluate the quality of clusters. It measures how similar an object is to its own cluster compared to other clusters. Scores range from -1 to +1, with a high value indicating dense and well-separated clusters.
            """)
        with tabs[2]:
            st.error("""🔴 **THE INCORRECT APPROACH: The "If It Ain't Broke..." Fallacy**
- An analyst discovers three distinct clusters of successful batches. A manager responds, *"Interesting, but all of those batches passed QC, so who cares? Let's move on."*
- **The Flaw:** This treats a treasure map as a doodle. The discovery is important *because* all batches passed! It means there are different-and potentially more or less robust-paths to success. One "regime" might be operating dangerously close to a failure cliff.""")
            st.success("""🟢 **THE GOLDEN RULE: A Cluster is a Clue, Not a Conclusion**
The discovery of clusters is the **start** of the investigation, not the end.
1.  **Find the Clusters:** Use an algorithm like K-Means and diagnostics like the Silhouette plot to find a statistically sound grouping.
2.  **Profile the Clusters (This is the most important step!):** Treat each cluster as a separate group. Ask:
    - Do batches in Cluster 1 use a different raw material lot than Cluster 2?
    - Were the batches in Cluster 3 all run by the night shift?
This profiling step is what turns a statistical finding into actionable process knowledge.""")

        with tabs[3]:
            st.markdown("""
            #### Historical Context: The Dawn of Unsupervised Learning
            **The Problem:** In the 1950s, the field of statistics was almost entirely focused on "supervised" problems-testing pre-defined hypotheses or building models to predict a known outcome. But what if you didn't have a hypothesis? What if you just had a mountain of data and wanted to know if there were any natural structures hidden within it?

            **The 'Aha!' Moment:** This was a challenge faced at Bell Labs, a hotbed of innovation. **Stuart Lloyd**, in a 1957 internal memo, proposed a simple, iterative algorithm to solve this problem for signal processing. His goal was to find a small set of "codebook vectors" (cluster centroids) that could efficiently represent a much larger set of signals, a form of data compression. Independently, others like E. W. Forgy and James MacQueen developed similar ideas. MacQueen was the first to coin the term "k-means" in a 1967 paper.
            
            **The Impact:** The development of K-Means was a direct consequence of the dawn of the digital computing age, as the iterative process was finally feasible. It became a canonical example of an **unsupervised learning** algorithm, a paradigm where the goal is not to predict a known label but to discover the inherent structure in the data itself. Along with Principal Component Analysis (PCA), K-Means helped lay the groundwork for the modern field of data mining and data science.
            """)
            st.markdown("#### Mathematical Basis")
            st.markdown("K-Means is an optimization algorithm. Its objective is to find the set of `k` cluster centroids `C` that minimizes the **Within-Cluster Sum of Squares (WCSS)**, also known as inertia.")
            st.latex(r"\text{WCSS} = \sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2")
            st.markdown("""
            -   `k`: The number of clusters.
            -   `Cᵢ`: The set of all points belonging to cluster `i`.
            -   `μᵢ`: The centroid (mean) of cluster `i`.
            The algorithm iteratively performs two steps until the cluster assignments no longer change:
            1.  **Assignment Step:** Assign each data point `x` to the cluster `Cᵢ` with the nearest centroid `μᵢ`.
            2.  **Update Step:** Recalculate the centroid `μᵢ` for each cluster by taking the mean of all points assigned to it.
            """)
        with tabs[4]:
            st.markdown("""
            These advanced analytical methods are key enablers for modern, data-driven approaches to process monitoring and control, as encouraged by global regulators.
            - **FDA Guidance for Industry - PAT — A Framework for Innovative Pharmaceutical Development, Manufacturing, and Quality Assurance:** This tool directly supports the PAT initiative's goal of understanding and controlling manufacturing processes through timely measurements to ensure final product quality.
            - **FDA Process Validation Guidance (Stage 3 - Continued Process Verification):** These advanced methods provide a more powerful way to meet the CPV requirement of continuously monitoring the process to ensure it remains in a state of control.
            - **ICH Q8(R2), Q9, Q10 (QbD Trilogy):** The use of sophisticated models for deep process understanding, real-time monitoring, and risk management is the practical implementation of the principles outlined in these guidelines.
            - **21 CFR Part 11 / GAMP 5:** If the model is used to make GxP decisions (e.g., real-time release), the underlying software and model must be fully validated as a computerized system.
            """)

def render_anomaly_detection():
    """Renders the module for unsupervised anomaly detection."""
    st.markdown("""
    #### Purpose & Application: The AI Bouncer
    **Purpose:** To deploy an **AI Bouncer** for your data-a smart system that identifies rare, unexpected observations (anomalies) without any prior knowledge of what "bad" looks like. It doesn't need a list of troublemakers; it learns the "normal vibe" of the crowd and flags anything that stands out.
    
    **Strategic Application:** This is a game-changer for monitoring complex processes where simple rule-based alarms are blind to new problems.
    - **Novel Fault Detection:** It can flag a completely new type of process failure the first time it occurs, because it looks for "weirdness," not pre-defined failures.
    - **"Golden Batch" Investigation:** Can find which batches, even if they passed all specifications, were statistically unusual. These "weird-but-good" batches often hold the secrets to improving process robustness.
    """)

    st.info("""
    **Interactive Demo:** Use the **Expected Contamination** slider to control the model's sensitivity.
    1.  Observe the final classification in the **top-left plot**.
    2.  See how the algorithm works in the **top-right plot**: outliers (anomalies) are isolated with very few "questions" (splits), resulting in short paths.
    3.  The **bottom plot** shows how this translates to a clean separation of anomaly scores. The slider moves the black decision threshold.
    """)

    with st.sidebar:
        st.sidebar.subheader("Anomaly Detection Controls")
        contamination_slider = st.sidebar.slider(
            "Expected Contamination (%)",
            min_value=1, max_value=25, value=10, step=1,
            help="Your assumption about the percentage of anomalies in the data. This tunes the model's sensitivity by moving the decision threshold."
        )

    fig, num_anomalies = plot_isolation_forest(contamination_rate=contamination_slider/100.0)
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "📋 Glossary", "✅ The Golden Rule", "📖 Theory & History", "🏛️ Regulatory & Compliance"])
        
        with tabs[0]:
            st.metric(label="Total Data Points Scanned", value="215")
            st.metric(label="Anomalies Flagged by Model", value=f"{num_anomalies}")

            st.markdown("""
            **Reading the Dashboard:**
            - **1. Detection Results:** The model successfully identifies the scattered outliers (red crosses) while classifying the main, correlated data cloud as normal (blue circles).
            - **2. Example Isolation Tree:** This visualizes how the algorithm thinks. It makes random splits to isolate points. Notice that the leaf nodes at the very top (shallow depth) contain the anomalies, as they are easy to single out.
            - **3. Score Distribution:** This plot shows the result of the isolation process. The true outliers (red histogram) have much higher anomaly scores because they are easy to isolate. The inliers (grey histogram) have lower scores. The black line is the decision threshold controlled by the `Contamination` slider.
            """)
        with tabs[1]:
            st.markdown("""
            ##### Glossary of Anomaly Terms
            - **Anomaly Detection:** The identification of rare items, events, or observations which raise suspicions by differing significantly from the majority of the data.
            - **Unsupervised Learning:** This approach is unsupervised because it does not require pre-labeled examples of "anomalies" to learn. It learns the structure of "normal" and flags anything that deviates.
            - **Isolation Forest:** An unsupervised anomaly detection algorithm based on the principle that anomalies are "few and different," making them easier to isolate than normal points.
            - **Isolation Tree:** A random binary tree used to partition the data. The path length from the root to a leaf node represents how easy it was to isolate a point.
            - **Anomaly Score:** A score derived from the average path length across all trees in the forest. Anomalies will have a short average path length and thus a high anomaly score.
            - **Contamination:** A user-defined parameter that sets the expected proportion of anomalies in the dataset. It is used to set the decision threshold on the anomaly scores.
            """)
        with tabs[2]:
            st.error("""🔴 **THE INCORRECT APPROACH: The "Glitch Hunter"**
When an anomaly is detected, the immediate reaction is to dismiss it as a data error.
- *"Oh, that's just a sensor glitch. Delete the point and move on."*
- *"Let's increase the contamination parameter until the alarms go away."*
This approach treats valuable signals as noise. It's like the bouncer seeing a problem, shrugging, and looking the other way. You are deliberately blinding yourself to potentially critical process information.""")
            st.success("""🟢 **THE GOLDEN RULE: An Anomaly is a Question, Not an Answer**
The goal is to treat every flagged anomaly as the start of a forensic investigation.
- **The anomaly is the breadcrumb:** When the bouncer flags someone, you ask questions. "What happened in the process at that exact time? Was it a specific operator? A new raw material lot?"
- **Investigate the weird-but-good:** If a batch that passed all specifications is flagged as an anomaly, it's a golden opportunity. What made it different? Understanding these "good" anomalies is a key to process optimization.""")

        with tabs[3]:
            st.markdown("""
            #### Historical Context: Flipping the Problem on its Head
            **The Problem:** For decades, "outlier detection" was a purely statistical affair, often done one variable at a time. This falls apart in high-dimensional data where an event might be anomalous not because of one value, but because of a strange *combination* of many values. Most methods focused on building a complex model of what "normal" data looks like and then flagging anything that didn't fit. This was often slow and brittle.

            **The 'Aha!' Moment:** In a 2008 paper, Fei Tony Liu, Kai Ming Ting, and Zhi-Hua Zhou introduced the **Isolation Forest** with a brilliantly counter-intuitive insight. Instead of trying to define "normal," they decided to just try to **isolate** every data point. They reasoned that anomalous points are, by definition, "few and different." This makes them much easier to separate from the rest of the data. Like finding a single red marble in a jar of blue ones, it's easy to "isolate" because it doesn't blend in.
            
            **The Impact:** This simple but powerful idea had huge consequences. The algorithm was extremely fast because it didn't need to model the whole dataset; it could often identify an anomaly in just a few steps. It worked well in high dimensions and didn't rely on any assumptions about the data's distribution. The Isolation Forest became a go-to method for unsupervised anomaly detection.
            """)
            st.markdown("#### Mathematical Basis")
            st.markdown("The algorithm is built on an ensemble of `iTrees` (Isolation Trees). Each `iTree` is a random binary tree built by recursively making random splits on random features.")
            st.markdown("The **path length** `h(x)` for a point `x` is the number of splits required to isolate it. Anomalies, being different, will have a much shorter average path length across all trees in the forest. The final anomaly score `s(x, n)` for a point is calculated based on its average path length `E(h(x))`:")
            st.latex(r"s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}")
            st.markdown("Where `c(n)` is a normalization factor based on the sample size `n`. Scores close to 1 are highly anomalous, while scores much smaller than 0.5 are normal.")
        with tabs[4]:
            st.markdown("""
            These advanced analytical methods are key enablers for modern, data-driven approaches to process monitoring and control, as encouraged by global regulators.
            - **FDA Guidance for Industry - PAT — A Framework for Innovative Pharmaceutical Development, Manufacturing, and Quality Assurance:** This tool directly supports the PAT initiative's goal of understanding and controlling manufacturing processes through timely measurements to ensure final product quality.
            - **FDA Process Validation Guidance (Stage 3 - Continued Process Verification):** These advanced methods provide a more powerful way to meet the CPV requirement of continuously monitoring the process to ensure it remains in a state of control.
            - **ICH Q8(R2), Q9, Q10 (QbD Trilogy):** The use of sophisticated models for deep process understanding, real-time monitoring, and risk management is the practical implementation of the principles outlined in these guidelines.
            - **21 CFR Part 11 / GAMP 5:** If the model is used to make GxP decisions (e.g., real-time release), the underlying software and model must be fully validated as a computerized system.
            """)   
            
def render_advanced_ai_concepts():
    """Renders the interactive dashboard for advanced AI concepts in a V&V context."""
    st.markdown("""
    #### Purpose & Application: A Glimpse into the AI Frontier for V&V
    **Purpose:** To provide a high-level, interactive overview of how cutting-edge AI architectures can solve some of the most difficult challenges in bioprocess V&V and technology transfer.
    """)
    st.info("""
    Select an AI concept below. The diagram will become an interactive simulation of a real-world biotech problem, and a dedicated set of controls will appear in the sidebar to let you explore the solution.
    """)

    concept_key = st.radio(
        "Select an Advanced AI Concept to Explore:", 
        ["Transformers", "Graph Neural Networks (GNNs)", "Reinforcement Learning (RL)", "Generative AI"],
        horizontal=True
    )
    
    p1, p2, p3 = 0, 0, 0
    with st.sidebar:
        st.subheader(f"{concept_key} Controls")
        if concept_key == "Transformers":
            p1 = st.select_slider("Focus Attention on Day:", options=list(range(14)), value=10,
                help="Select a day in the batch. The red bars show how much the model 'attends to' other days when making a prediction based on this day's data.")
        elif concept_key == "Graph Neural Networks (GNNs)":
            p1 = st.slider("Strength of Evidence", 0.0, 2.0, 1.0, 0.1,
                           help="How strongly to propagate the 'guilt' signal backwards from the failed QC test. High strength quickly pinpoints the likely source lots.")
        elif concept_key == "Reinforcement Learning (RL)":
            p1 = st.slider("Cost of Feed Waste ($/L)", 1.0, 10.0, 3.0, 0.5,
                           help="How 'expensive' wasted feed media is. Higher cost forces the AI agent to learn a more conservative, efficient feeding strategy.")
            p2 = st.slider("Process Variability", 0.1, 1.0, 0.3, 0.1,
                           help="The amount of random, unpredictable noise in the cell culture. High variability makes aggressive control strategies risky.")
        elif concept_key == "Generative AI":
            p1 = st.slider("Model 'Creativity' (Variance)", 0.1, 1.0, 0.3, 0.1,
                           help="Controls the diversity of the generated failure data. Low values are safe but less informative. High values are diverse but may be unrealistic.")
    
    fig = plot_advanced_ai_concepts(concept_key, p1, p2, p3)
    
    col1, col2 = st.columns([0.65, 0.35])
    with col1:
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        tabs = st.tabs(["💡 Key Insights", "📋 Glossary", "✅ The Golden Rule", "📖 Theory & History", "🏛️ Regulatory & Compliance"])
        
        with tabs[0]:
            if concept_key == "Transformers":
                st.metric(label="🧠 Core Concept", value="Self-Attention")
                st.markdown("**The AI Historian for Your Batch Record.** It learns long-range dependencies, such as how an early event in a batch influences the final outcome weeks later.")
            elif concept_key == "Graph Neural Networks (GNNs)":
                st.metric(label="🧠 Core Concept", value="Message Passing")
                st.markdown("**The System-Wide Process Cartographer.** It models your entire facility as a network to trace contamination or failures back to their root cause.")
            elif concept_key == "Reinforcement Learning (RL)":
                st.metric(label="🧠 Core Concept", value="Reward Maximization")
                st.markdown("**The AI Process Optimization Pilot.** It learns the optimal control strategy for a process by running millions of experiments in a safe, digital twin environment.")
            elif concept_key == "Generative AI":
                st.metric(label="🧠 Core Concept", value="Distribution Learning")
                st.markdown("**The Synthetic Data Factory.** It learns from a few examples of rare events and generates new, realistic synthetic examples to train more robust predictive models.")

        with tabs[1]:
            # --- THIS IS THE DYNAMIC GLOSSARY ---
            st.markdown("##### Glossary of Key Terms")
            if concept_key == "Transformers":
                st.markdown("- **Self-Attention:** A mechanism that allows a model to weigh the importance of different parts of the input sequence when processing a specific part. This is the core innovation of the Transformer.")
                st.markdown("- **Tokenization:** The process of converting a sequence of data (like a time series) into a series of discrete units or 'tokens' that the model can understand.")
            elif concept_key == "Graph Neural Networks (GNNs)":
                st.markdown("- **Graph:** A data structure consisting of 'nodes' (entities, e.g., raw material lots) and 'edges' (the relationships between them).")
                st.markdown("- **Message Passing:** The fundamental operation of a GNN, where each node aggregates information from its neighbors to update its own state or representation.")
            elif concept_key == "Reinforcement Learning (RL)":
                st.markdown("- **Agent:** The AI model that learns to make decisions (e.g., the feeding policy).")
                st.markdown("- **Environment:** The world the agent interacts with. In this case, a high-fidelity simulation of a bioreactor (a 'digital twin').")
                st.markdown("- **Reward:** A numerical signal that tells the agent how good or bad its last action was. The agent's goal is to maximize the cumulative reward over time.")
            elif concept_key == "Generative AI":
                st.markdown("- **Generative Model:** An AI model that learns the underlying probability distribution of a dataset and can generate new, synthetic samples from that distribution.")
                st.markdown("- **Discriminative Model:** The more common type of AI model, which learns to distinguish between different categories of data (e.g., a classifier that predicts Pass/Fail).")
            # --- END OF DYNAMIC GLOSSARY ---

        with tabs[2]:
            if concept_key == "Transformers":
                st.success("🟢 **THE GOLDEN RULE:** Tokenize Your Process Narrative. Convert continuous data into a discrete sequence of meaningful events (e.g., `[Feed_Event, pH_Excursion, Operator_Shift]`).")
            elif concept_key == "Graph Neural Networks (GNNs)":
                st.success("🟢 **THE GOLDEN RULE:** Your Graph IS Your Model. The most important work is defining the nodes (e.g., equipment, lots) and edges (e.g., 'used-in' relationships).")
            elif concept_key == "Reinforcement Learning (RL)":
                st.success("🟢 **THE GOLDEN RULE:** The Digital Twin is the Dojo. An RL agent must be trained in a high-fidelity simulation to learn optimal control strategies with zero real-world risk.")
            elif concept_key == "Generative AI":
                st.success("🟢 **THE GOLDEN RULE:** Validate the Forgeries. The generated data is only useful if it is proven to be statistically indistinguishable from real data.")
        
        with tabs[3]:
            if concept_key == "Transformers":
                st.markdown("The 2017 Google Brain paper, **'Attention Is All You Need,'** introduced the Transformer. It discarded recurrence and relied solely on **self-attention**, allowing every point in a sequence to directly 'look at' every other point. This is the foundation of all modern Large Language Models.")
            elif concept_key == "Graph Neural Networks (GNNs)":
                st.markdown("Pioneered around 2017, GNNs generalize deep learning to irregular graph data. The core insight is **message passing**, where each node updates its state by aggregating information from its neighbors, allowing complex relationships to be modeled.")
            elif concept_key == "Reinforcement Learning (RL)":
                st.markdown("With roots in control theory, the modern era of RL was sparked by **DeepMind's AlphaGo** (2016). By combining traditional RL with deep neural networks, they demonstrated that an AI agent could learn superhuman strategies in complex environments through self-play.")
            elif concept_key == "Generative AI":
                st.markdown("Revolutionized in 2014 by Ian Goodfellow's invention of **Generative Adversarial Networks (GANs)**, which pit a 'Generator' and a 'Discriminator' against each other. This adversarial game forces the generator to create incredibly realistic synthetic data.")
        
        with tabs[4]:
            st.markdown("""
            These advanced AI/ML methods are key enablers for modern, data-driven approaches to process monitoring and control, as encouraged by global regulators.
            - **FDA AI/ML Action Plan & GMLP:** These tools are part of the emerging field of AI/ML in regulated industries. They must align with principles of transparency, risk management, and model lifecycle management as defined in developing guidance like the FDA's Action Plan and Good Machine Learning Practice (GMLP).
            - **FDA Guidance for Industry - PAT — A Framework for Innovative Pharmaceutical Development, Manufacturing, and Quality Assurance:** This tool directly supports the PAT initiative's goal of understanding and controlling manufacturing processes through timely measurements to ensure final product quality.
            - **ICH Q8(R2), Q9, Q10 (QbD Trilogy):** The use of sophisticated models for deep process understanding, real-time monitoring, and risk management is the practical implementation of the principles outlined in these guidelines.
            - **21 CFR Part 11 / GAMP 5:** If the model is used to make GxP decisions (e.g., real-time release), the underlying software and model must be fully validated as a computerized system.
            """)
#==============================================================================================================================================================================================
#======================================================================NEW METHODS UI RENDERING ==============================================================================================
#=============================================================================================================================================================================================
# ==============================================================================
# UI RENDERING FUNCTION (Method 1)
# ==============================================================================
def render_mewma_xgboost():
    """Renders the MEWMA + XGBoost Diagnostics module."""
    st.markdown("""
    #### Purpose & Application: The AI First Responder
    **Purpose:** To create a two-stage "Detect and Diagnose" system. A **Multivariate EWMA (MEWMA)** chart acts as a highly sensitive alarm for small, coordinated drifts in a process. When it alarms, a pre-trained **XGBoost + SHAP model** instantly performs an automated root cause analysis.
    
    **Strategic Application:** This represents the state-of-the-art in intelligent process monitoring.
    - **Detect:** The MEWMA chart excels at finding subtle "stealth shifts" that individual charts would miss because it understands the process's normal correlation structure.
    - **Diagnose:** Instead of technicians guessing the cause of an alarm, the SHAP plot provides an immediate, data-driven "Top Suspects" list, dramatically accelerating troubleshooting.
    """)
    st.info("""**Interactive Demo:** Use the sliders to control the simulated process. **The tool is computationally intensive, and depending on latency, it may take a minute to load.**
    - **`Gradual Drift Magnitude`**: Controls how quickly the Temp and Pressure variables drift away from baseline.
    - **`MEWMA Lambda (λ)`**: Controls the "memory" of the chart. A smaller lambda makes it more sensitive to tiny, persistent drifts.
    """)
    with st.sidebar:
        st.sidebar.subheader("MEWMA + XGBoost Controls")
        drift_slider = st.sidebar.slider("Gradual Drift Magnitude", 0.0, 0.1, 0.03, 0.01, help="Controls the rate of the slow, creeping drift introduced into Temp & Pressure.")
        lambda_slider = st.sidebar.slider("MEWMA Lambda (λ)", 0.05, 0.5, 0.2, 0.05, help="Controls the 'memory' of the MEWMA chart. Smaller values give longer memory.")
    fig_dashboard, buf_waterfall, alarm_time = plot_mewma_xgboost(drift_magnitude=drift_slider, lambda_mewma=lambda_slider)
    st.plotly_chart(fig_dashboard, use_container_width=True)
    st.subheader("Diagnostic Analysis")
    if alarm_time:
        col1, col2 = st.columns([0.4, 0.6])
        with col1:
            st.metric("Alarm Status", "🚨 OUT-OF-CONTROL")
            st.metric("First Detection Time", f"Observation #{alarm_time}")
            st.markdown("""**Interpretation:**
            - **Top Plot:** The gradual drift is almost invisible to the naked eye.
            - **Bottom Plot:** The MEWMA chart successfully accumulates evidence of this drift until it crosses the control limit.""")
        with col2:
            st.markdown("##### Automated Root Cause Diagnosis (SHAP Waterfall)")
            st.image(buf_waterfall)
            st.markdown("The waterfall plot explains the alarm. Red bars (`Temp`, `Pressure`) pushed the risk of failure higher.")
    else:
        st.success("✅ IN-CONTROL: No alarm detected in the monitoring phase.")
    st.markdown("---")
    tabs = st.tabs(["💡 Key Insights", "📋 Glossary", "✅ The Golden Rule", "📖 Theory & History", "🏛️ Regulatory & Compliance"])
    with tabs[0]:
        st.markdown("""
        **A Realistic Workflow & Interpretation:**
        1.  **Phase 1 (Training):** A multivariate model of the normal process is built using data from validated, successful runs. A classifier model (XGBoost) is also trained to distinguish between "in-control" and "out-of-control" states based on historical data.
        2.  **Phase 2 (Monitoring):** The **MEWMA chart** monitors the process in real-time. It is the high-sensitivity "smoke detector."
        3.  **Phase 3 (Diagnosis):** When the MEWMA alarms, the data from that time point is fed to the **XGBoost + SHAP** model. The resulting waterfall plot provides an immediate, rank-ordered list of the variables that contributed most to the alarm, guiding the investigation.
        
        **The Strategic Insight:** This hybrid approach solves the biggest weakness of traditional multivariate SPC. While charts like Hotelling's T² or MEWMA are good at *detecting* a problem, they are poor at *diagnosing* it. By fusing a sensitive statistical detector with a powerful, explainable AI diagnostician, you get the best of both worlds: robust detection and rapid, data-driven root cause analysis.
        """)
    with tabs[1]:
        st.markdown("""
        ##### Glossary of Key Terms
        - **MEWMA (Multivariate EWMA):** A multivariate SPC chart that applies the "memory" of an EWMA to a vector of correlated process variables. It is highly sensitive to small, persistent drifts in the process mean.
        - **XGBoost (Extreme Gradient Boosting):** A powerful and efficient machine learning algorithm that builds a predictive model as an ensemble of many simple decision trees.
        - **SHAP (SHapley Additive exPlanations):** A game theory-based method for explaining the output of any machine learning model. It calculates the contribution of each feature to a specific prediction.
        - **Waterfall Plot:** A SHAP visualization that shows how each feature's SHAP value contributes to pushing the model's prediction from a baseline value to the final output.
        """)
    with tabs[2]:
        st.error("""🔴 **THE INCORRECT APPROACH: The 'Whack-a-Mole' Investigation**
A multivariate alarm sounds. Engineers frantically check dozens of individual parameter charts, trying to find a clear signal. They might chase a noisy pH sensor, ignoring the subtle, combined drift in Temp and Pressure that is the real root cause, wasting valuable time while the process may be producing non-conforming material.""")
        st.success("""🟢 **THE GOLDEN RULE: Detect Multivariately, Diagnose with Explainability**
1.  **Trust the multivariate alarm.** It sees the process holistically and can detect problems that are invisible to univariate charts.
2.  **Use the explainable AI diagnostic (SHAP) as your first investigative tool.** It instantly narrows the search space from all possible causes to the most probable ones based on historical patterns.
3.  **Confirm with SME knowledge.** The SHAP output is a powerful clue, not a final verdict. Use this data-driven starting point to guide the subject matter expert's investigation. This turns a slow, manual investigation into a rapid, data-driven confirmation.""")
    with tabs[3]:
        st.markdown("""
        #### Historical Context: The Diagnostic Bottleneck
        **The Problem:** By the 1980s, engineers had powerful multivariate detection tools like Hotelling's T² chart. However, these charts were slow to detect small, persistent drifts. The invention of the univariate EWMA chart was a major step forward, but the multivariate world was still waiting for its "high-sensitivity" detector.

        **The First Solution (MEWMA):** In 1992, Lowry et al. published their paper on the Multivariate EWMA (MEWMA) chart. The insight was a direct generalization: apply the "memory" and "weighting" concepts of EWMA to the vector of process variables. This created a chart that was exceptionally good at detecting small, coordinated shifts that T² would miss.

        **The New Problem (The Diagnostic Bottleneck):** But MEWMA created a new challenge. An alarm from a MEWMA chart is just a single number crossing a line. It tells you *that* the system has drifted, but gives no information about *which* variables are the cause. This has been the primary weakness of MSPC for decades.

        **The Modern Fusion:** This is where the AI revolution provided the missing piece. **XGBoost** (2014) offered a way to build highly accurate models to predict an alarm state, and **SHAP** (2017) provided the key to unlock that model's "black box." By fusing the robust statistical detection of MEWMA with the powerful, explainable diagnostics of XGBoost and SHAP, we finally solved the diagnostic bottleneck, creating a true "detect and diagnose" system.
        """)
    with tabs[4]:
        st.markdown("""
        This advanced hybrid system is a state-of-the-art implementation of the principles of modern process monitoring and control.
        - **FDA Process Validation Guidance (Stage 3 - Continued Process Verification):** This tool provides a highly effective method for meeting the CPV requirement of continuously monitoring the process to ensure it remains in a state of control, and for investigating any departures.
        - **FDA Guidance for Industry - PAT:** This "Detect and Diagnose" system is a direct implementation of the PAT goal of understanding and controlling manufacturing through timely measurements and feedback loops.
        - **ICH Q9 (Quality Risk Management):** By providing rapid root cause analysis, this system significantly reduces the risk associated with process deviations, minimizing their duration and impact.
        - **GAMP 5 & 21 CFR Part 11:** As this system uses an AI/ML model to provide diagnostic information for a GxP process, the model and the software it runs on would need to be validated as a Computerized System.
        """)
# ==============================================================================
# UI RENDERING FUNCTION (Method 2)
# ==============================================================================
def render_bocpd_ml_features():
    """Renders the Bayesian Online Change Point Detection module."""
    st.markdown("""
    #### Purpose & Application: The AI Seismologist
    **Purpose:** To provide a real-time, probabilistic assessment of process stability. Unlike traditional charts that give a binary "in/out" signal, **Bayesian Online Change Point Detection (BOCPD)** calculates the full probability distribution of the "current run length" (time since the last change). It acts like a seismologist, constantly looking for the tremors that signal a process earthquake.
    
    **Strategic Application:** This is a sophisticated method for monitoring high-value processes where understanding uncertainty is critical.
    - **Monitoring Model Performance:** Instead of monitoring raw data, this tool monitors the forecast errors (residuals) from a predictive ML model. BOCPD can detect when the model's performance suddenly degrades, signaling that the underlying process has changed in a way the model doesn't understand.
    - **Adaptive Alarming:** Instead of a fixed control limit, you can set alarms based on probability (e.g., "alarm if the probability of a recent changepoint exceeds 90%").
    """)

    st.info("""
    **Interactive Demo:** Use the sliders to control the nature of the process change at observation #100.
    - The top two plots show the raw data and the model residuals. Notice how the change is much more apparent in the residuals.
    - The bottom two plots show the BOCPD output. A high-confidence detection is marked by the **"Most Likely Run Length"** dropping to zero and the **"Probability of a Changepoint"** spiking towards 100%.
    """)

    with st.sidebar:
        st.sidebar.subheader("BOCPD Controls")
        autocorr_slider = st.sidebar.slider("Change in Autocorrelation", 0.0, 0.8, 0.4, 0.1,
            help="How much the process's 'memory' or smoothness changes at the changepoint. A large change is easier to detect.")
        noise_inc_slider = st.sidebar.slider("Increase in Noise (Factor)", 1.0, 5.0, 2.0, 0.5,
            help="The factor by which the process standard deviation increases after the change point. A value of 2 means the noise doubles.")

    fig, change_prob = plot_bocpd_ml_features(autocorr_shift=autocorr_slider, noise_increase=noise_inc_slider)
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Analysis & Interpretation")
        st.metric("True Changepoint Location", "Obs #100")
        st.metric("Detection Certainty at Changepoint", f"{change_prob:.1%}",
                  help="The posterior probability that a changepoint occurred at exactly observation #100.")
        
    st.divider()
    st.subheader("Deeper Dive")
    tabs = st.tabs(["💡 Key Insights", "📋 Glossary", "✅ The Golden Rule", "📖 Theory & History", "🏛️ Regulatory & Compliance"])
    with tabs[0]:
        st.markdown("""
        **A Realistic Workflow & Interpretation:**
        1.  **Raw Data (Plot 1):** Shows the simulated process. The change in dynamics at the red line (e.g., becoming less smooth and more noisy) is subtle and very difficult to spot by eye.
        2.  **Model Residuals (Plot 2):** We monitor the errors from a simple predictive model that was trained on the "normal" process. Because the process dynamics change at the changepoint, the model starts making bigger, more volatile errors, making the change much more visible.
        3.  **Most Likely Run Length (Plot 3):** The algorithm's "best guess" of how long the process has been stable. Note the sharp drop to zero at the changepoint, signaling a reset.
        4.  **Probability of a Changepoint (Plot 4):** This is the clearest signal. The algorithm becomes highly confident that a changepoint has just occurred right at observation #100.
        """)
        
    with tabs[1]:
        st.markdown("""
        ##### Glossary of Key Terms
        - **BOCPD (Bayesian Online Change Point Detection):** A real-time algorithm that calculates the full probability distribution of the "run length" (the time since the last statistically significant change in a process).
        - **Changepoint:** A point in time where the statistical properties of a time series (e.g., its mean, variance, or autocorrelation) change.
        - **Run Length:** The amount of time that has passed since the last changepoint. BOCPD's output is a probability distribution over this value.
        - **Residuals:** The difference between an observed value and a predicted value from a model. Monitoring residuals is a powerful way to detect when a process deviates from its expected behavior.
        - **Hazard Rate:** A parameter in the BOCPD algorithm that represents the prior probability of a changepoint occurring at any given step.
        """)
        
    with tabs[2]:
        st.error("""🔴 **THE INCORRECT APPROACH: The 'Delayed Reaction'**
Waiting for a traditional SPC chart to alarm on a complex signal (like a change in autocorrelation or variance) can take a very long time, if it alarms at all. The underlying assumptions of the chart are violated, but it may not produce a clear signal until significant out-of-spec material has been produced.""")
        st.success("""🟢 **THE GOLDEN RULE: Monitor Model Residuals, Not Just Raw Data**
For complex, dynamic processes, the most sensitive way to detect a change is to:
1.  **Build a model of the process's **normal** behavior.** This model (which could be a simple AR(1) model or a complex Neural Network) learns the "fingerprint" of a stable process.
2.  **Monitor the **residuals** (forecast errors) of that model.**
3.  When the process changes, the model's assumptions will be violated, and the residuals will change their behavior (e.g., become larger, more volatile, or biased). This provides a powerful and early signal that something is fundamentally wrong. BOCPD is an excellent algorithm for detecting this change in the residuals.""")

    with tabs[3]:
        st.markdown("""
        #### Historical Context: From Offline to Online
        **The Problem:** For decades, changepoint detection was primarily an *offline*, retrospective analysis. An engineer would collect an entire dataset, run a complex algorithm, and get a result like: "A significant change was detected at observation #152." While useful for forensic investigations after a failure, this was useless for preventing the failure in the first place.

        **The 'Aha!' Moment:** In their 2007 paper, "Bayesian Online Changepoint Detection," **Ryan P. Adams and David J.C. MacKay** presented a brilliantly elegant solution. Their key insight was to reframe the problem from finding a single "best" changepoint to calculating the full, evolving probability distribution of the "run length" (the time since the last changepoint). 
        
        **The Impact:** This probabilistic approach was a game-changer. It provided a much richer output than a simple binary alarm, and its recursive, online nature was computationally efficient enough to run in real-time on streaming data. It effectively transformed changepoint detection from a historical analysis tool into a modern, real-time process monitoring system, especially powerful when combined with feature extraction from modern ML models.
        """)
        
    with tabs[4]:
        st.markdown("""
        This advanced hybrid system is a state-of-the-art implementation of the principles of modern process monitoring and control.
        - **FDA Process Validation Guidance (Stage 3 - Continued Process Verification):** This tool provides a highly effective method for meeting the CPV requirement of continuously monitoring the process to ensure it remains in a state of control, and for investigating any departures.
        - **FDA Guidance for Industry - PAT:** This "Model and Monitor Residuals" approach is a direct implementation of the PAT goal of understanding and controlling manufacturing through timely measurements and feedback loops.
        - **ICH Q9 (Quality Risk Management):** By providing rapid and probabilistic detection of process changes, this system can significantly reduce the risk of producing non-conforming material.
        - **GAMP 5 & 21 CFR Part 11:** As this system uses an ML model to provide diagnostic information for a GxP process, the model and the software it runs on would need to be validated as a Computerized System.
        """)
# ==============================================================================
# UI RENDERING FUNCTION (Method 3)
# ==============================================================================
def render_kalman_nn_residual():
    """Renders the Kalman Filter + Residual Chart module."""
    st.markdown("""
    #### Purpose & Application: The AI Navigator
    **Purpose:** To track and predict the state of a dynamic process in real-time, even with noisy sensor data. The **Kalman Filter** acts as an optimal "navigator," constantly predicting the process's next move and then correcting its course based on the latest measurement. The key output is the **residual**—the degree of "surprise" at each measurement.
    
    **Strategic Application:** This is fundamental for state estimation in any time-varying system.
    - **Intelligent Filtering:** Provides a smooth, real-time estimate of a process's true state, filtering out sensor noise.
    - **Early Fault Detection:** By placing a control chart on the residuals, we create a highly sensitive alarm system. If the process behaves in a way the Kalman Filter didn't predict, the residuals will jump out of their normal range, signaling a fault long before the raw data looks abnormal.
    """)
    st.info("""
    **Interactive Demo:** Use the sliders to tune the Kalman Filter. At time #70, a sudden shock is introduced.
    - **`Process Noise (Q)`**: This is a critical tuning parameter. A **low Q** means you trust your model more (smoother estimate). A **high Q** means you trust your measurements more (estimate tracks noisy data).
    - **`Measurement Noise (R)`**: The known noise of your sensor.
    - **`Shock Magnitude`**: The size of the process fault.
    """)
    with st.sidebar:
        st.sidebar.subheader("Kalman Filter Controls")
        q_slider = st.sidebar.slider("Process Noise (Q)", 0.0, 0.5, 0.01, 0.005, format="%.4f",
            help="Model Uncertainty. How much you expect the true state to randomly change at each step. Higher Q makes the filter more responsive to measurements.")
        noise_slider = st.sidebar.slider("Measurement Noise (R)", 0.5, 5.0, 1.0, 0.5,
            help="Sensor Uncertainty. The known standard deviation of the measurement sensor.")
        shock_slider = st.sidebar.slider("Process Shock Magnitude", 1.0, 20.0, 10.0, 1.0,
            help="The magnitude of the sudden, unexpected event that occurs at time #70.")

    fig, alarm_time = plot_kalman_nn_residual(
        measurement_noise=noise_slider,
        shock_magnitude=shock_slider,
        process_noise_q=q_slider
    )
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Analysis & Interpretation")
        st.metric("Process Shock Event", "Time #70")
        st.metric("Alarm on Residuals", f"Time #{alarm_time}" if alarm_time else "Not Detected", help="The time the residual chart first detected the shock.")

    st.divider()
    st.subheader("Deeper Dive")
    tabs = st.tabs(["💡 Key Insights", "📋 Glossary", "✅ The Golden Rule", "📖 Theory & History", "🏛️ Regulatory & Compliance"])
    
    with tabs[0]:
        st.markdown("""
        **A Realistic Workflow & Interpretation:**
        1.  **State Estimation (Plot 1):** The red line is the Kalman Filter's "best guess" of the true state (black dashed line), created by optimally blending its internal model with the noisy measurements (grey dots). The shaded red area is the filter's own uncertainty about its estimate.
        2.  **Tuning the Filter (Sidebar):**
            - **Increase `Process Noise (Q)`:** Watch the red estimate line become "jittery" and follow the noisy measurements more closely. The uncertainty band also widens. This tells the filter "don't trust your model, trust the data."
            - **Decrease `Process Noise (Q)`:** The red line becomes much smoother, ignoring the noisy data. This tells the filter "my model is good, trust the prediction."
        3.  **Fault Detection (Plot 2):** This chart shows the "surprise" at each step. Notice the huge spike at time #70 when the process shock occurs—the measurement was far from what the filter predicted. This provides an unambiguous alarm.
        """)
        
    with tabs[1]:
        st.markdown("""
        ##### Glossary of Key Terms
        - **Kalman Filter:** An optimal algorithm for estimating the hidden state of a dynamic system from a series of noisy measurements. It operates in a two-step "predict-update" cycle.
        - **State Estimation:** The process of estimating the true, unobservable internal state of a system based on external, noisy measurements.
        - **Residual (or Innovation):** The difference between an actual measurement and the Kalman Filter's prediction of that measurement. A large residual indicates a "surprise" and signals that the process is not behaving as the model expected.
        - **Process Noise (Q):** A tuning parameter representing the uncertainty in the process model itself. It's the amount you expect the true state to randomly change between time steps.
        - **Measurement Noise (R):** A parameter representing the known uncertainty or variance of the measurement sensor.
        - **Kalman Gain:** A value calculated at each step that determines how much weight to give to the new measurement versus the model's prediction.
        """)
        
    with tabs[2]:
        st.error("""🔴 **THE INCORRECT APPROACH: Monitoring Raw, Noisy Data**
Charting the raw measurements (grey dots) directly would lead to a wide, insensitive control chart. The process shock might not even trigger an alarm if it's small relative to the measurement noise. You are blind to subtle deviations from the expected *behavior*.""")
        st.success("""🟢 **THE GOLDEN RULE: Model the Expected, Monitor the Unexpected**
1.  Use a dynamic model (like a Kalman Filter or a Neural Network) to capture the known, predictable behavior of your process.
2.  This model separates the signal into two streams: the predictable part (the state estimate) and the unpredictable part (the residuals).
3.  Place your high-sensitivity control chart on the **residuals**. This is monitoring the "unexplained" portion of the data, which is where novel faults will always appear first.""")

    with tabs[3]:
        st.markdown("""
        #### Historical Context: From Space Race to Bioreactor
        **The Problem:** During the height of the Cold War and the Space Race, a fundamental challenge was navigation. How could you guide a missile, or more inspiringly, a spacecraft to the Moon, using only a stream of noisy, imperfect sensor readings? You needed a way to fuse the predictions from a physical model (orbital mechanics) with the incoming data to get the best possible estimate of your true position and velocity.

        **The 'Aha!' Moment:** In 1960, **Rudolf E. Kálmán** published his landmark paper describing a recursive algorithm that provided the optimal solution to this problem. The **Kalman Filter** was born. Its elegant two-step "predict-update" cycle was computationally efficient enough to run on the primitive computers of the era.
        
        **The Impact:** The filter was almost immediately adopted by the aerospace industry and was a critical, mission-enabling component of the **NASA Apollo program**. Without the Kalman Filter to provide reliable real-time state estimation, the lunar landings would not have been possible. Its applications have since exploded into countless fields.
        
        **The Neural Network Connection:** The classic Kalman Filter assumes you have a good *linear* model of your system. But what about a complex, non-linear bioprocess? The modern approach is to replace the linear model with a **Recurrent Neural Network (RNN)**. The RNN *learns* the complex non-linear dynamics from data, and the Kalman Filter framework provides the mathematically optimal way to blend the RNN's predictions with new sensor measurements.
        """)
        
    with tabs[4]:
        st.markdown("""
        This advanced hybrid system is a state-of-the-art implementation of the principles of modern process monitoring and control.
        - **FDA Process Validation Guidance (Stage 3 - Continued Process Verification):** This tool provides a highly effective method for meeting the CPV requirement of continuously monitoring the process to ensure it remains in a state of control, and for investigating any departures.
        - **FDA Guidance for Industry - PAT:** This "Model and Monitor Residuals" approach is a direct implementation of the PAT goal of understanding and controlling manufacturing through timely measurements and feedback loops.
        - **ICH Q9 (Quality Risk Management):** By providing rapid detection of deviations from the expected process trajectory, this system can significantly reduce risk.
        - **GAMP 5 & 21 CFR Part 11:** As this system uses an AI/ML model to provide diagnostic information for a GxP process, the model and the software it runs on would need to be validated as a Computerized System.
        """)
# ==============================================================================
# UI RENDERING FUNCTION (Method 4)
# ==============================================================================
def render_rl_tuning():
    """Renders the Reinforcement Learning for Chart Tuning module."""
    st.markdown("""
    #### Purpose & Application: The AI Economist
    **Purpose:** To use **Reinforcement Learning (RL)** to automatically tune the parameters of a control chart (e.g., EWMA's `λ` and `L`) to achieve the best possible **economic performance**. It finds the optimal balance in the fundamental trade-off between reacting too quickly (costly false alarms) and reacting too slowly (costly missed signals).
    
    **Strategic Application:** This moves SPC from a purely statistical exercise to a business optimization problem.
    - **Customized Monitoring:** The RL agent designs a chart specifically tuned to your process's unique failure modes and your business's specific cost structure.
    - **Risk-Based Control:** For a high-value product, the cost of a missed signal is enormous, so the agent will design a highly sensitive chart. For a low-cost intermediate, it may design a less sensitive chart to avoid nuisance alarms.
    """)
    st.info("""
    **Interactive Demo:** You are the business manager. Use the sliders to define the economic reality and target failure mode for your process. The dashboard shows the full optimization landscape the RL agent explores to find the best solution.
    """)
    with st.sidebar:
        st.sidebar.subheader("RL Economic & Process Controls")
        cost_fa_slider = st.sidebar.slider("Cost of a False Alarm ($)", 1, 20, 1, 1,
            help="The economic cost ($) of a single false alarm (stopping the process, investigating a non-existent problem, scrapping material).")
        cost_delay_slider = st.sidebar.slider("Cost of Detection Delay ($/unit time)", 1, 20, 5, 1,
            help="The economic cost ($) incurred for *each time unit* that a real process shift goes undetected (e.g., cost of producing scrap).")
        shift_size_slider = st.sidebar.slider("Target Shift Size to Detect (σ)", 0.5, 3.0, 1.0, 0.25,
            help="The magnitude of the critical process shift you want to detect as quickly as possible.")

    fig_3d, fig_2d, opt_lambda, opt_L, min_cost = plot_rl_tuning(
        cost_false_alarm=cost_fa_slider,
        cost_delay_unit=cost_delay_slider,
        shift_size=shift_size_slider
    )
    
    st.header("Economic Optimization Landscape")
    col1, col2 = st.columns([0.5, 0.5])
    with col1:
        st.plotly_chart(fig_3d, use_container_width=True)
    with col2:
        st.subheader("Analysis & Interpretation")
        st.metric("Optimal λ Found by RL", f"{opt_lambda:.3f}", help="The optimal EWMA memory parameter.")
        st.metric("Optimal Limit Width (L) Found by RL", f"{opt_L:.2f}σ", help="The optimal control limit width in multiples of sigma.")
        st.metric("Minimum Achievable Cost", f"${min_cost:.3f}", help="The best possible economic performance for this chart.")
        st.markdown("""
        **The RL Agent's Solution:**
        The agent balances two competing goals: maximizing the time between false alarms (ARL₀) and minimizing the time to detect a real shift (ARL₁). The **3D Cost Surface** shows the combined economic result of this trade-off. The agent finds the `(λ, L)` combination at the bottom of this "cost valley" to design the most profitable control chart.
        """)

    st.header("Diagnostic Plots & Final Chart")
    st.plotly_chart(fig_2d, use_container_width=True)
    st.divider()
    st.subheader("Deeper Dive")
    tabs = st.tabs(["💡 Key Insights", "📋 Glossary", "✅ The Golden Rule", "📖 Theory & History", "🏛️ Regulatory & Compliance"])
    
    with tabs[0]:
        st.markdown("""
        **A Realistic Workflow & Interpretation:**
        1.  **Define the Costs:** The most critical step is for the business to provide realistic cost estimates for a false alarm and a detection delay.
        2.  **Explore the Landscape (Plot 1):** The 3D surface shows that poor parameter choices can be extremely costly. The agent's job is to find the lowest point.
        3.  **Understand the Trade-off (Plots 2 & 3):** The contour plots show the underlying statistical reality. Designs in the top-right have a very long time to a false alarm (high ARL₀) but are slow to detect real shifts (high ARL₁). Designs in the bottom-left are fast to detect shifts but have frequent false alarms.
        4.  **Deploy the Optimal Chart (Plot 4):** The final chart is the EWMA chart built with the economically optimal `λ` and `L` parameters.
        
        **The Strategic Insight:** Try increasing the **Cost of Detection Delay**. The RL agent will find a new optimum with a smaller `λ` and/or `L`, creating a more "nervous" and sensitive chart, because the business has decided that missing a shift is more costly than having a few extra false alarms.
        """)
        
    with tabs[1]:
        st.markdown("""
        ##### Glossary of Key Terms
        - **Reinforcement Learning (RL):** A field of AI where an "agent" learns to make optimal decisions by interacting with an environment (or a simulation) to maximize a cumulative reward.
        - **Economic Design of Control Charts:** A framework for selecting control chart parameters (like `λ` and `L`) to minimize a total cost function, rather than relying on purely statistical rules.
        - **Loss Function:** A mathematical function that quantifies the economic cost associated with a control chart's performance, typically including the cost of false alarms and the cost of detection delays.
        - **ARL₀ (Average Run Length - In Control):** The average number of samples taken before a false alarm occurs. A higher ARL₀ is better.
        - **ARL₁ (Average Run Length - Out of Control):** The average number of samples taken to detect a true process shift of a given magnitude. A lower ARL₁ is better.
        """)
        
    with tabs[2]:
        st.error("""🔴 **THE INCORRECT APPROACH: The 'Cookbook' Method**
A scientist reads a textbook that says 'use λ=0.2 and L=3 for EWMA charts.' They apply these default values to every process, regardless of the process stability, the value of the product, or the economic consequences of an error.""")
        st.success("""🟢 **THE GOLDEN RULE: Design the Chart to Match the Risk and the Business**
The control chart is not just a statistical tool; it's an economic asset. The tuning parameters should be deliberately chosen to minimize the total expected cost of quality for a specific process.
1.  **Quantify the Risk:** Work with stakeholders to define the costs of both Type I errors (false alarms) and Type II errors (missed signals).
2.  **Define the Target:** Identify the critical process shift that you must detect quickly.
3.  **Optimize:** Use a framework like this to find the chart parameters that provide the most cost-effective monitoring solution. This creates a highly defensible, data-driven control strategy.""")

    with tabs[3]:
        st.markdown("""
        #### Historical Context: The Unfulfilled Promise
        **The Problem:** The idea of designing control charts based on economics is surprisingly old, dating back to the work of **Acheson Duncan in the 1950s**. He recognized that the choice of chart parameters was an economic trade-off between the cost of looking for trouble (sampling and investigation) and the cost of not finding it in time (producing defective product). However, the mathematics required to find the optimal solution were incredibly complex and relied on many assumptions that were difficult to verify in practice. For decades, "Economic Design of Control Charts" remained an academically interesting but practically ignored field.

        **The 'Aha!' Moment (Simulation):** The modern solution came not from better math, but from more computing power. **Reinforcement Learning (RL)**, a field that exploded in the 2010s with successes like AlphaGo, provided a new paradigm. Instead of solving complex equations, an RL agent could learn the optimal strategy through millions of trial-and-error experiments in a fast, simulated "digital twin" of the manufacturing process. The agent's "reward" is simply the inverse of the economic loss function.
        
        **The Impact:** The rise of RL and high-fidelity process simulation has finally made the promise of economic design a practical reality. It allows engineers to move beyond statistical "rules of thumb" and design monitoring strategies that are provably optimized for their specific business and risk environment.
        """)
        
    with tabs[4]:
        st.markdown("""
        This advanced design methodology is a state-of-the-art implementation of the principles of modern process monitoring and control.
        - **ICH Q9 (Quality Risk Management):** This tool provides a direct, quantitative link between the business and patient risks (captured in the cost function) and the technical design of the control strategy. It is a perfect example of a risk-based approach to process monitoring.
        - **FDA Process Validation Guidance (Stage 3 - Continued Process Verification):** An economically designed chart provides a highly defensible rationale for the chosen monitoring strategy during CPV.
        - **FDA Guidance for Industry - PAT:** This tool supports the PAT goal of designing and controlling manufacturing based on a deep understanding of the process and the risks involved.
        - **GAMP 5 & 21 CFR Part 11:** If the RL agent and its simulation environment are used to formally set and justify control limits for a GxP process, the software and models would require validation as a Computerized System.
        """)
        
# ==============================================================================
# UI RENDERING FUNCTION (Method 5)
# ==============================================================================
def render_tcn_cusum():
    """Renders the TCN + CUSUM module."""
    st.markdown("""
    #### Purpose & Application: The AI Signal Processor
    **Purpose:** To create a powerful, hybrid system for detecting tiny, gradual drifts hidden within complex, non-linear, and seasonal time series data. A **Temporal Convolutional Network (TCN)** first learns and "subtracts" the complex but predictable patterns. Then, a **CUSUM chart** is applied to the remaining signal (the residuals) to detect any subtle, underlying drift.
    
    **Strategic Application:** This is for monitoring complex, dynamic processes like bioreactors or utility systems, where normal behavior is non-linear and cyclical.
    - **Bioreactor Monitoring:** A TCN can learn the complex S-shaped growth curve and daily cycles. The CUSUM on the residuals can then detect if the underlying cell growth rate is slowly starting to decline, signaling a problem with the media or culture health long before the raw data looks abnormal.
    """)
    st.info("""
    **Interactive Demo:** Use the sliders to control the simulated bioprocess.
    - **`Gradual Drift`**: Controls how quickly the hidden drift pulls the process away from its normal baseline.
    - **`Daily Cycle Strength`**: Controls the amplitude of the predictable, daily fluctuations. Notice that even with very strong cycles, the CUSUM chart on the TCN's residuals effectively detects the hidden drift.
    """)
    with st.sidebar:
        st.sidebar.subheader("TCN-CUSUM Controls")
        drift_slider = st.sidebar.slider("Gradual Drift Magnitude", 0.0, 0.2, 0.05, 0.01,
            help="The slope of the hidden linear trend added to the data. This is the subtle signal the CUSUM chart must find.")
        seasonality_slider = st.sidebar.slider("Daily Cycle Strength", 0.0, 5.0, 1.0, 0.5,
            help="Controls the amplitude of the cyclical patterns in the data. The TCN's job is to learn and remove this 'noise'.")

    fig, alarm_time = plot_tcn_cusum(drift_magnitude=drift_slider, daily_cycle_strength=seasonality_slider)
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Analysis & Interpretation")
        st.metric("Drift Detection Time", f"Hour #{alarm_time}" if alarm_time else "Not Detected",
                  help="The time the CUSUM chart first signaled a significant deviation in the residuals.")

    st.divider()
    st.subheader("Deeper Dive")
    tabs = st.tabs(["💡 Key Insights", "📋 Glossary", "✅ The Golden Rule", "📖 Theory & History", "🏛️ Regulatory & Compliance"])
    
    with tabs[0]:
        st.markdown("""
        **A Realistic Workflow & Interpretation:**
        1.  **Model the Predictable (Plot 1):** A TCN is trained on historical data from successful batches. It learns the complex but normal patterns, such as the S-shaped growth curve and daily cycles. The red dashed line is the TCN's forecast of what *should* be happening.
        2.  **Isolate the Unpredictable (Plot 3):** The model's forecast errors (the residuals) are calculated. This isolates the part of the signal that the model *cannot* explain. For a healthy process, these residuals should be random noise.
        3.  **Monitor for Drift (Plot 2):** The CUSUM chart is applied to these residuals. It is designed to ignore random noise but will accumulate the small, persistent signal caused by the hidden drift, eventually crossing the red control limit and firing a clear alarm (red 'X').

        **The Strategic Insight:** This hybrid approach allows you to apply a highly sensitive statistical tool (CUSUM) to a process that would normally violate all of its assumptions. The TCN acts as an intelligent "pre-processor," removing the complex, non-stationary patterns and allowing the simple, powerful CUSUM to do its job effectively.
        """)
        
    with tabs[1]:
        st.markdown("""
        ##### Glossary of Key Terms
        - **TCN (Temporal Convolutional Network):** A type of deep learning architecture that applies convolutional layers to time series data. It is known for its ability to capture long-range patterns efficiently.
        - **CUSUM (Cumulative Sum):** A "memory-based" control chart that plots the cumulative sum of deviations from a target. It is the fastest possible chart for detecting a shift of a specific, pre-defined magnitude.
        - **Residuals:** The difference between the actual data and the TCN's forecast. By applying CUSUM to the residuals, we monitor for changes in the part of the signal the TCN *could not* predict.
        - **Receptive Field:** The span of historical data that a TCN uses to make a single prediction. TCNs use dilated convolutions to achieve very large receptive fields efficiently.
        - **Non-Stationary Data:** A time series whose statistical properties such as mean and variance change over time. Bioprocess data is typically non-stationary.
        """)
        
    with tabs[2]:
        st.error("""🔴 **THE INCORRECT APPROACH: Charting the Raw Data**
Applying a CUSUM chart directly to the raw bioprocess data would be a disaster. The massive swings from the S-shaped growth curve and the daily cycles would cause constant false alarms, making the chart completely useless. The true, tiny drift signal would be completely buried in the predictable patterns.""")
        st.success("""🟢 **THE GOLDEN RULE: Separate the Predictable from the Unpredictable**
This is a fundamental principle of modern process monitoring for complex, dynamic systems.
1.  **Model the Known:** Use a sophisticated forecasting model (like a TCN or LSTM) to learn and mathematically remove the complex, **known patterns** (like growth curves and seasonality) from your data.
2.  **Monitor the Unknown:** Apply a sensitive change detection algorithm (like CUSUM or EWMA) to the model's **residuals** (the forecast errors). This focuses your monitoring on the part of the signal that is truly changing and unpredictable, where novel faults will always appear first.""")

    with tabs[3]:
        st.markdown("""
        #### Historical Context: The Evolution of Sequence Modeling
        **The Problem:** For years, Recurrent Neural Networks (RNNs) and their advanced variant, LSTMs, were the undisputed kings of sequence modeling. However, their inherently sequential nature-having to process time step `t` before moving to `t+1`-made them slow to train on very long sequences and difficult to parallelize on modern GPUs.

        **The 'Aha!' Moment:** In 2018, a paper by **Bai, Kolter, and Koltun**, "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling," showed that a different architecture could outperform LSTMs on many standard sequence tasks while being much faster. They systematized the **Temporal Convolutional Network (TCN)**. The key insight was to adapt techniques from computer vision (Convolutional Neural Networks) for time-series data. By using **causal convolutions** (to prevent seeing the future) and **dilated convolutions** (which exponentially increase the field of view), TCNs could learn very long-range patterns in parallel.

        **The Impact:** TCNs provided a powerful, fast, and often simpler alternative to LSTMs, becoming a go-to architecture for many time-series applications. Fusing this modern deep learning model with a classic, high-sensitivity statistical chart like **CUSUM (Page, 1954)** creates a hybrid system that leverages the best of both worlds: the pattern-recognition power of deep learning and the statistical rigor of classic SPC.
        """)
        
    with tabs[4]:
        st.markdown("""
        This advanced hybrid system is a state-of-the-art implementation of the principles of modern process monitoring and control.
        - **FDA Process Validation Guidance (Stage 3 - Continued Process Verification):** This tool provides a highly effective method for meeting the CPV requirement of continuously monitoring complex, non-stationary processes.
        - **FDA Guidance for Industry - PAT:** This "Model and Monitor Residuals" approach is a direct implementation of the PAT goal of understanding and controlling manufacturing through timely measurements and feedback loops.
        - **ICH Q9 (Quality Risk Management):** By providing early detection of subtle drifts, this system can significantly reduce the risk of producing non-conforming material.
        - **GAMP 5 & 21 CFR Part 11:** As this system uses an AI/ML model to provide diagnostic information for a GxP process, the model and the software it runs on would need to be validated as a Computerized System.
        """)
# ==============================================================================
# UI RENDERING FUNCTION (Method 6)
# ==============================================================================
def render_lstm_autoencoder_monitoring():
    """Renders the LSTM Autoencoder + Hybrid Monitoring module."""
    st.markdown("""
    #### Purpose & Application: The AI Immune System
    **Purpose:** To create a sophisticated, self-learning "immune system" for your process. An **LSTM Autoencoder** learns the normal, dynamic "fingerprint" of a healthy multivariate process. It then generates a single health score: the **reconstruction error**. We then deploy a **hybrid monitoring system** on this health score to detect different types of diseases (anomalies).
    
    **Strategic Application:** This is a state-of-the-art approach for unsupervised anomaly detection in multivariate time-series data, like that from a complex bioprocess.
    - **Learns Normal Dynamics:** The LSTM Autoencoder learns the complex, time-dependent correlations between many process parameters.
    - **One Health Score:** It distills hundreds of parameters into a single, chartable health score.
    - **Hybrid Detection:** An **EWMA chart** detects slow-onset diseases (gradual degradation), while a **BOCPD algorithm** detects acute events (sudden shocks).
    """)
    st.info("""
    **Interactive Demo:** Use the sliders to introduce two different types of anomalies into the multivariate process.
    - **`Gradual Drift Rate`**: Controls a slow, creeping deviation in both Temp and pH. Watch the **EWMA chart (orange)** in the bottom plot slowly rise to catch this.
    - **`Spike Magnitude`**: Controls a sudden shock at time #200. Watch the **BOCPD probability (purple)** in the bottom plot instantly react to this.
    """)
    with st.sidebar:
        st.sidebar.subheader("LSTM Anomaly Controls")
        drift_slider = st.sidebar.slider("Gradual Drift Rate", 0.0, 0.05, 0.02, 0.005,
            help="Controls how quickly the process drifts away from its normal behavior after time #100. Simulates gradual equipment degradation.")
        spike_slider = st.sidebar.slider("Spike Magnitude", 1.0, 5.0, 2.0, 0.5,
            help="Controls the size of the sudden shock at time #200. Simulates a sudden process fault or sensor failure.")

    fig, ewma_time, bocpd_time = plot_lstm_autoencoder_monitoring(drift_rate=drift_slider, spike_magnitude=spike_slider)
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        st.metric("EWMA Drift Detection Time", f"Hour #{ewma_time}" if ewma_time else "Not Detected",
                  help="Time the EWMA chart alarmed on the slow drift.")
        st.metric("BOCPD Spike Detection Time", f"Hour #{bocpd_time}" if bocpd_time else "Not Detected",
                  help="Time the BOCPD algorithm alarmed on the sudden spike.")

    st.divider()
    st.subheader("Deeper Dive")
    tabs = st.tabs(["💡 Key Insights", "📋 Glossary", "✅ The Golden Rule", "📖 Theory & History", "🏛️ Regulatory & Compliance"])
    with tabs[0]:
        st.markdown("""
        **A Realistic Workflow & Interpretation:**
        1.  **Learn the "Fingerprint" (Plot 1):** The LSTM Autoencoder is trained only on data from healthy, successful batches. It learns to reconstruct the normal, correlated dance between Temperature and pH. The dashed lines show the AI's reconstruction.
        2.  **Generate a Health Score (Plot 2):** When the live process data deviates from normal (due to the drift or spike), the AI struggles to reconstruct it. This failure is quantified as the **Reconstruction Error**, which serves as a single, powerful "health score" for the entire multivariate system.
        3.  **Deploy a Layered Defense (Plot 3):** Two specialized detectors monitor the health score:
            - The **EWMA chart (orange)** is insensitive to the sudden spike but accumulates the small, persistent signal from the gradual drift, eventually sounding an alarm.
            - The **BOCPD algorithm (purple)** ignores the slow drift but instantly detects the abrupt change caused by the spike, signaling an acute event.
        
        **The Strategic Insight:** This architecture creates a comprehensive "immune system" for your process. The LSTM Autoencoder is the T-cell that learns "self," and the hybrid monitoring charts are the antibodies and macrophages that detect and classify different types of "non-self" threats.
        """)
        
    with tabs[1]:
        st.markdown("""
        ##### Glossary of Key Terms
        - **LSTM (Long Short-Term Memory):** A type of recurrent neural network (RNN) that is excellent at learning long-range, time-dependent patterns in sequential data.
        - **Autoencoder:** An unsupervised neural network trained to reconstruct its input. It consists of an **Encoder** that compresses the data into a low-dimensional "fingerprint" and a **Decoder** that attempts to recreate the original data from that fingerprint.
        - **Reconstruction Error:** The difference between the original input data and the autoencoder's reconstructed output. It is a powerful anomaly score; if the model cannot reconstruct the data well, the error will be high, indicating an anomaly.
        - **Hybrid Monitoring:** The practice of applying multiple, specialized control charts (like EWMA for drifts and BOCPD for spikes) to a single "health score" like the reconstruction error to create a comprehensive detection system.
        """)
        
    with tabs[2]:
        st.error("""🔴 **THE INCORRECT APPROACH: The "One-Tool" Mindset**
An engineer tries to use a single Shewhart chart on the reconstruction error. It misses the slow drift entirely, and while it might catch the big spike, it gives no probabilistic context and provides no diagnostic information about the *type* of failure (chronic vs. acute).""")
        st.success("""🟢 **THE GOLDEN RULE: Use a Layered Defense for Anomaly Detection**
Different types of process failures leave different signatures in the data. A robust monitoring system must use a combination of tools, each specialized for a different type of signature. By running EWMA (for drifts) and BOCPD (for shocks) in parallel on the same AI-driven health score, you create a comprehensive immune system that can effectively detect and begin to classify both chronic and acute process diseases.""")

    with tabs[3]:
        st.markdown("""
        #### Historical Context: A Powerful Synthesis
        **The Problem:** Monitoring high-dimensional time-series data (like a bioreactor with hundreds of sensors) for anomalies is extremely difficult. A fault might not be a single sensor going haywire, but a subtle change in the *temporal correlation* between many sensors. How can you detect a deviation from a complex, dynamic "normal" state without having any examples of what "abnormal" looks like?

        **The 'Aha!' Moment (Synthesis):** This architecture became a popular and powerful technique in the late 2010s by intelligently combining three distinct ideas to solve the problem piece by piece:
        1.  **The Autoencoder:** A classic neural network design for unsupervised learning. It learns to compress data down to its essential features and then decompress it back to the original. Its ability to reconstruct the input serves as a measure of normalcy.
        2.  **The LSTM:** The Long Short-Term Memory network (**Hochreiter & Schmidhuber, 1997**) was the perfect choice to build the encoder and decoder, as it is specifically designed to learn the "grammar" and patterns of sequential data. Fusing these created the **LSTM Autoencoder**.
        3.  **Hybrid Monitoring:** The final piece was realizing that the autoencoder's output—the reconstruction error—is a single, powerful time series representing the health of the process. This allowed engineers to apply the best-in-class univariate monitoring tools, like **EWMA** and **BOCPD**, to this signal, creating a specialized, layered defense system.
        """)
        
    with tabs[4]:
        st.markdown("""
        This advanced hybrid system is a state-of-the-art implementation of the principles of modern process monitoring and control.
        - **FDA Process Validation Guidance (Stage 3 - Continued Process Verification):** This tool provides a highly effective method for meeting the CPV requirement of continuously monitoring complex, multivariate processes.
        - **FDA Guidance for Industry - PAT:** This "learn the fingerprint" approach is a direct implementation of the PAT goal of understanding and controlling manufacturing through timely measurements and feedback loops.
        - **ICH Q9 (Quality Risk Management):** By providing early detection of both gradual and sudden deviations, this system can significantly reduce the risk of producing non-conforming material.
        - **GAMP 5 & 21 CFR Part 11:** As this system uses an AI/ML model to provide diagnostic information for a GxP process, the model and the software it runs on would need to be validated as a Computerized System.
        """)

# ==============================================================================
# MAIN APP LOGIC AND LAYOUT
# ==============================================================================

# --- Initialize Session State ---
if 'current_view' not in st.session_state:
    st.session_state.current_view = 'Introduction'

# --- Sidebar Navigation ---
with st.sidebar:
    st.title("🧰 Toolkit Navigation")
    
    if st.sidebar.button("🚀 Project Framework", use_container_width=True):
        st.session_state.current_view = 'Introduction'
        st.rerun()

    st.divider()

    # --- FIX: all_tools dictionary and the for loop are now correctly indented inside the 'with st.sidebar:' block ---
    all_tools = {
        "ACT 0: PLANNING & STRATEGY": [
            "TPP & CQA Cascade",
            "Analytical Target Profile (ATP) Builder",
            "Quality Risk Management (QRM) Suite",
            "Design for Excellence (DfX)",
            "Validation Master Plan (VMP) Builder",
            "Requirements Traceability Matrix (RTM)"
        ],
        "ACT I: FOUNDATION & CHARACTERIZATION": [
            "Exploratory Data Analysis (EDA)",
            "Confidence Interval Concept",
            "Confidence Intervals for Proportions",
            "Core Validation Parameters",
            "LOD & LOQ",
            "Linearity & Range",
            "Non-Linear Regression (4PL/5PL)",
            "Gage R&R / VCA",
            "Attribute Agreement Analysis",
            "Comprehensive Diagnostic Validation",
            "ROC Curve Analysis",
            "Assay Robustness (DOE)",
            "Mixture Design (Formulations)",
            "Process Optimization: From DOE to AI",
            "Split-Plot Designs",
            "Causal Inference"
        ],
        "ACT II: TRANSFER & STABILITY": [
            "Sample Size for Qualification",
            "Advanced Stability Design",
            "Method Comparison",
            "Equivalence Testing (TOST)",
            "Statistical Equivalence for Process Transfer",
            "Process Stability (SPC)",
            "Process Capability (Cpk)",
            "First Time Yield & Cost of Quality",
            "Tolerance Intervals",
            "Bayesian Inference"
        ],
        "ACT III: LIFECYCLE & PREDICTIVE MGMT": [
            "Process Control Plan Builder",
            "Run Validation (Westgard)",
            "Small Shift Detection",
            "Multivariate SPC",
            "Stability Analysis (Shelf-Life)",
            "Reliability / Survival Analysis",
            "Time Series Analysis",
            "Multivariate Analysis (MVA)",
            "Predictive QC (Classification)",
            "Explainable AI (XAI)",
            "Clustering (Unsupervised)",
            "Anomaly Detection",
            "Advanced AI Concepts",
            "MEWMA + XGBoost Diagnostics",
            "BOCPD + ML Features",
            "Kalman Filter + Residual Chart",
            "RL for Chart Tuning",
            "TCN + CUSUM",
            "LSTM Autoencoder + Hybrid Monitoring"
        ]
    }

    for act_title, act_tools in all_tools.items():
        st.subheader(act_title)
        for tool in act_tools:
            if st.button(tool, key=tool, use_container_width=True):
                st.session_state.current_view = tool
                st.rerun()

# --- Main Content Area Dispatcher ---
view = st.session_state.current_view

if view == 'Introduction':
    render_introduction_content()
else:
    st.header(f"🔧 {view}")

    # --- FIX: Extra indentation at the end of this dictionary is removed ---
    PAGE_DISPATCHER = {
        # Act 0
        "TPP & CQA Cascade": render_tpp_cqa_cascade,
        "Analytical Target Profile (ATP) Builder": render_atp_builder,
        "Quality Risk Management (QRM) Suite": render_qrm_suite,
        "Design for Excellence (DfX)": render_dfx_dashboard,
        "Validation Master Plan (VMP) Builder": render_vmp_builder,
        "Requirements Traceability Matrix (RTM)": render_rtm_builder,
        
        # Act I
        "Exploratory Data Analysis (EDA)": render_eda_dashboard,
        "Confidence Interval Concept": render_ci_concept,
        "Confidence Intervals for Proportions": render_proportion_cis,
        "Core Validation Parameters": render_core_validation_params,
        "LOD & LOQ": render_lod_loq,
        "Linearity & Range": render_linearity,
        "Non-Linear Regression (4PL/5PL)": render_4pl_regression,
        "Gage R&R / VCA": render_gage_rr,
        "Attribute Agreement Analysis": render_attribute_agreement,
        "Comprehensive Diagnostic Validation": render_diagnostic_validation_suite,
        "ROC Curve Analysis": render_roc_curve,
        "Assay Robustness (DOE)": render_assay_robustness_doe,
        "Mixture Design (Formulations)": render_mixture_design,
        "Process Optimization: From DOE to AI": render_process_optimization_suite,
        "Split-Plot Designs": render_split_plot,
        "Causal Inference": render_causal_inference,
        
        # Act II
        "Sample Size for Qualification": render_sample_size_calculator,
        "Advanced Stability Design": render_stability_design,
        "Method Comparison": render_method_comparison,
        "Equivalence Testing (TOST)": render_tost,
        "Statistical Equivalence for Process Transfer": render_process_equivalence,
        "Process Stability (SPC)": render_spc_charts,
        "Process Capability (Cpk)": render_capability,
        "First Time Yield & Cost of Quality": render_fty_coq,
        "Tolerance Intervals": render_tolerance_intervals,
        "Bayesian Inference": render_bayesian,
        
        # Act III
        "Process Control Plan Builder": render_control_plan_builder,
        "Run Validation (Westgard)": render_multi_rule,
        "Small Shift Detection": render_ewma_cusum,
        "Multivariate SPC": render_multivariate_spc,
        "Stability Analysis (Shelf-Life)": render_stability_analysis,
        "Reliability / Survival Analysis": render_survival_analysis,
        "Time Series Analysis": render_time_series_analysis,
        "Multivariate Analysis (MVA)": render_mva_pls,
        "Predictive QC (Classification)": render_classification_models,
        "Explainable AI (XAI)": render_xai_shap,
        "Clustering (Unsupervised)": render_clustering,
        "Anomaly Detection": render_anomaly_detection,
        "Advanced AI Concepts": render_advanced_ai_concepts,
        "MEWMA + XGBoost Diagnostics": render_mewma_xgboost,
        "BOCPD + ML Features": render_bocpd_ml_features,
        "Kalman Filter + Residual Chart": render_kalman_nn_residual,
        "RL for Chart Tuning": render_rl_tuning,
        "TCN + CUSUM": render_tcn_cusum,
        "LSTM Autoencoder + Hybrid Monitoring": render_lstm_autoencoder_monitoring,
    }
    
    # --- FIX: This block is now correctly indented to be at the same level as the PAGE_DISPATCHER dictionary ---
    if view in PAGE_DISPATCHER:
        PAGE_DISPATCHER[view]()
    else:
        st.error("Error: Could not find the selected tool to render.")
        st.session_state.current_view = 'Introduction'
        st.rerun()
