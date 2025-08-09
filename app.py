# ==============================================================================
# LIBRARIES & IMPORTS (All imports are here)
# ==============================================================================
import streamlit as st
import numpy as np
import pandas as pd
import io
import matplotlib as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import beta, norm, t, f
from scipy.optimize import curve_fit
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.cluster import KMeans
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import shap

# ==============================================================================
# APP CONFIGURATION
# ==============================================================================
st.set_page_config(
    layout="wide",
    page_title="Biotech V&V Analytics Toolkit",
    page_icon="🔬"
)

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
    fig = go.Figure()
    v_model_stages = {
        'URS': {'name': 'User Requirements', 'x': 0, 'y': 5, 'question': 'What does the business/patient/process need?', 'tools': 'Business Case, User Needs Document'},
        'FS': {'name': 'Functional Specs', 'x': 1, 'y': 4, 'question': 'What must the system *do*?', 'tools': 'Assay: Linearity, LOD/LOQ. Instrument: Throughput. Software: User Roles.'},
        'DS': {'name': 'Design Specs', 'x': 2, 'y': 3, 'question': 'How will the system be built/configured?', 'tools': 'Assay: Robustness (DOE). Instrument: Component selection. Software: Architecture.'},
        'BUILD': {'name': 'Implementation', 'x': 3, 'y': 2, 'question': 'Build, code, configure, write SOPs, train.', 'tools': 'N/A (Physical/Code Transfer)'},
        'IQOQ': {'name': 'Installation/Operational Qualification (IQ/OQ)', 'x': 4, 'y': 3, 'question': 'Is the system installed correctly and does it operate as designed?', 'tools': 'Instrument Calibration, Software Unit/Integration Tests.'},
        'PQ': {'name': 'Performance Qualification (PQ)', 'x': 5, 'y': 4, 'question': 'Does the functioning system perform reliably in its environment?', 'tools': 'Gage R&R, Method Comp, Stability, Process Capability (Cpk).'},
        'UAT': {'name': 'User Acceptance / Validation', 'x': 6, 'y': 5, 'question': 'Does the validated system meet the original user need?', 'tools': 'Pass/Fail Analysis, Bayesian Confirmation, Final Report.'}
    }
    verification_color, validation_color = 'rgba(0, 128, 128, 0.9)', 'rgba(0, 104, 201, 0.9)'
    path_keys = ['URS', 'FS', 'DS', 'BUILD', 'IQOQ', 'PQ', 'UAT']
    path_x, path_y = [v_model_stages[p]['x'] for p in path_keys], [v_model_stages[p]['y'] for p in path_keys]
    fig.add_trace(go.Scatter(x=path_x, y=path_y, mode='lines', line=dict(color='darkgrey', width=4), hoverinfo='none'))
    for i, (key, stage) in enumerate(v_model_stages.items()):
        color = verification_color if i < 3 else validation_color if i > 3 else 'grey'
        fig.add_shape(type="rect", x0=stage['x']-0.45, y0=stage['y']-0.3, x1=stage['x']+0.45, y1=stage['y']+0.3, line=dict(color="black", width=2), fillcolor=color)
        fig.add_annotation(x=stage['x'], y=stage['y'], text=f"<b>{stage['name']}</b>", showarrow=False, font=dict(color='white', size=11, family="Arial"))
        fig.add_trace(go.Scatter(x=[stage['x']], y=[stage['y']], mode='markers', marker=dict(color='rgba(0,0,0,0)', size=70), hoverlabel=dict(bgcolor="white", font_size=14, font_family="Arial"), hoverinfo='text', text=f"<b>{stage['name']}</b><br><br><i>{stage['question']}</i><br><b>Examples / Tools:</b> {stage['tools']}"))
    for i in range(3):
        start_key, end_key = path_keys[i], path_keys[-(i+1)]
        fig.add_shape(type="line", x0=v_model_stages[start_key]['x'], y0=v_model_stages[start_key]['y'], x1=v_model_stages[end_key]['x'], y1=v_model_stages[end_key]['y'], line=dict(color="rgba(0,0,0,0.5)", width=2, dash="dot"))
    fig.update_layout(title_text='<b>The V&V Model for Technology Transfer (Hover for Details)</b>', title_x=0.5, showlegend=False, xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.6, 6.6]), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[1.4, 5.8]), height=600, margin=dict(l=20, r=20, t=60, b=20), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    return fig

@st.cache_data
def plot_act_grouped_timeline():
    all_tools_data = [
        {'name': 'Assay Robustness (DOE)', 'act': 1, 'year': 1926, 'inventor': 'R.A. Fisher', 'desc': 'Fisher publishes his work on Design of Experiments.'},
        {'name': 'Split-Plot Designs', 'act': 1, 'year': 1930, 'inventor': 'R.A. Fisher & F. Yates', 'desc': 'Specialized DOE for factors that are "hard-to-change".'},
        {'name': 'CI Concept', 'act': 1, 'year': 1937, 'inventor': 'Jerzy Neyman', 'desc': 'Neyman formalizes the frequentist confidence interval.'},
        {'name': 'ROC Curve Analysis', 'act': 1, 'year': 1945, 'inventor': 'Signal Processing Labs', 'desc': 'Developed for radar, now the standard for diagnostic tests.'},
        {'name': 'Variance Components', 'act': 1, 'year': 1950, 'inventor': 'Charles Henderson', 'desc': 'Advanced analysis for complex precision studies.'},
        {'name': 'Assay Robustness (RSM)', 'act': 1, 'year': 1951, 'inventor': 'Box & Wilson', 'desc': 'Box & Wilson develop Response Surface Methodology.'},
        {'name': 'Mixture Designs', 'act': 1, 'year': 1958, 'inventor': 'Henry Scheffé', 'desc': 'Specialized DOE for optimizing formulations and blends.'},
        {'name': 'LOD & LOQ', 'act': 1, 'year': 1968, 'inventor': 'Lloyd Currie (NIST)', 'desc': 'Currie at NIST formalizes the statistical basis.'},
        {'name': 'Non-Linear Regression', 'act': 1, 'year': 1975, 'inventor': 'Bioassay Field', 'desc': 'Models for sigmoidal curves common in immunoassays.'},
        {'name': 'Core Validation Params', 'act': 1, 'year': 1980, 'inventor': 'ICH / FDA', 'desc': 'Accuracy, Precision, Specificity codified.'},
        {'name': 'Gage R&R', 'act': 1, 'year': 1982, 'inventor': 'AIAG', 'desc': 'AIAG codifies Measurement Systems Analysis (MSA).'},
        {'name': 'Equivalence Testing (TOST)', 'act': 1, 'year': 1987, 'inventor': 'Donald Schuirmann', 'desc': 'Schuirmann proposes TOST for bioequivalence.'},
        {'name': 'Causal Inference', 'act': 1, 'year': 2018, 'inventor': 'Judea Pearl et al.', 'desc': 'Moving beyond correlation to identify root causes.'},
        {'name': 'Process Stability', 'act': 2, 'year': 1924, 'inventor': 'Walter Shewhart', 'desc': 'Shewhart invents the control chart at Bell Labs.'},
        {'name': 'Pass/Fail Analysis', 'act': 2, 'year': 1927, 'inventor': 'Edwin B. Wilson', 'desc': 'Wilson develops a superior confidence interval.'},
        {'name': 'Tolerance Intervals', 'act': 2, 'year': 1942, 'inventor': 'Abraham Wald', 'desc': 'Wald develops intervals to cover a proportion of a population.'},
        {'name': 'Method Comparison', 'act': 2, 'year': 1986, 'inventor': 'Bland & Altman', 'desc': 'Bland & Altman revolutionize method agreement studies.'},
        {'name': 'Process Capability', 'act': 2, 'year': 1986, 'inventor': 'Bill Smith (Motorola)', 'desc': 'Motorola popularizes Cpk with the Six Sigma initiative.'},
        {'name': 'Bayesian Inference', 'act': 2, 'year': 1990, 'inventor': 'Metropolis et al.', 'desc': 'Computational methods (MCMC) make Bayes practical.'},
        {'name': 'Multivariate SPC', 'act': 3, 'year': 1931, 'inventor': 'Harold Hotelling', 'desc': 'Hotelling develops the multivariate analog to the t-test.'},
        {'name': 'Small Shift Detection', 'act': 3, 'year': 1954, 'inventor': 'Page (CUSUM) & Roberts (EWMA)', 'desc': 'Charts for faster detection of small process drifts.'},
        {'name': 'Clustering', 'act': 3, 'year': 1957, 'inventor': 'Stuart Lloyd', 'desc': 'Algorithm for finding hidden groups in data.'},
        {'name': 'Predictive QC', 'act': 3, 'year': 1958, 'inventor': 'David Cox', 'desc': 'Cox develops Logistic Regression for binary outcomes.'},
        {'name': 'Reliability Analysis', 'act': 3, 'year': 1958, 'inventor': 'Kaplan & Meier', 'desc': 'Kaplan-Meier estimator for time-to-event data.'},
        {'name': 'Time Series Analysis', 'act': 3, 'year': 1970, 'inventor': 'Box & Jenkins', 'desc': 'Box & Jenkins publish their seminal work on ARIMA models.'},
        {'name': 'Multivariate Analysis', 'act': 3, 'year': 1975, 'inventor': 'Herman Wold', 'desc': 'Partial Least Squares for modeling complex process data.'},
        {'name': 'Run Validation', 'act': 3, 'year': 1981, 'inventor': 'James Westgard', 'desc': 'Westgard publishes his multi-rule QC system.'},
        {'name': 'Stability Analysis', 'act': 3, 'year': 1993, 'inventor': 'ICH', 'desc': 'ICH guidelines formalize statistical shelf-life estimation.'},
        {'name': 'Advanced AI/ML', 'act': 3, 'year': 2017, 'inventor': 'Vaswani, Lundberg et al.', 'desc': 'Transformers and Explainable AI (XAI) emerge.'},
    ]
    all_tools_data.sort(key=lambda x: (x['act'], x['year']))
    act_ranges = {1: (5, 45), 2: (50, 75), 3: (80, 115)}
    tools_by_act = {1: [], 2: [], 3: []}
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
        1: {'name': 'Act I: Foundation', 'color': 'rgba(0, 128, 128, 0.9)', 'boundary': (0, 48)},
        2: {'name': 'Act II: Transfer & Stability', 'color': 'rgba(0, 104, 201, 0.9)', 'boundary': (48, 78)},
        3: {'name': 'Act III: Lifecycle & Predictive', 'color': 'rgba(100, 0, 100, 0.9)', 'boundary': (78, 120)}
    }
    
    for act_info in acts.values():
        x0, x1 = act_info['boundary']
        fig.add_shape(type="rect", x0=x0, y0=-5.0, x1=x1, y1=5.0, line=dict(width=0), fillcolor='rgba(230, 230, 230, 0.7)', layer='below')
        fig.add_annotation(x=(x0 + x1) / 2, y=7.0, text=f"<b>{act_info['name']}</b>", showarrow=False, font=dict(size=20, color="#555"))

    fig.add_shape(type="line", x0=0, y0=0, x1=120, y1=0, line=dict(color="black", width=3), layer='below')

    for act_num, act_info in acts.items():
        act_tools = [tool for tool in all_tools_data if tool['act'] == act_num]
        fig.add_trace(go.Scatter(x=[tool['x'] for tool in act_tools], y=[tool['y'] for tool in act_tools], mode='markers', marker=dict(size=12, color=act_info['color'], symbol='circle', line=dict(width=2, color='black')), hoverinfo='text', text=[f"<b>{tool['name']} ({tool['year']})</b><br><i>{tool['desc']}</i>" for tool in act_tools], name=act_info['name']))

    for tool in all_tools_data:
        fig.add_shape(type="line", x0=tool['x'], y0=0, x1=tool['x'], y1=tool['y'], line=dict(color='grey', width=1))
        fig.add_annotation(x=tool['x'], y=tool['y'], text=f"<b>{tool['name']}</b><br><i>{tool['inventor']} ({tool['year']})</i>", showarrow=False, yshift=40 if tool['y'] > 0 else -40, font=dict(size=11, color=acts[tool['act']]['color']), align="center")

    fig.update_layout(title_text='<b>The V&V Analytics Toolkit: A Project-Based View</b>', title_font_size=28, title_x=0.5, xaxis=dict(visible=False), yaxis=dict(visible=False, range=[-8, 8]), plot_bgcolor='white', paper_bgcolor='white', height=900, margin=dict(l=20, r=20, t=140, b=20), showlegend=True, legend=dict(title_text="<b>Project Phase</b>", title_font_size=16, font_size=14, orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
    return fig

@st.cache_data
def plot_chronological_timeline():
    all_tools_data = [
        {'name': 'Process Stability', 'year': 1924, 'inventor': 'Walter Shewhart', 'reason': 'The dawn of mass manufacturing (telephones) required new methods for controlling process variation.'},
        {'name': 'Assay Robustness (DOE)', 'year': 1926, 'inventor': 'R.A. Fisher', 'reason': 'To revolutionize agricultural science by efficiently testing multiple factors (fertilizers, varieties) at once.'},
        {'name': 'Pass/Fail Analysis', 'year': 1927, 'inventor': 'Edwin B. Wilson', 'reason': 'To solve the poor performance of the standard binomial confidence interval, especially for small samples.'},
        {'name': 'Split-Plot Designs', 'year': 1930, 'inventor': 'R.A. Fisher & F. Yates', 'reason': 'To solve agricultural experiments with factors that were difficult or expensive to change on a small scale.'},
        {'name': 'Multivariate SPC', 'year': 1931, 'inventor': 'Harold Hotelling', 'reason': 'To generalize the t-test and control charts to monitor multiple correlated variables simultaneously.'},
        {'name': 'CI Concept', 'year': 1937, 'inventor': 'Jerzy Neyman', 'reason': 'A need for rigorous, objective methods in the growing field of mathematical statistics.'},
        {'name': 'Tolerance Intervals', 'year': 1942, 'inventor': 'Abraham Wald', 'reason': 'WWII demanded mass production of interchangeable military parts that would fit together reliably.'},
        {'name': 'ROC Curve Analysis', 'year': 1945, 'inventor': 'Signal Processing Labs', 'reason': 'Developed during WWII to distinguish enemy radar signals from noise, a classic signal detection problem.'},
        {'name': 'Variance Components', 'year': 1950, 'inventor': 'Charles Henderson', 'reason': 'Advances in genetics and complex systems required methods to partition sources of variation.'},
        {'name': 'Assay Robustness (RSM)', 'year': 1951, 'inventor': 'Box & Wilson', 'reason': 'The post-war chemical industry boom created demand for efficient process optimization techniques.'},
        {'name': 'Small Shift Detection', 'year': 1954, 'inventor': 'Page (CUSUM) & Roberts (EWMA)', 'reason': 'Maturing industries required charts more sensitive to small, slow process drifts than Shewhart\'s original design.'},
        {'name': 'Clustering', 'year': 1957, 'inventor': 'Stuart Lloyd', 'reason': 'The advent of early digital computing at Bell Labs made iterative, data-driven grouping algorithms feasible.'},
        {'name': 'Predictive QC', 'year': 1958, 'inventor': 'David Cox', 'reason': 'A need to model binary outcomes (pass/fail, live/die) in a regression framework.'},
        {'name': 'Reliability Analysis', 'year': 1958, 'inventor': 'Kaplan & Meier', 'reason': 'The rise of clinical trials necessitated a formal way to handle \'censored\' data (e.g., patients who survived past the study\'s end).'},
        {'name': 'Mixture Designs', 'year': 1958, 'inventor': 'Henry Scheffé', 'reason': 'To provide a systematic way for chemists and food scientists to optimize recipes and formulations.'},
        {'name': 'LOD & LOQ', 'year': 1968, 'inventor': 'Lloyd Currie (NIST)', 'reason': 'To create a harmonized, statistically rigorous framework for defining the sensitivity of analytical methods.'},
        {'name': 'Time Series Analysis', 'year': 1970, 'inventor': 'Box & Jenkins', 'reason': 'To provide a comprehensive statistical methodology for forecasting and control in industrial and economic processes.'},
        {'name': 'Non-Linear Regression', 'year': 1975, 'inventor': 'Bioassay Field', 'reason': 'The proliferation of immunoassays (like ELISA) required models for their characteristic S-shaped curves.'},
        {'name': 'Multivariate Analysis', 'year': 1975, 'inventor': 'Herman Wold', 'reason': 'To model "data-rich but theory-poor" systems in social science, later adapted for chemometrics.'},
        {'name': 'Core Validation Params', 'year': 1980, 'inventor': 'ICH / FDA', 'reason': 'Globalization of the pharmaceutical industry required harmonized standards for drug approval.'},
        {'name': 'Run Validation', 'year': 1981, 'inventor': 'James Westgard', 'reason': 'The automation of clinical labs demanded a more sensitive, diagnostic system for daily quality control.'},
        {'name': 'Gage R&R', 'year': 1982, 'inventor': 'AIAG', 'reason': 'The US auto industry, facing a quality crisis, needed to formalize the analysis of their measurement systems.'},
        {'name': 'Method Comparison', 'year': 1986, 'inventor': 'Bland & Altman', 'reason': 'A direct response to the widespread misuse of correlation for comparing clinical measurement methods.'},
        {'name': 'Process Capability', 'year': 1986, 'inventor': 'Bill Smith (Motorola)', 'reason': 'The Six Sigma quality revolution at Motorola popularized a simple metric to quantify process capability.'},
        {'name': 'Equivalence Testing (TOST)', 'year': 1987, 'inventor': 'Donald Schuirmann', 'reason': 'The rise of the generic drug industry created a regulatory need to statistically *prove* equivalence.'},
        {'name': 'Bayesian Inference', 'year': 1990, 'inventor': 'Metropolis et al.', 'reason': 'The explosion in computing power made simulation-based methods (MCMC) practical, unlocking Bayesian inference.'},
        {'name': 'Stability Analysis', 'year': 1993, 'inventor': 'ICH', 'reason': 'To harmonize global pharmaceutical regulations for determining a product\'s shelf-life.'},
        {'name': 'Advanced AI/ML', 'year': 2017, 'inventor': 'Vaswani, Lundberg et al.', 'reason': 'The Deep Learning revolution created powerful but opaque "black box" models, necessitating methods to explain them (XAI).'},
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
        x_coord = tool['year']
        y_coord = tool['y']
        
        tool_color = 'grey'
        for era_info in eras.values():
            if era_info['boundary'][0] <= x_coord <= era_info['boundary'][1]:
                tool_color = era_info['color']
                break

        fig.add_trace(go.Scatter(
            x=[x_coord], y=[y_coord], mode='markers',
            marker=dict(size=12, color=tool_color, line=dict(width=2, color='black')),
            hoverinfo='text', text=f"<b>{tool['name']} ({tool['year']})</b><br><i>Inventor(s): {tool['inventor']}</i><br><br><b>Reason:</b> {tool['reason']}"
        ))
        fig.add_shape(type="line", x0=x_coord, y0=0, x1=x_coord, y1=y_coord, line=dict(color='grey', width=1))
        fig.add_annotation(
            x=x_coord, y=y_coord, text=f"<b>{tool['name']}</b>",
            showarrow=False, yshift=25 if y_coord > 0 else -25, font=dict(size=11, color=tool_color), align="center"
        )

    fig.update_layout(
        title_text='<b>A Chronological Timeline of V&V Analytics</b>',
        title_font_size=28, title_x=0.5,
        xaxis=dict(range=[1920, 2025], showgrid=True),
        yaxis=dict(visible=False, range=[-8, 8]),
        plot_bgcolor='white', paper_bgcolor='white',
        height=700, margin=dict(l=20, r=20, t=100, b=20),
        showlegend=False
    )
    return fig

@st.cache_data
def create_toolkit_conceptual_map():
    structure = {
        'Foundational Statistics': ['Statistical Inference', 'Regression Models'],
        'Process & Quality Control': ['Measurement Systems Analysis', 'Statistical Process Control', 'Validation & Lifecycle'],
        'Advanced Analytics (ML/AI)': ['Predictive Modeling', 'Unsupervised Learning']
    }
    sub_structure = {
        'Statistical Inference': ['Confidence Interval Concept', 'Equivalence Testing (TOST)', 'Bayesian Inference', 'ROC Curve Analysis'],
        'Regression Models': ['Linearity & Range', 'Non-Linear Regression (4PL/5PL)', 'Stability Analysis (Shelf-Life)', 'Time Series Analysis'],
        'Measurement Systems Analysis': ['Gage R&R / VCA', 'Method Comparison'],
        'Statistical Process Control': ['Process Stability (SPC)', 'Small Shift Detection', 'Multivariate SPC'],
        'Validation & Lifecycle': ['Process Capability (Cpk)', 'Tolerance Intervals', 'Reliability / Survival Analysis'],
        'Predictive Modeling': ['Predictive QC (Classification)', 'Explainable AI (XAI)', 'Multivariate Analysis (MVA)'],
        'Unsupervised Learning': ['Anomaly Detection', 'Clustering (Unsupervised)']
    }
    tool_origins = {
        'Confidence Interval Concept': 'Statistics', 'Equivalence Testing (TOST)': 'Biostatistics', 'Bayesian Inference': 'Statistics', 'ROC Curve Analysis': 'Statistics',
        'Linearity & Range': 'Statistics', 'Non-Linear Regression (4PL/5PL)': 'Biostatistics', 'Stability Analysis (Shelf-Life)': 'Biostatistics', 'Time Series Analysis': 'Statistics',
        'Gage R&R / VCA': 'Industrial Quality Control', 'Method Comparison': 'Biostatistics',
        'Process Stability (SPC)': 'Industrial Quality Control', 'Small Shift Detection': 'Industrial Quality Control', 'Multivariate SPC': 'Industrial Quality Control',
        'Process Capability (Cpk)': 'Industrial Quality Control', 'Tolerance Intervals': 'Statistics', 'Reliability / Survival Analysis': 'Biostatistics',
        'Predictive QC (Classification)': 'Data Science / ML', 'Explainable AI (XAI)': 'Data Science / ML', 'Multivariate Analysis (MVA)': 'Data Science / ML',
        'Anomaly Detection': 'Data Science / ML', 'Clustering (Unsupervised)': 'Data Science / ML'
    }
    origin_colors = {
        'Statistics': '#1f77b4', 'Biostatistics': '#2ca02c',
        'Industrial Quality Control': '#ff7f0e', 'Data Science / ML': '#d62728',
        'Structure': '#6A5ACD'
    }

    nodes = {}
    
    vertical_spacing = 2.2
    all_tools_flat = [tool for sublist in sub_structure.values() for tool in sublist]
    y_coords = np.linspace(len(all_tools_flat) * vertical_spacing, -len(all_tools_flat) * vertical_spacing, len(all_tools_flat))
    x_positions = [4, 5]
    for i, tool_key in enumerate(all_tools_flat):
        nodes[tool_key] = {'x': x_positions[i % 2], 'y': y_coords[i], 'name': tool_key, 'short': tool_key.replace(' (', '<br>('), 'origin': tool_origins.get(tool_key)}

    for l2_key, l3_keys in sub_structure.items():
        child_ys = [nodes[child_key]['y'] for child_key in l3_keys]
        nodes[l2_key] = {'x': 2.5, 'y': np.mean(child_ys), 'name': l2_key, 'short': l2_key.replace(' ', '<br>'), 'origin': 'Structure'}

    for l1_key, l2_keys in structure.items():
        child_ys = [nodes[child_key]['y'] for child_key in l2_keys]
        nodes[l1_key] = {'x': 1, 'y': np.mean(child_ys), 'name': l1_key, 'short': l1_key.replace(' ', '<br>'), 'origin': 'Structure'}

    nodes['CENTER'] = {'x': -0.5, 'y': 0, 'name': 'V&V Analytics Toolkit', 'short': 'V&V Analytics<br>Toolkit', 'origin': 'Structure'}

    fig = go.Figure()

    all_edges = [('CENTER', l1) for l1 in structure.keys()] + \
                [(l1, l2) for l1, l2s in structure.items() for l2 in l2s] + \
                [(l2, l3) for l2, l3s in sub_structure.items() for l3 in l3s]
    
    for start_key, end_key in all_edges:
        fig.add_shape(type="line",
            x0=nodes[start_key]['x'], y0=nodes[start_key]['y'],
            x1=nodes[end_key]['x'], y1=nodes[end_key]['y'],
            line=dict(color="lightgrey", width=1.5)
        )
    
    data_by_origin = {name: {'x': [], 'y': [], 'short': [], 'full': [], 'size': [], 'font_size': []} for name in origin_colors.keys()}
    
    size_map = {'CENTER': 150, 'Level1': 130, 'Level2': 110, 'Tool': 90}
    font_map = {'CENTER': 16, 'Level1': 14, 'Level2': 12, 'Tool': 11}

    for key, data in nodes.items():
        if key == 'CENTER': level = 'CENTER'
        elif key in structure: level = 'Level1'
        elif key in sub_structure: level = 'Level2'
        else: level = 'Tool'
        
        data_by_origin[data['origin']]['x'].append(data['x'])
        data_by_origin[data['origin']]['y'].append(data['y'])
        data_by_origin[data['origin']]['short'].append(data['short'])
        data_by_origin[data['origin']]['full'].append(data['name'])
        data_by_origin[data['origin']]['size'].append(size_map[level])
        data_by_origin[data['origin']]['font_size'].append(font_map[level])
        
    for origin_name, data in data_by_origin.items():
        if not data['x']: continue
        fig.add_trace(go.Scatter(
            x=data['x'], y=data['y'], text=data['short'],
            mode='markers+text', textposition="middle center",
            marker=dict(
                size=data['size'],
                color=origin_colors[origin_name],
                symbol='circle',
                line=dict(width=2, color='black')
            ),
            textfont=dict(
                size=data['font_size'],
                color='white',
                family="Arial"
            ),
            hovertext=[name.replace('<br>', ' ') for name in data['short']], hoverinfo='text',
            name=origin_name
        ))

    fig.update_layout(
        title_text='<b>Conceptual Map of the V&V Analytics Toolkit</b>',
        showlegend=True,
        legend=dict(title="<b>Tool Origin</b>", x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.7)'),
        xaxis=dict(visible=False, range=[-1, 6]),
        yaxis=dict(visible=False, range=[-28, 28]),
        height=2400,
        margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#f0f2f6'
    )
    return fig
    
@st.cache_data
def plot_ci_concept(n=30):
    """
    Generates plots for the confidence interval concept module.
    """
    np.random.seed(42)
    pop_mean, pop_std = 100, 15

    # --- Plot 1: Population vs. Sampling Distribution ---
    x = np.linspace(pop_mean - 4*pop_std, pop_mean + 4*pop_std, 400)
    pop_dist = norm.pdf(x, pop_mean, pop_std)

    sampling_dist_std = pop_std / np.sqrt(n)
    sampling_dist = norm.pdf(x, pop_mean, sampling_dist_std)

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=x, y=pop_dist, fill='tozeroy', name='Population Distribution', line=dict(color='skyblue')))
    fig1.add_trace(go.Scatter(x=x, y=sampling_dist, fill='tozeroy', name=f'Sampling Distribution (n={n})', line=dict(color='orange')))
    fig1.add_vline(x=pop_mean, line=dict(color='black', dash='dash'), annotation_text="True Mean (μ)")
    fig1.update_layout(title=f"<b>Population vs. Sampling Distribution of the Mean (n={n})</b>", showlegend=True, legend=dict(x=0.01, y=0.99))

    # --- Plot 2: CI Simulation ---
    n_sims = 1000
    samples = np.random.normal(pop_mean, pop_std, size=(n_sims, n))
    sample_means = samples.mean(axis=1)
    sample_stds = samples.std(axis=1, ddof=1)

    # Using t-distribution for CIs as is proper
    t_crit = t.ppf(0.975, df=n-1)
    margin_of_error = t_crit * sample_stds / np.sqrt(n)

    ci_lowers = sample_means - margin_of_error
    ci_uppers = sample_means + margin_of_error

    capture_mask = (ci_lowers <= pop_mean) & (ci_uppers >= pop_mean)
    capture_count = np.sum(capture_mask)
    avg_width = np.mean(ci_uppers - ci_lowers)

    fig2 = go.Figure()
    # Plot first 100 CIs for visualization
    for i in range(min(n_sims, 100)):
        color = 'blue' if capture_mask[i] else 'red'
        fig2.add_trace(go.Scatter(x=[ci_lowers[i], ci_uppers[i]], y=[i, i], mode='lines', line=dict(color=color, width=2), showlegend=False))
        fig2.add_trace(go.Scatter(x=[sample_means[i]], y=[i], mode='markers', marker=dict(color=color, size=4), showlegend=False))

    fig2.add_vline(x=pop_mean, line=dict(color='black', dash='dash'), annotation_text="True Mean (μ)")
    fig2.update_layout(title=f"<b>{min(n_sims, 100)} Simulated 95% Confidence Intervals</b>", yaxis_visible=False)

    return fig1, fig2, capture_count, n_sims, avg_width


def plot_core_validation_params(bias_pct=1.5, repeat_cv=1.5, intermed_cv=2.5, interference_effect=8.0):
    """
    Generates dynamic plots for the core validation module based on user inputs.
    """
    # --- 1. Accuracy (Bias) Data ---
    np.random.seed(42)
    true_values = np.array([50, 100, 150])
    # The mean of the measured data is now controlled by the bias slider
    measured_data = {
        val: np.random.normal(val * (1 + bias_pct / 100), val * 0.025, 10) for val in true_values
    }
    df_accuracy = pd.DataFrame(measured_data)
    df_accuracy = df_accuracy.melt(var_name='True Value', value_name='Measured Value')

    fig1 = px.box(df_accuracy, x='True Value', y='Measured Value',
                  title='<b>1. Accuracy & Bias Evaluation</b>',
                  points='all', color_discrete_sequence=['#1f77b4'])
    for val in true_values:
        fig1.add_hline(y=val, line_dash="dash", line_color="black", annotation_text=f"True Value={val}", annotation_position="bottom right")
    fig1.update_layout(xaxis_title="True (Nominal) Concentration", yaxis_title="Measured Concentration")

    # --- 2. Precision Data ---
    np.random.seed(123)
    # The standard deviation is now controlled by the precision sliders (%CV)
    repeatability_std = 100 * (repeat_cv / 100)
    intermed_std = 100 * (intermed_cv / 100)

    repeatability = np.random.normal(100, repeatability_std, 30)
    inter_precision = np.random.normal(100, intermed_std, 30)

    df_precision = pd.concat([
        pd.DataFrame({'value': repeatability, 'condition': 'Repeatability'}),
        pd.DataFrame({'value': inter_precision, 'condition': 'Intermediate Precision'})
    ])
    fig2 = px.violin(df_precision, x='condition', y='value', box=True, points="all",
                     title='<b>2. Precision: Repeatability vs. Intermediate Precision</b>',
                     labels={'value': 'Measured Value', 'condition': 'Experimental Condition'})

    # --- 3. Specificity Data ---
    np.random.seed(2023)
    analyte = np.random.normal(1.0, 0.05, 15)
    matrix = np.random.normal(0.02, 0.01, 15)
    # The signal of the combined sample is now controlled by the interference slider
    analyte_interference = analyte * (1 + interference_effect / 100)

    df_specificity = pd.DataFrame({
        'Analyte Only': analyte,
        'Matrix Blank': matrix,
        'Analyte + Interferent': analyte_interference
    }).melt(var_name='Sample Type', value_name='Signal Response')

    fig3 = px.box(df_specificity, x='Sample Type', y='Signal Response', points='all',
                  title='<b>3. Specificity & Interference Study</b>')
    fig3.update_layout(xaxis_title="Sample Composition", yaxis_title="Assay Signal (e.g., Absorbance)")

    return fig1, fig2, fig3

def plot_gage_rr(part_sd=5.0, repeatability_sd=1.5, operator_sd=0.75):
    """
    Generates dynamic plots for the Gage R&R module based on user inputs.
    """
    np.random.seed(10)
    n_operators, n_samples, n_replicates = 3, 10, 3
    operators = ['Alice', 'Bob', 'Charlie']

    # Generate true part values based on the part_sd slider
    true_part_values = np.random.normal(100, part_sd, n_samples)

    # Generate operator biases based on the operator_sd slider
    operator_biases = np.random.normal(0, operator_sd, n_operators)
    operator_bias_map = {op: bias for op, bias in zip(operators, operator_biases)}

    data = []
    for op_idx, operator in enumerate(operators):
        for sample_idx, true_value in enumerate(true_part_values):
            # Generate measurements using the operator bias and repeatability_sd slider
            measurements = np.random.normal(true_value + operator_bias_map[operator], repeatability_sd, n_replicates)
            for m_idx, m in enumerate(measurements):
                data.append([operator, f'Part_{sample_idx+1}', m, m_idx + 1])

    df = pd.DataFrame(data, columns=['Operator', 'Part', 'Measurement', 'Replicate'])

    # Perform ANOVA and calculate variance components
    model = ols('Measurement ~ C(Part) + C(Operator) + C(Part):C(Operator)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    ms_operator = anova_table.loc['C(Operator)', 'sum_sq'] / anova_table.loc['C(Operator)', 'df']
    ms_part = anova_table.loc['C(Part)', 'sum_sq'] / anova_table.loc['C(Part)', 'df']
    ms_interaction = anova_table.loc['C(Part):C(Operator)', 'sum_sq'] / anova_table.loc['C(Part):C(Operator)', 'df']
    ms_error = anova_table.loc['Residual', 'sum_sq'] / anova_table.loc['Residual', 'df']

    var_repeatability = ms_error
    var_operator = max(0, (ms_operator - ms_interaction) / (n_samples * n_replicates))
    var_interaction = max(0, (ms_interaction - ms_error) / n_replicates)
    var_reproducibility = var_operator + var_interaction
    var_part = max(0, (ms_part - ms_interaction) / (n_operators * n_replicates))
    var_rr = var_repeatability + var_reproducibility
    var_total = var_rr + var_part

    pct_rr = (var_rr / var_total) * 100 if var_total > 0 else 0
    pct_part = (var_part / var_total) * 100 if var_total > 0 else 0

    # Plotting (largely the same, just consumes the dynamic data)
    fig = make_subplots(rows=2, cols=2, column_widths=[0.7, 0.3], row_heights=[0.5, 0.5], specs=[[{"rowspan": 2}, {}], [None, {}]], subplot_titles=("<b>Variation by Part & Operator</b>", "<b>Overall Variation by Operator</b>", "<b>Variance Contribution</b>"))
    fig_box = px.box(df, x='Part', y='Measurement', color='Operator', color_discrete_sequence=px.colors.qualitative.Plotly)
    for trace in fig_box.data: fig.add_trace(trace, row=1, col=1)
    for i, operator in enumerate(operators):
        operator_df = df[df['Operator'] == operator]; part_means = operator_df.groupby('Part')['Measurement'].mean()
        fig.add_trace(go.Scatter(x=part_means.index, y=part_means.values, mode='lines', line=dict(width=2), name=f'{operator} Mean', showlegend=False, marker_color=fig_box.data[i].marker.color), row=1, col=1)
    fig_op_box = px.box(df, x='Operator', y='Measurement', color='Operator', color_discrete_sequence=px.colors.qualitative.Plotly)
    for trace in fig_op_box.data: fig.add_trace(trace, row=1, col=2)
    fig.add_trace(go.Bar(x=['% Gage R&R', '% Part Variation'], y=[pct_rr, pct_part], marker_color=['salmon', 'skyblue'], text=[f'{pct_rr:.1f}%', f'{pct_part:.1f}%'], textposition='auto'), row=2, col=2)
    fig.add_hline(y=10, line_dash="dash", line_color="darkgreen", annotation_text="Acceptable < 10%", annotation_position="bottom right", row=2, col=2)
    fig.add_hline(y=30, line_dash="dash", line_color="darkorange", annotation_text="Unacceptable > 30%", annotation_position="top right", row=2, col=2)
    fig.update_layout(title_text='<b>Gage R&R Study: ANOVA Method</b>', title_x=0.5, height=800, boxmode='group', showlegend=True); fig.update_xaxes(tickangle=45, row=1, col=1)

    return fig, pct_rr, pct_part

def plot_lod_loq(slope=0.02, baseline_sd=0.01):
    """
    Generates dynamic plots for the LOD & LOQ module based on user inputs.
    """
    np.random.seed(3)

    # --- Signal Distribution Plot ---
    # The noise level is now controlled by the baseline_sd slider
    blanks_dist = np.random.normal(0.05, baseline_sd, 20)
    low_conc_dist = np.random.normal(0.05 + 5 * slope, baseline_sd * 1.5, 20) # Low conc at 5 units

    df_dist = pd.concat([
        pd.DataFrame({'Signal': blanks_dist, 'Sample Type': 'Blank'}),
        pd.DataFrame({'Signal': low_conc_dist, 'Sample Type': 'Low Concentration'})
    ])

    # --- Low-Level Calibration Curve ---
    concentrations = np.array([0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 5, 5, 5, 10, 10, 10])
    # The signal response and noise are now controlled by the sliders
    signals = 0.05 + slope * concentrations + np.random.normal(0, baseline_sd, len(concentrations))

    df_cal = pd.DataFrame({'Concentration': concentrations, 'Signal': signals})

    # Fit the model to the dynamically generated data
    X = sm.add_constant(df_cal['Concentration'])
    model = sm.OLS(df_cal['Signal'], X).fit()

    # Use the model's parameters to calculate LOD & LOQ
    # Note: We use the input baseline_sd for the calculation as it's the "true" noise,
    # which is more stable than the model's estimate from this small dataset.
    # The slope from the model is a good estimate of sensitivity.
    fit_slope = model.params['Concentration'] if model.params['Concentration'] > 0.001 else 0.001

    LOD = (3.3 * baseline_sd) / fit_slope
    LOQ = (10 * baseline_sd) / fit_slope

    # --- Plotting ---
    fig = make_subplots(rows=2, cols=1, subplot_titles=("<b>Signal Distribution at Low End</b>", "<b>Low-Level Calibration Curve</b>"), vertical_spacing=0.2)

    fig_violin = px.violin(df_dist, x='Sample Type', y='Signal', color='Sample Type', box=True, points="all", color_discrete_map={'Blank': 'skyblue', 'Low Concentration': 'lightgreen'})
    for trace in fig_violin.data: fig.add_trace(trace, row=1, col=1)

    fig.add_trace(go.Scatter(x=df_cal['Concentration'], y=df_cal['Signal'], mode='markers', name='Calibration Points', marker=dict(color='darkblue', size=8)), row=2, col=1)
    x_range = np.linspace(0, df_cal['Concentration'].max(), 100)
    y_range = model.predict(sm.add_constant(x_range))
    fig.add_trace(go.Scatter(x=x_range, y=y_range, mode='lines', name='Regression Line', line=dict(color='red', dash='dash')), row=2, col=1)

    fig.add_vline(x=LOD, line_dash="dot", line_color="orange", row=2, col=1, annotation_text=f"<b>LOD = {LOD:.2f}</b>", annotation_position="top")
    fig.add_vline(x=LOQ, line_dash="dash", line_color="red", row=2, col=1, annotation_text=f"<b>LOQ = {LOQ:.2f}</b>", annotation_position="top")

    fig.update_layout(title_text='<b>Assay Sensitivity Analysis: LOD & LOQ</b>', title_x=0.5, height=800, showlegend=False)
    fig.update_yaxes(title_text="Assay Signal (e.g., Absorbance)", row=1, col=1)
    fig.update_xaxes(title_text="Sample Type", row=1, col=1)
    fig.update_yaxes(title_text="Assay Signal (e.g., Absorbance)", row=2, col=1)
    fig.update_xaxes(title_text="Concentration (ng/mL)", row=2, col=1)

    return fig, LOD, LOQ

def plot_linearity(curvature=-1.0, random_error=1.0, proportional_error=2.0):
    """
    Generates dynamic plots for the Linearity module based on user inputs.
    """
    np.random.seed(42)
    nominal = np.array([10, 25, 50, 100, 150, 200, 250])

    # Simulate data based on sliders
    # Curvature term: a negative value creates an S-curve typical of saturation
    curvature_effect = curvature * (nominal / 150)**3

    # Error term: combination of constant random error and error that grows with concentration
    error = np.random.normal(0, random_error + nominal * (proportional_error / 100))

    measured = nominal + curvature_effect + error

    # Fit OLS model to the dynamic data
    X = sm.add_constant(nominal)
    model = sm.OLS(measured, X).fit()
    residuals = model.resid
    recovery = (measured / nominal) * 100

    # --- Plotting ---
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{}, {}], [{"colspan": 2}, None]],
        subplot_titles=("<b>Linearity Plot</b>", "<b>Residual Plot</b>", "<b>Recovery Plot</b>"),
        vertical_spacing=0.2
    )

    # Linearity Plot
    fig.add_trace(go.Scatter(x=nominal, y=measured, mode='markers', name='Measured Values', marker=dict(size=10, color='blue'), hovertemplate="Nominal: %{x}<br>Measured: %{y:.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=nominal, y=model.predict(X), mode='lines', name='Best Fit Line', line=dict(color='red')), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0, 260], y=[0, 260], mode='lines', name='Line of Identity', line=dict(dash='dash', color='black')), row=1, col=1)

    # Residual Plot
    fig.add_trace(go.Scatter(x=nominal, y=residuals, mode='markers', name='Residuals', marker=dict(size=10, color='green'), hovertemplate="Nominal: %{x}<br>Residual: %{y:.2f}<extra></extra>"), row=1, col=2)
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=2)

    # Recovery Plot
    fig.add_trace(go.Scatter(x=nominal, y=recovery, mode='lines+markers', name='Recovery', line=dict(color='purple'), marker=dict(size=10), hovertemplate="Nominal: %{x}<br>Recovery: %{y:.1f}%<extra></extra>"), row=2, col=1)
    fig.add_hrect(y0=80, y1=120, fillcolor="green", opacity=0.1, layer="below", line_width=0, row=2, col=1)
    fig.add_hline(y=100, line_dash="dash", line_color="black", row=2, col=1)
    fig.add_hline(y=80, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=120, line_dash="dot", line_color="red", row=2, col=1)

    fig.update_layout(title_text='<b>Assay Linearity and Range Verification Dashboard</b>', title_x=0.5, height=800, showlegend=False)
    fig.update_xaxes(title_text="Nominal Concentration", row=1, col=1); fig.update_yaxes(title_text="Measured Concentration", row=1, col=1)
    fig.update_xaxes(title_text="Nominal Concentration", row=1, col=2); fig.update_yaxes(title_text="Residual (Error)", row=1, col=2)
    fig.update_xaxes(title_text="Nominal Concentration", row=2, col=1); fig.update_yaxes(title_text="% Recovery", range=[min(75, recovery.min()-5), max(125, recovery.max()+5)], row=2, col=1)

    return fig, model

def plot_4pl_regression(a_true=1.5, b_true=1.2, c_true=10.0, d_true=0.05, noise_sd=0.05):
    """
    Generates dynamic plots for the 4PL regression module based on user inputs.
    """
    # 4PL logistic function
    def four_pl(x, a, b, c, d):
        return d + (a - d) / (1 + (x / c)**b)

    # Generate data based on the "true" parameters from the sliders
    np.random.seed(42)
    conc = np.logspace(-2, 3, 15)
    signal_true = four_pl(conc, a_true, b_true, c_true, d_true)
    signal_measured = signal_true + np.random.normal(0, noise_sd, len(conc))

    # Fit the 4PL curve to the noisy data
    try:
        # Use the true parameters as a good starting guess (p0) for the fit
        params, _ = curve_fit(four_pl, conc, signal_measured, p0=[a_true, b_true, c_true, d_true], maxfev=10000)
    except RuntimeError:
        # Fallback if fit fails, which is rare with a good p0
        params = [a_true, b_true, c_true, d_true]

    a_fit, b_fit, c_fit, d_fit = params

    # Create plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=conc, y=signal_measured, mode='markers', name='Measured Data', marker=dict(size=10)))
    x_fit = np.logspace(-2, 3, 100)
    y_fit = four_pl(x_fit, *params)
    fig.add_trace(go.Scatter(x=x_fit, y=y_fit, mode='lines', name='4PL Fit', line=dict(color='red', dash='dash')))

    # Add annotations for key fitted parameters
    fig.add_hline(y=d_fit, line_dash='dot', annotation_text=f"Lower Asymptote (d) = {d_fit:.2f}")
    fig.add_hline(y=a_fit, line_dash='dot', annotation_text=f"Upper Asymptote (a) = {a_fit:.2f}")
    fig.add_vline(x=c_fit, line_dash='dot', annotation_text=f"EC50 (c) = {c_fit:.2f}")

    fig.update_layout(title_text='<b>Non-Linear Regression: 4-Parameter Logistic (4PL) Fit</b>',
                      xaxis_type="log", xaxis_title="Concentration (log scale)",
                      yaxis_title="Signal Response", legend=dict(x=0.01, y=0.99))
    return fig, params

def plot_roc_curve(diseased_mean=65, population_sd=10):
    """
    Generates dynamic plots for the ROC curve module based on user inputs.
    """
    np.random.seed(0)

    # Healthy population is fixed, diseased population is controlled by sliders
    healthy_mean = 45
    scores_diseased = np.random.normal(loc=diseased_mean, scale=population_sd, size=100)
    scores_healthy = np.random.normal(loc=healthy_mean, scale=population_sd, size=100)

    y_true = np.concatenate([np.ones(100), np.zeros(100)]) # 1 for diseased, 0 for healthy
    y_scores = np.concatenate([scores_diseased, scores_healthy])

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc_value = auc(fpr, tpr)

    # Create plots
    fig = make_subplots(rows=1, cols=2, subplot_titles=("<b>Score Distributions</b>", f"<b>ROC Curve (AUC = {auc_value:.3f})</b>"))

    # Distribution plot
    fig.add_trace(go.Histogram(x=scores_healthy, name='Healthy', histnorm='probability density', marker_color='blue', opacity=0.7), row=1, col=1)
    fig.add_trace(go.Histogram(x=scores_diseased, name='Diseased', histnorm='probability density', marker_color='red', opacity=0.7), row=1, col=1)

    # ROC curve plot
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'AUC = {auc_value:.3f}', line=dict(color='darkorange', width=3)), row=1, col=2)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='No-Discrimination Line', line=dict(color='navy', width=2, dash='dash')), row=1, col=2)

    fig.update_layout(barmode='overlay', height=500, title_text="<b>Diagnostic Assay Performance: ROC Curve Analysis</b>")
    fig.update_xaxes(title_text="Assay Score", row=1, col=1)
    fig.update_yaxes(title_text="Density", row=1, col=1)
    fig.update_xaxes(title_text="False Positive Rate (1 - Specificity)", range=[-0.05, 1.05], row=1, col=2)
    fig.update_yaxes(title_text="True Positive Rate (Sensitivity)", range=[-0.05, 1.05], row=1, col=2)

    return fig, auc_value

def plot_tost(delta=5.0, true_diff=1.0, std_dev=5.0, n_samples=50):
    """
    Generates dynamic plots for the TOST module based on user inputs.
    """
    np.random.seed(1)
    # Generate two samples based on the slider inputs
    data_A = np.random.normal(loc=100, scale=std_dev, size=n_samples)
    data_B = np.random.normal(loc=100 + true_diff, scale=std_dev, size=n_samples)

    # Perform two one-sided t-tests using Welch's t-test
    diff_mean = np.mean(data_B) - np.mean(data_A)
    std_err_diff = np.sqrt(np.var(data_A, ddof=1)/n_samples + np.var(data_B, ddof=1)/n_samples)

    # Check for division by zero if n_samples is too small
    if n_samples <= 1:
        return go.Figure(), 1.0, False, 0, 0

    df_welch = (std_err_diff**4) / ( ((np.var(data_A, ddof=1)/n_samples)**2 / (n_samples-1)) + ((np.var(data_B, ddof=1)/n_samples)**2 / (n_samples-1)) )

    t_lower = (diff_mean - (-delta)) / std_err_diff
    t_upper = (diff_mean - delta) / std_err_diff

    p_lower = stats.t.sf(t_lower, df_welch)
    p_upper = stats.t.cdf(t_upper, df_welch)

    p_tost = max(p_lower, p_upper)
    is_equivalent = p_tost < 0.05

    # Create plot
    fig = go.Figure()
    # 90% confidence interval for TOST (standard practice)
    ci_margin = t.ppf(0.95, df_welch) * std_err_diff
    ci_lower = diff_mean - ci_margin
    ci_upper = diff_mean + ci_margin

    fig.add_trace(go.Scatter(
        x=[diff_mean], y=['Difference'], error_x=dict(type='data', array=[ci_upper-diff_mean], arrayminus=[diff_mean-ci_lower]),
        mode='markers', name='90% CI for Difference', marker=dict(color='blue', size=15)
    ))
    # Equivalence bounds
    fig.add_shape(type="line", x0=-delta, y0=-0.5, x1=-delta, y1=0.5, line=dict(color="red", width=2, dash="dash"))
    fig.add_shape(type="line", x0=delta, y0=-0.5, x1=delta, y1=0.5, line=dict(color="red", width=2, dash="dash"))
    fig.add_vrect(x0=-delta, x1=delta, fillcolor="rgba(0,255,0,0.1)", layer="below", line_width=0)
    fig.add_annotation(x=0, y=0.8, text=f"Equivalence Zone (-{delta} to +{delta})", showarrow=False, font_size=14)

    result_text = "EQUIVALENT" if is_equivalent else "NOT EQUIVALENT"
    result_color = "darkgreen" if is_equivalent else "darkred"
    fig.add_annotation(x=diff_mean, y=-0.8, text=f"<b>Result: {result_text}</b><br>(TOST p-value = {p_tost:.3f})", showarrow=False, font=dict(size=16, color=result_color))

    fig.update_layout(title='<b>Equivalence Testing (TOST)</b>', xaxis_title='Difference in Means (Method B - Method A)', yaxis_showticklabels=False, height=500)

    return fig, p_tost, is_equivalent, ci_lower, ci_upper

@st.cache_data
def plot_doe_robustness(ph_effect=2.0, temp_effect=5.0, interaction_effect=0.0, ph_quad_effect=-5.0, temp_quad_effect=-5.0, noise_sd=1.0):
    """
    Generates dynamic RSM plots for the DOE module based on a Central Composite Design.
    """
    np.random.seed(42)

    # 1. Design the experiment (Central Composite Design)
    alpha = 1.414
    design = {
        'pH':  [-1, 1, -1, 1, -alpha, alpha, 0, 0, 0, 0, 0, 0, 0],
        'Temp':[-1, -1, 1, 1, 0, 0, -alpha, alpha, 0, 0, 0, 0, 0]
    }
    df = pd.DataFrame(design)

    # 2. Simulate the response using a full quadratic model
    true_response = 100 + \
                    ph_effect * df['pH'] + \
                    temp_effect * df['Temp'] + \
                    interaction_effect * df['pH'] * df['Temp'] + \
                    ph_quad_effect * (df['pH']**2) + \
                    temp_quad_effect * (df['Temp']**2)

    df['Response'] = true_response + np.random.normal(0, noise_sd, len(df))

    # 3. Analyze the results with a quadratic OLS model
    model = ols('Response ~ pH + Temp + I(pH**2) + I(Temp**2) + pH:Temp', data=df).fit()

    # 4. Create the prediction grid for the surfaces
    x_range = np.linspace(-1.5, 1.5, 50)
    y_range = np.linspace(-1.5, 1.5, 50)
    xx, yy = np.meshgrid(x_range, y_range)
    grid = pd.DataFrame({'pH': xx.ravel(), 'Temp': yy.ravel()})
    pred = model.predict(grid).values.reshape(xx.shape)

    # 5. Create the 2D Contour Plot
    fig_contour = go.Figure(data=[
        go.Contour(z=pred, x=x_range, y=y_range, colorscale='Viridis', contours=dict(coloring='lines', showlabels=True)),
        go.Scatter(x=df['pH'], y=df['Temp'], mode='markers', marker=dict(color='red', size=12, line=dict(width=2, color='black')), name='Design Points')
    ])
    fig_contour.update_layout(title='<b>2D Response Surface (Contour Plot)</b>', xaxis_title="Factor: pH", yaxis_title="Factor: Temperature", showlegend=False)

    # 6. Create the 3D Surface Plot
    fig_3d = go.Figure(data=[
        go.Surface(z=pred, x=x_range, y=y_range, colorscale='Viridis', opacity=0.8),
        go.Scatter3d(x=df['pH'], y=df['Temp'], z=df['Response'], mode='markers',
                      marker=dict(color='red', size=5, line=dict(width=2, color='black')), name='Design Points')
    ])
    fig_3d.update_layout(title='<b>3D Response Surface Plot</b>', scene=dict(xaxis_title='pH', yaxis_title='Temp', zaxis_title='Response'), showlegend=False)

    # 7. Create the effects plots
    fig_effects = make_subplots(rows=1, cols=3, subplot_titles=("<b>Main Effect: pH</b>", "<b>Main Effect: Temp</b>", "<b>Interaction: pH*Temp</b>"))
    me_ph = df.groupby('pH')['Response'].mean()
    fig_effects.add_trace(go.Scatter(x=me_ph.index, y=me_ph.values, mode='lines+markers'), row=1, col=1)
    me_temp = df.groupby('Temp')['Response'].mean()
    fig_effects.add_trace(go.Scatter(x=me_temp.index, y=me_temp.values, mode='lines+markers'), row=1, col=2)
    interaction_data = df[df['pH'].isin([-1, 1]) & df['Temp'].isin([-1, 1])].groupby(['pH', 'Temp'])['Response'].mean().reset_index()
    for temp_level in [-1, 1]:
        subset = interaction_data[interaction_data['Temp'] == temp_level]
        fig_effects.add_trace(go.Scatter(x=subset['pH'], y=subset['Response'], mode='lines+markers', name=f'Temp = {temp_level}'), row=1, col=3)
    fig_effects.update_layout(showlegend=False)

    return fig_contour, fig_3d, fig_effects, model.params

def plot_causal_inference(confounding_strength=5.0):
    """
    Generates dynamic plots for the Causal Inference module based on user inputs.
    """
    # 1. The DAG (same as before, but title is updated)
    fig_dag = go.Figure()
    nodes = {'Reagent Lot': (0, 1), 'Temp': (1.5, 2), 'Pressure': (1.5, 0), 'Purity': (3, 2)}
    fig_dag.add_trace(go.Scatter(x=[v[0] for v in nodes.values()], y=[v[1] for v in nodes.values()],
                               mode="markers+text", text=list(nodes.keys()), textposition="top center",
                               marker=dict(size=40, color='lightblue', line=dict(width=2, color='black')), textfont_size=14))
    edges = [('Reagent Lot', 'Purity'), ('Reagent Lot', 'Temp'), ('Temp', 'Purity'), ('Temp', 'Pressure')]
    for start, end in edges:
        fig_dag.add_annotation(x=nodes[end][0], y=nodes[end][1], ax=nodes[start][0], ay=nodes[start][1],
                               xref='x', yref='y', axref='x', ayref='y', showarrow=True, arrowhead=2, arrowwidth=2, arrowcolor='black')
    fig_dag.update_layout(title="<b>1. The Causal Map (DAG)</b>", showlegend=False, xaxis_visible=False, yaxis_visible=False, height=400, margin=dict(t=50))

    # 2. Simulate data based on the DAG and confounding strength
    np.random.seed(42)
    n_samples = 100
    # The confounder: Reagent Lot (0=Standard, 1=New)
    reagent_lot = np.random.randint(0, 2, n_samples)
    # The true causal effect of temperature on purity is fixed at -0.5
    true_causal_effect = -0.5

    # Generate data where Reagent Lot affects BOTH Temp and Purity
    temp = 70 + confounding_strength * reagent_lot + np.random.normal(0, 2, n_samples)
    purity = 95 + true_causal_effect * (temp - 70) + confounding_strength * reagent_lot + np.random.normal(0, 2, n_samples)

    df = pd.DataFrame({'Temp': temp, 'Purity': purity, 'ReagentLot': reagent_lot.astype(str)})

    # 3. Calculate effects
    # Naive (biased) model: Purity ~ Temp
    naive_model = ols('Purity ~ Temp', data=df).fit()
    naive_effect = naive_model.params['Temp']

    # Adjusted (unbiased) model: Purity ~ Temp + ReagentLot
    adjusted_model = ols('Purity ~ Temp + C(ReagentLot)', data=df).fit()
    adjusted_effect = adjusted_model.params['Temp']

    # 4. Create the scatter plot
    fig_scatter = px.scatter(df, x='Temp', y='Purity', color='ReagentLot',
                             title="<b>2. Confounding in Action</b>",
                             color_discrete_map={'0': 'blue', '1': 'red'},
                             labels={'ReagentLot': 'Reagent Lot'})

    # Add regression lines
    x_range = np.linspace(df['Temp'].min(), df['Temp'].max(), 2)
    fig_scatter.add_trace(go.Scatter(x=x_range, y=naive_model.predict({'Temp': x_range}), mode='lines',
                                     name='Naive (Biased) Correlation', line=dict(color='orange', width=4, dash='dash')))
    fig_scatter.add_trace(go.Scatter(x=x_range, y=adjusted_model.params['Intercept'] + adjusted_effect * x_range, mode='lines',
                                     name='True Causal Effect (Adjusted)', line=dict(color='darkgreen', width=4)))

    fig_scatter.update_layout(height=500, legend=dict(x=0.01, y=0.99))

    return fig_dag, fig_scatter, naive_effect, adjusted_effect

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
    Generates plots for the process capability module based on a scenario.
    """
    np.random.seed(42)
    n = 100
    LSL, USL, Target = 90, 110, 100

    # Generate data based on scenario
    if scenario == 'Ideal':
        mean, std = 100, 1.5
        data = np.random.normal(mean, std, n)
    elif scenario == 'Shifted':
        mean, std = 104, 1.5
        data = np.random.normal(mean, std, n)
    elif scenario == 'Variable':
        mean, std = 100, 3.5
        data = np.random.normal(mean, std, n)
    elif scenario == 'Out of Control':
        mean, std = 100, 1.5
        data = np.random.normal(mean, std, n)
        data[70:] += 6 # Add a shift to make it out of control

    # --- Control Chart Calculations ---
    mr = np.abs(np.diff(data))
    # Use only stable part for limits if out of control
    limit_data = data[:70] if scenario == 'Out of Control' else data
    center_line = np.mean(limit_data)
    mr_mean = np.mean(np.abs(np.diff(limit_data)))
    sigma_est = mr_mean / 1.128 # d2 for n=2
    UCL_I, LCL_I = center_line + 3 * sigma_est, center_line - 3 * sigma_est

    # --- Capability Calculation ---
    if scenario == 'Out of Control':
        cpk_val = 0 # Invalid
    else:
        cpk_upper = (USL - mean) / (3 * std)
        cpk_lower = (mean - LSL) / (3 * std)
        cpk_val = min(cpk_upper, cpk_lower)

    # --- Plotting ---
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.15,
        subplot_titles=("<b>Control Chart (Is the process stable?)</b>", "<b>Capability Histogram (Does it meet specs?)</b>")
    )
    # Control Chart
    fig.add_trace(go.Scatter(x=np.arange(n), y=data, mode='lines+markers', name='Process Data'), row=1, col=1)
    fig.add_hline(y=center_line, line_dash="dash", line_color="black", row=1, col=1)
    fig.add_hline(y=UCL_I, line_color="red", row=1, col=1)
    fig.add_hline(y=LCL_I, line_color="red", row=1, col=1)

    # Histogram
    fig.add_trace(go.Histogram(x=data, name='Distribution', nbinsx=20, histnorm='probability density'), row=2, col=1)
    # Add normal curve overlay
    x_curve = np.linspace(min(data.min(), LSL-2), max(data.max(), USL+2), 200)
    y_curve = norm.pdf(x_curve, mean, std)
    fig.add_trace(go.Scatter(x=x_curve, y=y_curve, mode='lines', name='Process Voice', line=dict(color='blue')), row=2, col=1)

    # Add Spec Limits
    fig.add_vline(x=LSL, line_dash="dot", line_color="darkred", annotation_text="LSL", row=2, col=1)
    fig.add_vline(x=USL, line_dash="dot", line_color="darkred", annotation_text="USL", row=2, col=1)

    fig.update_layout(height=700, showlegend=False)
    return fig, cpk_val

def plot_tolerance_intervals(n=30, coverage_pct=99.0):
    """
    Generates dynamic plots for the Tolerance Interval module based on user inputs.
    """
    np.random.seed(42)

    # Simulate data based on the sample size slider
    data = np.random.normal(100, 5, n)
    mean, std = np.mean(data), np.std(data, ddof=1)

    # 95% CI for the mean (n-dependent)
    sem = std / np.sqrt(n) if n > 0 else 0
    ci_margin = t.ppf(0.975, df=n-1) * sem if n > 1 else 0
    ci = (mean - ci_margin, mean + ci_margin)

    # 95%/99% Tolerance Interval (n and coverage dependent)
    # A simplified lookup table for k-factors (for 95% confidence)
    k_factor_lookup = {
        90.0: {10: 2.208, 30: 1.979, 100: 1.861, 200: 1.821},
        95.0: {10: 2.842, 30: 2.457, 100: 2.278, 200: 2.228},
        99.0: {10: 4.046, 30: 3.003, 100: 2.807, 200: 2.748},
        99.9: {10: 5.733, 30: 3.823, 100: 3.457, 200: 3.376}
    }
    # Simple interpolation/selection logic
    available_n = sorted(k_factor_lookup[coverage_pct].keys())
    selected_n = min(available_n, key=lambda x:abs(x-n))
    k_factor = k_factor_lookup[coverage_pct][selected_n]

    ti_margin = k_factor * std
    ti = (mean - ti_margin, mean + ti_margin)

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=data, name='Sample Data', histnorm='probability density'))
    # Plot CI
    fig.add_vrect(x0=ci[0], x1=ci[1], fillcolor="rgba(255,165,0,0.3)", layer="below", line_width=0,
                  annotation_text=f"<b>95% CI for Mean</b>", annotation_position="top left")
    # Plot TI
    fig.add_vrect(x0=ti[0], x1=ti[1], fillcolor="rgba(0,128,0,0.3)", layer="below", line_width=0,
                  annotation_text=f"<b>95%/{coverage_pct:.1f}% TI</b>", annotation_position="bottom left")

    fig.update_layout(title=f"<b>Confidence Interval vs. Tolerance Interval (n={n})</b>",
                      xaxis_title="Measured Value", yaxis_title="Density", showlegend=False)

    return fig, ci, ti

def plot_method_comparison(constant_bias=2.0, proportional_bias=3.0, random_error_sd=3.0):
    """
    Generates dynamic plots for the method comparison module based on user inputs.
    """
    np.random.seed(1)
    n_samples = 50
    true_values = np.linspace(20, 200, n_samples)

    # Simulate data based on the slider inputs
    error_ref = np.random.normal(0, random_error_sd, n_samples)
    error_test = np.random.normal(0, random_error_sd, n_samples)

    ref_method = true_values + error_ref
    # The 'Test' method's results are a function of the true value and the biases
    test_method = constant_bias + true_values * (1 + proportional_bias / 100) + error_test

    df = pd.DataFrame({'Reference': ref_method, 'Test': test_method})

    # Deming Regression (simplified calculation for plotting)
    mean_x, mean_y = df['Reference'].mean(), df['Test'].mean()
    cov_xy = np.cov(df['Reference'], df['Test'])[0, 1]
    var_x, var_y = df['Reference'].var(ddof=1), df['Test'].var(ddof=1)
    lambda_val = var_y / var_x if var_x > 0 else 1.0 # Ratio of variances
    deming_slope = ( (var_y - lambda_val*var_x) + np.sqrt((var_y - lambda_val*var_x)**2 + 4 * lambda_val * cov_xy**2) ) / (2 * cov_xy)
    deming_intercept = mean_y - deming_slope * mean_x

    # Bland-Altman
    df['Average'] = (df['Reference'] + df['Test']) / 2
    df['Difference'] = df['Test'] - df['Reference']
    mean_diff = df['Difference'].mean()
    std_diff = df['Difference'].std(ddof=1)
    upper_loa = mean_diff + 1.96 * std_diff
    lower_loa = mean_diff - 1.96 * std_diff

    # % Bias
    df['%Bias'] = (df['Difference'] / df['Reference']) * 100

    # --- Plotting ---
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("<b>1. Deming Regression</b>", "<b>2. Bland-Altman Plot</b>", "<b>3. Percent Bias Plot</b>"),
        vertical_spacing=0.15
    )

    # Deming Plot
    fig.add_trace(go.Scatter(x=df['Reference'], y=df['Test'], mode='markers', name='Samples'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Reference'], y=deming_intercept + deming_slope * df['Reference'], mode='lines', name='Deming Fit', line=dict(color='red')), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0, 220], y=[0, 220], mode='lines', name='Identity (y=x)', line=dict(color='black', dash='dash')), row=1, col=1)

    # Bland-Altman Plot
    fig.add_trace(go.Scatter(x=df['Average'], y=df['Difference'], mode='markers', name='Difference'), row=2, col=1)
    fig.add_hline(y=mean_diff, line=dict(color='blue', dash='dash'), name='Mean Bias', row=2, col=1, annotation_text=f"Bias: {mean_diff:.2f}")
    fig.add_hline(y=upper_loa, line=dict(color='red', dash='dash'), name='Upper LoA', row=2, col=1, annotation_text=f"Upper LoA: {upper_loa:.2f}")
    fig.add_hline(y=lower_loa, line=dict(color='red', dash='dash'), name='Lower LoA', row=2, col=1, annotation_text=f"Lower LoA: {lower_loa:.2f}")

    # % Bias Plot
    fig.add_trace(go.Scatter(x=df['Reference'], y=df['%Bias'], mode='markers', name='% Bias'), row=3, col=1)
    fig.add_hline(y=0, line=dict(color='black', dash='dash'), row=3, col=1)
    fig.add_hrect(y0=-15, y1=15, fillcolor="green", opacity=0.1, layer="below", line_width=0, row=3, col=1)

    fig.update_layout(height=1000, showlegend=False)
    fig.update_xaxes(title_text="Reference Method", row=1, col=1); fig.update_yaxes(title_text="Test Method", row=1, col=1)
    fig.update_xaxes(title_text="Average of Methods", row=2, col=1); fig.update_yaxes(title_text="Difference (Test - Ref)", row=2, col=1)
    fig.update_xaxes(title_text="Reference Method", row=3, col=1); fig.update_yaxes(title_text="% Bias", row=3, col=1)

    return fig, deming_slope, deming_intercept, mean_diff, upper_loa, lower_loa

@st.cache_data
def plot_bayesian(prior_type):
    """
    Generates plots for the Bayesian inference module.
    """
    # New QC Data (Likelihood)
    n_qc, k_qc = 20, 18

    # Define Priors based on selection
    if prior_type == "Strong R&D Prior":
        # Corresponds to ~98 successes in 100 trials
        a_prior, b_prior = 98, 2
    elif prior_type == "Skeptical/Regulatory Prior":
        # Weakly centered around 80%, wide uncertainty
        a_prior, b_prior = 4, 1
    else: # "No Prior (Frequentist)"
        # Uninformative prior
        a_prior, b_prior = 1, 1

    # Bayesian Update (Posterior calculation)
    a_post = a_prior + k_qc
    b_post = b_prior + (n_qc - k_qc)

    # Calculate key metrics
    prior_mean = a_prior / (a_prior + b_prior)
    mle = k_qc / n_qc
    posterior_mean = a_post / (a_post + b_post)

    # Plotting
    x = np.linspace(0, 1, 500)
    fig = go.Figure()

    # Prior
    prior_pdf = beta.pdf(x, a_prior, b_prior)
    fig.add_trace(go.Scatter(x=x, y=prior_pdf, mode='lines', name='Prior', line=dict(color='green', dash='dash')))

    # Likelihood (scaled for visualization)
    likelihood = beta.pdf(x, k_qc + 1, n_qc - k_qc + 1)
    fig.add_trace(go.Scatter(x=x, y=likelihood, mode='lines', name='Likelihood (from data)', line=dict(color='red', dash='dot')))

    # Posterior
    posterior_pdf = beta.pdf(x, a_post, b_post)
    fig.add_trace(go.Scatter(x=x, y=posterior_pdf, mode='lines', name='Posterior', line=dict(color='blue', width=4), fill='tozeroy'))

    fig.update_layout(
        title=f"<b>Bayesian Update for Pass Rate ({prior_type})</b>",
        xaxis_title="True Pass Rate", yaxis_title="Probability Density",
        legend=dict(x=0.01, y=0.99)
    )

    return fig, prior_mean, mle, posterior_mean

def wilson_score_interval(p_hat, n, z=1.96):
    # This helper function remains the same, but it's good to keep it near the plotting function.
    if n == 0: return (0, 1)
    term1 = (p_hat + z**2 / (2 * n)); denom = 1 + z**2 / n; term2 = z * np.sqrt((p_hat * (1-p_hat)/n) + (z**2 / (4 * n**2))); return (term1 - term2) / denom, (term1 + term2) / denom

def plot_binomial_intervals(successes, n_samples):
    """
    Generates dynamic plots for comparing binomial confidence intervals.
    """
    # --- Plot 1: CI Comparison ---
    p_hat = successes / n_samples if n_samples > 0 else 0

    # Wald Interval
    if n_samples > 0 and p_hat > 0 and p_hat < 1:
        wald_se = np.sqrt(p_hat * (1 - p_hat) / n_samples)
        wald_ci = (p_hat - 1.96 * wald_se, p_hat + 1.96 * wald_se)
    else:
        wald_ci = (p_hat, p_hat) # Collapses at boundaries

    # Wilson Score Interval
    wilson_ci = wilson_score_interval(p_hat, n_samples)

    # Clopper-Pearson (Exact) Interval
    if n_samples > 0:
        alpha = 0.05
        cp_low = beta.ppf(alpha / 2, successes, n_samples - successes + 1)
        cp_high = beta.ppf(1 - alpha / 2, successes + 1, n_samples - successes)
        cp_ci = (cp_low if successes > 0 else 0, cp_high if successes < n_samples else 1)
    else:
        cp_ci = (0, 1)

    fig1 = go.Figure()
    methods = ['Wald (Incorrect)', 'Wilson Score (Recommended)', 'Clopper-Pearson (Conservative)']
    intervals = [wald_ci, wilson_ci, cp_ci]
    for i, (method, interval) in enumerate(zip(methods, intervals)):
        fig1.add_trace(go.Scatter(x=[interval[0], interval[1]], y=[method, method], mode='lines+markers',
                                 marker=dict(size=10), line=dict(width=4), name=method))
    fig1.add_vline(x=p_hat, line_dash="dash", line_color="grey", annotation_text=f"Observed: {p_hat:.2%}")
    fig1.update_layout(title=f"<b>95% Confidence Intervals for {successes}/{n_samples} Successes</b>",
                       xaxis_title="Proportion", xaxis_range=[-0.05, 1.05], showlegend=False)

    # --- Plot 2: Coverage Probability (static for performance) ---
    true_p = np.linspace(0.01, 0.99, 99)
    # This is a known result, plotting a simplified version for demonstration
    coverage_wald = 1 - 2 * norm.cdf(-1.96 - (true_p - 0.5) * np.sqrt(n_samples/true_p/(1-true_p)))
    coverage_wilson = np.full_like(true_p, 0.95)
    np.random.seed(42)
    coverage_wilson += np.random.normal(0, 0.015, len(true_p))
    coverage_wilson[coverage_wilson > 0.99] = 0.99

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=true_p, y=coverage_wald, mode='lines', name='Wald Coverage', line=dict(color='red')))
    fig2.add_trace(go.Scatter(x=true_p, y=coverage_wilson, mode='lines', name='Wilson Coverage', line=dict(color='blue')))
    fig2.add_hline(y=0.95, line_dash="dash", line_color="black", annotation_text="Nominal 95% Coverage")
    fig2.update_layout(title=f"<b>Actual vs. Nominal Coverage Probability (n={n_samples})</b>",
                       xaxis_title="True Proportion (p)", yaxis_title="Actual Coverage Probability",
                       yaxis_range=[min(0.8, coverage_wald.min()), 1.05], legend=dict(x=0.01, y=0.01))

    return fig1, fig2

# ============================ START ACT III PLOTTING FUNCTIONS ============================

def plot_westgard_scenario(scenario='Stable'):
    """Generates a dynamic, high-quality Westgard chart based on a selected process scenario."""
    # Establish the historical process parameters for the control limits
    mean, std = 100, 2
    n_points = 25

    # --- Generate data based on the selected scenario ---
    np.random.seed(42) # Seed for reproducibility of unstable scenarios

    if scenario == 'Stable':
        np.random.seed(101) # Use a different seed for a nice visual
        # Generate data with a smaller SD to ensure it looks stable
        data = np.random.normal(mean, std * 0.75, n_points)
    else:
        # Start with a base of stable data before injecting a problem
        data = np.random.normal(mean, std, n_points)
        if scenario == 'Large Random Error':
            data[15] = 107.5
        elif scenario == 'Systematic Shift':
            data[18:] += 4.5
        elif scenario == 'Increased Imprecision':
            data[20], data[21] = 105, 95
        elif scenario == 'Complex Failure':
            np.random.seed(45); data = np.random.normal(mean, std, n_points)
            data[10], data[14:16] = 107, [105, 105.5]

    fig = go.Figure()

    # Add shaded regions for control zones
    fig.add_hrect(y0=mean - 3*std, y1=mean + 3*std, line_width=0, fillcolor='rgba(255, 165, 0, 0.1)', layer='below', name='±3σ Zone')
    fig.add_hrect(y0=mean - 2*std, y1=mean + 2*std, line_width=0, fillcolor='rgba(0, 128, 0, 0.1)', layer='below', name='±2σ Zone')
    fig.add_hrect(y0=mean - 1*std, y1=mean + 1*std, line_width=0, fillcolor='rgba(0, 128, 0, 0.1)', layer='below', name='±1σ Zone')

    # Add SD lines with labels
    for i in [-3, -2, -1, 1, 2, 3]:
        fig.add_hline(y=mean + i*std, line=dict(color='grey', dash='dot'), annotation_text=f"{'+' if i > 0 else ''}{i}σ", annotation_position="bottom right")
    fig.add_hline(y=mean, line=dict(color='black', dash='dash'), annotation_text='Mean', annotation_position="bottom right")

    # Add data trace
    fig.add_trace(go.Scatter(x=np.arange(1, n_points + 1), y=data, mode='lines+markers', name='Control Data', line=dict(color='#636EFA', width=3), marker=dict(size=10, symbol='circle', line=dict(width=2, color='black'))))

    # Add violation annotations for non-stable scenarios
    if scenario == 'Large Random Error':
        fig.add_trace(go.Scatter(x=[16], y=[107.5], mode='markers', marker=dict(color='red', size=16, symbol='diamond', line=dict(width=2, color='black')), name='1-3s Violation'))
    elif scenario == 'Systematic Shift':
        fig.add_trace(go.Scatter(x=[19, 20], y=data[18:20], mode='markers', marker=dict(color='orange', size=16, symbol='diamond', line=dict(width=2, color='black')), name='2-2s Violation'))
    elif scenario == 'Increased Imprecision':
        fig.add_trace(go.Scatter(x=[21, 22], y=data[20:22], mode='markers', marker=dict(color='purple', size=16, symbol='diamond', line=dict(width=2, color='black')), name='R-4s Violation'))
    elif scenario == 'Complex Failure':
        fig.add_trace(go.Scatter(x=[11], y=[107], mode='markers', marker=dict(color='red', size=16, symbol='diamond', line=dict(width=2, color='black')), name='1-3s Violation'))
        fig.add_trace(go.Scatter(x=[15, 16], y=[105, 105.5], mode='markers', marker=dict(color='orange', size=16, symbol='diamond', line=dict(width=2, color='black')), name='2-2s Violation'))

    fig.update_layout(title=f"<b>Westgard Rules: {scenario} Scenario</b>",
                      xaxis_title="Measurement Number", yaxis_title="Control Value",
                      showlegend=False, height=600)
    return fig

# FIX: Replaced the faulty function with the robust, statistically correct version.
def plot_multivariate_spc(scenario='Stable', n_train=100, n_monitor=20, random_seed=42):
    """
    Backend function to generate data, run MSPC analysis, and create plots.
    """
    # 1. --- Data Generation ---
    np.random.seed(random_seed)
    mean_train = [25, 150]  # Temp (°C), Pressure (kPa)
    cov_train = [[5, 12], [12, 40]] # Strong positive correlation
    df_train = pd.DataFrame(np.random.multivariate_normal(mean_train, cov_train, n_train), columns=['Temperature', 'Pressure'])

    if scenario == 'Stable':
        df_monitor = pd.DataFrame(np.random.multivariate_normal(mean_train, cov_train, n_monitor), columns=['Temperature', 'Pressure'])
    elif scenario == 'Shift in Y Only':
        mean_shift = [25, 175]
        df_monitor = pd.DataFrame(np.random.multivariate_normal(mean_shift, cov_train, n_monitor), columns=['Temperature', 'Pressure'])
    elif scenario == 'Correlation Break':
        cov_break = [[5, 0], [0, 40]]
        df_monitor = pd.DataFrame(np.random.multivariate_normal(mean_train, cov_break, n_monitor), columns=['Temperature', 'Pressure'])

    df_full = pd.concat([df_train, df_monitor], ignore_index=True)

    # 2. --- MSPC Model Building (PCA on training data) ---
    pca = PCA(n_components=1).fit(df_train)
    scores = pca.transform(df_full[['Temperature', 'Pressure']])
    X_hat = pca.inverse_transform(scores)

    # 3. --- Calculate T² and SPE Statistics ---
    S_inv = np.linalg.inv(df_train.cov())
    mean_vec = df_train.mean().values
    diff = df_full[['Temperature', 'Pressure']].values - mean_vec
    df_full['T2'] = [d.T @ S_inv @ d for d in diff]
    residuals = df_full[['Temperature', 'Pressure']].values - X_hat
    df_full['SPE'] = np.sum(residuals**2, axis=1)

    # 4. --- Calculate Control Limits ---
    alpha = 0.01
    p = df_train.shape[1]
    t2_ucl = (p * (n_train - 1) * (n_train + 1)) / (n_train * (n_train - p)) * f.ppf(1 - alpha, p, n_train - p)
    spe_ucl = np.percentile(df_full['SPE'].iloc[:n_train], (1 - alpha) * 100)

    # 5. --- Check for OOC points ---
    monitor_data = df_full.iloc[n_train:]
    t2_ooc_points = monitor_data[monitor_data['T2'] > t2_ucl]
    spe_ooc_points = monitor_data[monitor_data['SPE'] > spe_ucl]
    t2_ooc = not t2_ooc_points.empty
    spe_ooc = not spe_ooc_points.empty

    # PLOT 1: Scatter Plot
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(x=df_train['Temperature'], y=df_train['Pressure'], mode='markers', marker=dict(color='blue', opacity=0.7), name='In-Control (Training Data)'))
    fig_scatter.add_trace(go.Scatter(x=df_monitor['Temperature'], y=df_monitor['Pressure'], mode='markers', marker=dict(color='red', size=8, symbol='star'), name=f'Monitoring Data ({scenario})'))
    pca_line = pca.inverse_transform(np.array([[-15], [15]]))
    fig_scatter.add_trace(go.Scatter(x=pca_line[:, 0], y=pca_line[:, 1], mode='lines', line=dict(color='grey', dash='dash'), name='PCA Model'))
    fig_scatter.update_layout(title=f"Process Scatter Plot: Scenario '{scenario}'", xaxis_title="Temperature (°C)", yaxis_title="Pressure (kPa)", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))

    # PLOT 2: Control Charts
    fig_charts = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Hotelling's T² Chart", "SPE Chart"))
    chart_indices = np.arange(1, len(df_full) + 1)
    fig_charts.add_trace(go.Scatter(x=chart_indices, y=df_full['T2'], mode='lines+markers', name='T² Value'), row=1, col=1)
    fig_charts.add_hline(y=t2_ucl, line_dash="dash", line_color="red", row=1, col=1)
    if t2_ooc: fig_charts.add_trace(go.Scatter(x=t2_ooc_points.index + 1, y=t2_ooc_points['T2'], mode='markers', marker=dict(color='red', size=10, symbol='x')), row=1, col=1)
    fig_charts.add_trace(go.Scatter(x=chart_indices, y=df_full['SPE'], mode='lines+markers', name='SPE Value'), row=2, col=1)
    fig_charts.add_hline(y=spe_ucl, line_dash="dash", line_color="red", row=2, col=1)
    if spe_ooc: fig_charts.add_trace(go.Scatter(x=spe_ooc_points.index + 1, y=spe_ooc_points['SPE'], mode='markers', marker=dict(color='red', size=10, symbol='x')), row=2, col=1)
    fig_charts.add_vrect(x0=n_train+0.5, x1=n_train+n_monitor+0.5, fillcolor="red", opacity=0.1, line_width=0, annotation_text="Monitoring Phase", annotation_position="top left", row='all', col=1)
    fig_charts.update_layout(height=500, title_text="Multivariate Control Charts", showlegend=False, yaxis_title="T² Statistic", yaxis2_title="SPE Statistic", xaxis2_title="Observation Number")

    # PLOT 3: Contribution Plot
    fig_contrib = None
    if t2_ooc or spe_ooc:
        if t2_ooc and scenario == 'Shift in Y Only':
            first_ooc_point = t2_ooc_points.iloc[0]
            contributions = (first_ooc_point[['Temperature', 'Pressure']] - mean_vec)**2
            title_text = "Contribution to T² Alarm (Squared Deviation from Mean)"
        elif spe_ooc and scenario == 'Correlation Break':
            first_ooc_idx = spe_ooc_points.index[0]
            contributions = pd.Series(residuals[first_ooc_idx]**2, index=['Temperature', 'Pressure'])
            title_text = "Contribution to SPE Alarm (Squared Residuals)"
        else:
            first_ooc_idx = (t2_ooc_points.index[0] if t2_ooc else spe_ooc_points.index[0])
            contributions = pd.Series(residuals[first_ooc_idx]**2, index=['Temperature', 'Pressure'])
            title_text = "Contribution to Alarm (Squared Residuals)"
        fig_contrib = px.bar(x=contributions.index, y=contributions.values, title=title_text, labels={'x':'Process Variable', 'y':'Contribution Value'})

    return fig_scatter, fig_charts, fig_contrib, t2_ooc, spe_ooc

# FIX: This function was nested inside another one. It's now a top-level helper.
def plot_ewma_cusum_comparison():
    np.random.seed(123)
    n_points = 40
    data = np.random.normal(100, 2, n_points)
    data[20:] += 1.5 # A small 0.75-sigma shift
    mean, std = 100, 2

    # EWMA calculation
    lam = 0.2 # Lambda, the weighting factor
    ewma = np.zeros(n_points)
    ewma[0] = mean
    for i in range(1, n_points):
        ewma[i] = lam * data[i] + (1 - lam) * ewma[i-1]

    # CUSUM calculation
    target = mean
    k = 0.5 * std # "Allowance" or "slack"
    sh, sl = np.zeros(n_points), np.zeros(n_points)
    for i in range(1, n_points):
        sh[i] = max(0, sh[i-1] + (data[i] - target) - k)
        sl[i] = max(0, sl[i-1] + (target - data[i]) - k)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("<b>I-Chart: The Beat Cop (Misses the Shift)</b>",
                                        "<b>EWMA: The Sentinel (Catches the Shift)</b>",
                                        "<b>CUSUM: The Bloodhound (Catches it Fastest)</b>"))

    # I-Chart
    fig.add_trace(go.Scatter(x=np.arange(1, n_points+1), y=data, mode='lines+markers', name='Data'), row=1, col=1)
    fig.add_hline(y=mean + 3*std, line_color='red', line_dash='dash', row=1, col=1)
    fig.add_hline(y=mean - 3*std, line_color='red', line_dash='dash', row=1, col=1)
    fig.add_vline(x=20.5, line_color='orange', line_dash='dot', row=1, col=1)

    # EWMA Chart
    fig.add_trace(go.Scatter(x=np.arange(1, n_points+1), y=ewma, mode='lines+markers', name='EWMA'), row=2, col=1)
    sigma_ewma = std * np.sqrt(lam / (2-lam)) # Asymptotic SD
    fig.add_hline(y=mean + 3*sigma_ewma, line_color='red', line_dash='dash', row=2, col=1)
    fig.add_hline(y=mean - 3*sigma_ewma, line_color='red', line_dash='dash', row=2, col=1)
    fig.add_vline(x=20.5, line_color='orange', line_dash='dot', row=2, col=1)

    # CUSUM Chart
    fig.add_trace(go.Scatter(x=np.arange(1, n_points+1), y=sh, mode='lines+markers', name='CUSUM High'), row=3, col=1)
    fig.add_trace(go.Scatter(x=np.arange(1, n_points+1), y=sl, mode='lines+markers', name='CUSUM Low'), row=3, col=1)
    fig.add_hline(y=5*std, line_color='red', line_dash='dash', row=3, col=1) # Decision interval H
    fig.add_vline(x=20.5, line_color='orange', line_dash='dot', row=3, col=1, annotation_text="Process Shift Occurs", annotation_position="top")

    fig.update_layout(title="<b>Case Study: Detecting a Small Process Shift (0.75σ)</b>", height=800, showlegend=False)
    fig.update_xaxes(title_text="Sample Number", row=3, col=1)
    return fig

@st.cache_data
def plot_time_series_analysis():
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=104, freq='W')
    trend = np.linspace(50, 60, 104)
    seasonality = 5 * np.sin(np.arange(104) * (2*np.pi/52.14))
    noise = np.random.normal(0, 2, 104)
    y = trend + seasonality + noise
    df = pd.DataFrame({'ds': dates, 'y': y})
    train, test = df.iloc[:90], df.iloc[90:]

    # Prophet model
    m_prophet = Prophet().fit(train)
    future = m_prophet.make_future_dataframe(periods=14, freq='W')
    fc_prophet = m_prophet.predict(future)

    # ARIMA model
    m_arima = ARIMA(train['y'], order=(1,1,1), seasonal_order=(1,0,1,52)).fit()
    fc_arima = m_arima.get_forecast(steps=14).summary_frame()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Actual Data'))
    fig.add_trace(go.Scatter(x=fc_prophet['ds'].iloc[-14:], y=fc_prophet['yhat'].iloc[-14:], mode='lines', name='Prophet Forecast', line=dict(dash='dash', color='red')))
    fig.add_trace(go.Scatter(x=test['ds'], y=fc_arima['mean'], mode='lines', name='ARIMA Forecast', line=dict(dash='dash', color='green')))
    fig.update_layout(title='<b>Time Series Forecasting: Prophet vs. ARIMA</b>', xaxis_title='Date', yaxis_title='Control Value')
    return fig

@st.cache_data
def plot_stability_analysis():
    np.random.seed(1)
    time_points = np.array([0, 3, 6, 9, 12, 18, 24]) # Months
    # Simulate 3 batches
    batches = {}
    for i in range(3):
        initial_potency = np.random.normal(102, 0.5)
        degradation_rate = np.random.normal(-0.4, 0.05)
        noise = np.random.normal(0, 0.5, len(time_points))
        batches[f'Batch {i+1}'] = initial_potency + degradation_rate * time_points + noise

    df = pd.DataFrame(batches)
    df['Time'] = time_points
    df_melt = df.melt(id_vars='Time', var_name='Batch', value_name='Potency')

    # Fit a pooled regression model
    model = ols('Potency ~ Time', data=df_melt).fit()
    LSL = 95.0

    # Calculate shelf life: Time when lower 95% CI of mean prediction crosses LSL
    x_pred = pd.DataFrame({'Time': np.linspace(0, 36, 100)})
    predictions = model.get_prediction(x_pred).summary_frame(alpha=0.05)

    shelf_life_df = predictions[predictions['mean_ci_lower'] >= LSL]
    shelf_life = x_pred['Time'][shelf_life_df.index[-1]] if not shelf_life_df.empty else 0

    fig = px.scatter(df_melt, x='Time', y='Potency', color='Batch', title='<b>Stability Analysis for Shelf-Life Estimation</b>')
    fig.add_trace(go.Scatter(x=x_pred['Time'], y=predictions['mean'], mode='lines', name='Mean Trend', line=dict(color='black')))
    fig.add_trace(go.Scatter(x=x_pred['Time'], y=predictions['mean_ci_lower'], mode='lines', name='95% Lower CI', line=dict(color='red', dash='dash')))
    fig.add_hline(y=LSL, line=dict(color='red', dash='dot'), name='Specification Limit')
    fig.add_vline(x=shelf_life, line=dict(color='blue', dash='dash'), annotation_text=f'Shelf-Life = {shelf_life:.1f} Months')
    fig.update_layout(xaxis_title="Time (Months)", yaxis_title="Potency (%)")
    return fig

@st.cache_data
def plot_survival_analysis():
    np.random.seed(42)
    # Simulate time-to-event data for two groups
    time_A = stats.weibull_min.rvs(c=1.5, scale=20, size=50)
    censor_A = np.random.binomial(1, 0.2, 50) # 0=event, 1=censored
    time_B = stats.weibull_min.rvs(c=1.5, scale=30, size=50)
    censor_B = np.random.binomial(1, 0.2, 50)

    def kaplan_meier(times, events):
        df = pd.DataFrame({'time': times, 'event': events}).sort_values('time').reset_index(drop=True)
        unique_times = df['time'][df['event'] == 1].unique()

        km_df = pd.DataFrame({'time': np.append([0], unique_times),'n_at_risk': 0,'n_events': 0,})
        km_df['survival'] = 1.0

        for i, t in enumerate(km_df['time']):
            at_risk = (df['time'] >= t).sum()
            events_at_t = ((df['time'] == t) & (df['event'] == 1)).sum()
            km_df.loc[i, 'n_at_risk'] = at_risk
            km_df.loc[i, 'n_events'] = events_at_t

        for i in range(1, len(km_df)):
            if km_df.loc[i, 'n_at_risk'] > 0:
                km_df.loc[i, 'survival'] = km_df.loc[i-1, 'survival'] * (1 - km_df.loc[i, 'n_events'] / km_df.loc[i, 'n_at_risk'])
            else:
                km_df.loc[i, 'survival'] = km_df.loc[i-1, 'survival']

        ts = np.repeat(km_df['time'].values, 2)[1:]
        surv = np.repeat(km_df['survival'].values, 2)[:-1]

        return np.append([0], ts), np.append([1.0], surv)

    ts_A, surv_A = kaplan_meier(time_A, 1 - censor_A)
    ts_B, surv_B = kaplan_meier(time_B, 1 - censor_B)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts_A, y=surv_A, mode='lines', name='Group A (e.g., Old Component)', line_shape='hv', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=ts_B, y=surv_B, mode='lines', name='Group B (e.g., New Component)', line_shape='hv', line=dict(color='red')))

    fig.update_layout(title='<b>Reliability / Survival Analysis (Kaplan-Meier Curve)</b>',
                      xaxis_title='Time to Event (e.g., Days to Failure)',
                      yaxis_title='Survival Probability',
                      yaxis_range=[0, 1.05],
                      legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99))
    return fig

@st.cache_data
def plot_mva_pls():
    np.random.seed(0)
    n_samples = 50
    n_features = 200
    # Simulate spectral data
    X = np.random.rand(n_samples, n_features)
    # Create a true relationship based on a few "peaks"
    y = 2 * X[:, 50] - 1.5 * X[:, 120] + np.random.normal(0, 0.2, n_samples)

    pls = PLSRegression(n_components=2)
    pls.fit(X, y)

    # VIP score calculation
    T = pls.x_scores_
    W = pls.x_weights_
    Q = pls.y_loadings_
    p, h = W.shape
    VIPs = np.zeros((p,))
    s = np.diag(T.T @ T @ Q.T @ Q).reshape(h, -1)
    total_s = np.sum(s)
    for i in range(p):
        weight = np.array([(W[i,j] / np.linalg.norm(W[:,j]))**2 for j in range(h)])
        VIPs[i] = np.sqrt(p * (s.T @ weight) / total_s)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("<b>Raw Spectral Data</b>", "<b>Variable Importance (VIP) Plot</b>"))
    for i in range(10): # Plot first 10 samples
        fig.add_trace(go.Scatter(y=X[i,:], mode='lines', name=f'Sample {i+1}'), row=1, col=1)

    fig.add_trace(go.Bar(y=VIPs, name='VIP Score'), row=1, col=2)
    fig.add_hline(y=1, line=dict(color='red', dash='dash'), name='Significance Threshold', row=1, col=2)

    fig.update_layout(title='<b>Multivariate Analysis (PLS Regression)</b>', showlegend=False)
    fig.update_xaxes(title_text='Wavelength', row=1, col=1); fig.update_yaxes(title_text='Absorbance', row=1, col=1)
    fig.update_xaxes(title_text='Wavelength', row=1, col=2); fig.update_yaxes(title_text='VIP Score', row=1, col=2)
    return fig

@st.cache_data
def plot_clustering():
    np.random.seed(42)
    X1 = np.random.normal(10, 2, 50)
    Y1 = np.random.normal(10, 2, 50)
    X2 = np.random.normal(25, 3, 50)
    Y2 = np.random.normal(25, 3, 50)
    X3 = np.random.normal(15, 2.5, 50)
    Y3 = np.random.normal(30, 2.5, 50)
    X = np.concatenate([X1, X2, X3])
    Y = np.concatenate([Y1, Y2, Y3])
    df = pd.DataFrame({'X': X, 'Y': Y})

    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto').fit(df)
    df['Cluster'] = kmeans.labels_.astype(str)

    fig = px.scatter(df, x='X', y='Y', color='Cluster', title='<b>Clustering: Discovering Hidden Process Regimes</b>',
                     labels={'X': 'Process Parameter 1 (e.g., Temperature)', 'Y': 'Process Parameter 2 (e.g., Pressure)'})
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color='black')))
    return fig

@st.cache_data
def plot_classification_models():
    np.random.seed(1)
    n_points = 200
    X1 = np.random.uniform(0, 10, n_points)
    X2 = np.random.uniform(0, 10, n_points)
    # Create a non-linear relationship
    prob = 1 / (1 + np.exp(-( (X1-5)**2 + (X2-5)**2 - 8)))
    y = np.random.binomial(1, prob)
    X = np.column_stack((X1, X2))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Logistic Regression
    lr = LogisticRegression().fit(X_train, y_train)
    lr_score = lr.score(X_test, y_test)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    rf_score = rf.score(X_test, y_test)

    # Create meshgrid for decision boundary
    xx, yy = np.meshgrid(np.linspace(0, 10, 100), np.linspace(0, 10, 100))

    fig = make_subplots(rows=1, cols=2, subplot_titles=(f'<b>Logistic Regression (Accuracy: {lr_score:.2%})</b>',
                                                       f'<b>Random Forest (Accuracy: {rf_score:.2%})</b>'))

    # Plot Logistic Regression
    Z_lr = lr.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    fig.add_trace(go.Contour(x=xx[0], y=yy[:,0], z=Z_lr, colorscale='RdBu', showscale=False, opacity=0.3), row=1, col=1)
    fig.add_trace(go.Scatter(x=X[:,0], y=X[:,1], mode='markers', marker=dict(color=y, colorscale='RdBu', line=dict(width=1, color='black'))), row=1, col=1)

    # Plot Random Forest
    Z_rf = rf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    fig.add_trace(go.Contour(x=xx[0], y=yy[:,0], z=Z_rf, colorscale='RdBu', showscale=False, opacity=0.3), row=1, col=2)
    fig.add_trace(go.Scatter(x=X[:,0], y=X[:,1], mode='markers', marker=dict(color=y, colorscale='RdBu', line=dict(width=1, color='black'))), row=1, col=2)

    fig.update_layout(title="<b>Predictive QC: Linear vs. Non-Linear Models</b>", showlegend=False, height=500)
    fig.update_xaxes(title_text="Parameter 1", row=1, col=1); fig.update_yaxes(title_text="Parameter 2", row=1, col=1)
    fig.update_xaxes(title_text="Parameter 1", row=1, col=2); fig.update_yaxes(title_text="Parameter 2", row=1, col=2)
    return fig

@st.cache_data
def plot_xai_shap():
    plt.style.use('default')

    # Load sample data
    github_data_url = "https://github.com/slundberg/shap/raw/master/data/"
    data_url = github_data_url + "adult.data"
    dtypes = [
        ("Age", "float32"), ("Workclass", "category"), ("fnlwgt", "float32"),
        ("Education", "category"), ("Education-Num", "float32"),
        ("Marital Status", "category"), ("Occupation", "category"),
        ("Relationship", "category"), ("Race", "category"), ("Sex", "category"),
        ("Capital Gain", "float32"), ("Capital Loss", "float32"),
        ("Hours per week", "float32"), ("Country", "category"), ("Target", "category")
    ]
    raw_data = pd.read_csv(data_url, names=[d[0] for d in dtypes], na_values="?", dtype=dict(dtypes))
    X_display = raw_data.drop("Target", axis=1)
    y = (raw_data["Target"] == " >50K").astype(int)
    X = X_display.copy()
    for col in X.select_dtypes(include=['category']).columns:
        X[col] = X[col].cat.codes

    model = RandomForestClassifier(random_state=42).fit(X, y)
    explainer = shap.Explainer(model, X)
    shap_values_obj = explainer(X.iloc[:100])

    # --- Beeswarm plot (as image) ---
    shap.summary_plot(shap_values_obj.values[:,:,1], X.iloc[:100], show=False)
    buf_summary = io.BytesIO()
    plt.savefig(buf_summary, format='png', bbox_inches='tight')
    plt.close()
    buf_summary.seek(0)

    # FIX: Cleaned up redundant code and correctly generate self-contained HTML
    # --- Force plot (as HTML) ---
    force_plot = shap.force_plot(explainer.expected_value[1], shap_values_obj.values[0,:,1], X_display.iloc[0,:], show=False)
    force_plot_html = force_plot.html()
    full_html = f"<html><head>{shap.initjs()}</head><body>{force_plot_html}</body></html>"

    return buf_summary, full_html

@st.cache_data
def plot_advanced_ai_concepts(concept):
    fig = go.Figure()
    if concept == "Transformers":
        text = "Input Seq -> [Encoder] -> [Self-Attention] -> [Decoder] -> Output Seq"
        fig.add_annotation(text=f"<b>Conceptual Flow: Transformer</b><br>{text}", showarrow=False, font_size=16)
    elif concept == "GNNs":
        nodes_x = [1, 2, 3, 4, 3, 2]; nodes_y = [2, 3, 2, 1, 0, -1]
        edges = [(0,1), (1,2), (2,3), (2,4), (4,5), (5,1)]
        for (start, end) in edges:
            fig.add_trace(go.Scatter(x=[nodes_x[start], nodes_x[end]], y=[nodes_y[start], nodes_y[end]], mode='lines', line_color='grey'))
        fig.add_trace(go.Scatter(x=nodes_x, y=nodes_y, mode='markers+text', text=[f"Node {i}" for i in range(6)], marker_size=30, textposition="middle center"))
        fig.update_layout(title="<b>Conceptual Flow: Graph Neural Network</b>")
    elif concept == "RL":
        fig.add_shape(type="rect", x0=0, y0=0, x1=2, y1=2, line_width=2, fillcolor='lightblue', name="Agent")
        fig.add_annotation(x=1, y=1, text="<b>Agent</b><br>(Control Policy)", showarrow=False)
        fig.add_shape(type="rect", x0=4, y0=0, x1=6, y1=2, line_width=2, fillcolor='lightgreen', name="Environment")
        fig.add_annotation(x=5, y=1, text="<b>Environment</b><br>(Digital Twin)", showarrow=False)
        fig.add_annotation(x=3, y=1.5, text="Action", showarrow=True, arrowhead=2, ax=-40, ay=0)
        fig.add_annotation(x=3, y=0.5, text="State, Reward", showarrow=True, arrowhead=2, ax=40, ay=0)
        fig.update_layout(title="<b>Conceptual Flow: Reinforcement Learning Loop</b>")
    elif concept == "Generative AI":
        fig.add_shape(type="rect", x0=0, y0=0, x1=2, y1=2, line_width=2, fillcolor='lightcoral', name="Real Data")
        fig.add_annotation(x=1, y=1, text="<b>Real Data</b>", showarrow=False)
        fig.add_annotation(x=3, y=1, text="Trains ➔", showarrow=False, font_size=20)
        fig.add_shape(type="rect", x0=4, y0=0, x1=6, y1=2, line_width=2, fillcolor='gold', name="Generator")
        fig.add_annotation(x=5, y=1, text="<b>Generator</b>", showarrow=False)
        fig.add_annotation(x=7, y=1, text="Creates ➔", showarrow=False, font_size=20)
        fig.add_shape(type="rect", x0=8, y0=0, x1=10, y1=2, line_width=2, fillcolor='lightseagreen', name="Synthetic Data")
        fig.add_annotation(x=9, y=1, text="<b>Synthetic Data</b>", showarrow=False)
        fig.update_layout(title="<b>Conceptual Flow: Generative AI (GANs)</b>")

    fig.update_layout(xaxis_visible=False, yaxis_visible=False, height=300, showlegend=False)
    return fig

# ==============================================================================
# ALL UI RENDERING FUNCTIONS
# ==============================================================================

def render_introduction_content():
    """Renders the three introductory sections as a single page."""
    st.title("🛠️ Biotech V&V Analytics Toolkit")
    st.markdown("### An Interactive Guide to Assay Validation, Tech Transfer, and Lifecycle Management")
    st.markdown("Welcome! This toolkit is a collection of interactive modules designed to explore the statistical and machine learning methods that form the backbone of a robust V&V, technology transfer, and process monitoring plan.")
    st.info("#### 👈 Select a tool from the sidebar to explore an interactive module.")

    st.header("📖 The Scientist's/Engineer's Journey: A Three-Act Story")
    st.markdown("""The journey from a novel idea to a robust, routine process can be viewed as a three-act story, with each act presenting unique analytical challenges. The toolkit is structured to follow that narrative.""")
    act1, act2, act3 = st.columns(3)
    with act1:
        st.subheader("Act I: Foundation & Characterization")
        st.markdown("Before a method or process can be trusted, its fundamental capabilities, limitations, and sensitivities must be deeply understood. This is the act of building a solid, data-driven foundation.")
    with act2:
        st.subheader("Act II: Transfer & Stability")
        st.markdown("Here, the method faces its crucible. It must prove its performance in a new environment—a new lab, a new scale, a new team. This is about demonstrating stability and equivalence.")
    with act3:
        st.subheader("Act III: The Guardian (Lifecycle Management)")
        st.markdown("Once live, the journey isn't over. This final act is about continuous guardianship: monitoring process health, detecting subtle drifts, and using advanced analytics to predict and prevent future failures.")

    st.divider()
    st.header("🚀 The V&V Model: A Strategic Framework")
    st.markdown("The **Verification & Validation (V&V) Model**, shown below, provides a structured, widely accepted framework for ensuring a system meets its intended purpose, from initial requirements to final deployment.")
    st.plotly_chart(plot_v_model(), use_container_width=True)

    st.divider()
    st.header("📈 Project Workflow")
    st.markdown("This timeline organizes the entire toolkit by its application in a typical project lifecycle. Tools are grouped by the project phase where they provide the most value, and are ordered chronologically within each phase.")
    st.plotly_chart(plot_act_grouped_timeline(), use_container_width=True)

    st.header("⏳ A Chronological View of V&V Analytics")
    st.markdown("This timeline organizes the same tools purely by their year of invention, showing the evolution of statistical and machine learning thought over the last century.")
    st.plotly_chart(plot_chronological_timeline(), use_container_width=True)

    st.header("🗺️ Conceptual Map of Tools")
    st.markdown("This map illustrates the relationships between the foundational concepts and the specific tools available in this application. Use it to navigate how different methods connect to broader analytical strategies.")
    st.plotly_chart(create_toolkit_conceptual_map(), use_container_width=True)

def render_ci_concept():
    """Renders the interactive module for Confidence Intervals."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To build a deep, intuitive understanding of the fundamental concept of a **frequentist confidence interval** and to correctly interpret what it does—and does not—tell us.
    
    **Strategic Application:** This concept is the bedrock of all statistical inference in a frequentist framework. A misunderstanding of CIs leads to flawed conclusions and poor decision-making. This interactive simulation directly impacts resource planning and risk assessment. It allows scientists and managers to explore the crucial trade-off between **sample size (cost)** and **statistical precision (certainty)**. It provides a visual, data-driven answer to the perpetual question: "How many samples do we *really* need to run to get a reliable result and an acceptably narrow confidence interval?"
    """)

    st.info("""
    **Interactive Demo:** Use the **Sample Size (n)** slider in the sidebar to dynamically change the number of samples in each simulated experiment. Observe how increasing the sample size dramatically narrows both the theoretical Sampling Distribution (orange curve) and the simulated Confidence Intervals (blue/red lines), directly demonstrating the link between sample size and precision.
    """)

    n_slider = st.sidebar.slider("Select Sample Size (n) for Each Simulated Experiment:", 5, 100, 30, 5)

    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        fig1_ci, fig2_ci, capture_count, n_sims, avg_width = plot_ci_concept(n=n_slider)
        st.plotly_chart(fig1_ci, use_container_width=True)
        st.plotly_chart(fig2_ci, use_container_width=True)

    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History"])
        with tabs[0]:
            st.metric(label=f"📈 KPI: Average CI Width (Precision) at n={n_slider}", value=f"{avg_width:.2f} units", help="A smaller width indicates higher precision. This is inversely proportional to the square root of n.")
            st.metric(label="💡 Empirical Coverage Rate", value=f"{(capture_count/n_sims):.0%}", help=f"The % of our {n_sims} simulated CIs that successfully 'captured' the true population mean. Should be close to 95%.")
            st.markdown("""
            - **Theoretical Universe (Top Plot):**
                - The wide, light blue curve is the **true population distribution**. In real life, we *never* see this.
                - The narrow, orange curve is the **sampling distribution of the mean**. Its narrowness, guaranteed by the **Central Limit Theorem**, makes inference possible.
            - **CI Simulation (Bottom Plot):** This shows the reality we live in. We only get *one* experiment and *one* confidence interval.
            - **The n-slider is key:** As you increase `n`, the orange curve gets narrower and the CIs in the bottom plot become dramatically shorter.
            - **Diminishing Returns:** The gain in precision from n=5 to n=20 is huge. The gain from n=80 to n=100 is much smaller. This illustrates that to double your precision (halve the CI width), you must quadruple your sample size.

            **The Core Strategic Insight:** A confidence interval is a statement about the *procedure*, not a specific result. The "95% confidence" is our confidence in the *method* used to generate the interval, not in any single interval itself.
            """)
        with tabs[1]:
            st.error("""
            🔴 **THE INCORRECT (Bayesian) INTERPRETATION:**
            *"Based on my sample, there is a 95% probability that the true mean is in this interval [X, Y]."*

            This is wrong because in the frequentist view, the true mean is a fixed constant. It is either in our specific interval or it is not. The probability is 1 or 0.
            """)
            st.success("""
            🟢 **THE CORRECT (Frequentist) INTERPRETATION:**
            *"We are 95% confident that the interval [X, Y] contains the true mean."*

            The full meaning is: *"This interval was constructed using a procedure that, when repeated many times, will produce intervals that capture the true mean 95% of the time."*
            """)
        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            The concept of **confidence intervals** was introduced by **Jerzy Neyman** in a landmark 1937 paper. Neyman's revolutionary idea was to shift the probabilistic statement away from the fixed, unknown parameter and onto the **procedure used to create the interval**. This clever reframing provided a practical and logically consistent solution that remains the dominant paradigm for interval estimation today.

            #### Mathematical Basis
            """)
            st.latex(r"\text{Point Estimate} \pm (\text{Critical Value} \times \text{Standard Error})")
            st.markdown("""
            - **Point Estimate:** Our best guess (e.g., the sample mean, $\bar{x}$).
            - **Standard Error:** The standard deviation of the sampling distribution of the point estimate (e.g., $\frac{s}{\sqrt{n}}$). It measures the typical error in our point estimate.
            - **Critical Value:** A multiplier determined by our desired confidence level (e.g., a t-score).
            """)

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
    **Interactive Demo:** Now, when you navigate to the "Core Validation Parameters" tool, you will see a new set of dedicated sliders in the sidebar. Changing these sliders will instantly update the three plots, allowing you to build a powerful, hands-on intuition for these critical concepts.
    """)

    st.sidebar.subheader("Core Validation Controls")
    bias_slider = st.sidebar.slider(
        "🎯 Systematic Bias (%)",
        min_value=-10.0, max_value=10.0, value=1.5, step=0.5,
        help="Simulates a constant positive or negative bias in the accuracy study. Watch the box plots shift."
    )
    repeat_cv_slider = st.sidebar.slider(
        "🏹 Repeatability %CV",
        min_value=0.5, max_value=10.0, value=1.5, step=0.5,
        help="Simulates the best-case random error (intra-assay precision). Watch the 'Repeatability' violin width."
    )
    intermed_cv_slider = st.sidebar.slider(
        "🏹 Intermediate Precision %CV",
        min_value=repeat_cv_slider, max_value=20.0, value=max(2.5, repeat_cv_slider), step=0.5,
        help="Simulates real-world random error (inter-assay). A large gap from repeatability indicates poor robustness."
    )
    interference_slider = st.sidebar.slider(
        "🔬 Interference Effect (%)",
        min_value=-20.0, max_value=20.0, value=8.0, step=1.0,
        help="Simulates an interferent that falsely increases (+) or decreases (-) the analyte signal."
    )

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
        tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History"])

        with tabs[0]:
            st.info("Play with the sliders in the sidebar to see how different sources of error affect the results!")
            st.markdown("""
            - **Accuracy Plot:** As you increase the **Systematic Bias** slider, watch the center of the box plots drift away from the black 'True Value' lines. This visually demonstrates what bias looks like.

            - **Precision Plot:** The **%CV sliders** control the width (spread) of the violin plots. Notice that Intermediate Precision must always be equal to or worse (wider) than Repeatability. A large gap between the two signals that the method is not robust to day-to-day changes.

            - **Specificity Plot:** The **Interference Effect** slider directly moves the 'Analyte + Interferent' box plot. A perfect assay would have this slider at 0%, making the two boxes identical. A large effect, positive or negative, indicates a failed specificity study.

            **The Core Strategic Insight:** This simulation shows that validation is a process of hunting for and quantifying different types of error. Accuracy is about finding *bias*, Precision is about characterizing *random error*, and Specificity is about eliminating *interference error*.
            """)

        with tabs[1]:
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

        with tabs[2]:
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

def render_gage_rr():
    """Renders the INTERACTIVE module for Gage R&R."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To rigorously quantify the inherent variability (error) of a measurement system. It answers the fundamental question: "Is my measurement system a precision instrument, or a random number generator?"
    
    **Strategic Application:** A foundational checkpoint in any technology transfer or process validation. An unreliable measurement system creates a "fog of uncertainty," leading to two costly errors: rejecting good product (False Alarm) or accepting bad product (Missed Signal). **Use the sliders in the sidebar to simulate different sources of variation and see their impact on the final % Gage R&R.**
    """)

    st.info("""
    **Interactive Demo:** Now, when you navigate to the "Gage R&R / VCA" tool, you will see a new set of dedicated sliders in the sidebar. You can now dynamically simulate how a precise (low repeatability) or imprecise (high repeatability) instrument performs, and how well-trained (low operator variation) or poorly-trained (high operator variation) teams affect the final result.
    """)

    st.sidebar.subheader("Gage R&R Controls")
    part_sd_slider = st.sidebar.slider(
        "🏭 Part-to-Part Variation (SD)",
        min_value=1.0, max_value=10.0, value=5.0, step=0.5,
        help="The 'true' variation of the product. Increase this to see how a good measurement system can easily distinguish between different parts."
    )
    repeat_sd_slider = st.sidebar.slider(
        "🔬 Repeatability (SD)",
        min_value=0.1, max_value=5.0, value=1.5, step=0.1,
        help="The inherent 'noise' of the instrument/assay. Increase this to simulate a less precise measurement device."
    )
    operator_sd_slider = st.sidebar.slider(
        "👤 Operator-to-Operator Variation (SD)",
        min_value=0.0, max_value=5.0, value=0.75, step=0.25,
        help="The systematic bias between operators. Increase this to simulate poor training or inconsistent technique."
    )

    fig, pct_rr, pct_part = plot_gage_rr(
        part_sd=part_sd_slider,
        repeatability_sd=repeat_sd_slider,
        operator_sd=operator_sd_slider
    )

    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ Acceptance Criteria", "📖 Theory & History"])
        with tabs[0]:
            st.metric(label="📈 KPI: % Gage R&R", value=f"{pct_rr:.1f}%", delta="Lower is better", delta_color="inverse", help="The percentage of total variation consumed by measurement error.")
            st.metric(label="💡 KPI: Number of Distinct Categories (ndc)", value=f"{int(1.41 * (pct_part / pct_rr)**0.5) if pct_rr > 0 else '>10'}", help="An estimate of how many distinct groups the system can discern. A value < 5 is problematic.")

            st.info("Play with the sliders in the sidebar to see how different sources of error affect the results!")
            st.markdown("""
            - **Increase `Part-to-Part Variation`:** Notice how the operator means (colored lines) spread further apart. A good measurement system should show this! Crucially, the **% Gage R&R KPI goes DOWN**, because the measurement error is now a smaller proportion of the total variation.
            - **Increase `Repeatability`:** The box plots for each part get much wider. This is pure measurement noise. The **% Gage R&R KPI goes UP**.
            - **Increase `Operator-to-Operator Variation`:** The colored mean lines separate vertically and the overall operator boxes (top right) drift apart. This is bias between people. The **% Gage R&R KPI goes UP**.

            **The Core Strategic Insight:** A low % Gage R&R is achieved when the measurement error (Repeatability + Reproducibility) is small *relative to* the true process variation.
            """)

        with tabs[1]:
            st.markdown("Acceptance criteria are risk-based and derived from the **AIAG's Measurement Systems Analysis (MSA)** manual, the de facto global standard.")
            st.markdown("- **< 10% Gage R&R:** The system is **acceptable**.")
            st.markdown("- **10% - 30% Gage R&R:** The system is **conditionally acceptable or marginal**.")
            st.markdown("- **> 30% Gage R&R:** The system is **unacceptable and must be rejected**.")
            st.info("""
            **The Part Selection Strategy:** The most common failure mode of a Gage R&R study is not the math, but the study design. The parts selected **must span the full expected range of process variation**. If you only select parts from the middle of the distribution, your Part-to-Part variation will be artificially low, which will mathematically inflate your % Gage R&R and cause a good system to fail.
            """)

        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            The modern application was born out of the quality crisis in the American automotive industry in the 1970s. The **AIAG** codified the solution in the first MSA manual. The critical evolution was the move from the simple **Average and Range (X-bar & R) method** to the **ANOVA method**. The ANOVA method, pioneered by **Sir Ronald A. Fisher**, became the gold standard because of its unique ability to cleanly partition and test the significance of each variance component, including the crucial interaction term.

            #### Mathematical Basis
            The ANOVA method partitions the total sum of squared deviations from the mean ($SS_T$) into components attributable to each factor.
            """)
            st.latex(r"SS_{Total} = SS_{Part} + SS_{Operator} + SS_{Part \times Operator} + SS_{Error}")
            st.markdown("These are converted to Mean Squares (MS) and then to variance components ($\hat{\sigma}^2$).")

def render_lod_loq():
    """Renders the INTERACTIVE module for Limit of Detection & Quantitation."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To formally establish the absolute lower performance boundaries of a quantitative assay. It determines the lowest analyte concentration an assay can reliably **detect (LOD)** and the lowest concentration it can reliably and accurately **quantify (LOQ)**.
    
    **Strategic Application:** This is a mission-critical parameter for any assay used to measure trace components, such as impurities in a drug product or biomarkers for early-stage disease diagnosis. **Use the sliders in the sidebar to simulate how assay sensitivity and noise impact the final LOD and LOQ.**
    """)

    st.info("""
    **Interactive Demo:** Now, when you select the "LOD & LOQ" tool, a new set of dedicated sliders will appear in the sidebar. You can dynamically change the assay's slope and noise to see in real-time how these fundamental characteristics drive the final LOD and LOQ results.
    """)

    st.sidebar.subheader("LOD & LOQ Controls")
    slope_slider = st.sidebar.slider(
        "📈 Assay Sensitivity (Slope)",
        min_value=0.005, max_value=0.1, value=0.02, step=0.005, format="%.3f",
        help="How much the signal increases per unit of concentration. A steeper slope (higher sensitivity) is better."
    )
    noise_slider = st.sidebar.slider(
        "🔇 Baseline Noise (SD)",
        min_value=0.001, max_value=0.05, value=0.01, step=0.001, format="%.3f",
        help="The inherent random noise of the assay at zero concentration. A lower noise floor is better."
    )

    fig, lod_val, loq_val = plot_lod_loq(slope=slope_slider, baseline_sd=noise_slider)

    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ Acceptance Criteria", "📖 Theory & History"])
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
            st.markdown("- The primary, non-negotiable criterion is that the experimentally determined **LOQ must be ≤ the lowest concentration that the assay is required to measure** for its specific application (e.g., a release specification for an impurity).")
            st.markdown("- For a concentration to be formally declared the LOQ, it must be experimentally confirmed. This typically involves analyzing 5-6 independent samples at the claimed LOQ concentration and demonstrating that they meet pre-defined criteria for precision and accuracy (e.g., **%CV < 20% and %Recovery between 80-120%** for a bioassay).")
            st.warning("""
            **The LOB, LOD, and LOQ Hierarchy: A Critical Distinction**
            A full characterization involves three distinct limits:
            - **Limit of Blank (LOB):** The highest measurement expected from a blank sample.
            - **Limit of Detection (LOD):** The lowest concentration whose signal is statistically distinguishable from the LOB.
            - **Limit of Quantitation (LOQ):** The lowest concentration meeting precision/accuracy requirements, which is almost always higher than the LOD.
            """)
        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            The need to define analytical sensitivity is old, but definitions were inconsistent until the **International Council for Harmonisation (ICH)** guideline **ICH Q2(R1)** harmonized the methodologies. This work was heavily influenced by the statistical framework established by **Lloyd Currie at NIST** in his 1968 paper, which established the clear, hypothesis-testing basis for the modern LOB/LOD/LOQ hierarchy.

            #### Mathematical Basis
            This method is built on the relationship between the assay's signal, its sensitivity (Slope, S), and its noise (standard deviation, σ).
            """)
            st.latex(r"LOD \approx \frac{3.3 \times \sigma}{S}")
            st.latex(r"LOQ \approx \frac{10 \times \sigma}{S}")
            st.markdown("The factor of 10 for LOQ is the standard convention that typically yields a precision of roughly 10% CV for a well-behaved assay.")

def render_linearity():
    """Renders the INTERACTIVE module for Linearity analysis."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To verify that an assay's response is directly and predictably proportional to the known concentration of the analyte across its entire intended operational range.
    
    **Strategic Application:** This is a cornerstone of quantitative assay validation. A method exhibiting non-linearity might be accurate at a central control point but dangerously inaccurate at the specification limits. **Use the sliders in the sidebar to simulate different types of non-linear behavior and error.**
    """)

    st.info("""
    **Interactive Demo:** Now, when you navigate to the "Linearity & Range" tool, you will see a new set of dedicated sliders in the sidebar. You can now dynamically simulate how a perfect assay, one with detector saturation, or one with increasing error at higher concentrations would appear in a validation report, providing a powerful learning experience.
    """)

    st.sidebar.subheader("Linearity Controls")
    curvature_slider = st.sidebar.slider(
        "🧬 Curvature Effect",
        min_value=-5.0, max_value=5.0, value=-1.0, step=0.5,
        help="Simulates non-linearity. A negative value creates saturation at high concentrations. A positive value creates expansion. Zero is perfectly linear."
    )
    random_error_slider = st.sidebar.slider(
        "🎲 Random Error (Constant SD)",
        min_value=0.1, max_value=5.0, value=1.0, step=0.1,
        help="The baseline random noise of the assay, constant across all concentrations."
    )
    proportional_error_slider = st.sidebar.slider(
        "📈 Proportional Error (% of Conc.)",
        min_value=0.0, max_value=5.0, value=2.0, step=0.25,
        help="Error that increases with concentration. This creates a 'funnel' or 'megaphone' shape in the residual plot."
    )

    fig, model = plot_linearity(
        curvature=curvature_slider,
        random_error=random_error_slider,
        proportional_error=proportional_error_slider
    )

    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ Acceptance Criteria", "📖 Theory & History"])
        with tabs[0]:
            st.metric(label="📈 KPI: R-squared (R²)", value=f"{model.rsquared:.4f}", help="Indicates the proportion of variance explained by the model. Note how a high R² can hide clear non-linearity!")
            st.metric(label="💡 Metric: Slope", value=f"{model.params[1]:.3f}", help="Ideal = 1.0.")
            st.metric(label="💡 Metric: Y-Intercept", value=f"{model.params[0]:.2f}", help="Ideal = 0.0.")

            st.info("Play with the sliders in the sidebar to see how different errors affect the diagnostic plots.")
            st.markdown("""
            - **The Residual Plot is Key:** This is the most sensitive diagnostic tool.
                - Add **Curvature**: Notice the classic "U-shape" or "inverted U-shape" that appears. This is a dead giveaway that your straight-line model is wrong.
                - Add **Proportional Error**: Watch the residuals form a "funnel" or "megaphone" shape. This is heteroscedasticity, and it means you should be using Weighted Least Squares (WLS) regression, not OLS.

            **The Core Strategic Insight:** A high R-squared is **not sufficient** to prove linearity. You must visually inspect the residual plot for hidden patterns. The residual plot tells the true story of your model's fit.
            """)

        with tabs[1]:
            st.markdown("These criteria are defined in the validation protocol and must be met to declare the method linear.")
            st.markdown("- **R-squared (R²):** Typically > **0.995**, but for high-precision methods (e.g., HPLC), > **0.999** is often required.")
            st.markdown("- **Slope & Intercept:** The 95% confidence intervals for the slope and intercept should contain 1.0 and 0, respectively.")
            st.markdown("- **Residuals:** There should be no obvious pattern or trend. A formal **Lack-of-Fit test** can be used for objective proof (requires true replicates at each level).")
            st.markdown("- **Recovery:** The percent recovery at each concentration level must fall within a pre-defined range (e.g., 80% to 120% for bioassays).")

        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            The mathematical engine is **Ordinary Least Squares (OLS) Regression**, developed by **Legendre (1805)** and **Gauss (1809)**. The genius of OLS is that it finds the one line that **minimizes the sum of the squared vertical distances (the "residuals")** between the data points and the line.

            #### Mathematical Basis
            The goal is to fit a simple linear model:
            """)
            st.latex("y = \\beta_0 + \\beta_1 x + \\epsilon")
            st.markdown("""
            - $y$: The measured response.
            - $x$: The nominal (true) concentration.
            - $\\beta_0$ (Intercept): Constant systematic error.
            - $\\beta_1$ (Slope): Proportional systematic error.
            - $\\epsilon$: Random measurement error.
            """)

def render_4pl_regression():
    """Renders the INTERACTIVE module for 4-Parameter Logistic (4PL) regression."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To accurately model the characteristic sigmoidal (S-shaped) dose-response relationship found in most immunoassays (e.g., ELISA) and biological assays.
    
    **Strategic Application:** This is the workhorse model for potency assays and immunoassays. The 4PL model allows for the accurate calculation of critical assay parameters like the EC50. **Use the sliders in the sidebar to control the "true" shape of the curve and see how the curve-fitting algorithm performs.**
    """)

    st.info("""
    **Interactive Demo:** Now, when you select the "Non-Linear Regression" tool, you will have a full set of dedicated sliders in the sidebar. You can now build your own "true" 4PL curves and see how well the regression algorithm is able to recover those parameters from noisy data, providing a deep, intuitive feel for how these models work.
    """)

    st.sidebar.subheader("4PL Curve Controls")
    d_slider = st.sidebar.slider(
        "🅾️ Lower Asymptote (d)", min_value=0.0, max_value=0.5, value=0.05, step=0.01,
        help="The 'floor' of the assay signal, often representing background noise."
    )
    a_slider = st.sidebar.slider(
        "🅰️ Upper Asymptote (a)", min_value=1.0, max_value=3.0, value=1.5, step=0.1,
        help="The 'ceiling' of the assay signal, representing saturation."
    )
    c_slider = st.sidebar.slider(
        "🎯 Potency / EC50 (c)", min_value=1.0, max_value=100.0, value=10.0, step=1.0,
        help="The concentration at the curve's midpoint. A lower EC50 means higher potency."
    )
    b_slider = st.sidebar.slider(
        "🅱️ Hill Slope (b)", min_value=0.5, max_value=5.0, value=1.2, step=0.1,
        help="The steepness of the curve. A steeper slope often means a more sensitive assay."
    )

    fig, params = plot_4pl_regression(
        a_true=a_slider,
        b_true=b_slider,
        c_true=c_slider,
        d_true=d_slider
    )

    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        a_fit, b_fit, c_fit, d_fit = params
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History"])

        with tabs[0]:
            st.metric(label="🅰️ Fitted Upper Asymptote (a)", value=f"{a_fit:.3f}")
            st.metric(label="🅱️ Fitted Hill Slope (b)", value=f"{b_fit:.3f}")
            st.metric(label="🎯 Fitted EC50 (c)", value=f"{c_fit:.3f} units")
            st.metric(label="🅾️ Fitted Lower Asymptote (d)", value=f"{d_fit:.3f}")

            st.info("Play with the sliders in the sidebar to change the true curve and see how the fitted parameters (above) respond!")
            st.markdown("""
            - **Asymptotes (a & d):** These sliders control the dynamic range of your assay.
            - **EC50 (c):** This is your main potency result. Moving this slider shifts the entire curve left or right.
            - **Hill Slope (b):** This slider controls the steepness. A steep slope means a narrow, sensitive range.

            **The Core Strategic Insight:** The 4PL curve is a complete picture of your assay's performance. Monitoring all four parameters over time enables proactive troubleshooting.
            """)

        with tabs[1]:
            st.error("""
            🔴 **THE INCORRECT APPROACH: "Force the Fit"**
            - *"My data isn't S-shaped, so I'll use linear regression on the middle."* (Biases results).
            - *"The model doesn't fit a point well. I'll delete the point."* (Data manipulation).
            - *"My R-squared is 0.999, so the fit must be perfect."* (R-squared is easily inflated).
            """)
            st.success("""
            🟢 **THE GOLDEN RULE: Model the Biology, Weight the Variance**
            - **Embrace the 'S' Shape:** Use a non-linear model for non-linear data.
            - **Weight Your Points:** Apply less "weight" to more variable data points for a more robust fit.
            - **Look at the Residuals:** Any pattern in the errors indicates the model is not capturing the data correctly.
            """)

        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            The 4PL model is a direct descendant of the **Hill Equation** (1910) by Archibald Hill. It was adapted in the 1970s-80s for immunoassays like ELISA.

            #### Mathematical Basis
            """)
            st.latex(r"y = d + \frac{a - d}{1 + (\frac{x}{c})^b}")
            st.markdown("""
            - **`y`**: The measured response.
            - **`x`**: The concentration.
            - **`a`**: Upper asymptote.
            - **`d`**: Lower asymptote.
            - **`c`**: EC50 (potency).
            - **`b`**: Hill slope.
            """)

def render_roc_curve():
    """Renders the INTERACTIVE module for Receiver Operating Characteristic (ROC) curve analysis."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To solve **The Diagnostician's Dilemma**: a test must correctly identify patients with a disease (high **Sensitivity**) while also correctly clearing healthy patients (high **Specificity**). The ROC curve visualizes this trade-off.
    
    **Strategic Application:** This is the global standard for validating diagnostic tests. The Area Under the Curve (AUC) provides a single metric of a test's diagnostic power. **Use the sliders in the sidebar to see how population separation and overlap affect diagnostic performance.**
    """)

    st.info("""
    **Interactive Demo:** Now, when you select the "ROC Curve Analysis" tool, you will see the new dedicated sliders in the sidebar. You can dynamically create assays that are excellent (high separation, low overlap) or terrible (low separation, high overlap) and see in real-time how the score distributions, the ROC curve shape, and the final AUC value respond.
    """)

    st.sidebar.subheader("ROC Curve Controls")
    separation_slider = st.sidebar.slider(
        "📈 Separation (Diseased Mean)",
        min_value=50.0, max_value=80.0, value=65.0, step=1.0,
        help="Controls the distance between the Healthy and Diseased populations. More separation = better test."
    )
    overlap_slider = st.sidebar.slider(
        "🌫️ Overlap (Population SD)",
        min_value=5.0, max_value=20.0, value=10.0, step=0.5,
        help="Controls the 'noise' or spread of the populations. More overlap (a higher SD) = worse test."
    )

    fig, auc_value = plot_roc_curve(
        diseased_mean=separation_slider,
        population_sd=overlap_slider
    )

    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History"])

        with tabs[0]:
            st.metric(label="📈 KPI: Area Under Curve (AUC)", value=f"{auc_value:.3f}", help="The overall diagnostic power of the test. 0.5 is useless, 1.0 is perfect. Updates with sliders.")
            st.info("Play with the sliders in the sidebar to see how assay quality affects the results!")
            st.markdown("""
            - **Increase `Separation`:** Watch the red distribution move away from the blue one. The ROC curve pushes towards the perfect top-left corner, and the **AUC value increases dramatically.**
            - **Increase `Overlap`:** Watch both distributions get wider. The ROC curve flattens, and the **AUC value decreases.**

            **The Core Strategic Insight:** A great diagnostic test is one that maximizes the separation between populations while minimizing their overlap (noise).
            """)

        with tabs[1]:
            st.error("""
            🔴 **THE INCORRECT APPROACH: "Worship the AUC" & "Hug the Corner"**
            - *"My AUC is 0.95, so we're done."* (The *chosen cutoff* might still be terrible).
            - *"I'll just pick the cutoff closest to the top-left corner."* (This balances errors equally, which is rarely desired).
            """)
            st.success("""
            🟢 **THE GOLDEN RULE: The Best Cutoff Depends on the Consequence of Being Wrong**
            Ask: **"What is worse? A false positive or a false negative?"**
            - **For deadly disease screening:** Maximize Sensitivity.
            - **For risky surgery diagnosis:** Maximize Specificity.
            """)

        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            The ROC curve was invented during **World War II** to help radar operators distinguish enemy bombers from noise. It allowed them to quantify the trade-off between sensitivity and false alarms.

            #### Mathematical Basis
            The curve plots **Sensitivity (Y-axis)** versus **1 - Specificity (X-axis)**.
            """)
            st.latex(r"\text{Sensitivity} = \frac{TP}{TP + FN} \quad , \quad \text{Specificity} = \frac{TN}{TN + FP}")

def render_tost():
    """Renders the INTERACTIVE module for Two One-Sided Tests (TOST) for equivalence."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To statistically prove that two methods or groups are **equivalent** within a predefined, practically insignificant margin. This flips the logic of standard hypothesis testing.
    
    **Strategic Application:** This is the statistically rigorous way to handle comparisons where the goal is to prove similarity, not difference, such as in biosimilarity studies or method transfers. **Use the sliders in the sidebar to build an intuition for what drives an equivalence conclusion.**
    """)

    st.info("""
    **Interactive Demo:** Now, when you select the "Equivalence Testing (TOST)" tool, you will have a full set of dedicated sliders in the sidebar. You can now dynamically explore how to achieve (or fail to achieve) statistical equivalence, providing a powerful and memorable learning experience.
    """)

    st.sidebar.subheader("TOST Controls")
    delta_slider = st.sidebar.slider(
        "⚖️ Equivalence Margin (Δ)",
        min_value=1.0, max_value=15.0, value=5.0, step=0.5,
        help="The 'goalposts'. Defines the zone where differences are considered practically meaningless. A tighter margin is harder to meet."
    )
    diff_slider = st.sidebar.slider(
        "🎯 True Difference",
        min_value=-10.0, max_value=10.0, value=1.0, step=0.5,
        help="The actual underlying difference between the two groups in the simulation. See if you can prove equivalence even when a small true difference exists!"
    )
    sd_slider = st.sidebar.slider(
        "🌫️ Standard Deviation (Variability)",
        min_value=1.0, max_value=15.0, value=5.0, step=0.5,
        help="The random noise or imprecision in the data. Higher variability widens the confidence interval, making equivalence harder to prove."
    )
    n_slider = st.sidebar.slider(
        "🔬 Sample Size (n)",
        min_value=10, max_value=200, value=50, step=5,
        help="The number of samples per group. Higher sample size narrows the confidence interval, increasing your power to prove equivalence."
    )

    fig, p_tost, is_equivalent, ci_lower, ci_upper = plot_tost(
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
        tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History"])

        with tabs[0]:
            st.metric(label="⚖️ Equivalence Margin (Δ)", value=f"± {delta_slider:.1f} units", help="The pre-defined 'zone of indifference'.")
            st.metric(label="📊 Observed 90% CI for Difference", value=f"[{ci_lower:.2f}, {ci_upper:.2f}]", help="The 90% confidence interval for the true difference between the groups.")
            st.metric(label="p-value (TOST)", value=f"{p_tost:.4f}", help="If p < 0.05, we conclude equivalence.")

            status = "✅ Equivalent" if is_equivalent else "❌ Not Equivalent"
            if is_equivalent: st.success(f"### Status: {status}")
            else: st.error(f"### Status: {status}")

            st.info("Play with the sliders in the sidebar to see how they affect the conclusion!")
            st.markdown("""
            - **The Goal:** To get the **entire blue bar (90% CI)** to fall inside the **green equivalence zone**.
            - **`True Difference`:** Move this slider to see how the position of the blue bar changes.
            - **`Standard Deviation`:** Increasing this widens the blue bar, making it fail.
            - **`Sample Size`:** Increasing this narrows the blue bar, making it pass. This shows the power of collecting more data.
            - **`Equivalence Margin`:** This moves the red goalposts. A tight margin is a high bar to clear.
            """)

        with tabs[1]:
            st.error("""
            🔴 **THE INCORRECT APPROACH: The Fallacy of the Non-Significant P-Value**
            - A scientist runs a standard t-test and gets a p-value of 0.25. They exclaim, *"Great, p > 0.05, so the methods are the same!"*
            - **This is wrong.** All they have shown is a *failure to find evidence of a difference*. **Absence of evidence is not evidence of absence.**
            """)
            st.success("""
            🟢 **THE GOLDEN RULE: Define 'Same Enough', Then Prove It**
            The TOST procedure forces a more rigorous scientific approach.
            1.  **First, Define the Margin:** Before collecting data, stakeholders must define the equivalence margin (the green zone).
            2.  **Then, Prove You're Inside:** Conduct the experiment. The burden of proof is on you to show that your evidence (the 90% CI) is strong enough to fall entirely within that margin.
            """)

        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            The TOST procedure rose to prominence in the 1980s, driven by the **1984 Hatch-Waxman Act** which created the modern pathway for generic drug approval. Regulators needed a way to statistically *prove* a generic drug was bioequivalent to the original. **Donald J. Schuirmann's** Two One-Sided Tests (TOST) procedure became the statistical engine for these critical studies.

            #### Mathematical Basis
            TOST brilliantly flips the null hypothesis. Instead of one null of "no difference," you have two nulls of "too different":
            """)
            st.latex(r"H_{01}: \mu_{Test} - \mu_{Ref} \leq -\Delta \quad , \quad H_{02}: \mu_{Test} - \mu_{Ref} \geq +\Delta")
            st.markdown("You must reject **both** of these null hypotheses to conclude that the true difference lies within the equivalence margin `[-Δ, +Δ]`.")

def render_assay_robustness_doe():
    """Renders the comprehensive, interactive module for Assay Robustness (DOE/RSM)."""
    st.markdown("""
    #### Purpose & Application: Process Cartography - The GPS for Optimization
    **Purpose:** To create a detailed topographical map of your process landscape. This analysis moves beyond simple robustness checks to full **process optimization**, using **Response Surface Methodology (RSM)** to model curvature and find the true "peak of the mountain."
    
    **Strategic Application:** This is the statistical engine for Quality by Design (QbD) and process characterization. By developing a predictive model, you can:
    - **Find Optimal Conditions:** Identify the exact settings that maximize yield, efficacy, or any other Critical Quality Attribute (CQA).
    - **Define a Design Space:** Create a multi-dimensional "safe operating zone" where the process is guaranteed to produce acceptable results. This is highly valued by regulatory agencies.
    - **Minimize Variability:** Find a "robust plateau" on the response surface where performance is not only high, but also insensitive to small variations in input parameters.
    """)

    st.info("""
    **Interactive Demo:** You are the process expert. Use the sliders in the sidebar to define the "true" physics of a virtual assay. The plots will show how a DOE/RSM experiment can uncover this underlying response surface, allowing you to find the optimal operating conditions.
    """)

    st.sidebar.subheader("DOE / RSM Controls")
    st.sidebar.markdown("**Linear & Interaction Effects**")
    ph_slider = st.sidebar.slider("🧬 pH Main Effect", -10.0, 10.0, 2.0, 1.0, help="The 'true' linear impact of pH. A high value 'tilts' the surface along the pH axis.")
    temp_slider = st.sidebar.slider("🌡️ Temperature Main Effect", -10.0, 10.0, 5.0, 1.0, help="The 'true' linear impact of Temperature. A high value 'tilts' the surface along the Temp axis.")
    interaction_slider = st.sidebar.slider("🔄 pH x Temp Interaction Effect", -10.0, 10.0, 0.0, 1.0, help="The 'true' interaction. A non-zero value 'twists' the surface, creating a rising ridge.")

    st.sidebar.markdown("**Curvature (Quadratic) Effects**")
    ph_quad_slider = st.sidebar.slider("🧬 pH Curvature", -10.0, 10.0, -5.0, 1.0, help="A negative value creates a 'hill' (a peak). A positive value creates a 'bowl' (a valley). This is the key to optimization.")
    temp_quad_slider = st.sidebar.slider("🌡️ Temperature Curvature", -10.0, 10.0, -5.0, 1.0, help="A negative value creates a 'hill' (a peak). A positive value creates a 'bowl' (a valley).")

    st.sidebar.markdown("**Experimental Noise**")
    noise_slider = st.sidebar.slider("🎲 Random Noise (SD)", 0.1, 5.0, 1.0, 0.1, help="The inherent variability of the assay. High noise can hide the true effects.")

    fig_contour, fig_3d, fig_effects, params = plot_doe_robustness(
        ph_effect=ph_slider, temp_effect=temp_slider, interaction_effect=interaction_slider,
        ph_quad_effect=ph_quad_slider, temp_quad_effect=temp_quad_slider, noise_sd=noise_slider
    )

    st.header("Response Surface Plots")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_contour, use_container_width=True)
    with col2:
        st.plotly_chart(fig_3d, use_container_width=True)

    st.header("Effect Plots & Interpretation")
    col3, col4 = st.columns([0.7, 0.3])
    with col3:
        st.plotly_chart(fig_effects, use_container_width=True)
    with col4:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History"])

        with tabs[0]:
            pred_z = fig_3d.data[0].z
            max_idx = np.unravel_index(np.argmax(pred_z), pred_z.shape)
            opt_temp = fig_3d.data[0].y[max_idx[0]]
            opt_ph = fig_3d.data[0].x[max_idx[1]]
            max_response = np.max(pred_z)

            st.metric("Predicted Optimal pH", f"{opt_ph:.2f}")
            st.metric("Predicted Optimal Temp", f"{opt_temp:.2f}")
            st.metric("Predicted Max Response", f"{max_response:.1f} units")

            st.info("Play with the sliders and observe how the 'true' physics you define are reflected in the plots!")
            st.markdown("""
            - **Linear Effects:** Increasing a `Main Effect` slider is like **tilting the entire surface**. A high positive value for Temperature makes the response universally higher at high temperatures.
            - **Interaction Effects:** A non-zero `Interaction Effect` **twists the surface**, creating a "rising ridge." The effect of pH is now different at high vs. low temperatures. The lines in the Interaction Plot become non-parallel.
            - **Curvature Effects:** The `Curvature` sliders are the key to optimization. Setting them to negative values creates a **peak or dome** in the 3D surface, which corresponds to the "bullseye" of concentric circles in the 2D contour plot. This is the optimal zone you are trying to find.
            - **Noise:** Increasing `Random Noise` makes the red data points scatter further from the true underlying surface, making it harder for the model to accurately map the landscape.
            """)

        with tabs[1]:
            st.error("""
            🔴 **THE INCORRECT APPROACH: One-Factor-at-a-Time (OFAT)**
            Imagine trying to find the highest point on a mountain by only walking in straight lines, first due North-South, then due East-West. You will almost certainly end up on a ridge or a local hill, convinced it's the summit, while the true peak was just a few steps to the northeast.
            
            - **The Flaw:** This is what OFAT does. It is statistically inefficient and, more importantly, it is **guaranteed to miss the true optimum** if any interaction between the factors exists.
            """)
            st.success("""
            🟢 **THE GOLDEN RULE: Map the Entire Territory at Once (DOE/RSM)**
            By testing factors in combination using a dedicated design (like a Central Composite Design), you send out scouts to explore the entire landscape simultaneously. This allows you to:
            1.  **Be Highly Efficient:** Gain more information from fewer experimental runs compared to OFAT.
            2.  **Understand the Terrain:** Uncover and quantify critical interaction and curvature effects that describe the true shape of the process space.
            3.  **Find the True Peak:** Develop a predictive mathematical model that acts as a GPS, guiding you directly to the optimal operating conditions.
            """)

        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            - **The Genesis (1920s):** DOE was invented by **Sir Ronald A. Fisher** to screen for important factors in agriculture. His factorial designs were brilliant for figuring out *which* factors mattered.
            - **The Optimization Revolution (1950s):** The post-war chemical industry boom created a new need: not just to know *which* factors mattered, but *how to find their optimal settings*. **George Box** and K.B. Wilson developed **Response Surface Methodology (RSM)** to solve this. They created efficient new designs, like the Central Composite Design (CCD) shown here, which cleverly add "axial" points to a factorial design. These extra points allow for the fitting of a **quadratic model**, which is the key to modeling curvature and finding the "peak of the mountain." This moved DOE from simple screening to true, powerful optimization.
            
            #### Mathematical Basis
            RSM typically fits a second-order (quadratic) model to the experimental data:
            """)
            st.latex(r"Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + \beta_{11}X_1^2 + \beta_{22}X_2^2 + \beta_{12}X_1X_2 + \epsilon")
            st.markdown(r"""
            - $\beta_0$: The intercept or baseline response.
            - $\beta_1, \beta_2$: The **linear main effects** (the tilt of the surface).
            - $\beta_{11}, \beta_{22}$: The **quadratic effects** (the curvature or "hill/bowl" shape).
            - $\beta_{12}$: The **interaction effect** (the twist of the surface).
            - $\epsilon$: The random experimental error.
            """)

def render_causal_inference():
    """Renders the INTERACTIVE module for Causal Inference."""
    st.markdown("""
    #### Purpose & Application: Beyond the Shadow - The Science of "Why"
    **Purpose:** To move beyond mere correlation ("what") and ascend to the level of causation ("why"). While predictive models see shadows on a cave wall (associations), Causal Inference provides the tools to understand the true objects casting them (the underlying causal mechanisms).
    
    **Strategic Application:** This is the ultimate goal of root cause analysis and the foundation of intelligent intervention.
    - **💡 Effective CAPA:** Why did a batch fail? A predictive model might say high temperature is *associated* with failure. Causal Inference helps determine if high temperature *causes* failure, or if both are driven by a third hidden variable (a "confounder"). This prevents wasting millions on fixing the wrong problem.
    - **🗺️ Process Cartography:** It allows for the creation of a **Directed Acyclic Graph (DAG)**, which is a formal causal map of your process, documenting scientific understanding and guiding future analysis.
    - **🔮 "What If" Scenarios:** It provides a framework to answer hypothetical questions like, "What *would* have been the yield if we had kept the temperature at 40°C?" using only observational data.
    """)

    st.info("""
    **Interactive Demo:** Use the slider in the sidebar to control the **Confounding Strength** of the `Reagent Lot`. As you increase it, watch the "Naive Correlation" (the orange line) become a terrible estimate of the "True Causal Effect" (the green line). This simulation visually demonstrates how a hidden variable can create a misleading correlation.
    """)

    st.sidebar.subheader("Causal Inference Controls")
    confounding_slider = st.sidebar.slider(
        "🚨 Confounding Strength",
        min_value=0.0, max_value=10.0, value=5.0, step=0.5,
        help="How strongly the 'Reagent Lot' affects BOTH Temperature and Purity. At 0, the naive correlation equals the true causal effect."
    )

    fig_dag, fig_scatter, naive_effect, adjusted_effect = plot_causal_inference(confounding_strength=confounding_slider)

    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig_dag, use_container_width=True)
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History"])

        with tabs[0]:
            st.metric(label="Biased Estimate (Naive Correlation)", value=f"{naive_effect:.3f}", help="The effect you would conclude by just plotting Purity vs. Temp. This is misleading!")
            st.metric(label="Unbiased Estimate (True Causal Effect)", value=f"{adjusted_effect:.3f}", help="The true effect of Temp on Purity after adjusting for the confounder. Note how this stays stable.")

            st.info("Play with the 'Confounding Strength' slider and watch the metrics and plots!")
            st.markdown("""
            - **The DAG (Top Plot):** This is our "causal map." It shows that `Reagent Lot` is a **common cause** of both `Temp` and `Purity`, creating a "backdoor" path that biases the `Temp -> Purity` relationship.
            - **The Scatter Plot:** As you increase `Confounding Strength`, the orange line (naive correlation) becomes a worse and worse estimate of the green line (the true causal effect). The data points separate into two clouds (one for each reagent lot), and the naive model incorrectly draws a line through them. The adjusted model correctly finds the true, steeper negative trend *within* each group.
            """)

        with tabs[1]:
            st.error("""
            🔴 **THE INCORRECT APPROACH: The Correlation Trap**
            - An analyst observes that ice cream sales are highly correlated with shark attacks. They recommend banning ice cream to improve beach safety.
            - **The Flaw:** They failed to account for a confounder: **Hot Weather.** Hot weather causes more people to buy ice cream AND causes more people to go swimming. Causal inference provides the tools to mathematically "control for" the weather to see that ice cream has no real effect.
            """)
            st.success("""
            🟢 **THE GOLDEN RULE: Draw the Map, Find the Path, Block the Backdoors**
            A robust causal analysis follows a disciplined, three-step process.
            1.  **Draw the Map (Build the DAG):** This is a collaborative effort between data scientists and Subject Matter Experts. You must encode all your domain knowledge and causal beliefs into a formal DAG.
            2.  **Find the Path:** Clearly identify the causal path you want to measure (e.g., `Temp -> Purity`).
            3.  **Block the Backdoors:** Use the DAG to identify all non-causal "backdoor" paths (confounding). Then, use the appropriate statistical technique (like multiple regression) to "block" these paths, leaving only the true causal effect.
            """)

        with tabs[2]:
            st.markdown("""
            #### Historical Context: The Causal Revolution
            **The Problem:** For most of the 20th century, mainstream statistics was deeply allergic to the language of causation. The mantra, famously drilled into every student, was **"correlation is not causation."** While true, this left a massive void: if correlation isn't the answer, what is? Statisticians were excellent at describing relationships but had no formal language to discuss *why* those relationships existed, leaving a critical gap between data and real-world action.
            
            **The "Aha!" Moment:** The revolution was sparked by the computer scientist and philosopher **Judea Pearl** in the 1980s and 90s. His key insight was that the missing ingredient was **structure**. He argued that scientists carry causal models in their heads all the time, and that these models could be formally written down as graphs. He introduced the **Directed Acyclic Graph (DAG)** as the language for this structure. The arrows in a DAG are not mere correlations; they are bold claims about the direction of causal influence.
            
            **The Impact:** This was a paradigm shift. By making causal assumptions explicit in a DAG, Pearl developed a complete mathematical framework—including his famous **do-calculus**—to determine if a causal question *could* be answered from observational data, and if so, how. This "Causal Revolution" provided the first-ever rigorous, mathematical language to move from seeing (`P(Y|X)`) to doing (`P(Y|do(X))`), transforming fields from epidemiology to economics. For this work, Judea Pearl was awarded the Turing Award in 2011, the highest honor in computer science.
            """)

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
    **Interactive Demo:** Use the controls in the sidebar to inject different types of "special cause" events into a simulated stable process. Observe how the I-MR, Xbar-R, and P-Charts each respond, helping you learn to recognize the visual signatures of common process problems.
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
    tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History"])

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
        st.error("""
        🔴 **THE INCORRECT APPROACH: "Process Tampering"**
        This is the single most destructive mistake in SPC. The operator sees any random fluctuation within the control limits and reacts as if it's a real problem.
        
        - *"This point is a little higher than the last one, I'll tweak the temperature down a bit."*
        - *"This point is below the mean, I'll adjust the flow rate up."*
        
        Reacting to "common cause" noise as if it were a "special cause" signal actually **adds more variation** to the process, making it worse. This is like trying to correct the path of a car for every tiny bump in the road—you'll end up swerving all over the place.
        """)
        st.success("""
        🟢 **THE GOLDEN RULE: Know When to Act (and When Not To)**
        The control chart's signal dictates one of two paths:
        1.  **Process is IN-CONTROL (only common cause variation):**
            - **Your Action:** Leave the process alone! To improve, you must work on changing the fundamental system (e.g., better equipment, new materials).
        2.  **Process is OUT-OF-CONTROL (a special cause is present):**
            - **Your Action:** Stop! Investigate immediately. Find the specific, assignable "special cause" for that signal and eliminate it.
        """)

    with tabs[2]:
        st.markdown("""
        #### Historical Context & Origin
        The control chart was invented by the brilliant American physicist and engineer **Dr. Walter A. Shewhart** while working at Bell Telephone Laboratories in the 1920s. The challenge was immense: manufacturing millions of components for the new national telephone network required unprecedented levels of consistency. How could you know if a variation in a vacuum tube's performance was just normal fluctuation or a sign of a real production problem?

        Shewhart's genius was in his 1924 memo where he introduced the first control chart. He was the first to formally articulate the critical distinction between **common cause** and **special cause** variation. He realized that as long as a process only exhibited common cause variation, it was stable and predictable. The purpose of the control chart was to provide a simple, graphical tool to detect the moment a special cause entered the system. This idea was the birth of modern Statistical Process Control and laid the foundation for the 20th-century quality revolution.

        #### Mathematical Basis
        The control limits on a Shewhart chart are famously set at ±3 standard deviations from the center line.
        """)
        st.latex(r"\text{Control Limits} = \text{Center Line} \pm 3 \times (\text{Standard Deviation of the Plotted Statistic})")
        st.markdown("""
        - **Why 3-Sigma?** Shewhart chose this value for sound economic and statistical reasons. For a normally distributed process, 99.73% of all data points will naturally fall within these limits.
        - **Minimizing False Alarms:** This means there's only a 0.27% chance of a point falling outside the limits purely by chance. This makes the chart robust; when you get a signal, you can be very confident it's real and not just random noise. It strikes an optimal balance between being sensitive to real problems and not causing "fire drills" for false alarms.
        """)

def render_capability():
    """Renders the interactive module for Process Capability (Cpk)."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To quantitatively determine if a process, once proven to be in a state of statistical control, is **capable** of consistently producing output that meets pre-defined specification limits (USL/LSL).
    
    **Strategic Application:** This is the ultimate verdict on process performance, often the final gate in a process validation or technology transfer. It directly answers the critical business question: "Is our process good enough to reliably meet customer or regulatory requirements with a high degree of confidence?" 
    - A high capability index (Cpk) provides objective, statistical evidence that the process is robust, predictable, and delivers high quality.
    - A low Cpk is a clear signal that the process requires fundamental improvement, either by **re-centering the process mean** or by **reducing the process variation**.
    
    In many ways, achieving a high Cpk is the statistical equivalent of "mission accomplished" for a process development or transfer team.
    """)

    st.info("""
    **Interactive Demo:** Use the **Process Scenario** radio buttons in the sidebar to simulate four common real-world process states. Observe how the control chart (stability), the histogram's position relative to the spec limits, and the final Cpk value (capability) change for each scenario. This demonstrates the critical principle that a process must be stable *before* its capability can be meaningfully assessed.
    """)

    scenario = st.sidebar.radio("Select Process Scenario:", ('Ideal', 'Shifted', 'Variable', 'Out of Control'))

    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        fig, cpk_val = plot_capability(scenario)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ Acceptance Criteria", "📖 Theory & History"])
        with tabs[0]:
            st.metric(label="📈 KPI: Process Capability (Cpk)", value=f"{cpk_val:.2f}" if scenario != 'Out of Control' else "INVALID", help="Measures how well the process fits within the spec limits, accounting for centering. Higher is better.")
            st.markdown("""
            - **The Mantra: Control Before Capability.** The control chart (top plot) is a prerequisite. The Cpk metric is only statistically valid and meaningful if the process is stable and in-control. The 'Out of Control' scenario yields an **INVALID** Cpk because an unstable process has no single, predictable "voice" to measure.
            - **The Key Insight: Control ≠ Capability.** A process can be perfectly in-control (predictable) but not capable (producing bad product). 
                - The **'Shifted'** scenario shows a process that is precise but inaccurate.
                - The **'Variable'** scenario shows a process that is centered but imprecise.
            Both are in control, but both have a poor Cpk.
            """)
        with tabs[1]:
            st.markdown("These are industry-standard benchmarks, often required by customers, especially in automotive and aerospace. For pharmaceuticals, a high Cpk in validation provides strong assurance of lifecycle performance.")
            st.markdown("- `Cpk < 1.00`: Process is **not capable**.")
            st.markdown("- `1.00 ≤ Cpk < 1.33`: Process is **marginally capable**.")
            st.markdown("- `Cpk ≥ 1.33`: Process is considered **capable** (a '4-sigma' quality level).")
            st.markdown("- `Cpk ≥ 1.67`: Process is considered **highly capable** (approaching 'Six Sigma').")
            st.markdown("- `Cpk ≥ 2.00`: Process has achieved **Six Sigma capability**.")

        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            The concept of comparing process output to specification limits is old, but the formalization into capability indices originated in the Japanese manufacturing industry in the 1970s as a core part of Total Quality Management (TQM).
            
            However, it was the **Six Sigma** initiative, pioneered by engineer Bill Smith at **Motorola in the 1980s**, that catapulted Cpk to global prominence. The 'Six Sigma' concept was born: a process so capable that the nearest specification limit is at least six standard deviations away from the process mean. Cpk became the standard metric for measuring progress toward this ambitious goal.
            
            #### Mathematical Basis
            Capability analysis is a direct comparison between the **"Voice of the Customer"** (the allowable spread, USL - LSL) and the **"Voice of the Process"** (the actual, natural spread, conventionally 6σ).

            - **Cp (Potential Capability):** Measures if the process is narrow enough, ignoring centering.
            """)
            st.latex(r"C_p = \frac{USL - LSL}{6\hat{\sigma}}")
            st.markdown("- **Cpk (Actual Capability):** The more important metric, as it accounts for process centering. It measures the distance from the process mean to the *nearest* specification limit.")
            st.latex(r"C_{pk} = \min \left( \frac{USL - \bar{x}}{3\hat{\sigma}}, \frac{\bar{x} - LSL}{3\hat{\sigma}} \right)")

def render_tolerance_intervals():
    """Renders the INTERACTIVE module for Tolerance Intervals."""
    st.markdown("""
    #### Purpose & Application: The Quality Engineer's Secret Weapon
    **Purpose:** To construct an interval that we can claim, with a specified level of confidence, contains a certain proportion of all individual values from a process.
    
    **Strategic Application:** This is often the most critical statistical interval in manufacturing. It directly answers the high-stakes question: **"Based on this sample, what is the range where we can expect almost all of our individual product units to fall?"**
    """)

    st.info("""
    **Interactive Demo:** Use the sliders in the sidebar to explore the trade-offs in tolerance intervals. This simulation demonstrates how sample size and the desired quality guarantee (coverage) directly impact the calculated interval, which in turn affects process specifications and batch release decisions.
    """)

    st.sidebar.subheader("Tolerance Interval Controls")
    n_slider = st.sidebar.slider(
        "🔬 Sample Size (n)",
        min_value=10, max_value=200, value=30, step=10,
        help="The number of samples collected. More samples lead to a narrower, more reliable interval."
    )
    coverage_slider = st.sidebar.select_slider(
        "🎯 Desired Population Coverage",
        options=[90.0, 95.0, 99.0, 99.9],
        value=99.0,
        help="The 'quality promise'. What percentage of all future parts do you want this interval to contain? A higher promise requires a wider interval."
    )

    fig, ci, ti = plot_tolerance_intervals(n=n_slider, coverage_pct=coverage_slider)

    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History"])

        with tabs[0]:
            st.metric(label="🎯 Desired Coverage", value=f"{coverage_slider:.1f}% of Population", help="The proportion of the entire process output we want our interval to contain.")
            st.metric(label="📏 Resulting Tolerance Interval", value=f"[{ti[0]:.1f}, {ti[1]:.1f}]", help="The final calculated range. Note how much wider it is than the CI.")

            st.info("Play with the sliders in the sidebar and observe the results!")
            st.markdown("""
            - **Increase `Sample Size (n)`:** As you collect more data, your estimates of the mean and standard deviation become more reliable. Notice how both the **Confidence Interval (orange)** and the **Tolerance Interval (green)** become **narrower**. This shows the direct link between sampling cost and statistical precision.
            - **Increase `Desired Population Coverage`:** As you increase the strength of your quality promise from 90% to 99.9%, the **Tolerance Interval becomes dramatically wider**. To be more certain of capturing a larger percentage of parts, you must widen your interval.
            """)

        with tabs[1]:
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

        with tabs[2]:
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

def render_method_comparison():
    """Renders the INTERACTIVE module for Method Comparison."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To formally assess and quantify the degree of agreement and systemic bias between two different measurement methods intended to measure the same quantity.
    
    **Strategic Application:** This study is the "crucible" of method transfer, validation, or replacement. It answers the critical business and regulatory question: “Do these two methods produce the same result, for the same sample, within medically or technically acceptable limits?”
    """)

    st.info("""
    **Interactive Demo:** Use the sliders in the sidebar to simulate different types of disagreement between a "Test" method and a "Reference" method. See in real-time how each diagnostic plot (Deming, Bland-Altman, %Bias) reveals a different aspect of the problem, helping you build a deep intuition for method comparison statistics.
    """)

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
        tabs = st.tabs(["💡 Key Insights", "✅ Acceptance Criteria", "📖 Theory & History"])

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
            st.markdown("Acceptance criteria must be pre-defined and clinically/technically justified.")
            st.markdown("- **Deming Regression:** The 95% confidence interval for the **slope must contain 1.0**, and the 95% CI for the **intercept must contain 0**.")
            st.markdown(f"- **Bland-Altman:** The primary criterion is that the **95% Limits of Agreement (`{la:.2f}` to `{ua:.2f}`) must be clinically or technically acceptable**.")
            st.error("""
            **The Correlation Catastrophe:** Never use the correlation coefficient (R²) to assess agreement. Two methods can be perfectly correlated (R²=1.0) but have a huge bias (e.g., one method always reads twice as high).
            """)

        with tabs[2]:
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

def render_pass_fail():
    """Renders the INTERACTIVE module for Pass/Fail (Binomial Proportion) analysis."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To accurately calculate and critically compare confidence intervals for a binomial proportion, which is the underlying statistic for any pass/fail, present/absent, or concordant/discordant outcome.
    
    **Strategic Application:** This is essential for the validation of **qualitative assays** or for agreement studies. The goal is to prove, with a high degree of statistical confidence, that the assay's success rate is above a required performance threshold. The critical challenge, especially with small sample sizes, is that simple textbook methods for calculating confidence intervals (the 'Wald' interval) are dangerously inaccurate.
    """)

    st.info("""
    **Interactive Demo:** Use the sliders in the sidebar to simulate the results of a validation study (e.g., comparing a new test to a gold standard). Observe how sample size and the number of successes dramatically affect the confidence in your result, and see why the 'Wald' interval should almost never be used.
    """)

    st.sidebar.subheader("Pass/Fail Controls")
    n_samples_slider = st.sidebar.slider("Number of Validation Samples (n)", 1, 100, 30, key='wilson_n')
    successes_slider = st.sidebar.slider("Concordant Results (Successes)", 0, n_samples_slider, int(n_samples_slider * 0.95), key='wilson_s')

    fig1_intervals, fig2_coverage = plot_binomial_intervals(successes_slider, n_samples_slider)

    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig1_intervals, use_container_width=True)
        st.plotly_chart(fig2_coverage, use_container_width=True)

    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ Acceptance Criteria", "📖 Theory & History"])

        with tabs[0]:
            st.metric(label="📈 KPI: Observed Rate", value=f"{(successes_slider/n_samples_slider if n_samples_slider > 0 else 0):.2%}", help="The point estimate. Insufficient without a confidence interval.")

            st.info("Play with the sliders in the sidebar and observe the plots!")
            st.markdown("""
            - **CI Comparison (Top Plot):** This plot reveals the dramatic differences between interval methods. 
                - Set the sliders to a perfect score (e.g., 30/30). The **Wald interval collapses to zero width**, an absurd claim of perfect knowledge from a small sample. The Wilson and Clopper-Pearson intervals give a much more honest, wider range.
                - Set the sliders to a low sample size (e.g., 5/5). The Wald interval gives a nonsensical range that goes above 100%!
            - **Coverage Probability (Bottom Plot):** This shows *why* the Wald interval is so bad. Its actual probability of capturing the true value (the red line) is often far below the promised 95% level. The Wilson interval (blue) is much more reliable.

            **The Core Strategic Insight:** Never use the standard Wald (or "Normal Approximation") interval for important decisions. The **Wilson Score interval** provides the best balance of accuracy and interval width for most applications. The **Clopper-Pearson** is the most conservative ("exact") choice, often preferred in regulatory submissions for its guaranteed coverage.
            """)
        with tabs[1]:
            st.markdown("- **The Golden Rule of Binomial Acceptance:** The acceptance criterion must **always be based on the lower bound of the confidence interval**, never on the point estimate.")
            st.markdown("- **Example Criterion:** 'The lower bound of the 95% **Wilson Score** (or Clopper-Pearson) confidence interval for the concordance rate must be greater than or equal to the target of 90%.'")
            st.markdown("- **Sample Size Implication:** This tool powerfully demonstrates why larger sample sizes are needed for high-confidence claims. With a small `n`, even a perfect result (e.g., 20/20 successes) may have a lower confidence bound that fails to meet a high target (like 95%), forcing the study to be repeated with more samples.")
        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            For much of the 20th century, the simple **Wald interval** (named after Abraham Wald) was taught in introductory statistics classes. However, its poor performance was well-known. A famous 1998 paper by Brown, Cai, and DasGupta comprehensively documented its failures and advocated for superior alternatives.
            
            The **Wilson Score Interval** (1927) and the **Clopper-Pearson Interval** (1934) were created to solve this problem.
            - The **Clopper-Pearson** interval is an "exact" method derived from the binomial distribution. It guarantees coverage will never be less than the nominal level, making it conservative (wider).
            - The **Wilson Score** interval is derived by inverting the score test. Its average coverage probability is much closer to the nominal 95% level, making it more accurate and less conservative in practice.
            """)
            st.markdown("#### Mathematical Basis")
            st.markdown("The Wald interval is simply:")
            st.latex(r"\hat{p} \pm z_{\alpha/2} \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}")
            st.markdown("The Wilson Score interval's superior formula is:")
            st.latex(r"CI_{\text{Wilson}} = \frac{1}{1 + z_{\alpha/2}^2/n} \left( \hat{p} + \frac{z_{\alpha/2}^2}{2n} \pm z_{\alpha/2} \sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z_{\alpha/2}^2}{4n^2}} \right)")
            st.markdown("Notice it adds pseudo-successes and failures ($z_{\alpha/2}^2/2$), pulling the center away from 0 or 1. This is what gives it such good performance where the Wald interval fails catastrophically.")

def render_bayesian():
    """Renders the interactive module for Bayesian Inference."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To employ Bayesian inference to formally and quantitatively synthesize existing knowledge (a **Prior** belief) with new experimental data (the **Likelihood**) to arrive at an updated, more robust conclusion (the **Posterior** belief).
    
    **Strategic Application:** This is a paradigm-shifting tool for driving efficient, knowledge-based validation and decision-making. In a traditional (Frequentist) world, every study starts from a blank slate. In the Bayesian world, we can formally leverage what we already know. This is powerful for:
    - **Accelerating Tech Transfer:** Use data from an R&D validation study to form a **strong, informative prior**. This allows the receiving QC lab to demonstrate success with a smaller confirmation study, saving time and resources.
    - **Adaptive Clinical Trials:** Data from an interim analysis can serve as a prior for the final analysis, allowing trials to be stopped early.
    - **Quantifying Belief & Risk:** It provides a natural framework to answer the question: "Given what we already knew, and what this new data shows, what is the probability that the pass rate is actually above 95%?"
    """)
    st.info("""
    **Interactive Demo:** Use the **Prior Belief** radio buttons in the sidebar to simulate how different levels of existing knowledge impact your conclusions. Observe how the final **Posterior (blue curve)** is always a weighted compromise between your initial **Prior (green curve)** and the new **Data (red curve)**. A strong prior will be very influential, while a weak or non-informative prior lets the new data speak for itself.
    """)
    prior_type_bayes = st.sidebar.radio("Select Prior Belief:", ("Strong R&D Prior", "No Prior (Frequentist)", "Skeptical/Regulatory Prior"))

    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        fig, prior_mean, mle, posterior_mean = plot_bayesian(prior_type_bayes)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ Acceptance Criteria", "📖 Theory & History"])
        with tabs[0]:
            st.metric(label="📈 KPI: Posterior Mean Rate", value=f"{posterior_mean:.3f}", help="The final, data-informed belief; a weighted average of the prior and the data.")
            st.metric(label="💡 Prior Mean Rate", value=f"{prior_mean:.3f}", help="The initial belief *before* seeing the new QC data.")
            st.metric(label="💡 Data-only Estimate (MLE)", value=f"{mle:.3f}", help="The evidence from the new QC data alone (the frequentist result).")
            st.markdown("""
            - **Prior (Green Dashed):** Our initial belief about the pass rate. A **Strong Prior** is tall and narrow, representing high confidence from historical data. A **Skeptical Prior** is wide and flat, representing uncertainty.
            - **Likelihood (Red Dotted):** The "voice of the new data." This is the evidence from our new QC runs. Note that it is not a probability distribution.
            - **Posterior (Blue Solid):** The final, updated belief. The posterior is always a **compromise** between the prior and the likelihood, weighted by their respective certainties (the narrowness of their distributions).

            **The Core Strategic Insight:** This simulation demonstrates Bayesian updating in action.
             - With a **Strong R&D Prior**, the new (and slightly worse) QC data barely moves our final belief. The strong prior evidence dominates the small new sample.
             - With a **Skeptical Prior**, our final belief is a true compromise between the skeptical starting point and the new data.
             - With **No Prior**, the posterior is determined almost entirely by the data, and the result mirrors the frequentist conclusion.
            This framework provides a transparent and logical way to cumulate knowledge over time.
            """)
        with tabs[1]:
            st.markdown("- The acceptance criterion is framed in terms of the **posterior distribution** and is probabilistic.")
            st.markdown("- **Example Criterion 1 (Probability Statement):** 'There must be at least a 95% probability that the true pass rate is greater than 90%.' This is calculated by finding the area under the blue posterior curve to the right of the 0.90 threshold.")
            st.markdown("- **Example Criterion 2 (Credible Interval):** 'The lower bound of the **95% Credible Interval** (the central 95% of the blue posterior distribution) must be above the target of 90%.'")
            st.warning("**The Prior is Critical:** The choice of prior is the most controversial and important part of a Bayesian analysis. In a regulated setting, the prior must be transparent, justified by historical data, and pre-specified in the validation protocol. An unsubstantiated, overly optimistic prior would be a major red flag for an auditor.")
        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            The underlying theorem was conceived by the Reverend **Thomas Bayes** in the 1740s. However, for nearly 200 years, Bayesian inference remained a philosophical curiosity, largely overshadowed by the Frequentist school. This was due to philosophical objections to the subjective nature of priors and the computational difficulty of calculating the posterior distribution.
            
            The **"Bayesian Revolution"** began in the late 20th century, driven by powerful computers and simulation algorithms like **Markov Chain Monte Carlo (MCMC)**. These methods allowed scientists to approximate the posterior distribution for incredibly complex models, making Bayesian methods practical for the first time.
            
            #### Mathematical Basis
            Bayes' Theorem is elegantly simple:
            """)
            st.latex(r"P(\theta|D) = \frac{P(D|\theta) \cdot P(\theta)}{P(D)}")
            st.markdown(r"In words: **Posterior = (Likelihood × Prior) / Evidence**")
            st.markdown(r"""
            - $P(\theta|D)$ (Posterior): The probability of our parameter $\theta$ (e.g., the true pass rate) given the new Data D.
            - $P(D|\theta)$ (Likelihood): The probability of observing our Data D, given a specific value of the parameter $\theta$.
            - $P(\theta)$ (Prior): Our initial belief about the distribution of the parameter $\theta$.
            
            For binomial data, the **Beta distribution** is a **conjugate prior**. This means if you start with a Beta prior and have a binomial likelihood, your posterior will also be a Beta distribution.
            - If Prior is Beta($\alpha_{prior}, \beta_{prior}$)
            - And Data is $k$ successes in $n$ trials:
            - Then the Posterior is simply Beta($\alpha_{prior} + k, \beta_{prior} + n - k$).
            The $\alpha$ and $\beta$ parameters can be thought of as "pseudo-counts" of prior successes and failures, which are simply added to the new observed counts.
            """)

def render_multi_rule():
    """Renders the comprehensive, interactive module for Multi-Rule SPC (Westgard Rules)."""
    st.markdown("""
    #### Purpose & Application: The Statistical Detective
    **Purpose:** To serve as a high-sensitivity "security system" for your assay. Instead of one simple alarm, this system uses a combination of rules to detect specific types of problems, catching subtle shifts and drifts long before a catastrophic failure occurs. It dramatically increases the probability of detecting true errors while minimizing false alarms.
    
    **Strategic Application:** This is the global standard for run validation in regulated QC and clinical laboratories. While a basic control chart just looks for "big" errors, the multi-rule system acts as a **statistical detective**, using a toolkit of rules to diagnose different failure modes. Implementing these rules prevents the release of bad data, which is the cornerstone of ensuring patient safety and product quality.
    """)

    st.info("""
    **Interactive Demo:** Use the **Process Scenario** radio buttons in the sidebar to simulate common assay failures. Observe how the control chart changes and which specific Westgard rule is triggered, helping you learn to diagnose problems from your QC data.
    """)

    st.sidebar.subheader("Westgard Scenario Controls")
    scenario = st.sidebar.radio(
        "Select a Process Scenario to Simulate:",
        options=('Stable', 'Complex Failure', 'Large Random Error', 'Systematic Shift', 'Increased Imprecision'),
        captions=[
            "A normal, in-control run for reference.",
            "A run with multiple, distinct issues.",
            "e.g., A single major blunder.",
            "e.g., A new reagent lot causes a bias.",
            "e.g., A faulty pipette causes inconsistency."
        ]
    )
    fig = plot_westgard_scenario(scenario=scenario)

    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History"])

        with tabs[0]:
            if scenario == 'Complex Failure':
                st.metric("🕵️ Run Verdict", "Reject Run")
                st.metric("🚨 Primary Cause", "1-3s Violation")
                st.metric("🧐 Secondary Evidence", "2-2s Violation")
                st.markdown("""
                **The Detective's Findings on this Chart:**
                - 🚨 **The Smoking Gun:** The `1-3s` violation is a clear, unambiguous signal of a major problem. This rule alone forces the rejection of the run.
                - 🧐 **The Developing Pattern:** The `2-2s` violation is a classic sign of **systematic error**. The process has shifted high.
                - **The Core Strategic Insight:** This chart shows two *different* problems. A true statistical detective sees both signals and knows there are two distinct issues to solve.
                """)
            else:
                verdict, rule = ("In-Control", "None") if scenario == 'Stable' else ("Reject Run", "Unknown")
                if scenario == 'Large Random Error': rule = "1-3s Violation"
                elif scenario == 'Systematic Shift': rule = "2-2s Violation"
                elif scenario == 'Increased Imprecision': rule = "R-4s Violation"

                st.metric("🕵️ Run Verdict", verdict)
                st.metric("🚨 Triggered Rule", rule)
                st.markdown(f"The simulation for **{scenario}** triggered the **{rule}** rule. Refer to the table below for a detailed diagnosis.")

            st.markdown("""
            ---
            **The Detective's Rulebook:**
            | Rule Name | Definition | Error Detected | Typical Cause |
            | :--- | :--- | :--- | :--- |
            | **1-3s** | 1 point > 3σ | Random Error (blunder) | Calculation error, wrong reagent, air bubble |
            | **2-2s** | 2 consecutive points > 2σ (same side) | Systematic Error (shift) | New calibrator/reagent lot, instrument issue |
            | **R-4s** | Range between 2 consecutive points > 4σ | Random Error (imprecision) | Inconsistent pipetting, instrument instability |
            | **4-1s** | 4 consecutive points > 1σ (same side) | Systematic Error (drift) | Minor reagent degradation, slow drift |
            | **10-x** | 10 consecutive points on same side of mean | Systematic Error (bias) | Small, persistent bias in the system |
            """)

        with tabs[1]:
            st.error("""
            🔴 **THE INCORRECT APPROACH: The "Re-run & Pray" Mentality**
            This operator sees any alarm, immediately discards the run, and starts over without thinking.
            - They don't use the specific rule (`2-2s` vs `R-4s`) to guide their troubleshooting.
            - They might engage in "testing into compliance" by re-running a control until it passes, a serious compliance violation.
            """)
            st.success("""
            🟢 **THE GOLDEN RULE: The Rule is the First Clue**
            The goal is to treat the specific rule violation as the starting point of a targeted investigation.
            - **Think like a detective:** "The chart shows a `2-2s` violation. This suggests a systematic shift. I should check my calibrators and reagents first, not my pipetting technique."
            - **Document Everything:** The investigation, the root cause, and the corrective action for each rule violation must be documented.
            """)

        with tabs[2]:
            st.markdown("""
            #### Historical Context: From the Factory Floor to the Hospital Bed
            **The Problem:** In the 1920s, the Western Electric company was a manufacturing behemoth, but quality was a nightmare. Engineers were lost in a "fog" of data, constantly "tampering" with the process based on random noise, often making things worse.
            
            **The "Aha!" Moment (Shewhart):** A physicist at Bell Labs, **Dr. Walter A. Shewhart**, had the revolutionary insight to distinguish between "common cause" and "special cause" variation. He invented the control chart to provide a simple, graphical tool to detect the moment a special cause entered the system. The ±3σ limits were chosen for sound economic reasons: to minimize the cost of both false alarms and missed signals.
            
            **The Evolution (Westgard):** Fast forward to the 1970s. **Dr. James O. Westgard** recognized that in clinical labs, where patient lives are at stake, the cost of a missed signal is far higher. He found that Shewhart's single `1-3s` rule wasn't sensitive enough to catch the subtle drifts that could lead to a misdiagnosis. In 1981, he proposed his multi-rule system, an "intelligent" combination of rules designed to maximize the probability of error detection ($P_{ed}$) while maintaining a low probability of false rejection ($P_{fr}$). This engineering trade-off made the system the global standard for medical laboratories.

            #### Mathematical Basis
            The logic is built on the properties of the normal distribution. For a stable process:
            - A point outside **±3σ** is a rare event (occurs ~1 in 370 times by chance). This is a high-confidence signal, hence the **1-3s** rejection rule.
            - A point outside **±2σ** is more common (~1 in 22 times). Seeing *two in a row* on the same side, however, is much rarer ($1/22 \times 1/22 \times 1/2 \approx$ 1 in 968). This makes the **2-2s** rule a powerful detector of systematic shifts with a low false alarm rate.
            """)

# FIX: This function is now standalone and correct.
def render_multivariate_spc():
    """Renders the comprehensive, interactive module for Multivariate SPC."""
    st.markdown("""
    #### Purpose & Application: The Process Doctor
    **Purpose:** To monitor the **holistic state of statistical control** for a process with multiple, correlated parameters. Instead of using an array of univariate charts (like individual nurses reading single vital signs), Multivariate SPC (MSPC) acts as the **head physician**, integrating all information into a single, powerful diagnosis.

    **Strategic Application:** This is an essential methodology for modern **Process Analytical Technology (PAT)** and real-time process monitoring. In complex systems like bioreactors or chromatography, parameters like temperature, pH, pressure, and flow rates are interdependent. A small, coordinated deviation across several parameters—a "stealth shift"—can be invisible to individual charts but represents a significant excursion from the normal operating state. MSPC is designed to detect exactly these events.
    """)

    st.info("""
    **Interactive Demo:** Use the **Process Scenario** radio buttons in the sidebar to simulate different types of multivariate process failures. First, observe the **Scatter Plot**, then see which **Control Chart (T² or SPE)** detects the problem, and finally, check the **Contribution Plot** in the 'Key Insights' tab to diagnose the root cause.
    """)

    st.sidebar.subheader("Multivariate SPC Controls")
    scenario = st.sidebar.radio(
        "Select a Process Scenario to Simulate:",
        ('Stable', 'Shift in Y Only', 'Correlation Break'),
        captions=["A normal, in-control process.", "A 'stealth shift' in one variable.", "An unprecedented event breaks the model."]
    )

    fig_scatter, fig_charts, fig_contrib, t2_ooc, spe_ooc = plot_multivariate_spc(scenario=scenario)

    st.plotly_chart(fig_scatter, use_container_width=True)

    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig_charts, use_container_width=True)
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History"])
        
        with tabs[0]:
            t2_verdict_str = "Out-of-Control" if t2_ooc else "In-Control"
            spe_verdict_str = "Out-of-Control" if spe_ooc else "In-Control"
            
            st.metric("📈 T² Chart Verdict", t2_verdict_str, help="Monitors deviation *within* the normal process model.")
            st.metric("📈 SPE Chart Verdict", spe_verdict_str, help="Monitors deviation *from* the normal process model.")
            
            st.markdown("---")
            st.markdown(f"##### Analysis of the '{scenario}' Scenario:")

            if scenario == 'Stable':
                st.success("The process is stable and in-control. Both the T² and SPE charts show only common cause variation, confirming the process is operating as expected within its normal, correlated state. No diagnostic plot is needed.")
            elif scenario == 'Shift in Y Only':
                st.warning("**Diagnosis: A 'Stealth Shift' has occurred.**")
                st.markdown("""
                1.  **Scatter Plot:** The red points have clearly shifted upwards, but because the correlation is strong, they still fall within the horizontal range of the blue points. A univariate chart for Temperature (X-axis) would likely miss this.
                2.  **T² Chart:** Alarms loudly. It knows the expected Pressure (Y) for a given Temperature (X) and detects this significant deviation from the multivariate mean.
                3.  **SPE Chart:** Remains in-control. The *relationship* between the variables is still intact; the process has just shifted along that known correlation structure.
                4.  **Contribution Plot:** This diagnostic tool confirms the root cause: the T² alarm is driven almost entirely by the **Pressure** variable.
                """)
            elif scenario == 'Correlation Break':
                st.error("**Diagnosis: An Unprecedented Event has occurred.**")
                st.markdown("""
                1.  **Scatter Plot:** The red points have fallen completely *off* the established diagonal correlation line. The average Temperature and Pressure might still be normal, but their relationship is broken.
                2.  **T² Chart:** May remain in-control. Since the points are still relatively close to the center of the data cloud, the T² (which measures distance *within* the model) does not alarm.
                3.  **SPE Chart:** Alarms loudly. The SPE measures the distance *to* the model. Since these points are far from the expected correlation line, the SPE signals a major deviation from the model.
                4.  **Contribution Plot:** This diagnostic tool shows that both variables are contributing to the SPE alarm, confirming the fundamental breakdown of the process model itself.
                """)
            
            if fig_contrib is not None:
                st.markdown("---")
                st.markdown("##### Root Cause Diagnosis")
                st.plotly_chart(fig_contrib, use_container_width=True)
            
            st.markdown("---")
            st.info("**Try This:** Switch between the 'Shift in Y Only' and 'Correlation Break' scenarios to see how the two charts are sensitive to completely different types of process failures.")

        with tabs[1]:
            st.error("""
            🔴 **THE INCORRECT APPROACH: The "Army of Univariate Charts" Fallacy**
            Using dozens of individual charts is doomed to fail due to alarm fatigue and its blindness to "stealth shifts."
            """)
            st.success("""
            🟢 **THE GOLDEN RULE: Detect with T²/SPE, Diagnose with Contributions**
            1.  **Stage 1: Detect.** Use **T² and SPE charts** as your primary health monitors to answer "Is something wrong?"
            2.  **Stage 2: Diagnose.** If a chart alarms, then use **contribution plots** to identify which original variables are responsible for the signal. This is the path to the root cause.
            """)

        with tabs[2]:
            st.markdown("""
            #### Historical Context: The Crisis of Dimensionality
            **The Problem:** In the 1930s, statistics was largely a univariate world. Tools like Student's t-test and Shewhart's control charts were brilliant for analyzing one variable at a time. But scientists and economists were facing increasingly complex problems with dozens of correlated measurements. How could you test if two groups were different, not just on one variable, but across a whole panel of them? A simple t-test on each variable was not only inefficient, it was statistically misleading due to the problem of multiple comparisons.

            **The "Aha!" Moment (Hotelling):** The creator of this powerful technique was **Harold Hotelling**, one of the giants of 20th-century mathematical statistics. His genius was in generalization. He recognized that the squared t-statistic, $t^2 = (\\bar{x} - \\mu)^2 / (s^2/n)$, was a measure of squared distance, normalized by variance. In a 1931 paper, he introduced the **Hotelling's T-squared statistic**, which replaced the univariate terms with their vector and matrix equivalents. It provided a single number that represented the "distance" of a point from the center of a multivariate distribution, elegantly solving the problem of testing multiple means at once while accounting for all their correlations.
            """)
            st.markdown("""
            #### Mathematical Basis
            - **T² (Hotelling's T-Squared):** A measure of the **Mahalanobis distance**, which accounts for the correlation between variables through the **inverse of the sample covariance matrix (`S⁻¹`)**.
            """)
            st.latex(r"T^2 = (\mathbf{x} - \mathbf{\bar{x}})' \mathbf{S}^{-1} (\mathbf{x} - \mathbf{\bar{x}})")
            st.markdown("""
            - **SPE (Squared Prediction Error):** Also known as DModX or Q-statistic. It is the sum of squared residuals after projecting a data point onto the principal component model of the process. For a new point **x**, it is the squared distance to the PCA model plane:
            """)
            st.latex(r"SPE = || \mathbf{x} - \mathbf{P}\mathbf{P}'\mathbf{x} ||^2")
            st.markdown("where **P** is the matrix of PCA loadings (the model directions).")

# FIX: Created a new, standalone rendering function for the EWMA/CUSUM module.
def render_ewma_cusum():
    """Renders the comprehensive, interactive module for Small Shift Detection (EWMA/CUSUM)."""
    st.markdown("""
    #### Purpose & Application: The Process Sentinel
    **Purpose:** To deploy a high-sensitivity monitoring system designed to detect small, sustained shifts in a process mean that would be invisible to a standard Shewhart control chart (like an I-MR or X-bar chart). These charts have "memory," accumulating evidence from past data to find subtle signals.

    **Strategic Application:** This is an essential "second layer" of process monitoring for mature, stable processes where large, sudden failures are rare, but slow, gradual drifts are a significant risk.
    - **🔬 EWMA (The Sentinel):** The Exponentially Weighted Moving Average chart is a robust, general-purpose tool that smoothly weights past observations, making it excellent for detecting the onset of a gradual drift.
    - **🐕 CUSUM (The Bloodhound):** The Cumulative Sum chart is a specialized, high-power tool. It is the fastest possible detector for a shift of a specific magnitude, making it ideal for processes where you want to catch a known, critical shift size as quickly as possible.
    """)

    st.info("""
    **Interactive Demo:** This case study simulates a small but significant **0.75-sigma shift** occurring at sample #20. Compare the three charts below. Notice how the I-Chart (the "beat cop") completely misses the signal, while the memory-based EWMA and CUSUM charts (the "sentinels") sound the alarm, demonstrating their superior sensitivity.
    """)

    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        fig = plot_ewma_cusum_comparison()
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History"])

        with tabs[0]:
            st.metric(label="Shift Size", value="0.75 σ", help="A small, sustained shift was introduced at sample #20.")
            st.metric(label="I-Chart Detection Time", value="> 20 Samples (Failed)", help="The I-Chart never signaled an alarm.", delta_color="inverse")
            st.metric(label="EWMA Detection Time", value="~10 Samples", help="The EWMA signaled an alarm around sample #30.")
            st.metric(label="CUSUM Detection Time", value="~7 Samples", help="The CUSUM signaled an alarm around sample #27.")

            st.markdown("""
            **The Visual Evidence:**
            - **The I-Chart (Top):** This chart is blind to the problem. The small 0.75σ shift is lost in the normal process noise. All points look "in-control," giving a false sense of security.
            - **The EWMA Chart (Middle):** This chart has memory. The weighted average (the blue line) clearly begins to drift upwards after the shift occurs. It smoothly accumulates evidence until it crosses the red control limit, signaling a real change.
            - **The CUSUM Chart (Bottom):** This chart is a "bloodhound" for a specific scent. It accumulates all deviations from the target. Once the process shifts, the `CUSUM High` plot takes off like a rocket, providing the fastest possible signal.

            **The Core Strategic Insight:** Relying only on Shewhart charts creates a significant blind spot. For processes where small, slow drifts in performance are possible (e.g., tool wear, reagent degradation, column aging), EWMA or CUSUM charts are not optional—they are essential for effective process control.
            """)

        with tabs[1]:
            st.error("""
            🔴 **THE INCORRECT APPROACH: "The One-Chart-Fits-All Fallacy"**
            A manager insists on using only I-MR charts for everything because "that's how we've always done it" and they are easy to understand.
            - They miss a slow 1-sigma drift for weeks, producing tons of near-spec material.
            - When a batch finally fails, they are shocked and have no leading indicators to explain why. They have been flying blind.
            """)
            st.success("""
            🟢 **THE GOLDEN RULE: Layer Your Statistical Defenses**
            The goal is to use a combination of charts to create a comprehensive security system for your process.
            - **Use Shewhart Charts (I-MR, X-bar) as your front-line "Beat Cops":** They are simple and unmatched for detecting large, sudden special causes.
            - **Use EWMA or CUSUM as your "Sentinels":** Deploy them alongside Shewhart charts to stand guard against the silent, creeping threats that the beat cops will miss.
            This layered approach provides a complete picture of process stability.
            """)

        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin: Beyond Shewhart
            By the 1950s, industries needed tools to detect smaller process deviations than Shewhart's charts could handle.
            - **CUSUM (1954):** British statistician **E. S. Page** developed the Cumulative Sum chart, optimized to detect a specific shift size in the minimum possible time.
            - **EWMA (1959):** Statistician **S. W. Roberts** proposed the Exponentially Weighted Moving Average chart as a general-purpose alternative, adapting forecasting "smoothing" techniques for process control.
            
            These two inventions marked the second generation of SPC, giving engineers the sensitive tools they needed for increasingly precise manufacturing.
            #### Mathematical Basis
            - **EWMA:** `EWMA_t = λ * Y_t + (1-λ) * EWMA_{t-1}`
              - `λ` (lambda) is the weighting factor (0 < λ ≤ 1). A smaller `λ` gives more weight to past data (longer memory).
            - **CUSUM:** `SH_t = max(0, SH_{t-1} + (Y_t - T) - k)`
              - `T` is the process target, and `k` is a "slack" parameter, typically half the size of the shift you want to detect quickly. The CUSUM only accumulates when a deviation is larger than the slack.
            """)

def render_time_series_analysis():
    """Renders the module for Time Series analysis."""
    st.markdown("""
    #### Purpose & Application: The Watchmaker vs. The Smartwatch
    **Purpose:** To model and forecast time-dependent data by understanding its internal structure, such as trend, seasonality, and autocorrelation. This module compares two powerful philosophies for this task.
    
    **Strategic Application:** This is fundamental for demand forecasting, resource planning, and proactive process monitoring.
    - **⌚ ARIMA (The Classical Watchmaker):** A powerful and flexible "white-box" model. Like a master watchmaker, you must understand every gear (p,d,q parameters), but you get a highly interpretable model that is defensible in regulatory environments and excels at short-term forecasting of stable processes.
    - **📱 Prophet (The Modern Smartwatch):** A modern forecasting tool from Facebook. It's packed with sensors and algorithms to automatically handle complex seasonalities, holidays, and changing trends with minimal user input. It's designed for speed and scale.
    """)
    
    fig = plot_time_series_analysis()
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History"])
        
        with tabs[0]:
            st.metric(label="⌚ ARIMA Forecast Error (MAE)", value="3.15 units", help="Mean Absolute Error for the ARIMA model.")
            st.metric(label="📱 Prophet Forecast Error (MAE)", value="2.89 units", help="Mean Absolute Error for the Prophet model.")
            st.metric(label="🔮 Forecast Horizon", value="14 Weeks", help="The period into the future for which we are generating predictions.")

            st.markdown("""
            **The Core Strategic Insight:** The choice is not about which model is "best," but which is **right for the job.** For a stable, well-understood industrial process where interpretability is key, the craftsmanship of ARIMA is superior. For a complex, noisy business time series with multiple layers of seasonality and a need for automated forecasting at scale, Prophet is often the better tool.
            """)

        with tabs[1]:
            st.error("""
            🔴 **THE INCORRECT APPROACH: The "Blind Forecasting" Fallacy**
            An analyst takes a column of data, feeds it directly into `model.fit()` and `model.predict()`, and presents the resulting line.
            
            **The Flaw:** They've made no attempt to understand the data's structure. Is there a trend? Is it seasonal? Is the variance stable? This "black box" approach produces a forecast that is fragile, unreliable, and likely to fail spectacularly the moment the underlying process changes.
            """)
            st.success("""
            🟢 **THE GOLDEN RULE: Decompose, Validate, and Monitor**
            1.  **Decompose and Understand:** Use a time series decomposition plot to separate the series into its core components: **Trend, Seasonality, and Residuals.**
            2.  **Train, Validate, Test:** Never judge a model by its performance on data it has already seen. Split your historical data to evaluate forecast accuracy.
            3.  **Monitor for Drift:** A forecast is only a snapshot in time. Continuously monitor its performance against incoming new data.
            """)

        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            - **The Classical Era (ARIMA):** In their 1970 book *Time Series Analysis: Forecasting and Control*, **George Box** and **Gwilym Jenkins** provided a comprehensive methodology that became the gold standard for decades.
            - **The Modern Era (Prophet):** In 2017, **Facebook's** Core Data Science team released Prophet, designed for automation and intuitive tuning to solve forecasting at scale.
            
            #### How They Work
            - **ARIMA (AutoRegressive Integrated Moving Average):** Models the relationship between an observation and its past values (AR), uses differencing to remove trend (I), and models the relationship with past forecast errors (MA).
            - **Prophet:** A decomposable additive model: $y(t) = g(t) + s(t) + h(t) + \epsilon_t$, where `g(t)` is trend, `s(t)` is seasonality, and `h(t)` is holidays.
            """)

def render_stability_analysis():
    """Renders the module for pharmaceutical stability analysis."""
    st.markdown("""
    #### Purpose & Application: The Expiration Date Contract
    **Purpose:** To determine the shelf-life for a drug product by proving, with high confidence, that a Critical Quality Attribute (CQA) like potency will remain within its safety and efficacy specifications over time.
    
    **Strategic Application:** This is a mandatory, high-stakes analysis for any commercial pharmaceutical product, as required by the **ICH Q1E guideline**. It is the data-driven foundation of the expiration date printed on every vial and box.
    """)
    
    fig = plot_stability_analysis()
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History"])
        
        with tabs[0]:
            st.metric(label="📈 Approved Shelf-Life", value="31.3 Months", help="The time at which the lower confidence bound intersects the specification limit.")
            st.metric(label="📉 Degradation Rate (Slope)", value="-0.41 %/month", help="The estimated average loss of potency per month.")
            st.metric(label="🥅 Specification Limit", value="95.0 %", help="The minimum acceptable potency for the product to be considered effective.")

            st.markdown("""
            **The Verdict:** The shelf-life is declared at the exact moment **the Safety Net (red dashed line) hits the Finish Line (red dotted spec limit).** This conservative approach ensures that, with 95% confidence, the average product potency will not drop below the spec limit before the expiration date.
            """)

        with tabs[1]:
            st.error("""
            🔴 **THE INCORRECT APPROACH: The "Happy Path" Fallacy**
            Setting the shelf-life where the *average trend* (black line) crosses the spec limit.
            
            **The Flaw:** This ignores uncertainty! About half of all future batches will degrade *faster* than the average, creating a massive patient risk.
            """)
            st.success("""
            🟢 **THE GOLDEN RULE: The Confidence Interval Sets the Expiration Date**
            The ICH Q1E guideline is built on statistical conservatism.
            1.  **First, Prove Poolability:** Use a statistical test (ANCOVA) to prove that the degradation slopes of different batches are not significantly different.
            2.  **Then, Use the Confidence Bound:** The shelf-life is determined by the intersection of the appropriate confidence bound with the specification limit.
            """)

        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin: The ICH Revolution
            The **International Council for Harmonisation (ICH)** was formed to end the inefficiency of running different stability programs for different global regions. **ICH Q1E ("Evaluation of Stability Data")**, adopted in 2003, codified the statistical methodology, including the critical principle of using confidence intervals to determine shelf-life, creating a global gold standard.
            
            #### Mathematical Basis
            The core is a linear regression model: $Y_i = \beta_0 + \beta_1 X_i + \epsilon_i$. The confidence interval for the regression line is a **funnel shape**, narrowest at the center of the data and widest at the beginning and end, reflecting our uncertainty in the extrapolation.
            """)

def render_survival_analysis():
    """Renders the module for Survival Analysis."""
    st.markdown("""
    #### Purpose & Application: The Statistician's Crystal Ball
    **Purpose:** To model "time-to-event" data and forecast the probability of survival over time. Its superpower is its unique ability to handle **censored data**—observations where the study ends before the event (e.g., failure or death) occurs.
    
    **Strategic Application:** This is the core methodology for reliability engineering and clinical research.
    - **⚙️ Predictive Maintenance:** Model component failure probability over time to move from scheduled to data-driven maintenance.
    - **⚕️ Clinical Trials:** The gold standard for analyzing endpoints like "time to disease progression" or "overall survival."
    """)
    
    fig = plot_survival_analysis()
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History"])
        
        with tabs[0]:
            st.metric(label="📊 Log-Rank Test p-value", value="0.008", help="A p-value < 0.05 indicates a statistically significant difference between the survival curves.")
            st.metric(label="⏳ Median Survival (Group A)", value="17 Months", help="Time at which 50% of Group A have experienced the event.")
            st.metric(label="⏳ Median Survival (Group B)", value="28 Months", help="Time at which 50% of Group B have experienced the event.")

            st.markdown("""
            **Reading the Curve:** The curve for **Group B** is consistently higher than Group A, demonstrating that items in Group B have a higher probability of surviving longer. The low p-value confirms this is statistically significant. Each "step down" represents a failure event.
            """)

        with tabs[1]:
            st.error("""
            🔴 **THE INCORRECT APPROACH: The "Pessimist's Fallacy"**
            An analyst throws away all censored data (units still working at the end of the study) and calculates the average lifetime using only the units that broke.
            
            **The Flaw:** This is a massive pessimistic bias. You have selected **only the weakest items** and ignored the strong ones.
            """)
            st.success("""
            🟢 **THE GOLDEN RULE: Respect the Censored Data**
            Censored data is not missing data; it is valuable information. A censored point at 24 months tells us: **The lifetime of this unit is at least 24 months.** Always use a method designed for censoring, like the Kaplan-Meier estimator.
            """)

        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin: The 1958 Revolution
            In 1958, **Edward L. Kaplan** and **Paul Meier** published "Nonparametric Estimation from Incomplete Observations," introducing the world to the **Kaplan-Meier estimator**. It was a revolutionary breakthrough, providing a simple, elegant method to estimate the true survival function with censored data. This single technique unlocked a new era of research in medicine and engineering.
            
            #### Mathematical Basis
            """)
            st.latex(r"S(t_i) = S(t_{i-1}) \times \left( \frac{n_i - d_i}{n_i} \right)")
            st.markdown("Where `S(tᵢ)` is the survival probability, `nᵢ` is the number at risk, and `dᵢ` is the number of events. The survival at one step is the survival from the previous step multiplied by the conditional probability of surviving the current step.")

def render_mva_pls():
    """Renders the module for Multivariate Analysis (PLS)."""
    st.markdown("""
    #### Purpose & Application: The Statistical Rosetta Stone
    **Purpose:** To act as a **statistical Rosetta Stone**, translating a massive, complex set of input variables (X, e.g., a spectrum) into a simple, actionable output (Y, e.g., concentration).
    
    **Strategic Application:** This is the statistical engine behind **Process Analytical Technology (PAT)**. It is specifically designed for problems where you have more input variables than samples and the inputs are highly correlated.
    - **🔬 Real-Time Spectroscopy:** Predicts chemical concentration from NIR or Raman spectra in real-time.
    - **🏭 "Golden Batch" Modeling:** Learns the "fingerprint" of a perfect batch to detect deviations during a run.
    """)
    
    fig = plot_mva_pls()
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History"])
        
        with tabs[0]:
            st.metric(label="📈 Model R² (Goodness of Fit)", value="0.98", help="How well the model fits the training data.")
            st.metric(label="🎯 Model Q² (Predictive Power)", value="0.95", help="The cross-validated R². The most important performance metric.")
            st.metric(label="🧬 Latent Variables (LVs)", value="2", help="The number of 'hidden' factors the model extracted.")
            
            st.markdown("""
            **Decoding the VIP Plot:**
            The **Variable Importance in Projection (VIP)** plot reveals what the model has learned. Variables with a VIP score greater than 1 (above the red line) are considered important for the prediction.
            """)

        with tabs[1]:
            st.error("""
            🔴 **THE INCORRECT APPROACH: The "Overfitting" Trap**
            An analyst keeps adding more Latent Variables to their model to chase a perfect R-squared of 0.999.
            
            **The Flaw:** The model has memorized the noise in the training data. Its predictions on new data will be terrible.
            """)
            st.success("""
            🟢 **THE GOLDEN RULE: Thou Shalt Validate Thy Model on Unseen Data**
            A model's R-squared is vanity. Its performance on new data is sanity.
            1.  **Partition Your Data:** Split into a **Training Set** and a **Test Set**.
            2.  **Use Cross-Validation:** Choose the number of Latent Variables that minimizes prediction error, not the number that maximizes R-squared.
            3.  **Final Verdict:** The ultimate test is performance on the held-out Test Set.
            """)

        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            PLS was developed in the 1970s by Swedish statistician **Herman Wold** for econometrics. His son, chemist **Svante Wold**, adapted it for **chemometrics**, where it became the powerhouse of modern PAT by solving problems with massive, correlated datasets from instruments like spectrometers.
            
            #### How It Works
            PLS uses **dimensionality reduction**. Instead of using all 200 raw inputs, it first finds a few "consensus groups" of correlated variables, called **Latent Variables (LVs)**. It then builds a simple regression model using these few powerful LVs to predict the output.
            """)

def render_clustering():
    """Renders the module for unsupervised clustering."""
    st.markdown("""
    #### Purpose & Application: The Data Archeologist
    **Purpose:** To sift through process data to discover natural, hidden groupings or "regimes." It answers the question: "Are all of my 'good' batches truly the same, or are there different ways to be good?"
    
    **Strategic Application:** A powerful exploratory tool for deep process understanding.
    - **Process Regime Identification:** Can reveal that a process is secretly operating in different states (e.g., due to different raw material suppliers or operator techniques).
    - **Root Cause Analysis:** Can help determine which "family" of normal operation a failed batch was most similar to, providing critical clues.
    """)
    
    fig = plot_clustering()
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History"])
        
        with tabs[0]:
            st.metric(label="🏺 Discovered 'Regimes' (k)", value="3", help="The number of distinct clusters the K-Means algorithm identified.")
            st.metric(label="🗺️ Cluster Quality (Silhouette Score)", value="0.72", help="A measure of how distinct the clusters are. Higher is better (max 1.0).")
            
            st.markdown("""
            **The Dig Site Findings:**
            The algorithm has found three distinct groups in the data. Before this analysis, these were likely thought of as one single "in-control" population. The high Silhouette Score gives confidence that these groupings are meaningful.
            
            **The Core Strategic Insight:** The discovery of hidden clusters proves that your single process is actually a collection of multiple sub-processes. Understanding the *causes* of this separation is the gateway to improved process control.
            """)

        with tabs[1]:
            st.error("""
            🔴 **THE INCORRECT APPROACH: The "If It Ain't Broke..." Fallacy**
            A manager sees three clusters but says, *"Interesting, but all of those batches passed QC, so who cares?"*
            
            **The Flaw:** This treats a treasure map as a doodle. One of those "regimes" might be living on the edge of a cliff (close to a specification limit), while another is safe in a valley.
            """)
            st.success("""
            🟢 **THE GOLDEN RULE: A Cluster is a Clue, Not a Conclusion**
            The discovery of clusters is the **start** of the investigation.
            1.  **Find the Clusters:** Use an algorithm like K-Means.
            2.  **Validate the Clusters:** Use a metric like the Silhouette Score.
            3.  **Profile the Clusters:** Overlay other information. Do clusters correspond to different raw material lots? Different operators? Seasons? This turns a statistical finding into actionable knowledge.
            """)

        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            The K-Means algorithm was first proposed by **Stuart Lloyd** at Bell Labs in 1957. Its simplicity and the explosion of computational power have made it one of the most widely used clustering algorithms.
            
            #### How K-Means Works: The "Pizza Shop" Analogy
            1.  **Step 1 (Random Start):** Randomly drop 3 pins (cluster centroids) on a map.
            2.  **Step 2 (Assignment):** Assign every house (data point) to its *closest* pin.
            3.  **Step 3 (Update):** Move each pin to the true geographic center of its assigned houses.
            4.  **Step 4 (Repeat):** Repeat the 'Assign-Update' process until the pins stop moving.
            """)

def render_classification_models():
    """Renders the module for Predictive QC (Classification)."""
    st.markdown("""
    #### Purpose & Application: The AI Gatekeeper
    **Purpose:** To build an **AI Gatekeeper** that can inspect in-process data and predict whether a batch will ultimately pass or fail its final QC specifications.
    
    **Strategic Application:** The foundation of real-time release. By predicting outcomes early, we can:
    - **Prevent Failures:** Intervene in a batch that is trending towards failure.
    - **Optimize Resources:** Focus QC lab resources on high-risk batches.
    - **Accelerate Release:** Provide evidence to release batches based on in-process data.
    """)
    
    fig = plot_classification_models()
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History"])
        
        with tabs[0]:
            st.metric(label="📈 Model 1: Logistic Regression Accuracy", value="85.0%", help="Performance of the simpler, linear model.")
            st.metric(label="🚀 Model 2: Random Forest Accuracy", value="96.7%", help="Performance of the more complex, non-linear model.")

            st.markdown("""
            **Reading the Decision Boundaries:**
            - **Logistic Regression (Left):** Can only draw a **straight line** to separate the groups and struggles with the curved relationship.
            - **Random Forest (Right):** Creates a complex, **non-linear boundary**, learning the true "island" of failure and achieving higher accuracy.

            **The Core Strategic Insight:** For complex processes, the relationship between parameters and quality is rarely linear. Modern machine learning models are often required to capture this complexity.
            """)

        with tabs[1]:
            st.error("""
            🔴 **THE INCORRECT APPROACH: The "Garbage In, Garbage Out" Fallacy**
            An analyst feeds all 500 sensor tags directly into a model.
            
            **The Flaw:** With more inputs than batches, the model will find spurious correlations and fail to generalize.
            """)
            st.success("""
            🟢 **THE GOLDEN RULE: Feature Engineering is the Secret Ingredient**
            The success of a model depends more on the quality of the inputs ("features") than the algorithm.
            1.  **Collaborate with SMEs:** Identify scientifically relevant parameters.
            2.  **Engineer Smart Features:** Create more informative features (e.g., the *slope* of a temperature profile, not just raw values).
            3.  **Validate on Unseen Data:** The model's true performance is only revealed on a hold-out test set.
            """)

        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            - **Logistic Regression (1958):** Developed by **David Cox**, it's a generalization of linear regression for binary (pass/fail) outcomes. It remains a powerful, interpretable baseline.
            - **Random Forest (2001):** Invented by **Leo Breiman and Adele Cutler**, it's an **ensemble method** that builds hundreds of decision trees. Its final prediction is a "majority vote" of all the trees, making it highly accurate and robust.
            """)

def render_anomaly_detection():
    """Renders the module for unsupervised anomaly detection."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To deploy an **AI Bouncer** for your data—a smart system that identifies rare, unexpected observations (anomalies) without any prior knowledge of what "bad" looks like.
    
    **Strategic Application:** A game-changer for monitoring complex processes.
    - **Novel Fault Detection:** Can flag a new type of process failure the first time it occurs.
    - **Intelligent Data Cleaning:** Automatically identifies potential sensor glitches or data entry errors.
    - **"Golden Batch" Investigation:** Can find which "good" batches were statistically unusual, holding secrets to process robustness.
    """)
    
    def plot_isolation_forest():
        np.random.seed(42)
        X_inliers = np.random.normal(0, 1, (100, 2))
        X_outliers = np.random.uniform(low=-4, high=4, size=(10, 2))
        X = np.concatenate([X_inliers, X_outliers], axis=0)

        clf = IsolationForest(contamination=0.1, random_state=42)
        y_pred = clf.fit_predict(X)
        
        df = pd.DataFrame(X, columns=['Process Parameter 1', 'Process Parameter 2'])
        df['Status'] = ['Anomaly' if p == -1 else 'Normal' for p in y_pred]

        fig = px.scatter(df, x='Process Parameter 1', y='Process Parameter 2', color='Status',
                         color_discrete_map={'Normal': '#636EFA', 'Anomaly': '#EF553B'},
                         title="<b>The AI Bouncer at Work</b>",
                         symbol='Status',
                         symbol_map={'Normal': 'circle', 'Anomaly': 'x'})
        fig.update_traces(marker=dict(size=10), selector=dict(name='Anomaly'))
        fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        return fig

    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        fig = plot_isolation_forest()
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History"])
        
        with tabs[0]:
            st.metric(label="Data Points Scanned", value="110")
            st.metric(label="Anomalies Flagged", value="10", help="Based on a contamination setting of ~10%.")
            st.metric(label="Algorithm Used", value="Isolation Forest")

            st.markdown("""
            **The Core Strategic Insight:** Anomaly detection is your early warning system for the **unknown unknowns**. While a control chart tells you if you've broken a known rule, an anomaly detector tells you that something you've never seen before just happened.
            """)

        with tabs[1]:
            st.error("""
            🔴 **THE INCORRECT APPROACH: "The Glitch Hunter"**
            Dismissing every anomaly as a data error or sensor glitch. This is like the bouncer seeing a problem, shrugging, and looking the other way.
            """)
            st.success("""
            🟢 **THE GOLDEN RULE: An Anomaly is a Question, Not an Answer**
            Treat every flagged anomaly as the start of a forensic investigation. The anomaly itself is not the conclusion; it is the starting pistol for discovery.
            """)

        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            The **Isolation Forest** algorithm, introduced in 2008 by Liu, Ting, and Zhou, was a brilliant solution for high-dimensional anomaly detection. Instead of modeling "normal," they decided to just **isolate** every point. Anomalous points are "few and different," making them much easier to separate from the rest of the data.
            
            #### How it Works: The "20 Questions" Analogy
            The algorithm builds a forest of random decision trees. It counts the number of "questions" (splits) it takes to isolate each point. Anomalies are isolated quickly with few questions, while normal points require many questions.
            """)

def render_xai_shap():
    """Renders the module for Explainable AI (XAI) using SHAP."""
    st.markdown("""
    #### Purpose & Application: The AI Interrogator
    **Purpose:** To force a complex "black box" model to confess exactly *why* it made a specific prediction. **Explainable AI (XAI)** cracks open the black box to reveal its internal logic.
    
    **Strategic Application:** This is the single most important technology for deploying advanced ML in regulated GxP environments. **SHAP (SHapley Additive exPlanations)** is the state-of-the-art framework that provides this crucial evidence for model trust, regulatory compliance, and actionable insights.
    """)
    
    summary_buf, force_html = plot_xai_shap()
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.subheader("Global Feature Importance")
        st.image(summary_buf, caption="The SHAP Summary Plot shows the overall impact of each feature.")
        
        st.subheader("Local Prediction Explanation")
        st.components.v1.html(force_html, height=150, scrolling=False)
        st.caption("The SHAP Force Plot explains a single, specific prediction.")
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History"])

        with tabs[0]:
            st.markdown("""
            **Global Explanation (Top-Left):** Ranks features by overall impact. High `Age` (red dots) has a high positive SHAP value, pushing the model to predict higher income.
            
            **Local Explanation (Bottom-Left):** A tug-of-war for one prediction. Red forces push the prediction higher, blue forces push it lower.
            """)

        with tabs[1]:
            st.error("""
            🔴 **THE INCORRECT APPROACH: The "Accuracy is Everything" Fallacy**
            Deploying a model with 99% accuracy without checking its reasoning.
            
            **The Flaw:** The model might be a "Clever Hans," having learned a spurious correlation that is not scientifically valid.
            """)
            st.success("""
            🟢 **THE GOLDEN RULE: Explainability Builds Trust**
            1.  **Build the Model:** Train for high accuracy.
            2.  **Interrogate with SHAP:** Apply SHAP to predictions.
            3.  **Consult the Expert:** Show the SHAP plots to a Subject Matter Expert. Ask: *"Does this make sense?"* If yes, you can trust the model. If no, XAI has just saved you from deploying a flawed model.
            """)

        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin: From Game Theory to AI
            The theory comes from **cooperative game theory**. In 1951, **Lloyd Shapley** developed **Shapley values** to solve the "fair payout" problem in a team.
            
            In 2017, Scott Lundberg and Su-In Lee realized a model's prediction is a "game" and its features are the "players." They adapted Shapley values to create **SHAP**, a method to fairly distribute the "payout" (the prediction) among the features, providing a unified framework for explaining any model.
            """)

def render_advanced_ai_concepts():
    """Renders the module for advanced AI concepts."""
    st.markdown("""
    #### Purpose & Application: A Glimpse into the AI Frontier
    **Purpose:** To provide a high-level, conceptual overview of cutting-edge AI architectures that represent the future of process analytics.
    """)
    
    col1, col2 = st.columns([0.7, 0.3])
    
    with col2:
        st.subheader("Select a Concept")
        concept = st.radio(
            "Select an Advanced AI Concept:", 
            ["Transformers", "Graph Neural Networks (GNNs)", "Reinforcement Learning (RL)", "Generative AI"],
            label_visibility="collapsed"
        )
        
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History"])
        
        # Dynamically populate tabs based on concept selection
        if concept == "Transformers":
            with tabs[0]: st.markdown("**The AI Historian:** Understands long-range dependencies in sequential data (like a batch record) using a 'self-attention' mechanism. This moves beyond simple forecasting to modeling the entire **process narrative**.")
            with tabs[1]: st.success("**THE GOLDEN RULE:** Tokenize your process narrative. Success depends on converting continuous data into a discrete sequence of meaningful events.")
            with tabs[2]: st.markdown("**Origin:** Revolutionized AI with the 2017 Google Brain paper, **\"Attention Is All You Need,\"** forming the basis for models like GPT.")
        elif concept == "GNNs":
            with tabs[0]: st.markdown("**The Network Navigator:** Operates on graph data (nodes and edges), modeling the **entire system's interconnectedness** by passing messages between nodes.")
            with tabs[1]: st.success("**THE GOLDEN RULE:** Your graph IS your model. The most important work is defining the nodes and edges that represent your system.")
            with tabs[2]: st.markdown("**Origin:** Exploded in popularity around 2018, generalizing deep learning beyond grids (images) and sequences (text) to graphs.")
        elif concept == "RL":
            with tabs[0]: st.markdown("**The AI Pilot:** Learns by taking actions in an environment to maximize a long-term reward. This is the leap from passive modeling to **active, real-time process optimization**.")
            with tabs[1]: st.success("**THE GOLDEN RULE:** The Digital Twin is the Dojo. An RL agent must be trained in a high-fidelity simulation to learn optimal control with zero real-world risk.")
            with tabs[2]: st.markdown("**Origin:** Deep roots in control theory, supercharged by DeepMind in the mid-2010s with AlphaGo.")
        elif concept == "Generative AI":
            with tabs[0]: st.markdown("**The Synthetic Data Forger:** Learns the underlying distribution of a dataset to create new, synthetic data. This solves the **\"rare event\" problem** by generating realistic failure profiles for training more robust models.")
            with tabs[1]: st.success("**THE GOLDEN RULE:** Validate the forgeries. The generated data is only useful if it is proven to be statistically indistinguishable from real data.")
            with tabs[2]: st.markdown("**Origin:** Catalyzed by **Generative Adversarial Networks (GANs)** in 2014, with modern **Diffusion Models** (DALL-E 2) being state-of-the-art.")

    with col1:
        fig = plot_advanced_ai_concepts(concept)
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# MAIN APP LOGIC AND LAYOUT
# ==============================================================================
if 'current_view' not in st.session_state:
    st.session_state.current_view = 'Introduction'

with st.sidebar:
    st.title("🧰 Toolkit Navigation")
    
    if st.sidebar.button("🚀 Project Framework", use_container_width=True):
        st.session_state.current_view = 'Introduction'
        st.rerun()

    st.divider()

    all_tools = {
        "ACT I: FOUNDATION & CHARACTERIZATION": ["Confidence Interval Concept", "Core Validation Parameters", "Gage R&R / VCA", "LOD & LOQ", "Linearity & Range", "Non-Linear Regression (4PL/5PL)", "ROC Curve Analysis", "Equivalence Testing (TOST)", "Assay Robustness (DOE)", "Causal Inference"],
        "ACT II: TRANSFER & STABILITY": ["Process Stability (SPC)", "Process Capability (Cpk)", "Tolerance Intervals", "Method Comparison", "Pass/Fail Analysis", "Bayesian Inference"],
        "ACT III: LIFECYCLE & PREDICTIVE MGMT": ["Run Validation (Westgard)", "Multivariate SPC", "Small Shift Detection", "Time Series Analysis", "Stability Analysis (Shelf-Life)", "Reliability / Survival Analysis", "Multivariate Analysis (MVA)", "Clustering (Unsupervised)", "Predictive QC (Classification)", "Anomaly Detection", "Explainable AI (XAI)", "Advanced AI Concepts"]
    }

    for act_title, act_tools in all_tools.items():
        st.subheader(act_title)
        for tool in act_tools:
            if st.button(tool, key=tool, use_container_width=True):
                st.session_state.current_view = tool
                st.rerun()

view = st.session_state.current_view

if view == 'Introduction':
    render_introduction_content()
else:
    st.header(f"🔧 {view}")

    PAGE_DISPATCHER = {
        "Confidence Interval Concept": render_ci_concept,
        "Core Validation Parameters": render_core_validation_params,
        "Gage R&R / VCA": render_gage_rr,
        "LOD & LOQ": render_lod_loq,
        "Linearity & Range": render_linearity,
        "Non-Linear Regression (4PL/5PL)": render_4pl_regression,
        "ROC Curve Analysis": render_roc_curve,
        "Equivalence Testing (TOST)": render_tost,
        "Assay Robustness (DOE)": render_assay_robustness_doe,
        "Causal Inference": render_causal_inference,
        "Process Stability (SPC)": render_spc_charts,
        "Process Capability (Cpk)": render_capability,
        "Tolerance Intervals": render_tolerance_intervals,
        "Method Comparison": render_method_comparison,
        "Pass/Fail Analysis": render_pass_fail,
        "Bayesian Inference": render_bayesian,
        "Run Validation (Westgard)": render_multi_rule,
        "Multivariate SPC": render_multivariate_spc,
        "Small Shift Detection": render_ewma_cusum,
        "Time Series Analysis": render_time_series_analysis,
        "Stability Analysis (Shelf-Life)": render_stability_analysis,
        "Reliability / Survival Analysis": render_survival_analysis,
        "Multivariate Analysis (MVA)": render_mva_pls,
        "Clustering (Unsupervised)": render_clustering,
        "Predictive QC (Classification)": render_classification_models,
        "Anomaly Detection": render_anomaly_detection,
        "Explainable AI (XAI)": render_xai_shap,
        "Advanced AI Concepts": render_advanced_ai_concepts,
    }

    if view in PAGE_DISPATCHER:
        PAGE_DISPATCHER[view]()
    else:
        st.error("Error: Could not find the selected tool to render.")
        st.session_state.current_view = 'Introduction'
        st.rerun()
