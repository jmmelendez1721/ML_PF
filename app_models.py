import warnings
warnings.filterwarnings("ignore")

from dash import Dash, dcc, html, Input, Output, dash_table
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
from itertools import combinations

from delong import delong_roc_test
from statsmodels.stats.multitest import multipletests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print(f"[STARTUP] BASE_DIR: {BASE_DIR}")
for root, dirs, files in os.walk(BASE_DIR):
    dirs[:] = [d for d in dirs if d not in ('__pycache__', '.git', 'venv', 'env', 'node_modules')]
    level = root.replace(BASE_DIR, '').count(os.sep)
    indent = '  ' * level
    print(f"[STARTUP] {indent}{os.path.basename(root) or 'ROOT'}/")
    for f in files:
        size = os.path.getsize(os.path.join(root, f))
        print(f"[STARTUP] {indent}  {f}  ({size:,} bytes)")

# ══════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════
app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# ══════════════════════════════════════════════
# DATOS — EDA
# ══════════════════════════════════════════════
df_raw = pd.read_csv(os.path.join(BASE_DIR, "HeartFailureDataset.csv"))
df_raw = df_raw.drop(columns=['id'])

NUM_COLS  = list(df_raw.select_dtypes(include=np.number).columns)
CAT_COLS  = [c for c in NUM_COLS if df_raw[c].nunique() <= 4]
CONT_COLS = [c for c in NUM_COLS if c not in CAT_COLS]

labels_map = {
    'gender': {
        1: 'Mujer',
        2: 'Hombre'
    },
    'cholesterol': {
        1: 'Normal',
        2: 'Sobre lo normal',
        3: 'Muy sobre lo normal'
    },
    'gluc': {
        1: 'Normal',
        2: 'Sobre lo normal',
        3: 'Muy sobre lo normal'
    },
    'smoke': {
        0: 'No fuma',
        1: 'Fuma'
    },
    'alco': {
        0: 'No consume',
        1: 'Consume'
    },
    'active': {
        0: 'Inactivo',
        1: 'Activo'
    },
    'cardio': {
        0: 'No enfermedad',
        1: 'Enfermedad'
    }
}

EDA_NUM_COLS = ['age', 'weight', 'height', 'ap_hi', 'ap_lo']

EDA_CAT_COLS = [
    'gender',
    'cholesterol',
    'gluc',
    'smoke',
    'alco',
    'active'
]


# ══════════════════════════════════════════════
# MODELOS — solo rutas a carpetas con CSVs
# No se carga ningún .pkl ni se hace inferencia
# ══════════════════════════════════════════════
MODEL_FOLDERS = {
    "Naive Bayes":         "bayesian",
    "Gradient Boosting":   "Gradient_Boosting",
    "KNN":                 "KNN",
    "Logistic Regression": "Logistic_Regression",
    "SVM":                 "SVM",
    "Random Forest":       "Random_Forests",
    "MLP":            "MLP",
    "FT-Transformer (Original)": "FTTransformer",
}

MODEL_COLORS = {
    "Naive Bayes":         "#4A9BBF",
    "Gradient Boosting":   "#E05C5C",
    "KNN":                 "#2ECC71",
    "Logistic Regression": "#9B59B6",
    "SVM":                 "#F39C12",
    "Random Forests":       "#1A3A5C",
    "MLP":            "#8E44AD",
    "FTTransformer": "#9467bd" ,
}

def model_path(name, *parts):
    return os.path.join(BASE_DIR, MODEL_FOLDERS[name], *parts)

def read_csv_safe(path):
    """Lee un CSV y devuelve DataFrame o None si no existe."""
    if not os.path.exists(path):
        print(f"[CSV] No encontrado: {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[CSV] Error leyendo {path}: {e}")
        return None

def load_metrics(name):
    return read_csv_safe(model_path(name, "metrics.csv"))

def load_roc(name):
    return read_csv_safe(model_path(name, "roc_curve.csv"))

def load_cr(name):
    return read_csv_safe(model_path(name, "classification_report.csv"))

def load_lime(name):
    return read_csv_safe(model_path(name, "lime", "lime_explanations.csv"))

def load_predictions(name):
    return read_csv_safe(model_path(name, "predictions.csv"))

# ══════════════════════════════════════════════
# PALETA Y ESTILOS
# ══════════════════════════════════════════════
COLOR_BG      = "#F7F9FC"
COLOR_CARD    = "#FFFFFF"
COLOR_PRIMARY = "#1A3A5C"
COLOR_ACCENT  = "#E05C5C"
COLOR_ACCENT2 = "#4A9BBF"
COLOR_TEXT    = "#2C3E50"
COLOR_MUTED   = "#7F8C8D"
COLOR_BORDER  = "#E0E6EF"
FONT_TITLE    = "'Playfair Display', serif"
FONT_BODY     = "'DM Sans', sans-serif"

GOOGLE_FONTS = html.Link(
    rel="stylesheet",
    href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap"
)

PLOT_BASE = dict(
    paper_bgcolor="white", plot_bgcolor="#F7F9FC",
    font=dict(family=FONT_BODY, color=COLOR_TEXT),
    margin=dict(l=40, r=40, t=50, b=40), height=440,
)

CHART_WRAP = {
    "background": COLOR_CARD, "borderRadius": "12px", "padding": "16px",
    "boxShadow": "0 2px 12px rgba(26,58,92,0.07)", "border": f"1px solid {COLOR_BORDER}"
}

# ══════════════════════════════════════════════
# HELPERS UI
# ══════════════════════════════════════════════
def card(children, extra=None):
    s = {"background": COLOR_CARD, "borderRadius": "12px", "padding": "28px 32px",
         "marginBottom": "24px", "boxShadow": "0 2px 12px rgba(26,58,92,0.07)",
         "border": f"1px solid {COLOR_BORDER}"}
    if extra: s.update(extra)
    return html.Div(children, style=s)

def sec_title(t):
    return html.H3(t, style={
        "fontFamily": FONT_TITLE, "color": COLOR_PRIMARY, "fontSize": "1.25rem",
        "fontWeight": "700", "marginBottom": "12px", "marginTop": "0",
        "borderLeft": f"4px solid {COLOR_ACCENT}", "paddingLeft": "12px"})

def body(t):
    return html.P(t, style={"fontFamily": FONT_BODY, "color": COLOR_TEXT,
        "fontSize": "0.95rem", "lineHeight": "1.75", "marginTop": "0", "fontWeight": "300"})

def lbl(t):
    return html.Label(t, style={"fontFamily": FONT_BODY, "fontWeight": "600",
        "color": COLOR_PRIMARY, "fontSize": "0.82rem", "textTransform": "uppercase",
        "letterSpacing": "0.08em", "marginBottom": "8px", "display": "block"})

def two_col(left, right):
    return html.Div([
        html.Div([left],  style={"flex": "1", "marginRight": "12px", "minWidth": "0"}),
        html.Div([right], style={"flex": "1", "marginLeft":  "12px", "minWidth": "0"}),
    ], style={"display": "flex", "alignItems": "stretch"})

LOREM = ("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor "
         "incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud "
         "exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.")

# ══════════════════════════════════════════════
# DICCIONARIO
# ══════════════════════════════════════════════
variables = [
    {"Variable":"Id",          "Tipo":"Numérica",              "Unidad":"NA",           "Significado":"Indicador único para cada paciente."},
    {"Variable":"Age",         "Tipo":"Numérica",              "Unidad":"Días",         "Significado":"Edad del paciente."},
    {"Variable":"Gender",      "Tipo":"Categórica",            "Unidad":"NA",           "Significado":"Variable binaria: 1 = Mujer, 2 = Hombre."},
    {"Variable":"Height",      "Tipo":"Numérica",              "Unidad":"Centímetros",  "Significado":"Altura del paciente."},
    {"Variable":"Weight",      "Tipo":"Numérica",              "Unidad":"Kilogramos",   "Significado":"Peso del paciente."},
    {"Variable":"ap_hi",       "Tipo":"Numérica",              "Unidad":"mmHg",         "Significado":"Medición más alta de la presión sanguínea sistólica."},
    {"Variable":"ap_lo",       "Tipo":"Numérica",              "Unidad":"mmHg",         "Significado":"Medición más alta de la presión sanguínea diastólica."},
    {"Variable":"cholesterol", "Tipo":"Categórica",            "Unidad":"NA",           "Significado":"Nivel de colesterol: 1 = Normal, 2 = Sobre lo normal, 3 = Muy sobre lo normal."},
    {"Variable":"gluc",        "Tipo":"Categórica",            "Unidad":"NA",           "Significado":"Nivel de glucosa: 1 = Normal, 2 = Sobre lo normal, 3 = Muy sobre lo normal."},
    {"Variable":"smoke",       "Tipo":"Categórica",            "Unidad":"NA",           "Significado":"Hábito de fumar: 0 = Fumador, 1 = No fumador."},
    {"Variable":"alco",        "Tipo":"Categórica",            "Unidad":"NA",           "Significado":"Consumo de alcohol: 0 = Bebedor, 1 = No bebedor."},
    {"Variable":"active",      "Tipo":"Categórica",            "Unidad":"NA",           "Significado":"Actividad física: 0 = Activo, 1 = No activo."},
    {"Variable":"cardio",      "Tipo":"Categórica (objetivo)", "Unidad":"NA",           "Significado":"Presencia de enfermedad cardiaca (variable objetivo de clasificación)."},
]
df_vars = pd.DataFrame(variables)

# ══════════════════════════════════════════════
# TAB 1
# ══════════════════════════════════════════════
def make_tab1():
    img_card = lambda src: html.Div([
        html.Img(src=src, style={"width":"100%","height":"100%","minHeight":"220px",
            "objectFit":"cover","borderRadius":"8px","display":"block"}),
    ], style={"background":COLOR_CARD,"borderRadius":"12px","padding":"16px",
        "boxShadow":"0 2px 12px rgba(26,58,92,0.07)","border":f"1px solid {COLOR_BORDER}",
        "height":"100%","boxSizing":"border-box"})

    return html.Div([
        html.Div([
            html.Div([
                html.Span("Análisis Clínico", style={"fontFamily":FONT_BODY,"color":COLOR_ACCENT,
                    "fontSize":"0.85rem","fontWeight":"600","letterSpacing":"0.15em","textTransform":"uppercase"}),
                html.H1("Falla Cardíaca & Machine Learning", style={"fontFamily":FONT_TITLE,"color":"white",
                    "fontSize":"2.4rem","fontWeight":"700","marginTop":"8px","marginBottom":"8px","lineHeight":"1.2"}),
                html.P("Proyecto de análisis exploratorio y modelado predictivo sobre diagnóstico de insuficiencia cardíaca",
                    style={"fontFamily":FONT_BODY,"color":"rgba(255,255,255,0.7)","fontSize":"1rem","fontWeight":"300","marginTop":"0"}),
            ], style={"flex":"1"}),
            html.Div("♥", style={"fontSize":"6rem","color":"white","opacity":"0.12","lineHeight":"1","alignSelf":"center","marginLeft":"24px"}),
        ], style={"display":"flex","justifyContent":"space-between",
            "background":f"linear-gradient(135deg,{COLOR_PRIMARY} 0%,#2A5A8C 100%)",
            "borderRadius":"16px","padding":"40px 48px","marginBottom":"28px"}),

        html.Div([
            html.Div([card([
                sec_title("Definición del Problema de Investigación"),
                body("El problema abordado consiste en el desarrollo de un modelo de clasificación supervisada capaz de determinar la presencia o ausencia de enfermedad cardíaca a partir de variables clínicas y fisiológicas, como colesterol, glucosa, género, altura, peso y hábitos de salud del paciente. Formalmente, se busca modelar la relación entre un conjunto de características y una variable objetivo binaria asociada al diagnóstico cardiovascular."),
                body("Este problema tiene alta relevancia debido a que las enfermedades cardiovasculares representan una de las principales causas de mortalidad a nivel mundial. Por ello, un modelo predictivo confiable puede contribuir a la detección temprana, el diagnóstico oportuno y la optimización de recursos en los sistemas de salud. Asimismo, es fundamental garantizar un buen desempeño del modelo, ya que errores de clasificación podrían generar diagnósticos omitidos o intervenciones médicas innecesarias."),
            ], extra={"marginBottom":"0","height":"100%","boxSizing":"border-box"})],
            style={"flex":"1","marginRight":"16px"}),
            html.Div([img_card("https://media.istockphoto.com/id/2165374222/es/foto/mujer-con-dolor-en-el-coraz%C3%B3n-mujer-madura-presiona-la-mano-contra-el-pecho-tiene-un-ataque-al.webp?a=1&b=1&s=612x612&w=0&k=20&c=g7q2cvFhnAkbYunzjWGWDWHNzAbEYY0cAyKV_9vJJMc=")], style={"flex":"1"}),
        ], style={"display":"flex","marginBottom":"24px","alignItems":"stretch"}),

        html.Div([
            html.Div([img_card("https://media.istockphoto.com/id/2223338895/es/foto/ataque-card%C3%ADaco-y-enfermedad-card%C3%ADaca-fondo-de-ecg-ilustraci%C3%B3n-3d.webp?a=1&b=1&s=612x612&w=0&k=20&c=3RNc2lyyG5Nk6Cc0YIePvuJ_ZzzqZMr7zSS9ibsLGsk=")], style={"flex":"1","marginRight":"16px"}),
            html.Div([card([
                sec_title("Dataset Escogido"),
                body("El conjunto de datos seleccionado corresponde a Heart Failure Diagnosis Data for Machine Learning, disponible en Kaggle y publicado por Alam Shihab. Este dataset contiene información de 70,000 individuos e incluye variables demográficas, antropométricas y clínicas relacionadas con el estado de salud cardiovascular de los pacientes."),
                body("Su estructura lo hace adecuado para el desarrollo de modelos de clasificación binaria orientados a la predicción de enfermedades cardíacas mediante técnicas de machine learning."),
                html.A("Ver dataset en Kaggle →",
                    href="https://www.kaggle.com/datasets/alamshihab075/heart-failure-diagnosis-data-for-machine-learning/data",
                    target="_blank",
                    style={"fontFamily":FONT_BODY,"color":COLOR_ACCENT2,"fontSize":"0.9rem","fontWeight":"500","textDecoration":"none"}),
            ], extra={"marginBottom":"0","height":"100%","boxSizing":"border-box"})],
            style={"flex":"1"}),
        ], style={"display":"flex","marginBottom":"24px","alignItems":"stretch"}),

        card([
            sec_title("Diccionario de Variables"),
            html.P("Descripción de cada variable del dataset: tipo, unidad de medida y significado.",
                style={"fontFamily":FONT_BODY,"color":COLOR_MUTED,"fontSize":"0.88rem","marginBottom":"16px","marginTop":"0"}),
            dash_table.DataTable(
                data=df_vars.to_dict("records"),
                columns=[{"name":c,"id":c} for c in df_vars.columns],
                style_table={"overflowX":"auto","borderRadius":"8px","border":f"1px solid {COLOR_BORDER}"},
                style_header={"backgroundColor":COLOR_PRIMARY,"color":"white","fontWeight":"600",
                    "fontFamily":FONT_BODY,"fontSize":"0.82rem","padding":"12px 16px",
                    "border":"none","textTransform":"uppercase","letterSpacing":"0.05em"},
                style_cell={"fontFamily":FONT_BODY,"fontSize":"0.88rem","color":COLOR_TEXT,
                    "padding":"10px 16px","border":f"1px solid {COLOR_BORDER}",
                    "textAlign":"left","whiteSpace":"normal","height":"auto"},
                style_data_conditional=[
                    {"if":{"row_index":"odd"},"backgroundColor":"#F7F9FC"},
                    {"if":{"column_id":"Variable"},"fontWeight":"600","color":COLOR_PRIMARY},
                    {"if":{"column_id":"Tipo","filter_query":'{Tipo} contains "objetivo"'},
                     "backgroundColor":"#FFF0F0","color":COLOR_ACCENT,"fontWeight":"600"},
                ],
            ),
        ]),
    ], style={"padding":"32px 40px","background":COLOR_BG,"minHeight":"100vh"})

# ══════════════════════════════════════════════
# TAB 2
# ══════════════════════════════════════════════
def make_tab2():

    return html.Div([

        html.H2(
            "EDA — Análisis Exploratorio de Datos",
            style={
                "fontFamily": FONT_TITLE,
                "color": COLOR_PRIMARY,
                "fontSize": "1.8rem",
                "marginBottom": "4px",
                "marginTop": "0"
            }
        ),

        html.P(
            f"Dataset: {len(df_raw)} registros · {len(df_raw.columns)} variables",
            style={
                "fontFamily": FONT_BODY,
                "color": COLOR_MUTED,
                "fontSize": "0.9rem",
                "marginBottom": "28px"
            }
        ),

        # =====================================================
        # SELECTOR PRINCIPAL
        # =====================================================

        card([

            lbl("Tipo de análisis"),

            dcc.Dropdown(
                id="eda-section",
                options=[
                    {
                        "label": "📊 Análisis Unidimensional",
                        "value": "uni"
                    },
                    {
                        "label": "🔀 Análisis Bidimensional",
                        "value": "bi"
                    },
                    {
                        "label": "🧩 Correlación",
                        "value": "corr"
                    }
                ],
                value="uni",
                clearable=False,
                style={
                    "fontFamily": FONT_BODY,
                    "fontSize": "0.95rem"
                }
            )

        ], extra={"maxWidth": "420px"}),

        # =====================================================
        # CONTENIDO DINÁMICO
        # =====================================================

        html.Div(id="eda-dynamic-content")

    ], style={
        "padding": "32px 40px",
        "background": COLOR_BG,
        "minHeight": "100vh"
    })
    
# ══════════════════════════════════════════════
# TAB 3 — MODELOS (sin inferencia, solo CSVs)
# ══════════════════════════════════════════════
def make_tab3():
    return html.Div([
        html.Div([
            html.Div([
                html.Span("Machine Learning", style={"fontFamily":FONT_BODY,"color":COLOR_ACCENT,
                    "fontSize":"0.85rem","fontWeight":"600","letterSpacing":"0.15em","textTransform":"uppercase"}),
                html.H1("Modelos Predictivos", style={"fontFamily":FONT_TITLE,"color":"white",
                    "fontSize":"2.2rem","fontWeight":"700","marginTop":"8px","marginBottom":"8px","lineHeight":"1.2"}),
                html.P("Evaluación comparativa de clasificadores para diagnóstico de insuficiencia cardíaca",
                    style={"fontFamily":FONT_BODY,"color":"rgba(255,255,255,0.7)","fontSize":"1rem","fontWeight":"300","marginTop":"0"}),
            ], style={"flex":"1"}),
            html.Div("⚙", style={"fontSize":"5rem","color":"white","opacity":"0.1",
                                  "lineHeight":"1","alignSelf":"center","marginLeft":"24px"}),
        ], style={"display":"flex","justifyContent":"space-between",
            "background":f"linear-gradient(135deg,{COLOR_PRIMARY} 0%,#2A5A8C 100%)",
            "borderRadius":"16px","padding":"40px 48px","marginBottom":"28px"}),

        card([
            sec_title("Seleccionar Modelos"),
            html.P("Elige uno o más modelos para comparar. Las métricas detalladas corresponden al último modelo seleccionado.",
                style={"fontFamily":FONT_BODY,"color":COLOR_MUTED,"fontSize":"0.88rem","marginTop":"0","marginBottom":"16px"}),
            dcc.Dropdown(
                id="model-selector",
                options=[{"label":k,"value":k} for k in MODEL_FOLDERS],
                value=["Random Forest"],
                multi=True, clearable=False,
                style={"fontFamily":FONT_BODY,"borderRadius":"8px","fontSize":"0.95rem"},
            ),
        ]),

        html.Div(id="models-output"),
    ], style={"padding":"32px 40px","background":COLOR_BG,"minHeight":"100vh"})

# ══════════════════════════════════════════════
# ESTILOS TABS
# ══════════════════════════════════════════════
TAB_S   = {"fontFamily":FONT_BODY,"fontWeight":"500","fontSize":"0.9rem","color":COLOR_MUTED,
           "padding":"14px 28px","borderBottom":"3px solid transparent","background":"transparent","border":"none"}
TAB_SEL = {**TAB_S,"color":COLOR_PRIMARY,"borderBottom":f"3px solid {COLOR_ACCENT}","fontWeight":"600"}

# ══════════════════════════════════════════════
# LAYOUT
# ══════════════════════════════════════════════
app.layout = html.Div([
    GOOGLE_FONTS,
    html.Div([
        html.Div([
            html.Span("♥", style={"color":COLOR_ACCENT,"fontSize":"1.2rem","marginRight":"8px"}),
            html.Span("CardioML", style={"fontFamily":FONT_TITLE,"color":COLOR_PRIMARY,"fontWeight":"700","fontSize":"1.1rem"}),
        ], style={"display":"flex","alignItems":"center"}),
        html.Span("Heart Failure Prediction · Dashboard",
            style={"fontFamily":FONT_BODY,"color":COLOR_MUTED,"fontSize":"0.82rem"}),
    ], style={"display":"flex","justifyContent":"space-between","alignItems":"center",
        "padding":"16px 40px","background":COLOR_CARD,"borderBottom":f"1px solid {COLOR_BORDER}",
        "position":"sticky","top":"0","zIndex":"100","boxShadow":"0 2px 8px rgba(26,58,92,0.05)"}),

    dcc.Tabs(id="main-tabs", value="tab-intro", children=[
        dcc.Tab(label="Introducción", value="tab-intro", style=TAB_S, selected_style=TAB_SEL),
        dcc.Tab(label="EDA",          value="tab-eda",   style=TAB_S, selected_style=TAB_SEL),
        dcc.Tab(label="Modelos",      value="tab-models",style=TAB_S, selected_style=TAB_SEL),
    ], style={"background":COLOR_CARD,"paddingLeft":"32px","borderBottom":f"1px solid {COLOR_BORDER}"}),

    html.Div(id="tab-content"),
], style={"background":COLOR_BG,"minHeight":"100vh"})

# ══════════════════════════════════════════════
# CALLBACKS
# ══════════════════════════════════════════════
@app.callback(Output("tab-content","children"), Input("main-tabs","value"))
def render_tab(tab):
    if tab=="tab-intro": return make_tab1()
    elif tab=="tab-eda": return make_tab2()
    else:                return make_tab3()

# ══════════════════════════════════════════════
# IMPORTS NECESARIOS
# ══════════════════════════════════════════════
from dash.exceptions import PreventUpdate


# ══════════════════════════════════════════════
# CALLBACK — RENDER DINÁMICO DEL EDA
# ══════════════════════════════════════════════
@app.callback(
    Output("eda-dynamic-content", "children"),
    Input("eda-section", "value")
)
def render_eda_section(section):

    # =====================================================
    # ANÁLISIS UNIDIMENSIONAL
    # =====================================================

    if section == "uni":

        return html.Div([

            # ─────────────────────────────────────
            # VARIABLES NUMÉRICAS
            # ─────────────────────────────────────

            card([

                sec_title("Variables Numéricas"),

                body(
                    "Distribución univariada y densidades "
                    "condicionadas por enfermedad cardiovascular."
                ),

                dcc.Dropdown(
                    id="num-var",
                    options=[
                        {"label": c, "value": c}
                        for c in EDA_NUM_COLS
                    ],
                    value=EDA_NUM_COLS[0],
                    clearable=False,
                    style={"marginBottom": "20px"}
                ),

                html.Div([

                    html.Div(
                        dcc.Graph(id="hist-graph"),
                        style={
                            "flex": "1",
                            "marginRight": "12px"
                        }
                    ),

                    html.Div(
                        dcc.Graph(id="box-graph"),
                        style={"flex": "1"}
                    )

                ], style={"display": "flex"}),

                dcc.Graph(id="density-graph")

            ]),

            # ─────────────────────────────────────
            # VARIABLES CATEGÓRICAS
            # ─────────────────────────────────────

            card([

                sec_title("Variables Categóricas"),

                body(
                    "Distribución de variables categóricas."
                ),

                dcc.Dropdown(
                    id="cat-var",
                    options=[
                        {"label": c, "value": c}
                        for c in EDA_CAT_COLS
                    ],
                    value=EDA_CAT_COLS[0],
                    clearable=False,
                    style={"marginBottom": "20px"}
                ),

                dcc.Graph(id="cat-graph")

            ])

        ])

    # =====================================================
    # ANÁLISIS BIDIMENSIONAL
    # =====================================================

    elif section == "bi":

        return html.Div([

            # ─────────────────────────────────────
            # SCATTER
            # ─────────────────────────────────────

            card([

                sec_title("Scatter Plot"),

                html.Div([

                    html.Div([

                        lbl("Variable X"),

                        dcc.Dropdown(
                            id="scatter-x",
                            options=[
                                {"label": c, "value": c}
                                for c in CONT_COLS
                            ],
                            value=CONT_COLS[0],
                            clearable=False
                        )

                    ], style={
                        "flex": "1",
                        "marginRight": "12px"
                    }),

                    html.Div([

                        lbl("Variable Y"),

                        dcc.Dropdown(
                            id="scatter-y",
                            options=[
                                {"label": c, "value": c}
                                for c in CONT_COLS
                            ],
                            value=CONT_COLS[1],
                            clearable=False
                        )

                    ], style={"flex": "1"})

                ], style={"display": "flex"}),

                dcc.Graph(id="scatter-graph")

            ]),

            # ─────────────────────────────────────
            # BOXPLOT NUM vs CAT
            # ─────────────────────────────────────

            card([

                sec_title("Boxplot — Numérica vs Categórica"),

                html.Div([

                    html.Div([

                        lbl("Variable Numérica"),

                        dcc.Dropdown(
                            id="box-num",
                            options=[
                                {"label": c, "value": c}
                                for c in EDA_NUM_COLS
                            ],
                            value=EDA_NUM_COLS[0],
                            clearable=False
                        )

                    ], style={
                        "flex": "1",
                        "marginRight": "12px"
                    }),

                    html.Div([

                        lbl("Variable Categórica"),

                        dcc.Dropdown(
                            id="box-cat",
                            options=[
                                {"label": c, "value": c}
                                for c in EDA_CAT_COLS
                            ],
                            value=EDA_CAT_COLS[0],
                            clearable=False
                        )

                    ], style={"flex": "1"})

                ], style={"display": "flex"}),

                dcc.Graph(id="box-cat-num-graph")

            ]),

            # ─────────────────────────────────────
            # CONTINGENCIA
            # ─────────────────────────────────────

            card([

                sec_title("Tablas de Contingencia"),

                html.Div([

                    html.Div([

                        lbl("Variable 1"),

                        dcc.Dropdown(
                            id="cont-cat1",
                            options=[
                                {"label": c, "value": c}
                                for c in EDA_CAT_COLS
                            ],
                            value=EDA_CAT_COLS[0],
                            clearable=False
                        )

                    ], style={
                        "flex": "1",
                        "marginRight": "12px"
                    }),

                    html.Div([

                        lbl("Variable 2"),

                        dcc.Dropdown(
                            id="cont-cat2",
                            options=[
                                {"label": c, "value": c}
                                for c in EDA_CAT_COLS
                            ],
                            value=EDA_CAT_COLS[1],
                            clearable=False
                        )

                    ], style={"flex": "1"})

                ], style={"display": "flex"}),

                html.Div(id="contingency-table")

            ])

        ])

    # =====================================================
    # CORRELACIÓN
    # =====================================================

    elif section == "corr":

        return html.Div([

            card([

                sec_title("Matriz de Correlación"),

                body(
                    "Correlación lineal entre todas las variables."
                ),

                dcc.Graph(id="corr-matrix")

            ])

        ])

    raise PreventUpdate


# ══════════════════════════════════════════════
# CALLBACK — VARIABLES NUMÉRICAS
# ══════════════════════════════════════════════
@app.callback(
    Output("hist-graph", "figure"),
    Output("box-graph", "figure"),
    Output("density-graph", "figure"),
    Input("num-var", "value")
)
def update_numeric(var):

    if var is None:
        raise PreventUpdate

    # HISTOGRAMA
    hist = px.histogram(
        df_raw,
        x=var,
        nbins=30,
        color_discrete_sequence=[COLOR_ACCENT2]
    )

    hist.update_layout(
        title=f"Histograma — {var}",
        **PLOT_BASE
    )

    # BOXPLOT
    box = px.box(
        df_raw,
        y=var,
        color_discrete_sequence=[COLOR_ACCENT]
    )

    box.update_layout(
        title=f"Boxplot — {var}",
        **PLOT_BASE
    )

    # DENSIDAD
    df_plot = df_raw.copy()

    df_plot["cardio_label"] = (
        df_plot["cardio"]
        .map(labels_map["cardio"])
        .astype(str)
    )

    density = px.histogram(
        df_plot,
        x=var,
        color="cardio_label",
        marginal="violin",
        histnorm="density",
        opacity=0.55,
        barmode="overlay"
    )

    density.update_layout(
        title=f"Densidad por cardio — {var}",
        **PLOT_BASE
    )

    return hist, box, density


# ══════════════════════════════════════════════
# CALLBACK — VARIABLES CATEGÓRICAS
# ══════════════════════════════════════════════
@app.callback(
    Output("cat-graph", "figure"),
    Input("cat-var", "value")
)
def update_cat(cat):

    if cat is None:
        raise PreventUpdate

    df_plot = df_raw.copy()

    df_plot[cat] = (
        df_plot[cat]
        .map(labels_map[cat])
        .astype(str)
    )

    fig = px.histogram(
        df_plot,
        x=cat,
        color=cat,
        text_auto=True
    )

    fig.update_layout(
        title=f"Distribución de {cat}",
        showlegend=False,
        **PLOT_BASE
    )

    return fig


# ══════════════════════════════════════════════
# CALLBACK — SCATTER
# ══════════════════════════════════════════════
@app.callback(
    Output("scatter-graph", "figure"),
    Input("scatter-x", "value"),
    Input("scatter-y", "value")
)
def update_scatter(x, y):

    if x is None or y is None:
        raise PreventUpdate

    fig = px.scatter(
        df_raw,
        x=x,
        y=y,
        color=df_raw["cardio"].astype(str),
        opacity=0.6
    )

    fig.update_layout(
        title=f"{x} vs {y}",
        **PLOT_BASE
    )

    return fig


# ══════════════════════════════════════════════
# CALLBACK — BOXPLOT NUM vs CAT
# ══════════════════════════════════════════════
@app.callback(
    Output("box-cat-num-graph", "figure"),
    Input("box-num", "value"),
    Input("box-cat", "value")
)
def update_box_cat(num_var, cat_var):

    if num_var is None or cat_var is None:
        raise PreventUpdate

    df_plot = df_raw.copy()

    df_plot[cat_var] = (
        df_plot[cat_var]
        .map(labels_map[cat_var])
        .astype(str)
    )

    fig = px.box(
        df_plot,
        x=cat_var,
        y=num_var,
        color=cat_var
    )

    fig.update_layout(
        title=f"{num_var} vs {cat_var}",
        showlegend=False,
        **PLOT_BASE
    )

    return fig


# ══════════════════════════════════════════════
# CALLBACK — TABLA DE CONTINGENCIA
# ══════════════════════════════════════════════
@app.callback(
    Output("contingency-table", "children"),
    Input("cont-cat1", "value"),
    Input("cont-cat2", "value")
)
def update_contingency(cat1, cat2):

    if cat1 is None or cat2 is None:
        raise PreventUpdate

    df_plot = df_raw.copy()

    df_plot[cat1] = (
        df_plot[cat1]
        .map(labels_map[cat1])
        .astype(str)
    )

    df_plot[cat2] = (
        df_plot[cat2]
        .map(labels_map[cat2])
        .astype(str)
    )

    ct = pd.crosstab(
        df_plot[cat1],
        df_plot[cat2]
    ).reset_index()

    return dash_table.DataTable(

        data=ct.to_dict("records"),

        columns=[
            {"name": c, "id": c}
            for c in ct.columns
        ],

        style_table={
            "overflowX": "auto",
            "marginTop": "20px"
        },

        style_header={
            "backgroundColor": COLOR_PRIMARY,
            "color": "white",
            "fontWeight": "600"
        },

        style_cell={
            "textAlign": "center",
            "padding": "12px",
            "fontFamily": FONT_BODY
        }
    )


# ══════════════════════════════════════════════
# CALLBACK — MATRIZ DE CORRELACIÓN
# ══════════════════════════════════════════════
@app.callback(
    Output("corr-matrix", "figure"),
    Input("eda-section", "value")
)
def update_corr(_):

    corr = df_raw.corr(numeric_only=True)

    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r"
    )

    layout_corr = PLOT_BASE.copy()
    layout_corr["height"] = 700

    fig.update_layout(
        title="Correlation Matrix",
        **layout_corr
    )

    return fig

# ── CALLBACK MODELOS — solo lectura de CSVs ──
@app.callback(
    Output("models-output","children"),
    Input("model-selector","value"),
)
def render_models(selected_models):
    if not selected_models:
        return html.P("Selecciona al menos un modelo.",
            style={"fontFamily":FONT_BODY,"color":COLOR_MUTED})

    last_name = selected_models[-1]

    # ── Cargar métricas de todos los modelos seleccionados ──
    all_metrics = []
    for name in selected_models:
        df_m = load_metrics(name)
        if df_m is None:
            continue
        row = df_m.iloc[0].to_dict()
        row["Modelo"] = name
        all_metrics.append(row)

    if not all_metrics:
        return html.P("No se encontraron archivos de métricas para los modelos seleccionados.",
            style={"fontFamily":FONT_BODY,"color":COLOR_ACCENT})

    # ── Métricas del último modelo ──
    last_metrics_df = load_metrics(last_name)
    if last_metrics_df is None:
        last_name = next(r["Modelo"] for r in all_metrics)
        last_metrics_df = load_metrics(last_name)

    last_m = last_metrics_df.iloc[0].to_dict()

    # ════════════════════════════════════════════
    # 1. CONFUSION MATRIX — desde TN, FP, FN, TP del CSV
    # ════════════════════════════════════════════
    tn = int(last_m.get("TN", 0))
    fp = int(last_m.get("FP", 0))
    fn = int(last_m.get("FN", 0))
    tp = int(last_m.get("TP", 0))
    cm = np.array([[tn, fp],[fn, tp]])
    labels = ["No Heart Failure (0)", "Heart Failure (1)"]

    cm_fig = go.Figure()
    cm_fig.add_trace(go.Heatmap(
        z=[[tn,fp],[fn,tp]], x=labels, y=labels,
        colorscale=[[0,"#EBF5FF"],[0.5,"#4A9BBF"],[1,"#1A3A5C"]],
        showscale=False,
    ))
    annotations = []
    subcaptions = {(0,0):"VN",(0,1):"FP",(1,0):"FN",(1,1):"VP"}
    for i in range(2):
        for j in range(2):
            val = cm[i,j]
            annotations.append(dict(
                x=labels[j], y=labels[i], text=f"<b>{val}</b>",
                showarrow=False,
                font=dict(size=26, family=FONT_TITLE,
                    color="white" if val > cm.max()/2 else COLOR_PRIMARY)))
            annotations.append(dict(
                x=labels[j], y=labels[i], text=subcaptions[(i,j)],
                showarrow=False, yshift=-22,
                font=dict(size=11, family=FONT_BODY,
                    color="rgba(255,255,255,0.7)" if val > cm.max()/2 else COLOR_MUTED)))
    cm_fig.update_layout(
        title=f"Confusion Matrix — {last_name}",
        xaxis_title="Predicción", yaxis_title="Real",
        annotations=annotations,
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family=FONT_BODY, color=COLOR_TEXT),
        margin=dict(l=80,r=40,t=60,b=60), height=380,
        xaxis=dict(side="bottom"), yaxis=dict(autorange="reversed"),
    )

    acc         = float(last_m.get("Accuracy",   0))
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    def metric_badge(value, label, color):
        return html.Div([
            html.P(value, style={"fontFamily":FONT_TITLE,"fontSize":"1.5rem","fontWeight":"700","color":color,"margin":"0"}),
            html.P(label, style={"fontFamily":FONT_BODY,"fontSize":"0.72rem","color":COLOR_MUTED,"margin":"0","textTransform":"uppercase"}),
        ], style={"flex":"1","textAlign":"center","padding":"16px","background":"#F7F9FC",
            "borderRadius":"8px","border":f"1px solid {COLOR_BORDER}"})

    quick_metrics = html.Div([
        metric_badge(f"{acc:.2%}",         "Accuracy",              COLOR_PRIMARY),
        metric_badge(f"{sensitivity:.2%}", "Sensibilidad (Recall 1)", COLOR_ACCENT2),
        metric_badge(f"{specificity:.2%}", "Especificidad (Recall 0)", COLOR_ACCENT),
        metric_badge(str(tp + fp),         "Predichos positivos",   COLOR_PRIMARY),
    ], style={"display":"flex","gap":"12px","marginTop":"16px"})

    cm_block = card([
        sec_title(f"Confusion Matrix — {last_name}"),
        html.Div([dcc.Graph(figure=cm_fig, config={"displayModeBar":False})], style=CHART_WRAP),
        quick_metrics,
    ])

    # ════════════════════════════════════════════
    # 2. TABLA COMPARATIVA DE MÉTRICAS
    # ════════════════════════════════════════════
    metric_cols = [
    "Modelo",
    "ROC_AUC_Test",
    "Accuracy",
    "Precision",
    "Recall",
    "F1_Score"
]

    col_labels = {
    "ROC_AUC_Test": "AUC-ROC",
    "F1_Score": "F1 Score"
}

    df_table = pd.DataFrame(all_metrics)
    # Conservar solo columnas que existan
    keep = [c for c in metric_cols if c in df_table.columns]
    df_table = df_table[keep].copy()
    df_table.rename(columns=col_labels, inplace=True)

    # ── Reordenar columnas dejando AUC-ROC al inicio ──
    ordered_cols = [
    "Modelo",
    "AUC-ROC",
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]

    df_table = df_table[ordered_cols]

    # Redondear numéricas
    for col in df_table.columns:
        if col != "Modelo":
            df_table[col] = pd.to_numeric(df_table[col], errors="coerce").round(4)

    cond_styles = [{"if":{"row_index":"odd"},"backgroundColor":"#F7F9FC"}]
    for i, row in df_table.iterrows():
        if row["Modelo"] == last_name:
            cond_styles.append({"if":{"row_index":i},"backgroundColor":"#EBF5FF","fontWeight":"600"})

    metrics_block = card([
        sec_title("Comparativa de Métricas"),
        html.P("La fila resaltada en azul corresponde al último modelo seleccionado.",
            style={"fontFamily":FONT_BODY,"color":COLOR_MUTED,"fontSize":"0.85rem","marginTop":"0","marginBottom":"16px"}),
        dash_table.DataTable(
            data=df_table.to_dict("records"),
            columns=[{"name":c,"id":c} for c in df_table.columns],
            style_table={"overflowX":"auto","borderRadius":"8px","border":f"1px solid {COLOR_BORDER}"},
            style_header={"backgroundColor":COLOR_PRIMARY,"color":"white","fontWeight":"600",
                "fontFamily":FONT_BODY,"fontSize":"0.82rem","padding":"12px 16px",
                "border":"none","textTransform":"uppercase","letterSpacing":"0.05em"},
            style_cell={"fontFamily":FONT_BODY,"fontSize":"0.9rem","color":COLOR_TEXT,
                "padding":"12px 16px","border":f"1px solid {COLOR_BORDER}","textAlign":"center"},
            style_cell_conditional=[{"if":{"column_id":"Modelo"},"textAlign":"left","fontWeight":"500"}],
            style_data_conditional=cond_styles,
        ),
    ])

    # ════════════════════════════════════════════
    # 3. CURVA ROC — desde roc_curve.csv
    # ════════════════════════════════════════════
    roc_fig = go.Figure()
    roc_fig.add_shape(type="line",x0=0,y0=0,x1=1,y1=1,
        line=dict(color=COLOR_BORDER,width=1.5,dash="dash"))

    for name in selected_models:
        df_roc = load_roc(name)
        if df_roc is None:
            continue
        # Normalizar nombres de columnas (FPR/fpr, TPR/tpr)
        df_roc.columns = [c.strip().lower() for c in df_roc.columns]
        fpr_col = next((c for c in df_roc.columns if "fpr" in c or "false" in c), None)
        tpr_col = next((c for c in df_roc.columns if "tpr" in c or "true"  in c), None)
        if fpr_col is None or tpr_col is None:
            continue
        fpr = df_roc[fpr_col].values
        tpr = df_roc[tpr_col].values
        # AUC desde métricas CSV — búsqueda flexible del nombre de columna
        df_m2 = load_metrics(name)
        roc_auc_val = 0.0
        if df_m2 is not None:
            roc_col = next((c for c in df_m2.columns if "roc" in c.lower() or "auc" in c.lower()), None)
            if roc_col:
                roc_auc_val = float(df_m2.iloc[0][roc_col])
        roc_fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"{name} (AUC={roc_auc_val:.3f})",
            line=dict(color=MODEL_COLORS.get(name, COLOR_ACCENT2), width=2.5),
        ))

    roc_fig.update_layout(
        title="Curvas ROC-AUC",
        xaxis_title="Tasa de Falsos Positivos (FPR)",
        yaxis_title="Tasa de Verdaderos Positivos (TPR)",
        legend=dict(x=0.62,y=0.08,bgcolor="rgba(255,255,255,0.85)",
            bordercolor=COLOR_BORDER,borderwidth=1,font=dict(family=FONT_BODY,size=11)),
        **PLOT_BASE,
    )
    roc_block = card([
        sec_title("Curva ROC-AUC"),
        html.Div([dcc.Graph(figure=roc_fig, config={"displayModeBar":False})], style=CHART_WRAP),
    ])

    # ════════════════════════════════════════════
    # 4. CLASSIFICATION REPORT — desde Classification_report.csv
    # ════════════════════════════════════════════
    df_cr = load_cr(last_name)
    if df_cr is not None:
        # Limpiar nombres de columnas
        df_cr.columns = [c.strip() for c in df_cr.columns]

        # La primera columna (puede venir como "Unnamed: 0" o vacía) → renombrar a "Clase"
        first_col = df_cr.columns[0]
        df_cr = df_cr.rename(columns={first_col: "Clase"})
        class_col = "Clase"

        # Mapear etiquetas de clase
        label_map = {
            "0":           "Clase 0 — No Heart Failure",
            "1":           "Clase 1 — Heart Failure",
            "macro avg":   "Macro Avg",
            "weighted avg":"Weighted Avg",
            "accuracy":    "Accuracy",
        }
        df_cr[class_col] = df_cr[class_col].astype(str).map(lambda x: label_map.get(x.strip(), x))

        # Columnas numéricas del CR (todo excepto "Clase")
        num_cr_cols = [c for c in df_cr.columns if c != class_col]

        df_cr[num_cr_cols] = df_cr[num_cr_cols].astype(object)

        # Round a 4 decimales en columnas numéricas
        for col in num_cr_cols:
            df_cr[col] = pd.to_numeric(df_cr[col], errors="coerce").round(4)

        # Para la fila "Accuracy": colocar "--" en todas las columnas excepto
        # la penúltima (f1-score) y la última (support), siguiendo el
        # formato estándar del classification_report de sklearn.
        # La penúltima columna numérica = el score de accuracy; la última = support.
        acc_mask = df_cr[class_col] == "Accuracy"
        if acc_mask.any() and len(num_cr_cols) >= 2:
            score_col   = num_cr_cols[-2]   # penúltima → f1-score / accuracy value
            support_col = num_cr_cols[-1]    # última    → support
            blank_cols  = num_cr_cols[:-2]   # todas las anteriores → "--"
            for col in blank_cols:
                df_cr.loc[acc_mask, col] = "--"

        cr_cond = [
            {"if":{"row_index":"odd"},"backgroundColor":"#F7F9FC"},
            {"if":{"filter_query":f'{{{class_col}}} contains "Weighted"'},
             "backgroundColor":"#EBF5FF","fontWeight":"600"},
            {"if":{"filter_query":f'{{{class_col}}} contains "Macro"'},
             "backgroundColor":"#FFF0F0"},
        ]
        cr_content = dash_table.DataTable(
            data=df_cr.to_dict("records"),
            columns=[{"name":c,"id":c} for c in df_cr.columns],
            style_table={"overflowX":"auto","borderRadius":"8px","border":f"1px solid {COLOR_BORDER}"},
            style_header={"backgroundColor":COLOR_PRIMARY,"color":"white","fontWeight":"600",
                "fontFamily":FONT_BODY,"fontSize":"0.82rem","padding":"12px 16px",
                "border":"none","textTransform":"uppercase","letterSpacing":"0.05em"},
            style_cell={"fontFamily":FONT_BODY,"fontSize":"0.9rem","color":COLOR_TEXT,
                "padding":"12px 16px","border":f"1px solid {COLOR_BORDER}","textAlign":"center"},
            style_cell_conditional=[{"if":{"column_id":class_col},"textAlign":"left","fontWeight":"500"}],
            style_data_conditional=cr_cond,
        )
    else:
        cr_content = html.P(f"No se encontró Classification_report.csv para {last_name}.",
            style={"fontFamily":FONT_BODY,"color":COLOR_ACCENT,"fontSize":"0.9rem"})

    cr_block = card([sec_title(f"Classification Report — {last_name}"), cr_content])

    # ════════════════════════════════════════════
    # 5. LIME — desde lime/lime_explanations.csv
    # Agrupa por feature y promedia absolute_weight
    # ════════════════════════════════════════════
    df_lime = load_lime(last_name)
    if df_lime is not None:
        df_lime.columns = [c.strip().lower() for c in df_lime.columns]

        feat_col  = next((c for c in df_lime.columns if "feature" in c), None)
        weight_col= next((c for c in df_lime.columns if "absolute" in c), None) or \
                    next((c for c in df_lime.columns if "weight"   in c), None)

        if feat_col and weight_col:
            feat_df = (
                df_lime.groupby(feat_col)[weight_col]
                .mean()
                .reset_index()
                .rename(columns={feat_col:"Feature", weight_col:"Importancia"})
                .sort_values("Importancia")
            )
            total = feat_df["Importancia"].sum() or 1
            feat_df["Importancia"] = feat_df["Importancia"] / total

            lime_fig = go.Figure(go.Bar(
                x=feat_df["Importancia"],
                y=feat_df["Feature"],
                orientation="h",
                marker=dict(
                    color=feat_df["Importancia"],
                    colorscale=[[0,"#EBF5FF"],[0.5,COLOR_ACCENT2],[1,COLOR_PRIMARY]],
                    showscale=False,
                ),
                text=[f"{v:.3f}" for v in feat_df["Importancia"]],
                textposition="outside",
                textfont=dict(family=FONT_BODY, size=11, color=COLOR_TEXT),
            ))
            lime_fig.update_layout(
                title=f"Feature Importance (LIME) — {last_name}",
                xaxis_title="Importancia media absoluta normalizada",
                yaxis_title="",
                paper_bgcolor="white", plot_bgcolor="#F7F9FC",
                font=dict(family=FONT_BODY, color=COLOR_TEXT),
                margin=dict(l=160, r=80, t=60, b=40),
                height=max(360, len(feat_df) * 36),
                xaxis=dict(showgrid=True, gridcolor=COLOR_BORDER),
                yaxis=dict(showgrid=False),
            )
            n_samples = df_lime["sample_id"].nunique() if "sample_id" in df_lime.columns else "N/A"
            lime_block = card([
                sec_title(f"Feature Importance (LIME) — {last_name}"),
                html.P(f"Importancia media absoluta normalizada calculada sobre {n_samples} muestras del conjunto de test.",
                    style={"fontFamily":FONT_BODY,"color":COLOR_MUTED,
                           "fontSize":"0.85rem","marginTop":"0","marginBottom":"16px"}),
                html.Div([dcc.Graph(figure=lime_fig, config={"displayModeBar":False})], style=CHART_WRAP),
            ])
        else:
            lime_block = card([sec_title("Feature Importance (LIME)"),
                html.P("Columnas esperadas ('feature', 'absolute_weight') no encontradas en el CSV.",
                    style={"fontFamily":FONT_BODY,"color":COLOR_ACCENT,"fontSize":"0.9rem"})])
    else:
        lime_block = card([sec_title("Feature Importance (LIME)"),
            html.P(f"No se encontró lime/lime_explanations.csv para {last_name}.",
                style={"fontFamily":FONT_BODY,"color":COLOR_ACCENT,"fontSize":"0.9rem"})])

    # ── Layout final ──────────────────────────────
        # ════════════════════════════════════════════
    # 6. DELONG TEST
    # ════════════════════════════════════════════

    delong_block = html.Div()

    if len(selected_models) >= 2:

        pred_data = {}

        for model_name in selected_models:

            df_pred = load_predictions(model_name)

            if df_pred is None:
                continue

            df_pred.columns = [c.strip().lower() for c in df_pred.columns]

            y_col = "y_true"
            prob_col = "y_proba"

            if y_col is None or prob_col is None:
                continue

            pred_data[model_name] = {
                "y_true": df_pred[y_col].values,
                "y_prob": df_pred[prob_col].values
            }

        results = []
        raw_pvalues = []
        
        model_pairs = list(combinations(pred_data.keys(), 2))
        
        for m1, m2 in model_pairs:
        
            y_true_1 = pred_data[m1]["y_true"]
            y_true_2 = pred_data[m2]["y_true"]
        
            p1 = pred_data[m1]["y_prob"]
            p2 = pred_data[m2]["y_prob"]
        
            # =====================================================
            # PARCHE: mínimo común de longitud
            # =====================================================
            min_len = min(len(y_true_1), len(y_true_2))
        
            y_true = y_true_1[:min_len]
            p1 = p1[:min_len]
            p2 = p2[:min_len]
        
            # =====================================================
            # DeLong test
            # =====================================================
            p = delong_roc_test(y_true, p1, p2)
        
            p_value = float(p)
        
            raw_pvalues.append(p_value)
        
            results.append({
                "Model_1": m1,
                "Model_2": m2,
                "p_raw": p_value
            })

        # ─────────────────────────────────────
        # Corrección Bonferroni
        # ─────────────────────────────────────

        corrected = multipletests(
            raw_pvalues,
            method='bonferroni'
        )[1]

        for i in range(len(results)):
            results[i]["p_corrected"] = corrected[i]

        # ─────────────────────────────────────
        # Heatmap
        # ─────────────────────────────────────

        matrix = pd.DataFrame(
            np.ones((len(selected_models), len(selected_models))),
            index=selected_models,
            columns=selected_models
        )

        for r in results:

            matrix.loc[r["Model_1"], r["Model_2"]] = r["p_corrected"]
            matrix.loc[r["Model_2"], r["Model_1"]] = r["p_corrected"]

        heatmap_fig = px.imshow(
            matrix,
            text_auto=".3g",
            color_continuous_scale="RdBu_r",
            zmin=0,
            zmax=1,
            aspect="auto"
        )

        layout_corr = PLOT_BASE.copy()
        layout_corr["height"] = 700
        
        heatmap_fig.update_layout(
            title="DeLong Test — p-values",
            xaxis_title="Modelo",
            yaxis_title="Modelo",
            **layout_corr
        )

        delong_table = pd.DataFrame(results)

        
        delong_table.rename(columns={
            "Model_1": "Modelo 1",
            "Model_2": "Modelo 2",
            "p_raw": "P_value",
            "p_corrected": "P corregido"
        }, inplace=True)

        delong_table["Significativo"] = delong_table["P corregido"] < 0.05
        
        delong_table["P_value"] = (
                    delong_table["P_value"]
                    .astype(float)
                    .round(6)
                )

        delong_table["P corregido"] = (
                    delong_table["P corregido"]
                    .astype(float)
                    .round(6)
                )

        delong_block = card([

            sec_title("Prueba Estadística DeLong"),

            html.P(
                "Comparación estadística entre curvas ROC-AUC. "
                "Se aplica corrección de Bonferroni para múltiples comparaciones.",
                style={
                    "fontFamily": FONT_BODY,
                    "color": COLOR_MUTED,
                    "fontSize": "0.9rem",
                    "marginBottom": "18px"
                }
            ),

            html.Div([
                dcc.Graph(
                    figure=heatmap_fig,
                    config={"displayModeBar": False}
                )
            ], style=CHART_WRAP),

            html.Br(),

            dash_table.DataTable(

                data=delong_table.to_dict("records"),

                columns=[
                    {"name": c, "id": c}
                    for c in delong_table.columns
                ],

                style_table={
                    "overflowX": "auto",
                    "borderRadius": "8px",
                    "border": f"1px solid {COLOR_BORDER}"
                },

                style_header={
                    "backgroundColor": COLOR_PRIMARY,
                    "color": "white",
                    "fontWeight": "600",
                    "fontFamily": FONT_BODY
                },

                style_cell={
                    "fontFamily": FONT_BODY,
                    "fontSize": "0.88rem",
                    "padding": "10px",
                    "textAlign": "center"
                },

                style_data_conditional=[
                    {
                        "if": {
                            "filter_query": "{Significativo} = True"
                        },
                        "backgroundColor": "#FFECEC",
                        "color": "#C0392B",
                        "fontWeight": "600"
                    }
                ]
            )

        ])

    # ──────────────────────────────────────────
    # Layout final
    # ──────────────────────────────────────────

    return html.Div([
        two_col(cm_block, metrics_block),
        two_col(roc_block, cr_block),
        lime_block,
        delong_block,
    ])

    


if __name__ == '__main__':
    app.run(debug=False)