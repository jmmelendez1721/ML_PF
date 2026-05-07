import warnings
warnings.filterwarnings("ignore")   # suprimir OHE unknown-categories y otros warnings

from dash import Dash, dcc, html, Input, Output, State, dash_table, ctx
import colorlover as cl
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pickle, os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc
)
import lime
import lime.lime_tabular

# ══════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════
app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# ══════════════════════════════════════════════
# DATOS
# ══════════════════════════════════════════════
df_raw = pd.read_csv("HeartFailureDataset.csv")
df_raw = df_raw.drop(columns=['id'])

X = df_raw.iloc[:, :-1]
Y = df_raw.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, stratify=Y, test_size=0.2, random_state=66
)
FEATURE_NAMES = list(X.columns)

# ══════════════════════════════════════════════
# COLUMNAS EDA
# ══════════════════════════════════════════════
NUM_COLS  = list(df_raw.select_dtypes(include=np.number).columns)
CAT_COLS  = [c for c in NUM_COLS if df_raw[c].nunique() <= 4]
CONT_COLS = [c for c in NUM_COLS if c not in CAT_COLS]

# ══════════════════════════════════════════════
# MODELOS
# ══════════════════════════════════════════════
MODEL_PATHS = {
    "Naive Bayes":          "bayesian/naive_bayes_gridsearch.pkl",
    "Gradient Boosting":    "Gradient Boosting/gradient_boosting_gridsearch.pkl",
    "KNN":                  "KNN/knn_cardio_pipeline.pkl",
    "Logistic Regression":  "Logistic Regression/logistic_regression.pkl",
    "SVM":                  "SVM/svm_gridsearch.pkl",
    "Random Forest":        "Random Forest/random_forest_gridsearch.pkl",
}
# Caché de feature importance por modelo (se calcula una vez y se reutiliza)
FEAT_CACHE = {}

MODEL_COLORS = {
    "Naive Bayes":         "#4A9BBF",
    "Gradient Boosting":   "#E05C5C",
    "KNN":                 "#2ECC71",
    "Logistic Regression": "#9B59B6",
    "SVM":                 "#F39C12",
    "Random Forest":       "#1A3A5C",
}

def load_model(name):
    path = MODEL_PATHS[name]
    if not os.path.exists(path):
        return None
    # joblib es más robusto que pickle para objetos de scikit-learn
    # y resuelve el error STACK_GLOBAL requires str
    try:
        return joblib.load(path)
    except Exception:
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

def get_proba(model, X):
    """Obtener probabilidades compatible con pipelines y modelos sin predict_proba."""
    try:
        return model.predict_proba(X)[:, 1]
    except AttributeError:
        try:
            scores = model.decision_function(X)
            return (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        except Exception:
            return None

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

LOREM = ("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor "
         "incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud "
         "exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.")

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

def img_placeholder(n):
    return html.Div([
        html.Div("🖼", style={"fontSize": "2rem", "marginBottom": "8px"}),
        html.P(f"Imagen de referencia {n}", style={"fontFamily": FONT_BODY, "color": COLOR_MUTED, "fontSize": "0.85rem", "margin": "0"}),
        html.P("(reemplazar con html.Img)", style={"fontFamily": FONT_BODY, "color": COLOR_BORDER, "fontSize": "0.75rem", "margin": "0"}),
    ], style={
        "background": "#F0F4F8", "border": f"2px dashed {COLOR_BORDER}", "borderRadius": "10px",
        "height": "180px", "display": "flex", "flexDirection": "column",
        "alignItems": "center", "justifyContent": "center",
    })

# ══════════════════════════════════════════════
# DICCIONARIO
# ══════════════════════════════════════════════
variables = [
    {"Variable":"Id",          "Tipo":"Numérica",   "Unidad":"NA",            "Significado":"Indicador único para cada paciente."},
    {"Variable":"Age",         "Tipo":"Numérica",   "Unidad":"Días",          "Significado":"Edad del paciente."},
    {"Variable":"Gender",      "Tipo":"Categórica", "Unidad":"NA",            "Significado":"Variable binaria: 1 = Mujer, 2 = Hombre."},
    {"Variable":"Height",      "Tipo":"Numérica",   "Unidad":"Centímetros",   "Significado":"Altura del paciente."},
    {"Variable":"Weight",      "Tipo":"Numérica",   "Unidad":"Kilogramos",    "Significado":"Peso del paciente."},
    {"Variable":"ap_hi",       "Tipo":"Numérica",   "Unidad":"mmHg",          "Significado":"Medición más alta de la presión sanguínea sistólica."},
    {"Variable":"ap_lo",       "Tipo":"Numérica",   "Unidad":"mmHg",          "Significado":"Medición más alta de la presión sanguínea diastólica."},
    {"Variable":"cholesterol", "Tipo":"Categórica", "Unidad":"NA",            "Significado":"Nivel de colesterol: 1 = Normal, 2 = Sobre lo normal, 3 = Muy sobre lo normal."},
    {"Variable":"gluc",        "Tipo":"Categórica", "Unidad":"NA",            "Significado":"Nivel de glucosa: 1 = Normal, 2 = Sobre lo normal, 3 = Muy sobre lo normal."},
    {"Variable":"smoke",       "Tipo":"Categórica", "Unidad":"NA",            "Significado":"Hábito de fumar: 0 = Fumador, 1 = No fumador."},
    {"Variable":"alco",        "Tipo":"Categórica", "Unidad":"NA",            "Significado":"Consumo de alcohol: 0 = Bebedor, 1 = No bebedor."},
    {"Variable":"active",      "Tipo":"Categórica", "Unidad":"NA",            "Significado":"Actividad física: 0 = Activo, 1 = No activo."},
    {"Variable":"cardio",      "Tipo":"Categórica (objetivo)", "Unidad":"NA", "Significado":"Presencia de enfermedad cardiaca (variable objetivo de clasificación)."},
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
        # Hero
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

        # Fila 1: Definición del problema | Imagen 1
        html.Div([
            html.Div([card([
                sec_title("Definición del Problema de Investigación"),
                body("El problema abordado consiste en el desarrollo de un modelo de clasificación supervisada capaz de determinar la presencia o ausencia de enfermedad cardíaca a partir de variables clínicas y fisiológicas, como colesterol, glucosa, género, altura, peso y hábitos de salud del paciente. Formalmente, se busca modelar la relación entre un conjunto de características y una variable objetivo binaria asociada al diagnóstico cardiovascular."),
                body("Este problema tiene alta relevancia debido a que las enfermedades cardiovasculares representan una de las principales causas de mortalidad a nivel mundial. Por ello, un modelo predictivo confiable puede contribuir a la detección temprana, el diagnóstico oportuno y la optimización de recursos en los sistemas de salud. Asimismo, es fundamental garantizar un buen desempeño del modelo, ya que errores de clasificación podrían generar diagnósticos omitidos o intervenciones médicas innecesarias."),
            ], extra={"marginBottom":"0","height":"100%","boxSizing":"border-box"})],
            style={"flex":"1","marginRight":"16px"}),
            html.Div([img_card("https://media.istockphoto.com/id/2165374222/es/foto/mujer-con-dolor-en-el-coraz%C3%B3n-mujer-madura-presiona-la-mano-contra-el-pecho-tiene-un-ataque-al.webp?a=1&b=1&s=612x612&w=0&k=20&c=g7q2cvFhnAkbYunzjWGWDWHNzAbEYY0cAyKV_9vJJMc=")], style={"flex":"1"}),
        ], style={"display":"flex","marginBottom":"24px","alignItems":"stretch"}),

        # Fila 2: Imagen 2 | Dataset escogido
        html.Div([
            html.Div([img_card("https://media.istockphoto.com/id/2223338895/es/foto/ataque-card%C3%ADaco-y-enfermedad-card%C3%ADaca-fondo-de-ecg-ilustraci%C3%B3n-3d.webp?a=1&b=1&s=612x612&w=0&k=20&c=3RNc2lyyG5Nk6Cc0YIePvuJ_ZzzqZMr7zSS9ibsLGsk=")], style={"flex":"1","marginRight":"16px"}),
            html.Div([card([
                sec_title("Dataset Escogido"),
                body("El conjunto de datos seleccionado corresponde a Heart Failure Diagnosis Data for Machine Learning, disponible en Kaggle y publicado por Alam Shihab. Este dataset contiene información de 70,000 individuos e incluye variables demográficas, antropométricas y clínicas relacionadas con el estado de salud cardiovascular de los pacientes."),
                body("Su estructura lo hace adecuado para el desarrollo de modelos de clasificación binaria orientados a la predicción de enfermedades cardíacas mediante técnicas de machine learning."),
                html.A("Ver dataset en Kaggle →", href="https://www.kaggle.com/datasets/alamshihab075/heart-failure-diagnosis-data-for-machine-learning/data", target="_blank",
                    style={"fontFamily":FONT_BODY,"color":COLOR_ACCENT2,"fontSize":"0.9rem",
                           "fontWeight":"500","textDecoration":"none"}),
            ], extra={"marginBottom":"0","height":"100%","boxSizing":"border-box"})],
            style={"flex":"1"}),
        ], style={"display":"flex","marginBottom":"24px","alignItems":"stretch"}),

        # Diccionario
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
                    {"if":{"column_id":"Tipo","filter_query":'{Tipo} contains "objetivo"'},"backgroundColor":"#FFF0F0","color":COLOR_ACCENT,"fontWeight":"600"},
                ],
            ),
        ]),
    ], style={"padding":"32px 40px","background":COLOR_BG,"minHeight":"100vh"})


# ══════════════════════════════════════════════
# TAB 2
# ══════════════════════════════════════════════
def make_tab2():
    ctrl = {"background":COLOR_CARD,"borderRadius":"12px","padding":"20px 24px",
            "marginBottom":"24px","boxShadow":"0 2px 12px rgba(26,58,92,0.07)",
            "border":f"1px solid {COLOR_BORDER}"}
    return html.Div([
        html.H2("EDA — Análisis Exploratorio de Datos",style={"fontFamily":FONT_TITLE,"color":COLOR_PRIMARY,
            "fontSize":"1.8rem","marginBottom":"4px","marginTop":"0"}),
        html.P(f"Dataset: {len(df_raw)} registros · {len(df_raw.columns)} variables",
            style={"fontFamily":FONT_BODY,"color":COLOR_MUTED,"fontSize":"0.9rem","marginBottom":"28px"}),

        html.Div([lbl("Tipo de análisis"),
            dcc.Dropdown(id="eda-tipo",
                options=[{"label":"📊  Análisis Unidimensional","value":"uni"},
                         {"label":"🔀  Análisis Bidimensional","value":"bi"}],
                value="uni",clearable=False,
                style={"fontFamily":FONT_BODY,"fontSize":"0.95rem","borderRadius":"8px"}),
        ],style={**ctrl,"maxWidth":"420px"}),

        html.Div(id="uni-panel",children=[
            html.Div([
                html.Div([lbl("Variable"),dcc.Dropdown(id="uni-var",
                    options=[{"label":c,"value":c} for c in NUM_COLS],
                    value=NUM_COLS[0] if NUM_COLS else None,clearable=False,
                    style={"fontFamily":FONT_BODY,"borderRadius":"8px"})],
                    style={"flex":"1","marginRight":"16px"}),
                html.Div([lbl("Tipo de gráfica"),dcc.Dropdown(id="uni-chart-type",
                    options=[{"label":"Histograma","value":"hist"},
                             {"label":"Box Plot","value":"box"},
                             {"label":"Violin Plot","value":"violin"}],
                    value="hist",clearable=False,
                    style={"fontFamily":FONT_BODY,"borderRadius":"8px"})],
                    style={"flex":"1"}),
            ],style={"display":"flex"}),
        ],style=ctrl),

        html.Div(id="bi-panel",children=[
            html.Div([
                html.Div([lbl("Variable X"),dcc.Dropdown(id="bi-var-x",
                    options=[{"label":c,"value":c} for c in CONT_COLS],
                    value=CONT_COLS[0] if CONT_COLS else None,clearable=False,
                    style={"fontFamily":FONT_BODY,"borderRadius":"8px"})],
                    style={"flex":"1","marginRight":"16px"}),
                html.Div([lbl("Variable Y"),dcc.Dropdown(id="bi-var-y",
                    options=[{"label":c,"value":c} for c in CONT_COLS],
                    value=CONT_COLS[1] if len(CONT_COLS)>1 else (CONT_COLS[0] if CONT_COLS else None),
                    clearable=False,style={"fontFamily":FONT_BODY,"borderRadius":"8px"})],
                    style={"flex":"1","marginRight":"16px"}),
                html.Div([lbl("Color por"),dcc.Dropdown(id="bi-color",
                    options=[{"label":"Ninguno","value":"none"}]+[{"label":c,"value":c} for c in CAT_COLS],
                    value=CAT_COLS[-1] if CAT_COLS else "none",clearable=False,
                    style={"fontFamily":FONT_BODY,"borderRadius":"8px"})],
                    style={"flex":"1"}),
            ],style={"display":"flex"}),
        ],style={**ctrl,"display":"none"}),

        html.Div(id="eda-chart"),
    ],style={"padding":"32px 40px","background":COLOR_BG,"minHeight":"100vh"})

# ══════════════════════════════════════════════
# TAB 3 — MODELOS
# ══════════════════════════════════════════════
def make_tab3():
    model_options = [{"label": k, "value": k} for k in MODEL_PATHS]
    return html.Div([
        # Header
        html.Div([
            html.Div([
                html.Span("Machine Learning", style={"fontFamily":FONT_BODY,"color":COLOR_ACCENT,
                    "fontSize":"0.85rem","fontWeight":"600","letterSpacing":"0.15em","textTransform":"uppercase"}),
                html.H1("Modelos Predictivos", style={"fontFamily":FONT_TITLE,"color":"white",
                    "fontSize":"2.2rem","fontWeight":"700","marginTop":"8px","marginBottom":"8px","lineHeight":"1.2"}),
                html.P("Evaluación comparativa de clasificadores para diagnóstico de insuficiencia cardíaca",
                    style={"fontFamily":FONT_BODY,"color":"rgba(255,255,255,0.7)","fontSize":"1rem","fontWeight":"300","marginTop":"0"}),
            ],style={"flex":"1"}),
            html.Div("⚙", style={"fontSize":"5rem","color":"white","opacity":"0.1","lineHeight":"1",
                                   "alignSelf":"center","marginLeft":"24px"}),
        ],style={"display":"flex","justifyContent":"space-between",
            "background":f"linear-gradient(135deg,{COLOR_PRIMARY} 0%,#2A5A8C 100%)",
            "borderRadius":"16px","padding":"40px 48px","marginBottom":"28px"}),

        # Selector de modelos
        card([
            sec_title("Seleccionar Modelos"),
            html.P("Elige uno o más modelos para comparar. Las métricas detalladas corresponden al último modelo seleccionado.",
                style={"fontFamily":FONT_BODY,"color":COLOR_MUTED,"fontSize":"0.88rem","marginTop":"0","marginBottom":"16px"}),
            dcc.Dropdown(
                id="model-selector",
                options=model_options,
                value=["Random Forest"],
                multi=True,
                clearable=False,
                style={"fontFamily":FONT_BODY,"borderRadius":"8px","fontSize":"0.95rem"},
            ),
        ]),

        # Zona de resultados (se rellena con callback)
        html.Div(id="models-output"),

    ],style={"padding":"32px 40px","background":COLOR_BG,"minHeight":"100vh"})

# ══════════════════════════════════════════════
# ESTILOS TABS
# ══════════════════════════════════════════════
TAB_S   = {"fontFamily":FONT_BODY,"fontWeight":"500","fontSize":"0.9rem","color":COLOR_MUTED,
           "padding":"14px 28px","borderBottom":"3px solid transparent","background":"transparent","border":"none"}
TAB_SEL = {**TAB_S,"color":COLOR_PRIMARY,"borderBottom":f"3px solid {COLOR_ACCENT}","fontWeight":"600"}

# ══════════════════════════════════════════════
# LAYOUT PRINCIPAL
# ══════════════════════════════════════════════
app.layout = html.Div([
    GOOGLE_FONTS,
    html.Div([
        html.Div([
            html.Span("♥",style={"color":COLOR_ACCENT,"fontSize":"1.2rem","marginRight":"8px"}),
            html.Span("CardioML",style={"fontFamily":FONT_TITLE,"color":COLOR_PRIMARY,"fontWeight":"700","fontSize":"1.1rem"}),
        ],style={"display":"flex","alignItems":"center"}),
        html.Span("Heart Failure Prediction · Dashboard",
            style={"fontFamily":FONT_BODY,"color":COLOR_MUTED,"fontSize":"0.82rem"}),
    ],style={"display":"flex","justifyContent":"space-between","alignItems":"center",
        "padding":"16px 40px","background":COLOR_CARD,"borderBottom":f"1px solid {COLOR_BORDER}",
        "position":"sticky","top":"0","zIndex":"100","boxShadow":"0 2px 8px rgba(26,58,92,0.05)"}),

    dcc.Tabs(id="main-tabs",value="tab-intro",children=[
        dcc.Tab(label="Introducción",value="tab-intro",style=TAB_S,selected_style=TAB_SEL),
        dcc.Tab(label="EDA",         value="tab-eda",  style=TAB_S,selected_style=TAB_SEL),
        dcc.Tab(label="Modelos",     value="tab-models",style=TAB_S,selected_style=TAB_SEL),
    ],style={"background":COLOR_CARD,"paddingLeft":"32px","borderBottom":f"1px solid {COLOR_BORDER}"}),

    html.Div(id="tab-content"),
],style={"background":COLOR_BG,"minHeight":"100vh"})

# ══════════════════════════════════════════════
# CALLBACKS
# ══════════════════════════════════════════════

@app.callback(Output("tab-content","children"), Input("main-tabs","value"))
def render_tab(tab):
    if tab=="tab-intro": return make_tab1()
    elif tab=="tab-eda": return make_tab2()
    else:                return make_tab3()


@app.callback(
    Output("uni-panel","style"), Output("bi-panel","style"),
    Input("eda-tipo","value"),
)
def toggle_panels(tipo):
    base={"background":COLOR_CARD,"borderRadius":"12px","padding":"20px 24px",
          "marginBottom":"24px","boxShadow":"0 2px 12px rgba(26,58,92,0.07)","border":f"1px solid {COLOR_BORDER}"}
    show={**base,"display":"block"}; hide={**base,"display":"none"}
    return (show,hide) if tipo=="uni" else (hide,show)


@app.callback(
    Output("eda-chart","children"),
    Input("eda-tipo","value"), Input("uni-var","value"), Input("uni-chart-type","value"),
    Input("bi-var-x","value"), Input("bi-var-y","value"), Input("bi-color","value"),
)
def render_chart(tipo, uni_var, uni_chart_type, bi_var_x, bi_var_y, bi_color):
    if tipo == "uni":
        if not uni_var: return html.Div()
        if uni_chart_type == "hist":
            fig = go.Figure(go.Histogram(x=df_raw[uni_var],nbinsx=30,marker_color=COLOR_ACCENT2,opacity=0.85,name=uni_var))
            fig.update_layout(title=f"Distribución de {uni_var}",**PLOT_BASE)
            stats = df_raw[uni_var].describe().round(3)
            stat_row = html.Div([
                html.Div([
                    html.P(str(round(v,2)),style={"fontFamily":FONT_TITLE,"fontSize":"1.2rem","fontWeight":"700","color":COLOR_PRIMARY,"margin":"0"}),
                    html.P(k,style={"fontFamily":FONT_BODY,"fontSize":"0.7rem","color":COLOR_MUTED,"margin":"0","textTransform":"uppercase"}),
                ],style={"flex":"1","background":"#F7F9FC","borderRadius":"8px","padding":"10px 14px","textAlign":"center","border":f"1px solid {COLOR_BORDER}"})
                for k,v in stats.items()
            ],style={"display":"flex","gap":"8px","marginBottom":"14px"})
            return html.Div([stat_row,html.Div([dcc.Graph(figure=fig,config={"displayModeBar":False})],style=CHART_WRAP)])
        elif uni_chart_type == "box":
            fig = go.Figure(go.Box(y=df_raw[uni_var],name=uni_var,marker_color=COLOR_ACCENT,line_color=COLOR_PRIMARY,fillcolor=COLOR_ACCENT2,opacity=0.7))
            fig.update_layout(title=f"Box Plot – {uni_var}",**PLOT_BASE)
        else:
            fig = go.Figure(go.Violin(y=df_raw[uni_var],name=uni_var,box_visible=True,meanline_visible=True,fillcolor=COLOR_ACCENT2,line_color=COLOR_PRIMARY,opacity=0.75))
            fig.update_layout(title=f"Violin Plot – {uni_var}",**PLOT_BASE)
        return html.Div([dcc.Graph(figure=fig,config={"displayModeBar":False})],style=CHART_WRAP)
    else:
        if not bi_var_x or not bi_var_y: return html.Div()
        color_arg = None if (not bi_color or bi_color=="none") else bi_color
        if color_arg and color_arg in df_raw.columns:
            df_plot = df_raw.copy(); df_plot[color_arg] = df_plot[color_arg].astype(str)
            fig = px.scatter(df_plot,x=bi_var_x,y=bi_var_y,color=color_arg,opacity=0.65,
                title=f"{bi_var_x} vs {bi_var_y}  ·  color: {color_arg}",
                color_discrete_sequence=px.colors.qualitative.Set2)
        else:
            fig = px.scatter(df_raw,x=bi_var_x,y=bi_var_y,opacity=0.6,
                color_discrete_sequence=[COLOR_ACCENT2],title=f"{bi_var_x} vs {bi_var_y}")
        fig.update_traces(marker=dict(size=5)); fig.update_layout(**PLOT_BASE)
        corr = df_raw[[bi_var_x,bi_var_y]].corr().iloc[0,1]
        badge = html.Div([
            html.P(f"{corr:.4f}",style={"fontFamily":FONT_TITLE,"fontSize":"1.5rem","fontWeight":"700","color":COLOR_PRIMARY,"margin":"0"}),
            html.P("Correlación de Pearson",style={"fontFamily":FONT_BODY,"fontSize":"0.72rem","color":COLOR_MUTED,"margin":"0","textTransform":"uppercase"}),
        ],style={"background":"#F7F9FC","borderRadius":"8px","padding":"14px 24px",
            "border":f"1px solid {COLOR_BORDER}","display":"inline-block","marginBottom":"14px"})
        return html.Div([badge,html.Div([dcc.Graph(figure=fig,config={"displayModeBar":False})],style=CHART_WRAP)])


# ── CALLBACK PRINCIPAL DE MODELOS ─────────────
@app.callback(
    Output("models-output", "children"),
    Input("model-selector", "value"),
)
def render_models(selected_models):
    if not selected_models:
        return html.P("Selecciona al menos un modelo.", style={"fontFamily":FONT_BODY,"color":COLOR_MUTED})

    # ── Cargar modelos y calcular métricas ──────
    loaded    = {}
    metrics   = []
    last_name = selected_models[-1]

    for name in selected_models:
        model = load_model(name)
        if model is None:
            continue
        loaded[name] = model
        preds = model.predict(X_test)
        metrics.append({
            "Modelo":    name,
            "Accuracy":  round(accuracy_score(y_test, preds), 4),
            "Precision": round(precision_score(y_test, preds, average="weighted", zero_division=0), 4),
            "Recall":    round(recall_score(y_test, preds, average="weighted", zero_division=0), 4),
            "F1 Score":  round(f1_score(y_test, preds, average="weighted", zero_division=0), 4),
            "F1 Macro":  round(f1_score(y_test, preds, average="macro", zero_division=0), 4),
        })

    if not loaded:
        return html.P("No se pudieron cargar los modelos seleccionados.", style={"fontFamily":FONT_BODY,"color":COLOR_ACCENT})

    last_model = loaded.get(last_name)
    if last_model is None:
        last_name  = list(loaded.keys())[-1]
        last_model = loaded[last_name]

    last_preds = last_model.predict(X_test)

    # ════════════════════════════════════════════
    # 1. CONFUSION MATRIX (último modelo)
    # ════════════════════════════════════════════
    cm = confusion_matrix(y_test, last_preds)
    tn, fp, fn, tp = cm.ravel()
    labels = ["No Heart Failure (0)", "Heart Failure (1)"]

    cm_fig = go.Figure()
    # Fondo coloreado
    cm_fig.add_trace(go.Heatmap(
        z=[[tn, fp],[fn, tp]],
        x=labels, y=labels,
        colorscale=[[0,"#EBF5FF"],[0.5,"#4A9BBF"],[1,"#1A3A5C"]],
        showscale=False,
        hoverongaps=False,
    ))
    # Anotaciones estilizadas
    annotations = [
        dict(x=labels[j], y=labels[i],
             text=f"<b>{cm[i,j]}</b>",
             showarrow=False,
             font=dict(size=26, family=FONT_TITLE,
                       color="white" if cm[i,j] > cm.max()/2 else COLOR_PRIMARY))
        for i in range(2) for j in range(2)
    ]
    subcaptions = {(0,0):"VN",(0,1):"FP",(1,0):"FN",(1,1):"VP"}
    for i in range(2):
        for j in range(2):
            annotations.append(dict(
                x=labels[j], y=labels[i],
                text=subcaptions[(i,j)],
                showarrow=False, yshift=-22,
                font=dict(size=11, family=FONT_BODY,
                          color="rgba(255,255,255,0.7)" if cm[i,j] > cm.max()/2 else COLOR_MUTED)
            ))

    cm_fig.update_layout(
        title=f"Confusion Matrix — {last_name}",
        xaxis_title="Predicción", yaxis_title="Real",
        annotations=annotations,
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family=FONT_BODY, color=COLOR_TEXT),
        margin=dict(l=80, r=40, t=60, b=60),
        height=380,
        xaxis=dict(side="bottom"),
        yaxis=dict(autorange="reversed"),
    )

    # Métricas rápidas bajo la CM
    acc_last = accuracy_score(y_test, last_preds)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

    quick_metrics = html.Div([
        html.Div([
            html.P(f"{acc_last:.2%}",style={"fontFamily":FONT_TITLE,"fontSize":"1.5rem","fontWeight":"700","color":COLOR_PRIMARY,"margin":"0"}),
            html.P("Accuracy",style={"fontFamily":FONT_BODY,"fontSize":"0.72rem","color":COLOR_MUTED,"margin":"0","textTransform":"uppercase"}),
        ],style={"flex":"1","textAlign":"center","padding":"16px","background":"#F7F9FC","borderRadius":"8px","border":f"1px solid {COLOR_BORDER}"}),
        html.Div([
            html.P(f"{sensitivity:.2%}",style={"fontFamily":FONT_TITLE,"fontSize":"1.5rem","fontWeight":"700","color":COLOR_ACCENT2,"margin":"0"}),
            html.P("Sensibilidad (Recall 1)",style={"fontFamily":FONT_BODY,"fontSize":"0.72rem","color":COLOR_MUTED,"margin":"0","textTransform":"uppercase"}),
        ],style={"flex":"1","textAlign":"center","padding":"16px","background":"#F7F9FC","borderRadius":"8px","border":f"1px solid {COLOR_BORDER}"}),
        html.Div([
            html.P(f"{specificity:.2%}",style={"fontFamily":FONT_TITLE,"fontSize":"1.5rem","fontWeight":"700","color":COLOR_ACCENT,"margin":"0"}),
            html.P("Especificidad (Recall 0)",style={"fontFamily":FONT_BODY,"fontSize":"0.72rem","color":COLOR_MUTED,"margin":"0","textTransform":"uppercase"}),
        ],style={"flex":"1","textAlign":"center","padding":"16px","background":"#F7F9FC","borderRadius":"8px","border":f"1px solid {COLOR_BORDER}"}),
        html.Div([
            html.P(f"{tp+fp}",style={"fontFamily":FONT_TITLE,"fontSize":"1.5rem","fontWeight":"700","color":COLOR_PRIMARY,"margin":"0"}),
            html.P("Predichos positivos",style={"fontFamily":FONT_BODY,"fontSize":"0.72rem","color":COLOR_MUTED,"margin":"0","textTransform":"uppercase"}),
        ],style={"flex":"1","textAlign":"center","padding":"16px","background":"#F7F9FC","borderRadius":"8px","border":f"1px solid {COLOR_BORDER}"}),
    ],style={"display":"flex","gap":"12px","marginTop":"16px"})

    cm_block = card([
        sec_title(f"Confusion Matrix — {last_name}"),
        html.Div([dcc.Graph(figure=cm_fig,config={"displayModeBar":False})],style=CHART_WRAP),
        quick_metrics,
    ])

    # ════════════════════════════════════════════
    # 2. TABLA COMPARATIVA DE MÉTRICAS
    # ════════════════════════════════════════════
    df_metrics = pd.DataFrame(metrics)

    # Colorear la fila del último modelo
    cond_styles = [{"if":{"row_index":"odd"},"backgroundColor":"#F7F9FC"}]
    for i, row in df_metrics.iterrows():
        if row["Modelo"] == last_name:
            cond_styles.append({"if":{"row_index":i},"backgroundColor":"#EBF5FF","fontWeight":"600"})

    metrics_block = card([
        sec_title("Comparativa de Métricas"),
        html.P("La fila resaltada en azul corresponde al último modelo seleccionado.",
            style={"fontFamily":FONT_BODY,"color":COLOR_MUTED,"fontSize":"0.85rem","marginTop":"0","marginBottom":"16px"}),
        dash_table.DataTable(
            data=df_metrics.to_dict("records"),
            columns=[{"name":c,"id":c} for c in df_metrics.columns],
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
    # 3. CURVA ROC-AUC (todos los modelos)
    # ════════════════════════════════════════════
    roc_fig = go.Figure()
    roc_fig.add_shape(type="line",x0=0,y0=0,x1=1,y1=1,
        line=dict(color=COLOR_BORDER,width=1.5,dash="dash"))

    for name, model in loaded.items():
        proba = get_proba(model, X_test)
        if proba is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_auc = auc(fpr, tpr)
        roc_fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={roc_auc:.3f})",
            line=dict(color=MODEL_COLORS.get(name, COLOR_ACCENT2), width=2.5),
        ))

    roc_fig.update_layout(
        title="Curvas ROC-AUC",
        xaxis_title="Tasa de Falsos Positivos (FPR)",
        yaxis_title="Tasa de Verdaderos Positivos (TPR)",
        legend=dict(x=0.62, y=0.08, bgcolor="rgba(255,255,255,0.85)",
                    bordercolor=COLOR_BORDER, borderwidth=1,
                    font=dict(family=FONT_BODY, size=11)),
        **PLOT_BASE,
    )

    roc_block = card([
        sec_title("Curva ROC-AUC"),
        html.Div([dcc.Graph(figure=roc_fig,config={"displayModeBar":False})],style=CHART_WRAP),
    ])

    # ════════════════════════════════════════════
    # 4. CLASSIFICATION REPORT (último modelo)
    # ════════════════════════════════════════════
    cr = classification_report(y_test, last_preds, output_dict=True, zero_division=0)
    cr_rows = []
    label_map = {"0":"Clase 0 — No Heart Failure","1":"Clase 1 — Heart Failure",
                 "macro avg":"Macro Avg","weighted avg":"Weighted Avg","accuracy":"Accuracy"}
    for k, v in cr.items():
        if k == "accuracy":
            cr_rows.append({"Clase":label_map.get(k,k),"Precision":"—","Recall":"—",
                            "F1-Score":round(v,4),"Support":"—"})
        elif isinstance(v, dict):
            cr_rows.append({"Clase":label_map.get(str(k),str(k)),
                            "Precision":round(v["precision"],4),
                            "Recall":round(v["recall"],4),
                            "F1-Score":round(v["f1-score"],4),
                            "Support":int(v["support"])})

    cr_cond = [
        {"if":{"row_index":"odd"},"backgroundColor":"#F7F9FC"},
        {"if":{"filter_query":'{Clase} contains "Weighted"'},"backgroundColor":"#EBF5FF","fontWeight":"600"},
        {"if":{"filter_query":'{Clase} contains "Macro"'},   "backgroundColor":"#FFF0F0"},
    ]

    cr_block = card([
        sec_title(f"Classification Report — {last_name}"),
        dash_table.DataTable(
            data=cr_rows,
            columns=[{"name":c,"id":c} for c in ["Clase","Precision","Recall","F1-Score","Support"]],
            style_table={"overflowX":"auto","borderRadius":"8px","border":f"1px solid {COLOR_BORDER}"},
            style_header={"backgroundColor":COLOR_PRIMARY,"color":"white","fontWeight":"600",
                "fontFamily":FONT_BODY,"fontSize":"0.82rem","padding":"12px 16px",
                "border":"none","textTransform":"uppercase","letterSpacing":"0.05em"},
            style_cell={"fontFamily":FONT_BODY,"fontSize":"0.9rem","color":COLOR_TEXT,
                "padding":"12px 16px","border":f"1px solid {COLOR_BORDER}","textAlign":"center"},
            style_cell_conditional=[{"if":{"column_id":"Clase"},"textAlign":"left","fontWeight":"500"}],
            style_data_conditional=cr_cond,
        ),
    ])

    # ════════════════════════════════════════════
    # 5. FEATURE IMPORTANCE
    # Estrategia:
    #   - Modelos rápidos (tree-based, logistic): LIME (5 muestras)
    #   - Modelos lentos con OHE (SVM, KNN):     Permutation Importance
    # Los resultados se cachean en FEAT_CACHE para no recomputar
    # ════════════════════════════════════════════
    SLOW_MODELS = {"SVM", "KNN"}

    lime_block = html.Div()

    if last_name in FEAT_CACHE:
        # Usar resultado cacheado
        feat_df, method_label = FEAT_CACHE[last_name]

    elif last_name in SLOW_MODELS:
        # ── Permutation Importance (rápido, modelo-agnóstico) ──
        method_label = "Permutation Importance"
        try:
            from sklearn.inspection import permutation_importance
            result = permutation_importance(
                last_model, X_test, y_test,
                n_repeats=10, random_state=66, scoring="f1_weighted", n_jobs=1
            )
            feat_df = pd.DataFrame({
                "Feature":     FEATURE_NAMES,
                "Importancia": np.maximum(result.importances_mean, 0),
            }).sort_values("Importancia")
            # Normalizar
            total = feat_df["Importancia"].sum() or 1
            feat_df["Importancia"] = feat_df["Importancia"] / total
            FEAT_CACHE[last_name] = (feat_df, method_label)
        except Exception as e:
            feat_df = None
            method_label = f"Error: {e}"

    else:
        # ── LIME (solo para modelos rápidos) ──
        method_label = "LIME"
        try:
            explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=X_train.values,
                feature_names=FEATURE_NAMES,
                class_names=["No Heart Failure", "Heart Failure"],
                mode="classification",
                random_state=66,
            )
            n_lime = 5
            sample_idx = np.random.RandomState(42).choice(len(X_test), n_lime, replace=False)
            feat_weights = {f: 0.0 for f in FEATURE_NAMES}

            def predict_fn(x):
                arr = pd.DataFrame(x, columns=FEATURE_NAMES)
                for col in CAT_COLS:
                    if col in arr.columns:
                        arr[col] = arr[col].round().clip(
                            int(X_train[col].min()), int(X_train[col].max())
                        ).astype(int)
                try:
                    return last_model.predict_proba(arr)
                except AttributeError:
                    preds = last_model.predict(arr).astype(float)
                    return np.column_stack([1 - preds, preds])

            for idx in sample_idx:
                exp = explainer.explain_instance(
                    X_test.values[idx], predict_fn,
                    num_features=len(FEATURE_NAMES), top_labels=1,
                )
                label_used = exp.available_labels()[0]
                for feat, weight in exp.as_list(label=label_used):
                    feat_name = feat.split(" ")[0].strip()
                    match = next(
                        (f for f in FEATURE_NAMES if feat_name == f),
                        next((f for f in FEATURE_NAMES if feat_name in f), None)
                    )
                    if match:
                        feat_weights[match] += abs(weight)

            total = sum(feat_weights.values()) or 1
            feat_df = pd.DataFrame(
                {"Feature": list(feat_weights.keys()),
                 "Importancia": [v/total for v in feat_weights.values()]}
            ).sort_values("Importancia")
            FEAT_CACHE[last_name] = (feat_df, method_label)

        except Exception as e:
            feat_df = None
            method_label = f"Error: {e}"

    # ── Renderizar gráfico ──
    if feat_df is not None and not feat_df.empty:
        feat_fig = go.Figure(go.Bar(
            x=feat_df["Importancia"],
            y=feat_df["Feature"],
            orientation="h",
            marker=dict(
                color=feat_df["Importancia"],
                colorscale=[[0, "#EBF5FF"], [0.5, COLOR_ACCENT2], [1, COLOR_PRIMARY]],
                showscale=False,
            ),
            text=[f"{v:.3f}" for v in feat_df["Importancia"]],
            textposition="outside",
            textfont=dict(family=FONT_BODY, size=11, color=COLOR_TEXT),
        ))
        feat_fig.update_layout(
            title=f"Feature Importance ({method_label}) — {last_name}",
            xaxis_title="Importancia relativa normalizada",
            yaxis_title="",
            paper_bgcolor="white", plot_bgcolor="#F7F9FC",
            font=dict(family=FONT_BODY, color=COLOR_TEXT),
            margin=dict(l=160, r=80, t=60, b=40),
            height=max(360, len(FEATURE_NAMES) * 36),
            xaxis=dict(showgrid=True, gridcolor=COLOR_BORDER),
            yaxis=dict(showgrid=False),
        )
        lime_block = card([
            sec_title(f"Feature Importance — {last_name}"),
            html.P(
                f"Método: {method_label}. "
                + ("Importancia de permutación sobre el conjunto de test (n_repeats=10)."
                   if method_label == "Permutation Importance"
                   else f"Importancia media absoluta LIME calculada sobre 5 muestras del test."),
                style={"fontFamily": FONT_BODY, "color": COLOR_MUTED,
                       "fontSize": "0.85rem", "marginTop": "0", "marginBottom": "16px"},
            ),
            html.Div([dcc.Graph(figure=feat_fig, config={"displayModeBar": False})], style=CHART_WRAP),
        ])
    else:
        lime_block = card([
            sec_title("Feature Importance"),
            html.P(
                f"No se pudo calcular para {last_name}: {method_label}",
                style={"fontFamily": FONT_BODY, "color": COLOR_ACCENT, "fontSize": "0.9rem"},
            ),
        ])

    # ════════════════════════════════════════════
    # LAYOUT FINAL — pares de 2 columnas
    # Fila 1: Confusion Matrix | Comparativa
    # Fila 2: Curva ROC        | Classification Report
    # Fila 3: LIME (ancho completo)
    # ════════════════════════════════════════════
    def two_col(left, right):
        return html.Div([
            html.Div([left],  style={"flex": "1", "marginRight": "12px", "minWidth": "0"}),
            html.Div([right], style={"flex": "1", "marginLeft":  "12px", "minWidth": "0"}),
        ], style={"display": "flex", "alignItems": "stretch", "marginBottom": "0"})

    return html.Div([
        two_col(cm_block, metrics_block),
        two_col(roc_block, cr_block),
        lime_block,
    ])


if __name__ == '__main__':
    app.run(debug=True)