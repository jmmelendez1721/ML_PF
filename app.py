from dash import Dash, dcc, html, Input, Output, State, dash_table, ctx
import colorlover as cl
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server

df_raw = pd.read_csv("HeartFailureDataset.csv")
df_raw = df_raw.drop(columns=['id'])
# ─────────────────────────────────────────────
# COLUMNAS — detección dinámica desde el dataset real
# ─────────────────────────────────────────────
NUM_COLS  = list(df_raw.select_dtypes(include=np.number).columns)
# Columnas categóricas: numéricas con 4 o menos valores únicos (binarias/ordinales)
CAT_COLS  = [c for c in NUM_COLS if df_raw[c].nunique() <= 4]
CONT_COLS = [c for c in NUM_COLS if c not in CAT_COLS]

# ─────────────────────────────────────────────
# PALETA Y ESTILOS
# ─────────────────────────────────────────────
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
         "exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure "
         "dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.")

PLOT_BASE = dict(
    paper_bgcolor="white", plot_bgcolor="#F7F9FC",
    font=dict(family=FONT_BODY, color=COLOR_TEXT),
    margin=dict(l=40, r=40, t=50, b=40), height=440,
)

CHART_WRAP = {
    "background": COLOR_CARD, "borderRadius": "12px", "padding": "16px",
    "boxShadow": "0 2px 12px rgba(26,58,92,0.07)", "border": f"1px solid {COLOR_BORDER}"
}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# DICCIONARIO DE VARIABLES
# ─────────────────────────────────────────────
variables = [
    {"Variable":"age","Tipo":"Numérica continua","Unidad":"años","Significado":"Edad del paciente"},
    {"Variable":"anaemia","Tipo":"Binaria","Unidad":"0/1","Significado":"Disminución de glóbulos rojos o hemoglobina"},
    {"Variable":"creatinine_phosphokinase","Tipo":"Numérica discreta","Unidad":"mcg/L","Significado":"Nivel de la enzima CPK en sangre"},
    {"Variable":"diabetes","Tipo":"Binaria","Unidad":"0/1","Significado":"Presencia de diabetes mellitus"},
    {"Variable":"ejection_fraction","Tipo":"Numérica discreta","Unidad":"%","Significado":"Porcentaje de sangre que sale del corazón por contracción"},
    {"Variable":"high_blood_pressure","Tipo":"Binaria","Unidad":"0/1","Significado":"Presencia de hipertensión arterial"},
    {"Variable":"platelets","Tipo":"Numérica continua","Unidad":"kiloplatelets/mL","Significado":"Recuento de plaquetas en sangre"},
    {"Variable":"serum_creatinine","Tipo":"Numérica continua","Unidad":"mg/dL","Significado":"Nivel de creatinina sérica"},
    {"Variable":"serum_sodium","Tipo":"Numérica discreta","Unidad":"mEq/L","Significado":"Nivel de sodio en suero"},
    {"Variable":"sex","Tipo":"Binaria","Unidad":"0/1","Significado":"Sexo biológico (0=mujer, 1=hombre)"},
    {"Variable":"smoking","Tipo":"Binaria","Unidad":"0/1","Significado":"Si el paciente fuma o fumó"},
    {"Variable":"time","Tipo":"Numérica discreta","Unidad":"días","Significado":"Período de seguimiento clínico"},
    {"Variable":"DEATH_EVENT","Tipo":"Binaria (objetivo)","Unidad":"0/1","Significado":"Si el paciente falleció durante el seguimiento"},
]
df_vars = pd.DataFrame(variables)

# ══════════════════════════════════════════════
# TAB 1 — INTRODUCCIÓN
# Layout:
#   Fila 1: [Definición del problema] [Imagen ref 1]
#   Fila 2: [Imagen ref 2]            [Justificación]
#   Fila 3: Diccionario de variables
# ══════════════════════════════════════════════
def make_tab1():
    return html.Div([
        # Hero
        html.Div([
            html.Div([
                html.Span("Análisis Clínico", style={
                    "fontFamily": FONT_BODY, "color": COLOR_ACCENT,
                    "fontSize": "0.85rem", "fontWeight": "600",
                    "letterSpacing": "0.15em", "textTransform": "uppercase"}),
                html.H1("Falla Cardíaca & Machine Learning", style={
                    "fontFamily": FONT_TITLE, "color": "white", "fontSize": "2.4rem",
                    "fontWeight": "700", "marginTop": "8px", "marginBottom": "8px", "lineHeight": "1.2"}),
                html.P("Proyecto de análisis exploratorio y modelado predictivo sobre diagnóstico de insuficiencia cardíaca",
                    style={"fontFamily": FONT_BODY, "color": "rgba(255,255,255,0.7)",
                           "fontSize": "1rem", "fontWeight": "300", "marginTop": "0"}),
            ], style={"flex": "1"}),
            html.Div("♥", style={"fontSize": "6rem", "color": "white", "opacity": "0.12",
                                  "lineHeight": "1", "alignSelf": "center", "marginLeft": "24px"}),
        ], style={
            "display": "flex", "justifyContent": "space-between",
            "background": f"linear-gradient(135deg,{COLOR_PRIMARY} 0%,#2A5A8C 100%)",
            "borderRadius": "16px", "padding": "40px 48px", "marginBottom": "28px",
        }),

        # Fila 1: Definición del problema | Imagen 1
        html.Div([
            html.Div([
                card([sec_title("Definición del Problema de Investigación"), body(LOREM), body(LOREM)],
                     extra={"marginBottom": "0", "height": "100%", "boxSizing": "border-box"})
            ], style={"flex": "1", "marginRight": "16px"}),
            html.Div([
                html.Div([img_placeholder(1)], style={
                    "background": COLOR_CARD, "borderRadius": "12px", "padding": "28px 32px",
                    "boxShadow": "0 2px 12px rgba(26,58,92,0.07)", "border": f"1px solid {COLOR_BORDER}",
                    "height": "100%", "boxSizing": "border-box",
                })
            ], style={"flex": "1", "marginLeft": "0"}),
        ], style={"display": "flex", "marginBottom": "24px", "alignItems": "stretch"}),

        # Fila 2: Imagen 2 | Justificación del dataset
        html.Div([
            html.Div([
                html.Div([img_placeholder(2)], style={
                    "background": COLOR_CARD, "borderRadius": "12px", "padding": "28px 32px",
                    "boxShadow": "0 2px 12px rgba(26,58,92,0.07)", "border": f"1px solid {COLOR_BORDER}",
                    "height": "100%", "boxSizing": "border-box",
                })
            ], style={"flex": "1", "marginRight": "16px"}),
            html.Div([
                card([sec_title("Justificación del Dataset"), body(LOREM), body(LOREM)],
                     extra={"marginBottom": "0", "height": "100%", "boxSizing": "border-box"})
            ], style={"flex": "1", "marginLeft": "0"}),
        ], style={"display": "flex", "marginBottom": "24px", "alignItems": "stretch"}),

        # Diccionario
        card([
            sec_title("Diccionario de Variables"),
            html.P("Descripción de cada variable: tipo, unidad de medida y significado clínico.",
                style={"fontFamily": FONT_BODY, "color": COLOR_MUTED, "fontSize": "0.88rem",
                       "marginBottom": "16px", "marginTop": "0"}),
            dash_table.DataTable(
                data=df_vars.to_dict("records"),
                columns=[{"name": c, "id": c} for c in df_vars.columns],
                style_table={"overflowX": "auto", "borderRadius": "8px", "border": f"1px solid {COLOR_BORDER}"},
                style_header={
                    "backgroundColor": COLOR_PRIMARY, "color": "white", "fontWeight": "600",
                    "fontFamily": FONT_BODY, "fontSize": "0.82rem", "padding": "12px 16px",
                    "border": "none", "textTransform": "uppercase", "letterSpacing": "0.05em"},
                style_cell={
                    "fontFamily": FONT_BODY, "fontSize": "0.88rem", "color": COLOR_TEXT,
                    "padding": "10px 16px", "border": f"1px solid {COLOR_BORDER}",
                    "textAlign": "left", "whiteSpace": "normal", "height": "auto"},
                style_data_conditional=[
                    {"if": {"row_index": "odd"}, "backgroundColor": "#F7F9FC"},
                    {"if": {"column_id": "Variable"}, "fontWeight": "600", "color": COLOR_PRIMARY},
                    {"if": {"column_id": "Tipo", "filter_query": '{Tipo} contains "objetivo"'},
                     "backgroundColor": "#FFF0F0", "color": COLOR_ACCENT, "fontWeight": "600"},
                ],
            ),
        ]),
    ], style={"padding": "32px 40px", "background": COLOR_BG, "minHeight": "100vh"})


# ══════════════════════════════════════════════
# TAB 2 — EDA
# ══════════════════════════════════════════════
def make_tab2():
    ctrl_style = {
        "background": COLOR_CARD, "borderRadius": "12px", "padding": "20px 24px",
        "marginBottom": "24px", "boxShadow": "0 2px 12px rgba(26,58,92,0.07)",
        "border": f"1px solid {COLOR_BORDER}",
    }
    return html.Div([
        html.H2("EDA — Análisis Exploratorio de Datos", style={
            "fontFamily": FONT_TITLE, "color": COLOR_PRIMARY,
            "fontSize": "1.8rem", "marginBottom": "4px", "marginTop": "0"}),
        html.P(f"Dataset: {len(df_raw)} registros · {len(df_raw.columns)} variables",
            style={"fontFamily": FONT_BODY, "color": COLOR_MUTED,
                   "fontSize": "0.9rem", "marginBottom": "28px"}),

        # Selector principal
        html.Div([
            lbl("Tipo de análisis"),
            dcc.Dropdown(id="eda-tipo",
                options=[
                    {"label": "📊  Análisis Unidimensional", "value": "uni"},
                    {"label": "🔀  Análisis Bidimensional",  "value": "bi"},
                ],
                value="uni", clearable=False,
                style={"fontFamily": FONT_BODY, "fontSize": "0.95rem", "borderRadius": "8px"}),
        ], style={**ctrl_style, "maxWidth": "420px"}),

        # Panel unidimensional — visible por defecto
        html.Div(id="uni-panel", children=[
            html.Div([
                html.Div([
                    lbl("Variable"),
                    dcc.Dropdown(id="uni-var",
                        options=[{"label": c, "value": c} for c in NUM_COLS],
                        value=NUM_COLS[0] if NUM_COLS else None,
                        clearable=False,
                        style={"fontFamily": FONT_BODY, "borderRadius": "8px"}),
                ], style={"flex": "1", "marginRight": "16px"}),
                html.Div([
                    lbl("Tipo de gráfica"),
                    dcc.Dropdown(id="uni-chart-type",
                        options=[
                            {"label": "Histograma", "value": "hist"},
                            {"label": "Box Plot",   "value": "box"},
                            {"label": "Violin Plot","value": "violin"},
                        ],
                        value="hist", clearable=False,
                        style={"fontFamily": FONT_BODY, "borderRadius": "8px"}),
                ], style={"flex": "1"}),
            ], style={"display": "flex"}),
        ], style=ctrl_style),

        # Panel bidimensional — oculto por defecto
        html.Div(id="bi-panel", children=[
            html.Div([
                html.Div([
                    lbl("Variable X"),
                    dcc.Dropdown(id="bi-var-x",
                        options=[{"label": c, "value": c} for c in CONT_COLS],
                        value=CONT_COLS[0] if CONT_COLS else None,
                        clearable=False,
                        style={"fontFamily": FONT_BODY, "borderRadius": "8px"}),
                ], style={"flex": "1", "marginRight": "16px"}),
                html.Div([
                    lbl("Variable Y"),
                    dcc.Dropdown(id="bi-var-y",
                        options=[{"label": c, "value": c} for c in CONT_COLS],
                        value=CONT_COLS[1] if len(CONT_COLS) > 1 else (CONT_COLS[0] if CONT_COLS else None),
                        clearable=False,
                        style={"fontFamily": FONT_BODY, "borderRadius": "8px"}),
                ], style={"flex": "1", "marginRight": "16px"}),
                html.Div([
                    lbl("Color por"),
                    dcc.Dropdown(id="bi-color",
                        # Opciones dinámicas basadas en columnas reales del dataset
                        options=[{"label": "Ninguno", "value": "none"}] +
                                [{"label": c, "value": c} for c in CAT_COLS],
                        value=CAT_COLS[-1] if CAT_COLS else "none",
                        clearable=False,
                        style={"fontFamily": FONT_BODY, "borderRadius": "8px"}),
                ], style={"flex": "1"}),
            ], style={"display": "flex"}),
        ], style={**ctrl_style, "display": "none"}),

        html.Div(id="eda-chart"),

    ], style={"padding": "32px 40px", "background": COLOR_BG, "minHeight": "100vh"})


# ══════════════════════════════════════════════
# TAB 3 — MODELOS
# ══════════════════════════════════════════════
def make_tab3():
    return html.Div([
        html.H2("Modelos Predictivos", style={
            "fontFamily": FONT_TITLE, "color": COLOR_PRIMARY,
            "fontSize": "1.8rem", "marginBottom": "4px", "marginTop": "0"}),
        html.P("Construcción y evaluación de modelos de machine learning",
            style={"fontFamily": FONT_BODY, "color": COLOR_MUTED, "marginBottom": "28px"}),
        card([sec_title("Descripción"), body(LOREM), body(LOREM), body(LOREM)]),
    ], style={"padding": "32px 40px", "background": COLOR_BG, "minHeight": "100vh"})


# ─────────────────────────────────────────────
# ESTILOS TABS
# ─────────────────────────────────────────────
TAB_S = {
    "fontFamily": FONT_BODY, "fontWeight": "500", "fontSize": "0.9rem", "color": COLOR_MUTED,
    "padding": "14px 28px", "borderBottom": "3px solid transparent",
    "background": "transparent", "border": "none"
}
TAB_SEL = {**TAB_S, "color": COLOR_PRIMARY, "borderBottom": f"3px solid {COLOR_ACCENT}", "fontWeight": "600"}

# ─────────────────────────────────────────────
# LAYOUT PRINCIPAL
# ─────────────────────────────────────────────
app.layout = html.Div([
    GOOGLE_FONTS,
    html.Div([
        html.Div([
            html.Span("♥", style={"color": COLOR_ACCENT, "fontSize": "1.2rem", "marginRight": "8px"}),
            html.Span("CardioML", style={"fontFamily": FONT_TITLE, "color": COLOR_PRIMARY,
                                          "fontWeight": "700", "fontSize": "1.1rem"}),
        ], style={"display": "flex", "alignItems": "center"}),
        html.Span("Heart Failure Prediction · Dashboard",
            style={"fontFamily": FONT_BODY, "color": COLOR_MUTED, "fontSize": "0.82rem"}),
    ], style={
        "display": "flex", "justifyContent": "space-between", "alignItems": "center",
        "padding": "16px 40px", "background": COLOR_CARD, "borderBottom": f"1px solid {COLOR_BORDER}",
        "position": "sticky", "top": "0", "zIndex": "100",
        "boxShadow": "0 2px 8px rgba(26,58,92,0.05)",
    }),

    dcc.Tabs(id="main-tabs", value="tab-intro", children=[
        dcc.Tab(label="Introducción", value="tab-intro", style=TAB_S, selected_style=TAB_SEL),
        dcc.Tab(label="EDA",          value="tab-eda",   style=TAB_S, selected_style=TAB_SEL),
        dcc.Tab(label="Modelos",      value="tab-models",style=TAB_S, selected_style=TAB_SEL),
    ], style={"background": COLOR_CARD, "paddingLeft": "32px", "borderBottom": f"1px solid {COLOR_BORDER}"}),

    html.Div(id="tab-content"),
], style={"background": COLOR_BG, "minHeight": "100vh"})

# ══════════════════════════════════════════════
# CALLBACKS
# ══════════════════════════════════════════════

@app.callback(Output("tab-content", "children"), Input("main-tabs", "value"))
def render_tab(tab):
    if tab == "tab-intro": return make_tab1()
    elif tab == "tab-eda": return make_tab2()
    else:                  return make_tab3()


@app.callback(
    Output("uni-panel", "style"),
    Output("bi-panel",  "style"),
    Input("eda-tipo", "value"),
)
def toggle_panels(tipo):
    base = {
        "background": COLOR_CARD, "borderRadius": "12px", "padding": "20px 24px",
        "marginBottom": "24px", "boxShadow": "0 2px 12px rgba(26,58,92,0.07)",
        "border": f"1px solid {COLOR_BORDER}",
    }
    show = {**base, "display": "block"}
    hide = {**base, "display": "none"}
    return (show, hide) if tipo == "uni" else (hide, show)


@app.callback(
    Output("eda-chart", "children"),
    Input("eda-tipo",       "value"),
    Input("uni-var",        "value"),
    Input("uni-chart-type", "value"),
    Input("bi-var-x",       "value"),
    Input("bi-var-y",       "value"),
    Input("bi-color",       "value"),
)
def render_chart(tipo, uni_var, uni_chart_type, bi_var_x, bi_var_y, bi_color):

    # ── UNIDIMENSIONAL ──────────────────────────────────
    if tipo == "uni":
        if not uni_var:
            return html.Div()

        if uni_chart_type == "hist":
            fig = go.Figure(go.Histogram(
                x=df_raw[uni_var], nbinsx=30,
                marker_color=COLOR_ACCENT2, opacity=0.85, name=uni_var))
            fig.update_layout(title=f"Distribución de {uni_var}", **PLOT_BASE)
            stats = df_raw[uni_var].describe().round(3)
            stat_row = html.Div([
                html.Div([
                    html.P(str(round(v, 2)), style={"fontFamily": FONT_TITLE, "fontSize": "1.2rem",
                        "fontWeight": "700", "color": COLOR_PRIMARY, "margin": "0"}),
                    html.P(k, style={"fontFamily": FONT_BODY, "fontSize": "0.7rem",
                        "color": COLOR_MUTED, "margin": "0", "textTransform": "uppercase"}),
                ], style={"flex": "1", "background": "#F7F9FC", "borderRadius": "8px",
                    "padding": "10px 14px", "textAlign": "center", "border": f"1px solid {COLOR_BORDER}"})
                for k, v in stats.items()
            ], style={"display": "flex", "gap": "8px", "marginBottom": "14px"})
            return html.Div([
                stat_row,
                html.Div([dcc.Graph(figure=fig, config={"displayModeBar": False})], style=CHART_WRAP),
            ])

        elif uni_chart_type == "box":
            fig = go.Figure(go.Box(y=df_raw[uni_var], name=uni_var,
                marker_color=COLOR_ACCENT, line_color=COLOR_PRIMARY,
                fillcolor=COLOR_ACCENT2, opacity=0.7))
            fig.update_layout(title=f"Box Plot – {uni_var}", **PLOT_BASE)

        else:  # violin
            fig = go.Figure(go.Violin(y=df_raw[uni_var], name=uni_var,
                box_visible=True, meanline_visible=True,
                fillcolor=COLOR_ACCENT2, line_color=COLOR_PRIMARY, opacity=0.75))
            fig.update_layout(title=f"Violin Plot – {uni_var}", **PLOT_BASE)

        return html.Div([dcc.Graph(figure=fig, config={"displayModeBar": False})], style=CHART_WRAP)

    # ── BIDIMENSIONAL ───────────────────────────────────
    else:
        if not bi_var_x or not bi_var_y:
            return html.Div()

        # Convertir la columna de color a string para que px.scatter
        # la trate siempre como categórica (evita el error de tipo)
        color_arg = None if (not bi_color or bi_color == "none") else bi_color

        if color_arg and color_arg in df_raw.columns:
            df_plot = df_raw.copy()
            df_plot[color_arg] = df_plot[color_arg].astype(str)
            fig = px.scatter(
                df_plot, x=bi_var_x, y=bi_var_y, color=color_arg,
                opacity=0.65,
                title=f"{bi_var_x} vs {bi_var_y}  ·  color: {color_arg}",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
        else:
            fig = px.scatter(
                df_raw, x=bi_var_x, y=bi_var_y, opacity=0.6,
                color_discrete_sequence=[COLOR_ACCENT2],
                title=f"{bi_var_x} vs {bi_var_y}",
            )

        fig.update_traces(marker=dict(size=5))
        fig.update_layout(**PLOT_BASE)

        corr = df_raw[[bi_var_x, bi_var_y]].corr().iloc[0, 1]
        badge = html.Div([
            html.P(f"{corr:.4f}", style={"fontFamily": FONT_TITLE, "fontSize": "1.5rem",
                "fontWeight": "700", "color": COLOR_PRIMARY, "margin": "0"}),
            html.P("Correlación de Pearson", style={"fontFamily": FONT_BODY, "fontSize": "0.72rem",
                "color": COLOR_MUTED, "margin": "0", "textTransform": "uppercase"}),
        ], style={"background": "#F7F9FC", "borderRadius": "8px", "padding": "14px 24px",
            "border": f"1px solid {COLOR_BORDER}", "display": "inline-block", "marginBottom": "14px"})

        return html.Div([
            badge,
            html.Div([dcc.Graph(figure=fig, config={"displayModeBar": False})], style=CHART_WRAP),
        ])


if __name__ == '__main__':
    app.run(debug=True)