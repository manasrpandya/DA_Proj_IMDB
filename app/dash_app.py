from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go

# --------------------
# data
# --------------------
DATA_MAIN = Path("data/processed/imdb_movies_clean.parquet")
DATA_GENRE_LONG = Path("data/processed/imdb_movies_genre_long.parquet")

df = pd.read_parquet(DATA_MAIN)
g_long = pd.read_parquet(DATA_GENRE_LONG)

# guards
df["startYear"] = pd.to_numeric(df["startYear"], errors="coerce").astype("Int64")
df["runtimeMinutes"] = pd.to_numeric(df["runtimeMinutes"], errors="coerce")
df["averageRating"] = pd.to_numeric(df["averageRating"], errors="coerce")
df["numVotes"] = pd.to_numeric(df["numVotes"], errors="coerce")

# optional columns
if "genres" not in df.columns:
    df["genres"] = ""
if "decade" not in df.columns and df["startYear"].notna().any():
    df["decade"] = ((df["startYear"].astype("float") // 10) * 10).astype("Int64")

MIN_YEAR = int(np.nanmin(df["startYear"])) if df["startYear"].notna().any() else 1900
MAX_YEAR = int(np.nanmax(df["startYear"])) if df["startYear"].notna().any() else 2025

genres_sorted = g_long["genre"].dropna().value_counts().index.tolist()
decades_sorted = (
    df["decade"].dropna().drop_duplicates().astype(int).sort_values().tolist()
    if "decade" in df.columns else []
)

# --------------------
# global style and constants
# --------------------
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = px.colors.qualitative.Set2

CARD_HEIGHT = 380          # fixed → no vertical stretching
UIREV = "lock"             # persist zoom pan unless we programmatically set ranges

# --------------------
# helper: data-aware axis ranges
# --------------------
def _pad_span(lo: float, hi: float, pad_frac: float, min_span: float) -> tuple[float, float]:
    if np.isnan(lo) or np.isnan(hi):
        return lo, hi
    span = hi - lo
    if span < min_span:
        mid = (hi + lo) / 2.0
        half = max(min_span / 2.0, span / 2.0)
        lo, hi = mid - half, mid + half
    pad = max(pad_frac * (hi - lo), 1e-6)
    return lo - pad, hi + pad

def x_range_rating(d: pd.DataFrame) -> list[float]:
    s = d["averageRating"].dropna().astype(float)
    if s.empty:
        return [1.0, 10.0]
    lo, hi = np.nanmin(s), np.nanmax(s)
    lo, hi = _pad_span(lo, hi, pad_frac=0.05, min_span=0.5)
    return [max(1.0, lo), min(10.0, hi)]

def y_range_rating(d: pd.DataFrame) -> list[float]:
    # same as x but clamp to [1,10]
    return x_range_rating(d)

def x_range_runtime(d: pd.DataFrame) -> list[float]:
    s = d["runtimeMinutes"].dropna().astype(float)
    if s.empty:
        return [40, 240]
    lo, hi = np.nanpercentile(s, [1, 99])
    lo, hi = _pad_span(lo, hi, pad_frac=0.05, min_span=20.0)
    return [max(0.0, lo), min(600.0, hi)]

def y_range_votes_log(d: pd.DataFrame) -> list[float]:
    s = d["numVotes"].dropna().astype(float)
    s = s[s > 0]
    if s.empty:
        return [0, 1]  # log axis handles this as a tiny span; fixed height prevents stretch
    lo, hi = np.nanpercentile(s, [1, 99])
    lo = max(lo, 1.0)
    lo_log, hi_log = np.log10(lo), np.log10(hi)
    lo_log, hi_log = _pad_span(lo_log, hi_log, pad_frac=0.05, min_span=0.5)
    return [lo_log, hi_log]

def y_max_hist_count(d: pd.DataFrame) -> int:
    # upper bound for histogram y via count estimate; prevents huge global cap
    n = int(d.shape[0])
    return max(10, int(np.ceil(n * 1.10)))

def y_max_yearly_count(d: pd.DataFrame) -> int:
    if d.empty or d["startYear"].isna().all():
        return 10
    c = d.groupby("startYear", dropna=True)["tconst"].count()
    return max(10, int(np.ceil(c.max() * 1.10)))

def x_max_director_count(d: pd.DataFrame) -> int:
    dd = d.copy()
    if "director_name" in dd.columns:
        dd["director_name"] = dd["director_name"].fillna("Unknown")
    else:
        dd["director_name"] = "Unknown"
    c = dd.groupby("director_name")["tconst"].count()
    return max(10, int(np.ceil(c.max() * 1.10))) if not c.empty else 10

# --------------------
# layout helpers
# --------------------
def base_layout(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title=title,
        height=CARD_HEIGHT,
        margin=dict(l=50, r=20, t=50, b=40),
        font=dict(size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest",
        transition={"duration": 0},
        uirevision=UIREV,
    )
    return fig

# --------------------
# app
# --------------------
app = Dash(__name__)
app.title = "IMDb Movie Analytics"

controls = html.Div(
    [
        html.H2("Filters", style={"marginTop": 0}),
        html.Label("Year range"),
        dcc.RangeSlider(
            id="year-range",
            min=MIN_YEAR, max=MAX_YEAR, step=1,
            value=[max(MIN_YEAR, MAX_YEAR - 30), MAX_YEAR],
            allowCross=False,
            tooltip={"placement": "bottom"}
        ),
        html.Div(id="year-range-label", style={"marginBottom": "10px"}),

        html.Label("Genres"),
        dcc.Dropdown(
            id="genre-select",
            options=[{"label": g, "value": g} for g in genres_sorted],
            value=[],
            multi=True,
            placeholder="All genres",
        ),

        html.Label("Min votes"),
        dcc.Slider(
            id="min-votes", min=50, max=50000, step=50, value=500,
            tooltip={"placement": "bottom"}
        ),
        html.Div(id="min-votes-label", style={"marginBottom": "10px"}),

        html.Label("Decade"),
        dcc.Dropdown(
            id="decade-select",
            options=[{"label": str(d), "value": d} for d in decades_sorted],
            value=None, placeholder="All decades", clearable=True
        ),

        html.Label("Director (substring match)"),
        dcc.Input(id="director-like", type="text", placeholder="e.g., nolan", debounce=True),

        html.Hr(),
        html.Button("Download filtered CSV", id="dl-btn"),
        dcc.Download(id="dl-data"),
    ],
    style={
        "width": "300px",
        "padding": "12px",
        "borderRight": "1px solid #eee",
        "background": "#fafafa",
        "flex": "0 0 300px"
    },
)

grid = html.Div(
    [
        html.Div([dcc.Graph(id="hist", config={"displayModeBar": False}), html.Div(id="hist-note")], className="card"),
        html.Div([dcc.Graph(id="yearly", config={"displayModeBar": False}), html.Div(id="yearly-note")], className="card"),
        html.Div([dcc.Graph(id="genrebar", config={"displayModeBar": False}), html.Div(id="genre-note")], className="card"),
        html.Div([dcc.Graph(id="scatter", config={"displayModeBar": False}), html.Div(id="scatter-note")], className="card"),
        html.Div([dcc.Graph(id="runtime2d", config={"displayModeBar": False}), html.Div(id="runtime-note")], className="card"),
        html.Div([dcc.Graph(id="directors", config={"displayModeBar": False}), html.Div(id="directors-note")], className="card"),
    ],
    id="grid",
    style={
        "display": "grid",
        "gridTemplateColumns": "minmax(520px, 1fr) minmax(520px, 1fr)",
        "gap": "16px",
        "alignItems": "start",
        "padding": "12px",
        "boxSizing": "border-box",
        "minWidth": "1080px",
    },
)

app.layout = html.Div(
    [
        html.H1("IMDb Movie Analytics", style={"margin": "12px 12px 0 12px"}),
        html.Div(
            [controls, grid],
            style={
                "display": "flex",
                "alignItems": "flex-start",
                "gap": "0px",
                "overflowX": "auto",
                "width": "100%",
            },
        ),
        html.Div(
            "Data: IMDb official datasets. Defaults: last ~30 years, all genres, min 500 votes.",
            style={"padding": "8px 12px", "borderTop": "1px solid #eee", "marginTop": "4px", "fontSize": "12px"}
        )
    ],
    style={"width": "100%", "maxWidth": "1600px", "margin": "0 auto"}
)

# --------------------
# figure builders (now with dynamic ranges)
# --------------------
def fig_ratings_hist(d):
    fig = px.histogram(d, x="averageRating", nbins=40, opacity=0.9)
    xr = x_range_rating(d)
    yr_max = y_max_hist_count(d)
    fig.update_xaxes(title="Rating", range=xr, fixedrange=True)
    fig.update_yaxes(title="Count", range=[0, yr_max], fixedrange=True)
    return base_layout(fig, "Ratings distribution")

def fig_yearly_counts_mean(d):
    by_year = (
        d.groupby("startYear", dropna=True)
        .agg(count=("tconst", "count"), mean_rating=("averageRating", "mean"))
        .reset_index()
        .sort_values("startYear")
    )
    fig = go.Figure()
    fig.add_bar(x=by_year["startYear"], y=by_year["count"], name="Movie count", marker=dict(line=dict(width=0)))
    fig.add_trace(go.Scatter(
        x=by_year["startYear"], y=by_year["mean_rating"], name="Mean rating", yaxis="y2", mode="lines"
    ))
    yr_max = y_max_yearly_count(d)
    # x range: clamp to selected year slider to keep domain tight
    if by_year["startYear"].notna().any():
        xlo, xhi = int(by_year["startYear"].min()), int(by_year["startYear"].max())
    else:
        xlo, xhi = MIN_YEAR, MAX_YEAR
    fig.update_layout(
        xaxis=dict(title="Year", range=[xlo - 0.5, xhi + 0.5], fixedrange=True, constrain="domain"),
        yaxis=dict(title="Count", range=[0, yr_max], fixedrange=True),
        yaxis2=dict(title="Mean rating", range=y_range_rating(d), overlaying="y", side="right", fixedrange=True),
    )
    return base_layout(fig, "Movies per year with mean rating")

def fig_mean_rating_by_genre(d):
    dd = d.merge(g_long[["tconst", "genre"]], on="tconst", how="left")
    grp = (
        dd.dropna(subset=["genre"])
        .groupby("genre")
        .agg(mean_rating=("averageRating", "mean"), n=("tconst", "count"))
        .reset_index()
    )
    grp = grp[grp["n"] >= 100].sort_values("mean_rating", ascending=False).head(20)
    fig = px.bar(grp, x="mean_rating", y="genre", orientation="h", text="n")
    xr = y_range_rating(d)  # reuse rating bounds
    fig.update_xaxes(title="Mean rating", range=xr, fixedrange=True)
    fig.update_yaxes(title="", fixedrange=True, automargin=True)
    return base_layout(fig, "Mean rating by genre (n ≥ 100)")

def fig_votes_vs_rating(d):
    fig = px.scatter(d, x="averageRating", y="numVotes", hover_name="primaryTitle", opacity=0.6)
    fig.update_xaxes(title="Rating", range=x_range_rating(d), fixedrange=True)
    fig.update_yaxes(title="Votes (log)", type="log", range=y_range_votes_log(d), fixedrange=True)
    return base_layout(fig, "Votes vs Rating (log y)")

def fig_runtime_vs_rating(d):
    fig = px.density_heatmap(d, x="runtimeMinutes", y="averageRating", nbinsx=40, nbinsy=40, histnorm="")
    fig.update_xaxes(title="Runtime (min)", range=x_range_runtime(d), fixedrange=True)
    fig.update_yaxes(title="Rating", range=y_range_rating(d), fixedrange=True)
    fig.update_traces(colorbar_title="Count")
    return base_layout(fig, "Runtime vs Rating (2D density)")

def fig_directors_bubble(d):
    dd = d.copy()
    if "director_name" in dd.columns:
        dd["director_name"] = dd["director_name"].fillna("Unknown")
    else:
        dd["director_name"] = "Unknown"
    grp = (
        dd.groupby("director_name")
        .agg(n=("tconst", "count"), avg_rating=("averageRating", "mean"), avg_votes=("numVotes", "mean"))
        .reset_index()
    )
    grp = grp[grp["n"] >= 5].sort_values("n", ascending=False).head(100)
    fig = px.scatter(grp, x="n", y="avg_rating", size="avg_votes", hover_name="director_name")
    fig.update_xaxes(title="Film count", range=[0, x_max_director_count(d)], fixedrange=True)
    fig.update_yaxes(title="Mean rating", range=y_range_rating(d), fixedrange=True)
    return base_layout(fig, "Directors: output vs average rating (n ≥ 5)")

# --------------------
# filtering
# --------------------
def apply_filters(year_range, genres, min_votes, decade, director_like):
    d = df
    if year_range and len(year_range) == 2:
        d = d[(d["startYear"] >= year_range[0]) & (d["startYear"] <= year_range[1])]
    if genres:
        mask = pd.Series(False, index=d.index)
        for g in genres:
            mask = mask | d["genres"].fillna("").str.contains(fr"(^|,){g}(,|$)")
        d = d[mask]
    if min_votes:
        d = d[d["numVotes"] >= min_votes]
    if decade is not None and "decade" in d.columns:
        d = d[d["decade"] == decade]
    if director_like:
        if "director_name" in d.columns:
            m1 = d["director_name"].fillna("").str.contains(director_like, case=False, na=False)
        else:
            m1 = pd.Series(False, index=d.index)
        m2 = d["directors"].fillna("").str.contains(director_like, case=False, na=False) if "directors" in d.columns else pd.Series(False, index=d.index)
        d = d[m1 | m2]
    return d

# --------------------
# callbacks
# --------------------
@app.callback(
    Output("year-range-label", "children"),
    Output("min-votes-label", "children"),
    Input("year-range", "value"),
    Input("min-votes", "value"),
)
def _labels(year_range, min_votes):
    yr = f"{year_range[0]}–{year_range[1]}" if year_range else "All"
    return f"Selected: {yr}", f"Min votes: {min_votes}"

@app.callback(
    Output("hist", "figure"),
    Output("hist-note", "children"),
    Output("yearly", "figure"),
    Output("yearly-note", "children"),
    Output("genrebar", "figure"),
    Output("genre-note", "children"),
    Output("scatter", "figure"),
    Output("scatter-note", "children"),
    Output("runtime2d", "figure"),
    Output("runtime-note", "children"),
    Output("directors", "figure"),
    Output("directors-note", "children"),
    Input("year-range", "value"),
    Input("genre-select", "value"),
    Input("min-votes", "value"),
    Input("decade-select", "value"),
    Input("director-like", "value"),
)
def _update_all(year_range, genres, min_votes, decade, director_like):
    d = apply_filters(year_range, genres, min_votes, decade, director_like)
    return (
        fig_ratings_hist(d), "Distribution adapts to filter; axes tighten automatically.",
        fig_yearly_counts_mean(d), "Right axis shows mean rating; left axis scales to filtered counts.",
        fig_mean_rating_by_genre(d), "Low-n genres (n < 100) hidden to reduce noise.",
        fig_votes_vs_rating(d), "Log votes with data-driven bounds; no vertical stretch.",
        fig_runtime_vs_rating(d), "Runtime and rating bounds adapt with percentile clipping.",
        fig_directors_bubble(d), "Directors with ≥5 films; x-range fits filtered output size.",
    )

@app.callback(
    Output("dl-data", "data"),
    Input("dl-btn", "n_clicks"),
    State("year-range", "value"),
    State("genre-select", "value"),
    State("min-votes", "value"),
    State("decade-select", "value"),
    State("director-like", "value"),
    prevent_initial_call=True
)
def _download(n_clicks, year_range, genres, min_votes, decade, director_like):
    d = apply_filters(year_range, genres, min_votes, decade, director_like)
    return dict(content=d.to_csv(index=False), filename="imdb_filtered.csv")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
