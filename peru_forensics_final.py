"""
peru_forensics_final.py — Peru 2021 Segunda Vuelta Corrected Analysis
======================================================================
Sample  : CONTABILIZADA + COMPUTADA RESUELTA, domestic mesas only
Dropped : ANULADA (213), EN PROCESO (20), SIN INSTALAR (6)
Overseas: excluded from main, reported separately
Variable: contested = 1 if COMPUTADA RESUELTA, 0 if CONTABILIZADA
"""

import sys, re, unicodedata
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import seaborn as sns
import geopandas as gpd
from scipy import stats

# ── 0. Style ──────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.1,
})

FINAL_STATUS = {"CONTABILIZADA", "COMPUTADA RESUELTA"}
OVERSEAS     = {"AMERICA", "EUROPA", "ASIA", "OCEANIA", "AFRICA"}
COSTA  = {"LIMA","LA LIBERTAD","LAMBAYEQUE","PIURA","ICA","AREQUIPA","MOQUEGUA",
          "TACNA","TUMBES","ANCASH","LIMA PROVINCIAS","CALLAO"}
SELVA  = {"LORETO","UCAYALI","MADRE DE DIOS","AMAZONAS","SAN MARTIN","HUANUCO","SAN MARTÍN"}

REG_COLOR  = {"Costa":"#2166ac", "Sierra":"#d73027", "Selva":"#1a9850"}
REG_MARKER = {"Costa":"o",       "Sierra":"s",       "Selva":"^"}

def region(dep):
    d = dep.strip().upper()
    return "Costa" if d in COSTA else ("Selva" if d in SELVA else "Sierra")

def norm(s):
    """Uppercase, strip accents, remove ALL non-A-Z (inc. garbled ¿ chars)."""
    s = str(s).upper().strip()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s

# Province-name override: (ONPE_DEPT_NORM, ONPE_PROV_NORM) → (GADM_DEPT_NORM, GADM_PROV_NORM)
# The ¿¿ garbling: MARAÑON→MARAON, FERREÑAFE→FERREAFE, CAÑETE→CAETE after norm()
PROV_NORM_FIX = {
    ("ANCASH",      "ANTONIORAIMONDI"): ("ANCASH",       "ANTONIORAYMONDI"),
    ("HUANUCO",     "HUANUCO"):         ("HUANUCO",      "HUENUCO"),     # dept-named province
    ("ICA",         "NASCA"):           ("ICA",          "NAZCA"),
    ("HUANUCO",     "MARAON"):          ("HUANUCO",      "MARANON"),     # garbled MARAÑON
    ("LAMBAYEQUE",  "FERREAFE"):        ("LAMBAYEQUE",   "FERRENAFE"),   # garbled FERREÑAFE
    ("LIMA",        "CAETE"):           ("LIMA",         "CANETE"),      # garbled CAÑETE
    # LIMA dept / LIMA prov in ONPE = Lima City province → GADM "Lima" dept / "Lima" prov
    # GADM "LimaProvince"/"Lima" = rural Lima surrounding provinces → no direct ONPE equiv
    # (all rural Lima provinces in ONPE are under dept "LIMA", not "LIMA PROVINCIAS")
    # PUNO / LAGOTITICACA in GADM = lake polygon, no voters → left gray
}

def save_fig(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}")
    print(f"  → {stem}.png / .pdf")

def sep(title=""):
    print(f"\n{'='*68}")
    if title:
        print(f"  {title}")
        print(f"{'='*68}")

def ols(y, Xs, names):
    """
    OLS with HC0-robust SEs. Returns (betas, SEs, t-stats, p-values, R2).
    Xs: list of 1D arrays (each a column); intercept must be included.
    Prints formatted table.
    """
    X = np.column_stack(Xs)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y = X[mask], y[mask]
    n, k = X.shape

    b   = np.linalg.lstsq(X, y, rcond=None)[0]
    yh  = X @ b
    e   = y - yh
    ss_r = (e**2).sum()
    ss_t = ((y - y.mean())**2).sum()
    r2   = 1 - ss_r / ss_t

    # HC0 robust variance
    XtX_inv = np.linalg.inv(X.T @ X)
    meat = (X * e[:, None]).T @ (X * e[:, None])
    V    = XtX_inv @ meat @ XtX_inv
    se   = np.sqrt(np.diag(V))

    t_s  = b / se
    p_v  = 2 * (1 - stats.t.cdf(np.abs(t_s), df=n - k))

    print(f"  N = {n:,}   R² = {r2:.4f}")
    print(f"  {'Variable':<28} {'Coef':>9} {'HC0-SE':>9} {'t':>7} {'p':>8}  sig")
    print(f"  {'-'*65}")
    for nm, bi, si, ti, pi in zip(names, b, se, t_s, p_v):
        sig = "***" if pi<0.001 else "**" if pi<0.01 else "*" if pi<0.05 else ""
        print(f"  {nm:<28} {bi:>9.4f} {si:>9.4f} {ti:>7.2f} {pi:>8.4f}  {sig}")
    return b, se, t_s, p_v, r2

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("Loading segunda vuelta...")
df_raw = pd.read_csv(
    "segunda_vuelta_data/Resultados_2da_vuelta_Version_PCM .csv",
    encoding="latin-1", sep=";", quotechar='"', index_col=False,
    dtype={"UBIGEO": str, "MESA_DE_VOTACION": str}
)
for c in ["N_CVAS","N_ELEC_HABIL","VOTOS_P1","VOTOS_P2","VOTOS_VB","VOTOS_VN","VOTOS_VI"]:
    df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce").fillna(0)
df_raw.rename(columns={
    "N_ELEC_HABIL":"REG_VOTERS", "N_CVAS":"TOTAL_VOTES",
    "VOTOS_P1":"CASTILLO",      "VOTOS_P2":"FUJIMORI",
    "VOTOS_VN":"NULL_VOTES",    "VOTOS_VB":"BLANK_VOTES",
    "VOTOS_VI":"CHALLENGED",
}, inplace=True)

stat_u = df_raw["DESCRIP_ESTADO_ACTA"].str.strip().str.upper()
dept_u = df_raw["DEPARTAMENTO"].str.strip().str.upper()

df_overseas = df_raw[stat_u.isin(FINAL_STATUS) & dept_u.isin(OVERSEAS)].copy()
df = df_raw[stat_u.isin(FINAL_STATUS) & ~dept_u.isin(OVERSEAS)].copy()

df["contested"]   = (df["DESCRIP_ESTADO_ACTA"].str.strip().str.upper() == "COMPUTADA RESUELTA").astype(int)
df["VALID_VOTES"] = df["CASTILLO"] + df["FUJIMORI"]
df["TURNOUT"]     = df["TOTAL_VOTES"] / df["REG_VOTERS"].replace(0, np.nan)
df["SHARE_C"]     = df["CASTILLO"]   / df["VALID_VOTES"].replace(0, np.nan)
df["SHARE_F"]     = df["FUJIMORI"]   / df["VALID_VOTES"].replace(0, np.nan)
df["NULL_SHARE"]  = df["NULL_VOTES"] / df["TOTAL_VOTES"].replace(0, np.nan)
df["BLANK_SHARE"] = df["BLANK_VOTES"] / df["TOTAL_VOTES"].replace(0, np.nan)
df["REGION"]      = df["DEPARTAMENTO"].str.strip().str.upper().map(region)
df["UBIGEO"]      = df["UBIGEO"].str.zfill(6)
df["DEPT_NORM"]   = df["DEPARTAMENTO"].apply(norm)

# Apply province norm with fix
df["PROV_NORM_RAW"] = df["PROVINCIA"].apply(norm)
df["PROV_NORM"] = df.apply(
    lambda r: PROV_NORM_FIX.get((r["DEPT_NORM"], r["PROV_NORM_RAW"]),
                                (r["DEPT_NORM"], r["PROV_NORM_RAW"]))[1], axis=1
)
df["DEPT_NORM_GADM"] = df.apply(
    lambda r: PROV_NORM_FIX.get((r["DEPT_NORM"], r["PROV_NORM_RAW"]),
                                (r["DEPT_NORM"], r["PROV_NORM_RAW"]))[0], axis=1
)

# ══════════════════════════════════════════════════════════════════════════════
# 2. AUDIT NOTE
# ══════════════════════════════════════════════════════════════════════════════
sep("AUDIT NOTE — Row accounting")
print("Formulas:")
print("  TURNOUT     = TOTAL_VOTES (N_CVAS) / REG_VOTERS (N_ELEC_HABIL)")
print("  VALID_VOTES = CASTILLO (VOTOS_P1) + FUJIMORI (VOTOS_P2)")
print("  SHARE_C     = CASTILLO / VALID_VOTES")
print("  NULL_SHARE  = NULL_VOTES (VOTOS_VN) / TOTAL_VOTES")
print("  BLANK_SHARE = BLANK_VOTES (VOTOS_VB) / TOTAL_VOTES")
print("  contested   = 1 if COMPUTADA RESUELTA, 0 if CONTABILIZADA")
print()
print("Merge keys:")
print("  Province map: (DEPT_NORM_GADM, PROV_NORM) on GADM (DEPT_NORM, PROV_NORM)")
print("  Primera-segunda: UBIGEO (6-digit, zero-padded string)")
print()

vc = df_raw["DESCRIP_ESTADO_ACTA"].str.strip().str.upper().value_counts()
print("All 86,488 rows by DESCRIP_ESTADO_ACTA:")
for s, n_ in vc.items():
    tag = "KEPT (final-status)" if s in FINAL_STATUS else "DROPPED"
    print(f"  {s:<30} {n_:>6,}  ← {tag}")
print()
print("Final-status overseas mesas (kept separately):")
for dep, n_ in df_overseas["DEPARTAMENTO"].value_counts().items():
    print(f"  {dep:<20} {n_:>6,}")
print(f"  {'TOTAL overseas':<20} {len(df_overseas):>6,}")
print()
print(f"Main domestic final-status sample:  {len(df):>7,} mesas")
print(f"  CONTABILIZADA:                    {(df['contested']==0).sum():>7,}")
print(f"  COMPUTADA RESUELTA:               {(df['contested']==1).sum():>7,}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — National totals (domestic, all final-status)
# ══════════════════════════════════════════════════════════════════════════════
sep("SECTION 1 — National totals (domestic final-status mesas)")

def totals_block(df_x, label=""):
    reg   = df_x["REG_VOTERS"].sum()
    cast  = df_x["TOTAL_VOTES"].sum()
    c_v   = df_x["CASTILLO"].sum()
    f_v   = df_x["FUJIMORI"].sum()
    valid = c_v + f_v
    nul   = df_x["NULL_VOTES"].sum()
    blk   = df_x["BLANK_VOTES"].sum()
    chal  = df_x["CHALLENGED"].sum()
    if label: print(f"  {label}")
    print(f"  Mesas:                      {len(df_x):>9,}")
    print(f"  Registered voters:          {reg:>9,.0f}")
    print(f"  Votes cast (TOTAL_VOTES):   {cast:>9,.0f}   turnout {cast/reg:.4%}")
    print(f"  Valid votes (C + F):        {valid:>9,.0f}")
    print(f"  Castillo  (VOTOS_P1):       {c_v:>9,.0f}   {c_v/valid:.6%} of valid")
    print(f"  Fujimori  (VOTOS_P2):       {f_v:>9,.0f}   {f_v/valid:.6%} of valid")
    print(f"  Margin (Castillo − Fujimori): {c_v-f_v:>7,.0f}   {(c_v-f_v)/valid:.6%}")
    print(f"  Null votes  (VOTOS_VN):     {nul:>9,.0f}   {nul/cast:.4%} of cast")
    print(f"  Blank votes (VOTOS_VB):     {blk:>9,.0f}   {blk/cast:.4%} of cast")
    print(f"  Challenged  (VOTOS_VI):     {chal:>9,.0f}   {chal/cast:.4%} of cast")
    return c_v, f_v, valid

print()
c_dom, f_dom, v_dom = totals_block(df, "DOMESTIC (main analysis sample)")
print()
totals_block(df_overseas, "OVERSEAS (separate, not in main analysis)")
print()

# Reconciliation with official figure
print("  Reconciliation with official ONPE margin of 44,263:")
c_all = df_raw["CASTILLO"].sum()
f_all = df_raw["FUJIMORI"].sum()
v_all = c_all + f_all
print(f"  All 86,488 rows: Castillo {c_all:,.0f}  Fujimori {f_all:,.0f}  margin {c_all-f_all:,.0f}")
print(f"  Our domestic final-status:          margin {c_dom-f_dom:,.0f}")
print(f"  Difference vs official 44,263:      {abs(c_dom-f_dom - 44263):,.0f} votes")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Contested vs uncontested comparison
# ══════════════════════════════════════════════════════════════════════════════
sep("SECTION 2 — Contested vs Uncontested comparison")

def grp_stats(g, label):
    n_      = len(g)
    c_v     = g["CASTILLO"].sum()
    f_v     = g["FUJIMORI"].sum()
    valid_  = c_v + f_v
    return pd.Series({
        "Group":           label,
        "N mesas":         n_,
        "Avg turnout":     g["TURNOUT"].mean(),
        "Avg Castillo %":  g["SHARE_C"].mean(),
        "Avg Fujimori %":  g["SHARE_F"].mean(),
        "Avg null %":      g["NULL_SHARE"].mean(),
        "Avg blank %":     g["BLANK_SHARE"].mean(),
        "Total Castillo":  c_v,
        "Total Fujimori":  f_v,
        "Net margin (C-F)": c_v - f_v,
    })

rows = [
    grp_stats(df[df["contested"]==0], "Uncontested (CONTABILIZADA)"),
    grp_stats(df[df["contested"]==1], "Contested   (COMP. RESUELTA)"),
    grp_stats(df,                     "TOTAL"),
]
tbl = pd.DataFrame(rows).set_index("Group")
print()
print(tbl.to_string(
    formatters={
        "N mesas":         "{:,.0f}".format,
        "Avg turnout":     "{:.4%}".format,
        "Avg Castillo %":  "{:.4%}".format,
        "Avg Fujimori %":  "{:.4%}".format,
        "Avg null %":      "{:.4%}".format,
        "Avg blank %":     "{:.4%}".format,
        "Total Castillo":  "{:,.0f}".format,
        "Total Fujimori":  "{:,.0f}".format,
        "Net margin (C-F)":"{:+,.0f}".format,
    }))

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Last-digit tests (by contested status)
# ══════════════════════════════════════════════════════════════════════════════
sep("SECTION 3 — Last-digit uniformity tests")

def lastdigit_full(series, label):
    vals   = series.dropna().astype(int)
    n_zero = (vals == 0).sum()
    vals   = vals[vals > 0]
    last_d = vals % 10
    counts = last_d.value_counts().sort_index().reindex(range(10), fill_value=0)
    n      = counts.sum()
    chi2, p_val = stats.chisquare(counts.values)
    heap   = (counts[0] + counts[5]) / n
    print(f"\n  {label}  (n={n:,}, zeros excluded={n_zero:,})")
    print(f"  Chi2={chi2:.4f}  p={p_val:.6f}  {'REJECT' if p_val<0.05 else 'fail to reject'} uniform")
    print(f"  0+5 heaping: {100*heap:.2f}%  (expected 20.00%,  excess {100*(heap-0.2):+.2f}pp)")
    print(f"  Digit | Count  |   Pct  | Deviation")
    for d in range(10):
        pct = 100*counts[d]/n
        dev = pct - 10.0
        bar = "▊" * int(abs(dev)*10) if abs(dev) >= 0.05 else ""
        print(f"    {d}   |{counts[d]:6,} | {pct:5.2f}% | {dev:+.2f}pp {bar}")
    return counts, chi2, p_val, heap

print()
unc = df[df["contested"]==0]
con = df[df["contested"]==1]

print("── CASTILLO VOTES ─────────────────────────────────────")
ld_cu, chi_cu, p_cu, h_cu = lastdigit_full(unc["CASTILLO"], "Castillo / UNCONTESTED")
ld_cc, chi_cc, p_cc, h_cc = lastdigit_full(con["CASTILLO"], "Castillo / CONTESTED")

print("\n── FUJIMORI VOTES ─────────────────────────────────────")
ld_fu, chi_fu, p_fu, h_fu = lastdigit_full(unc["FUJIMORI"], "Fujimori / UNCONTESTED")
ld_fc, chi_fc, p_fc, h_fc = lastdigit_full(con["FUJIMORI"], "Fujimori / CONTESTED")

print()
print("  NOTE: Contested sample n≈1,386 — low power; chi2 test has ~25%")
print("  power to detect 20% heaping at this sample size (vs >99% for n=84,xxx)")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Primera vuelta: load, merge, regression
# ══════════════════════════════════════════════════════════════════════════════
sep("SECTION 4 — Primera vuelta merge and regression")

print("Loading primera vuelta (same status filter)...")
df1_raw = pd.read_csv(
    "primera_vuelta_data/Resultados_1ra_vuelta_Version_PCM.csv",
    encoding="latin-1", sep=";", quotechar='"', index_col=False, dtype={"UBIGEO": str}
)
for c in df1_raw.columns:
    if c.startswith("VOTOS") or c in ["N_CVAS","N_ELEC_HABIL"]:
        df1_raw[c] = pd.to_numeric(df1_raw[c], errors="coerce").fillna(0)

stat1_u = df1_raw["DESCRIP_ESTADO_ACTA"].str.strip().str.upper()
dept1_u = df1_raw["DEPARTAMENTO"].str.strip().str.upper()
df1 = df1_raw[stat1_u.isin(FINAL_STATUS) & ~dept1_u.isin(OVERSEAS)].copy()

print(f"Primera vuelta domestic final-status: {len(df1):,} rows")
print(f"  CONTABILIZADA:       {(stat1_u[stat1_u.isin(FINAL_STATUS) & ~dept1_u.isin(OVERSEAS)] == 'CONTABILIZADA').sum():,}")
print(f"  COMPUTADA RESUELTA:  {(stat1_u[stat1_u.isin(FINAL_STATUS) & ~dept1_u.isin(OVERSEAS)] == 'COMPUTADA RESUELTA').sum():,}")

# VOTOS_P16 = Castillo (Peru Libre) — confirmed by 18.9% national share
# VOTOS_P11 = Fujimori (Fuerza Popular) — confirmed by 13.4%
p_cols = [c for c in df1.columns if c.startswith("VOTOS_P")]
df1["C1_VALID_ALL"] = df1[p_cols].sum(axis=1)   # computed BEFORE rename
df1.rename(columns={
    "VOTOS_P16":"C1_CASTILLO", "N_ELEC_HABIL":"C1_REG", "N_CVAS":"C1_TOTAL"
}, inplace=True)
df1["UBIGEO"] = df1["UBIGEO"].str.zfill(6)

# Verify identification
c1_nat = df1["C1_CASTILLO"].sum()
v1_nat = df1["C1_VALID_ALL"].sum()
print(f"\nVerification — Castillo (P16) primera vuelta national:")
print(f"  {c1_nat:,.0f} / {v1_nat:,.0f} = {c1_nat/v1_nat:.4%}  (official: ~18.9%)")

# District-level aggregation
grp2 = df.groupby("UBIGEO").agg(
    castillo_2v  = ("CASTILLO","sum"),
    fujimori_2v  = ("FUJIMORI","sum"),
    reg_2v       = ("REG_VOTERS","sum"),
    n_contested  = ("contested","sum"),
    n_total      = ("contested","count"),
).reset_index()
grp2["valid_2v"]     = grp2["castillo_2v"] + grp2["fujimori_2v"]
grp2["share_2v"]     = grp2["castillo_2v"] / grp2["valid_2v"].replace(0, np.nan)
grp2["share_cont"]   = grp2["n_contested"] / grp2["n_total"]  # fraction contested

grp1 = df1.groupby("UBIGEO").agg(
    castillo_1v  = ("C1_CASTILLO","sum"),
    valid_all_1v = ("C1_VALID_ALL","sum"),
    reg_1v       = ("C1_REG","sum"),
).reset_index()
grp1["share_1v"] = grp1["castillo_1v"] / grp1["valid_all_1v"].replace(0, np.nan)

dept_map = df.groupby("UBIGEO")[["DEPARTAMENTO","REGION"]].first().reset_index()
dist = grp1.merge(grp2, on="UBIGEO").merge(dept_map, on="UBIGEO", how="left")
dist = dist.dropna(subset=["share_1v","share_2v"])
dist = dist[(dist["share_1v"] > 0) & (dist["valid_2v"] > 10)]

print(f"\nDistricts in 1v ∩ 2v (domestic, >10 valid 2v votes, C>0 in 1v): {len(dist):,}")
print(f"Districts with at least 1 contested mesa: {(dist['n_contested']>0).sum():,}")
print(f"Districts with share_cont > 10%: {(dist['share_cont']>0.1).sum():,}")

dist["interaction"] = dist["share_1v"] * dist["share_cont"]

print("\n──────────────────────────────────────────────────────────────────")
print("Spec 1:  share_2v ~ share_1v")
b1, se1, t1, p1, r2_1 = ols(
    dist["share_2v"],
    [np.ones(len(dist)), dist["share_1v"]],
    ["intercept", "share_1v"]
)

print("\nSpec 2:  share_2v ~ share_1v + share_cont + share_1v × share_cont")
b2, se2, t2, p2, r2_2 = ols(
    dist["share_2v"],
    [np.ones(len(dist)), dist["share_1v"], dist["share_cont"], dist["interaction"]],
    ["intercept", "share_1v", "share_cont", "share_1v × share_cont"]
)
print()
print("  Interpretation of Spec 2:")
print(f"  share_cont coeff = {b2[2]:+.4f}  p={p2[2]:.4f}: "
      f"{'significant' if p2[2]<0.05 else 'NOT significant'} baseline shift")
print(f"  interaction coeff= {b2[3]:+.4f}  p={p2[3]:.4f}: "
      f"{'significant' if p2[3]<0.05 else 'NOT significant'} slope shift")
print("  γ>0 would mean higher contested share → higher Castillo 2v share (fraud-consistent)")
print("  γ<0 would mean higher contested share → lower Castillo 2v share (challenge-consistent)")

# Top residuals
dist["resid"] = dist["share_2v"] - (b1[0] + b1[1]*dist["share_1v"])
top10  = dist.nlargest(10, "resid")
bot10  = dist.nsmallest(10, "resid")
print("\n  Top 10 over-performers (Spec1 residual):")
print(top10[["UBIGEO","DEPARTAMENTO","share_1v","share_2v","share_cont","resid"]].to_string(
    index=False,
    formatters={"share_1v":"{:.3f}".format,"share_2v":"{:.3f}".format,
                "share_cont":"{:.3f}".format,"resid":"{:+.3f}".format}))

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Geographic distribution of contested mesas
# ══════════════════════════════════════════════════════════════════════════════
sep("SECTION 5 — Geographic distribution of contested mesas")

dept_geo = df.groupby("DEPARTAMENTO").agg(
    n_cont   = ("contested","sum"),
    n_total  = ("contested","count"),
    castillo = ("CASTILLO","sum"),
    fujimori = ("FUJIMORI","sum"),
    region   = ("REGION","first"),
).reset_index()
dept_geo["n_unc"] = dept_geo["n_total"] - dept_geo["n_cont"]
dept_geo["share_cont"]  = dept_geo["n_cont"] / dept_geo["n_total"]
dept_geo["valid"]       = dept_geo["castillo"] + dept_geo["fujimori"]
dept_geo["margin_c"]    = (dept_geo["castillo"] - dept_geo["fujimori"]) / dept_geo["valid"]

print()
print("By department (sorted by share_contested descending):")
dg = dept_geo.sort_values("share_cont", ascending=False)
print(f"  {'DEPT':<22} {'n_cont':>7} {'n_unc':>7} {'n_total':>8} {'%cont':>7} {'margin_C%':>10}")
print(f"  {'-'*65}")
for _, r in dg.iterrows():
    print(f"  {r['DEPARTAMENTO']:<22} {r['n_cont']:>7,} {r['n_unc']:>7,} "
          f"{r['n_total']:>8,} {100*r['share_cont']:>7.2f}% {100*r['margin_c']:>+10.2f}%")

print()
print("By macro-region:")
reg_geo = df.groupby("REGION").agg(
    n_cont  = ("contested","sum"),
    n_total = ("contested","count"),
).reset_index()
reg_geo["share_cont"] = reg_geo["n_cont"] / reg_geo["n_total"]
for _, r in reg_geo.iterrows():
    print(f"  {r['REGION']:<8}: {r['n_cont']:>5,} contested / {r['n_total']:>6,} total "
          f"({100*r['share_cont']:.2f}%)")

print()
print("Contested mesa Castillo margin vs uncontested, by department:")
tmp = df.groupby(["DEPARTAMENTO","contested"]).agg(
    castillo=("CASTILLO","sum"), fujimori=("FUJIMORI","sum")
).reset_index()
tmp["valid"]    = tmp["castillo"] + tmp["fujimori"]
tmp["margin_c"] = (tmp["castillo"] - tmp["fujimori"]) / tmp["valid"]
piv = tmp.pivot(index="DEPARTAMENTO", columns="contested", values="margin_c")
piv.columns = ["margin_uncontested","margin_contested"]
piv["diff"] = piv["margin_contested"] - piv["margin_uncontested"]
piv = piv.sort_values("diff", na_position="last")
print(f"  {'DEPT':<22} {'uncont margin':>14} {'cont margin':>12} {'diff':>7}")
print(f"  {'-'*57}")
for dept_, r in piv.iterrows():
    if pd.isna(r["margin_contested"]):
        print(f"  {dept_:<22} {100*r['margin_uncontested']:>+13.2f}% {'(no cont)':>13}")
    else:
        print(f"  {dept_:<22} {100*r['margin_uncontested']:>+13.2f}% "
              f"{100*r['margin_contested']:>+11.2f}% {100*r['diff']:>+6.2f}pp")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Province aggregates and shapefile prep
# ══════════════════════════════════════════════════════════════════════════════
sep("SECTION 6 — Province aggregates and shapefile join")

prov = df.groupby(["DEPT_NORM_GADM","PROV_NORM","DEPARTAMENTO","PROVINCIA"]).agg(
    castillo    = ("CASTILLO","sum"),
    fujimori    = ("FUJIMORI","sum"),
    null_votes  = ("NULL_VOTES","sum"),
    total_cast  = ("TOTAL_VOTES","sum"),
    reg_voters  = ("REG_VOTERS","sum"),
    n_contested = ("contested","sum"),
    n_total     = ("contested","count"),
).reset_index()
prov.rename(columns={"DEPT_NORM_GADM":"DEPT_NORM"}, inplace=True)
prov["valid"]         = prov["castillo"] + prov["fujimori"]
prov["share_c"]       = prov["castillo"]  / prov["valid"].replace(0, np.nan)
prov["null_share"]    = prov["null_votes"] / prov["total_cast"].replace(0, np.nan)
prov["turnout"]       = prov["total_cast"] / prov["reg_voters"].replace(0, np.nan)
prov["share_cont"]    = prov["n_contested"] / prov["n_total"]

# Load shapefiles
print("Loading shapefiles...")
gdf_prov = gpd.read_file("gadm_peru_prov.json")
gdf_dept = gpd.read_file("gadm_peru_dept.json")
gdf_prov["DEPT_NORM"] = gdf_prov["NAME_1"].apply(norm)
gdf_prov["PROV_NORM"] = gdf_prov["NAME_2"].apply(norm)
gdf_dept["DEPT_NORM"] = gdf_dept["NAME_1"].apply(norm)

# Primary merge
data_cols = ["DEPT_NORM","PROV_NORM","share_c","null_share","share_cont",
             "castillo","fujimori","valid","turnout","total_cast","n_contested","n_total"]
gdf_m = gdf_prov.merge(prov[data_cols], on=["DEPT_NORM","PROV_NORM"], how="left")

n_matched   = gdf_m["share_c"].notna().sum()
unmatched   = gdf_m[gdf_m["share_c"].isna()][["NAME_1","NAME_2","DEPT_NORM","PROV_NORM"]]
print(f"\nPrimary match: {n_matched}/{len(gdf_prov)} provinces matched ({100*n_matched/len(gdf_prov):.1f}%)")
print(f"Remaining unmatched ({len(unmatched)}):")
print(unmatched.to_string(index=False))

# Build ONPE lookup by raw (dept, prov) for secondary matching
onpe_lookup = {}
for _, r in prov.iterrows():
    k = (r["DEPARTAMENTO"].strip().upper(), r["PROVINCIA"].strip().upper())
    onpe_lookup[k] = r

# Manual secondary matching for known mismatches
SECONDARY = {
    # (GADM NAME_1 upper, GADM NAME_2 upper) : list of ONPE (dept, prov) to try
    ("ANCASH",       "ANTONIORAYMONDI"):  [("ANCASH",       "ANTONIO RAIMONDI")],
    ("HUÁNUCO",      "HUENUCO"):          [("HUANUCO",      "HUANUCO")],
    ("ICA",          "NAZCA"):            [("ICA",          "NASCA")],
    ("LAMBAYEQUE",   "FERREÑAFE"):        [("LAMBAYEQUE",   "FERRE\u00bf\u00bfAFE"), ("LAMBAYEQUE","FERRENAFE")],
    ("HUÁNUCO",      "MARAÑÓN"):          [("HUANUCO",      "MARA\u00bf\u00bfON"),   ("HUANUCO","MARANON")],
    ("LIMA",         "CAÑETE"):           [("LIMA",         "CA\u00bf\u00bfETE"),    ("LIMA","CANETE")],
    ("LIMA PROVINCE","LIMA"):             [("LIMA",         "LIMA"),  ("LIMA PROVINCIAS","LIMA")],
    ("PUNO",         "LAGO TITICACA"):    [],  # lake — no voters
}

n_secondary = 0
for idx in gdf_m[gdf_m["share_c"].isna()].index:
    g1 = gdf_m.loc[idx, "NAME_1"].strip().upper()
    g2 = gdf_m.loc[idx, "NAME_2"].strip().upper()
    candidates = SECONDARY.get((g1, g2), [])
    matched_row = None
    for (od, op) in candidates:
        if (od, op) in onpe_lookup:
            matched_row = onpe_lookup[(od, op)]
            break
    if matched_row is not None:
        for col in ["share_c","null_share","share_cont","castillo","fujimori",
                    "valid","turnout","total_cast","n_contested","n_total"]:
            if col in matched_row.index:
                gdf_m.loc[idx, col] = matched_row[col]
        n_secondary += 1
        print(f"  Secondary match: GADM {g1}/{g2} → ONPE {candidates[0]}")

print(f"\nAfter secondary matching: {gdf_m['share_c'].notna().sum()}/{len(gdf_prov)} matched")
print(f"Still unmatched ({gdf_m['share_c'].isna().sum()}):")
still = gdf_m[gdf_m["share_c"].isna()][["NAME_1","NAME_2"]]
print(still.to_string(index=False))

# Project to UTM zone 18S
PERU_CRS = "EPSG:32718"
gdf_p  = gdf_m.to_crs(PERU_CRS)
gdf_dp = gdf_dept.to_crs(PERU_CRS)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════
sep("FIGURES")

# ── FIG 1 — Castillo vote share choropleth ────────────────────────────────────
print("\nFig 1: Castillo vote share by province")
fig, ax = plt.subplots(figsize=(7, 10))
ax.set_axis_off()
cmap1  = mpl.colormaps["RdBu_r"]
norm1  = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
gdf_p.plot(column="share_c", cmap=cmap1, norm=norm1, ax=ax,
           missing_kwds={"color":"#cccccc"}, linewidth=0.15, edgecolor="white")
gdf_dp.boundary.plot(ax=ax, linewidth=0.7, color="black", alpha=0.7)
sm = mpl.cm.ScalarMappable(cmap=cmap1, norm=norm1)
sm.set_array([])
cb = fig.colorbar(sm, ax=ax, shrink=0.45, pad=0.01, aspect=25, location="right")
cb.set_label("Castillo share of valid votes", fontsize=9)
cb.set_ticks([0, 0.25, 0.50, 0.75, 1.0])
cb.set_ticklabels(["0%\n(Fujimori)", "25%", "50%\n(Tie)", "75%", "100%\n(Castillo)"])
cb.ax.tick_params(labelsize=8)
# title removed — caption in LaTeX
ax.annotate(f"Department boundaries in black. Gray = unmatched ({gdf_p['share_c'].isna().sum()} provs).",
            xy=(0.01,0.01), xycoords="axes fraction", fontsize=7, color="#555", va="bottom")
save_fig(fig, "fig1_map_castillo")
plt.close()

# ── FIG 2 — Klimek plots with contested highlighted ───────────────────────────
print("Fig 2: Klimek fingerprint plots (contested highlighted)")

def klimek_panel_v2(ax_main, ax_top, ax_right, share_col, candidate_name):
    sub_unc = df[df["contested"]==0][[
        "TURNOUT", share_col, "REGION"]].dropna()
    sub_unc = sub_unc[(sub_unc["TURNOUT"]>0.05)&(sub_unc["TURNOUT"]<=1.02)
                      &(sub_unc[share_col]>0)&(sub_unc[share_col]<1)]
    sub_con = df[df["contested"]==1][[
        "TURNOUT", share_col, "REGION"]].dropna()
    sub_con = sub_con[(sub_con["TURNOUT"]>0.05)&(sub_con["TURNOUT"]<=1.02)
                      &(sub_con[share_col]>0)&(sub_con[share_col]<1)]

    # Gray cloud: uncontested
    ax_main.scatter(sub_unc["TURNOUT"], sub_unc[share_col],
                    c="#aaaaaa", alpha=0.05, s=3, rasterized=True, linewidths=0,
                    label=f"Uncontested (n={len(sub_unc):,})")
    # Orange layer: contested
    ax_main.scatter(sub_con["TURNOUT"], sub_con[share_col],
                    c="#e67e00", alpha=0.55, s=12, rasterized=True, linewidths=0.3,
                    edgecolors="#c0392b", zorder=5,
                    label=f"Contested (n={len(sub_con):,})")

    # Contours on uncontested only
    h, _, _ = np.histogram2d(sub_unc["TURNOUT"].clip(0,1),
                              sub_unc[share_col].clip(0,1), bins=65)
    ax_main.contour(np.linspace(0,1,65), np.linspace(0,1,65), h.T,
                    levels=[5,25,80,200], colors=["#333"], linewidths=[0.3,0.5,0.7,0.9], alpha=0.5)

    ax_main.axhline(0.5, color="black", lw=0.6, ls="--", alpha=0.4)
    ax_main.set_xlim(0.05, 1.02); ax_main.set_ylim(0.0, 1.0)
    ax_main.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax_main.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax_main.set_xlabel("Turnout (votes cast / registered voters)", fontsize=9)
    ax_main.set_ylabel(f"{candidate_name} share of valid votes", fontsize=9)
    # panel title removed — caption in LaTeX
    ax_main.legend(fontsize=8, markerscale=2,
                   loc="upper center", bbox_to_anchor=(0.5, -0.12),
                   ncol=2, framealpha=0.9)

    # Marginals
    all_sub = pd.concat([sub_unc, sub_con])
    sns.kdeplot(all_sub["TURNOUT"].clip(0.05,1.0), ax=ax_top,
                color="#555", lw=1.2, fill=True, alpha=0.2)
    ax_top.set_xlim(0.05, 1.02); ax_top.set_axis_off()
    sns.kdeplot(y=all_sub[share_col].clip(0,1), ax=ax_right,
                color="#555", lw=1.2, fill=True, alpha=0.2)
    ax_right.set_ylim(0.0, 1.0); ax_right.set_axis_off()

fig2 = plt.figure(figsize=(14, 6.5))
# suptitle removed — caption in LaTeX
gs2 = GridSpec(2, 4, width_ratios=[4,0.65,4,0.65], height_ratios=[0.65,4],
               hspace=0.05, wspace=0.05, left=0.07, right=0.97, top=0.92, bottom=0.10)
axm_c = fig2.add_subplot(gs2[1,0]); axt_c = fig2.add_subplot(gs2[0,0], sharex=axm_c)
axr_c = fig2.add_subplot(gs2[1,1], sharey=axm_c)
axm_f = fig2.add_subplot(gs2[1,2]); axt_f = fig2.add_subplot(gs2[0,2], sharex=axm_f)
axr_f = fig2.add_subplot(gs2[1,3], sharey=axm_f)
klimek_panel_v2(axm_c, axt_c, axr_c, "SHARE_C", "Castillo")
klimek_panel_v2(axm_f, axt_f, axr_f, "SHARE_F", "Fujimori")
axm_f.set_ylabel("")
save_fig(fig2, "fig2_klimek")
plt.close()

# ── FIG 3 — Last-digit (4-panel) ─────────────────────────────────────────────
print("Fig 3: Last-digit 4-panel")

def ld_panel(ax, series, label, color):
    vals   = series.dropna().astype(int)
    vals   = vals[vals > 0]
    last_d = vals % 10
    counts = last_d.value_counts().sort_index().reindex(range(10), fill_value=0)
    n      = len(vals)
    pct    = 100 * counts / n
    chi2, p_val = stats.chisquare(counts.values)
    bar_c  = [color if d not in [0,5] else "#555555" for d in range(10)]
    ax.bar(range(10), pct, color=bar_c, edgecolor="white", linewidth=0.4, width=0.7, zorder=3)
    ax.axhline(10.0, color="black", lw=1.2, ls="--", zorder=4)
    ax.set_xticks(range(10))
    ax.set_ylim(0, 16)
    ax.set_xlabel("Last digit", fontsize=8)
    ax.set_ylabel("Share (%)", fontsize=8)
    # panel title removed — caption in LaTeX
    ax.text(0.02, 0.98, label.replace("\n", " | "), transform=ax.transAxes,
            ha="left", va="top", fontsize=7.5, color="#333",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))
    ax.grid(axis="y", alpha=0.3, zorder=0)
    verdict = "Uniform" if p_val > 0.05 else "REJECT uniform"
    ax.text(0.97, 0.97, f"$\\chi^2$={chi2:.2f}\np={p_val:.3f}\nn={n:,}\n{verdict}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", alpha=0.9))

fig3, axes3 = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
# suptitle removed — caption in LaTeX
ld_panel(axes3[0,0], unc["CASTILLO"], "Castillo — Uncontested\n(CONTABILIZADA)", "#c0392b")
ld_panel(axes3[0,1], con["CASTILLO"], "Castillo — Contested\n(COMPUTADA RESUELTA)", "#e67e00")
ld_panel(axes3[1,0], unc["FUJIMORI"], "Fujimori — Uncontested\n(CONTABILIZADA)", "#2980b9")
ld_panel(axes3[1,1], con["FUJIMORI"], "Fujimori — Contested\n(COMPUTADA RESUELTA)", "#f39c12")
plt.tight_layout()
save_fig(fig3, "fig3_lastdigit")
plt.close()

# ── FIG 4 — Primera vs Segunda scatterplot ────────────────────────────────────
print("Fig 4: Primera vs segunda vuelta scatter")

fig4, ax4 = plt.subplots(figsize=(8, 7))
sizes4 = np.sqrt(dist["reg_2v"].clip(100)) / 5
for rn in ("Costa","Sierra","Selva"):
    m = dist["REGION"] == rn
    ax4.scatter(dist.loc[m,"share_1v"], dist.loc[m,"share_2v"],
                c=REG_COLOR[rn], marker=REG_MARKER[rn],
                s=sizes4[m], alpha=0.3, lw=0, rasterized=True, label=rn)
xs4 = np.linspace(dist["share_1v"].min(), dist["share_1v"].max(), 100)
ax4.plot(xs4, b1[0]+b1[1]*xs4, "k-", lw=1.8, zorder=5,
         label=f"OLS: $\\hat{{y}} = {b1[0]:.3f} + {b1[1]:.3f}x$\n$R^2 = {r2_1:.3f}$, N={len(dist):,}")
for _, row in top10.iterrows():
    dept_s = row["DEPARTAMENTO"][:4].title()
    ax4.scatter(row["share_1v"], row["share_2v"],
                c="none", edgecolors="black", s=55, lw=1.2, zorder=6)
    ax4.annotate(f"{row['UBIGEO']} ({dept_s})",
                 xy=(row["share_1v"], row["share_2v"]),
                 xytext=(5,3), textcoords="offset points", fontsize=6.5,
                 arrowprops=dict(arrowstyle="-", lw=0.5, color="gray"))
ax4.set_xlabel("Castillo 1st-round share (of all 18-candidate valid votes)", fontsize=9.5)
ax4.set_ylabel("Castillo 2nd-round share (of Castillo + Fujimori)", fontsize=9.5)
ax4.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax4.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
# title removed — caption in LaTeX
handles4, labels4 = ax4.get_legend_handles_labels()
ax4.legend(handles4, labels4, fontsize=8.5,
           loc="upper center", bbox_to_anchor=(0.5, -0.12),
           ncol=2, framealpha=0.9)
save_fig(fig4, "fig4_primera_vs_segunda")
plt.close()

# ── FIG 5 — Turnout distribution (1v vs 2v) ──────────────────────────────────
print("Fig 5: Turnout distribution")

t2v = df["TURNOUT"].dropna().clip(0.05, 1.0)
t1v = df1["C1_TOTAL"].div(df1["C1_REG"].replace(0, np.nan)).dropna().clip(0.05, 1.0)

fig5, ax5 = plt.subplots(figsize=(9, 5))
bins5 = np.linspace(0.05, 1.0, 75)
ax5.hist(t1v, bins=bins5, alpha=0.45, color="#4575b4",
         label=f"Primera vuelta  (med={t1v.median():.1%})", density=True,
         edgecolor="white", linewidth=0.2)
ax5.hist(t2v, bins=bins5, alpha=0.45, color="#d73027",
         label=f"Segunda vuelta  (med={t2v.median():.1%})", density=True,
         edgecolor="white", linewidth=0.2)
sns.kdeplot(t1v, ax=ax5, color="#4575b4", lw=1.8, label="")
sns.kdeplot(t2v, ax=ax5, color="#d73027", lw=1.8, label="")
for val, col, ls in [(t1v.mean(),"#4575b4","--"),(t2v.mean(),"#d73027","--"),
                     (t1v.median(),"#4575b4",":"),(t2v.median(),"#d73027",":")]:
    ax5.axvline(val, color=col, lw=1.2, ls=ls, alpha=0.85)
ax5.set_xlabel("Turnout (votes cast / registered voters)", fontsize=10)
ax5.set_ylabel("Density", fontsize=10)
ax5.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
# title removed — caption in LaTeX
ax5.legend(handles=[
    mpatches.Patch(facecolor="#4575b4", alpha=0.6, label=f"Primera vuelta (n={len(t1v):,})"),
    mpatches.Patch(facecolor="#d73027", alpha=0.6, label=f"Segunda vuelta (n={len(t2v):,})"),
    Line2D([0],[0], color="gray", lw=1.2, ls="--", label="Mean"),
    Line2D([0],[0], color="gray", lw=1.2, ls=":", label="Median"),
], fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, framealpha=0.9)
ax5.set_xlim(0.05, 1.0)
save_fig(fig5, "fig5_turnout_dist")
plt.close()

# ── FIG 6 — Department bar chart ─────────────────────────────────────────────
print("Fig 6: Department bar chart")

dept6 = df.groupby("DEPARTAMENTO").agg(
    castillo=("CASTILLO","sum"), fujimori=("FUJIMORI","sum")
).reset_index()
dept6["valid"]   = dept6["castillo"] + dept6["fujimori"]
dept6["share_c"] = dept6["castillo"] / dept6["valid"]
dept6 = dept6.sort_values("share_c").reset_index(drop=True)

fig6, ax6 = plt.subplots(figsize=(9, 8))
colors6 = ["#c0392b" if s > 0.5 else "#2980b9" for s in dept6["share_c"]]
ax6.barh(dept6["DEPARTAMENTO"], (dept6["share_c"] - 0.5)*100, left=50,
         color=colors6, edgecolor="white", linewidth=0.4, height=0.7)
ax6.axvline(50, color="black", lw=1.0)
for i, row in dept6.iterrows():
    ax6.text(101, i, f"{row['valid']/1e3:,.0f}K", va="center", fontsize=7.5, color="#444")
ax6.set_xlabel("Castillo vote share of valid votes (%)", fontsize=10)
ax6.set_xlim(10, 105)
ax6.set_xticks([20,30,40,50,60,70,80,90])
ax6.xaxis.set_major_formatter(mtick.FormatStrFormatter("%d%%"))
# title removed — caption in LaTeX
ax6.legend(handles=[mpatches.Patch(color="#c0392b",label="Castillo majority"),
                    mpatches.Patch(color="#2980b9",label="Fujimori majority")],
           fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=2, framealpha=0.9)
ax6.text(101.5, len(dept6)-0.5, "Valid\nvotes", ha="left", va="top", fontsize=8, color="#444")
ax6.spines["right"].set_visible(False); ax6.spines["top"].set_visible(False)
plt.tight_layout()
save_fig(fig6, "fig6_dept_bars")
plt.close()

# ── FIG 7 — Null vote rate choropleth ────────────────────────────────────────
print("Fig 7: Null vote rate by province")

fig7, ax7 = plt.subplots(figsize=(7, 10))
ax7.set_axis_off()
cmap7 = mpl.colormaps["YlOrRd"]
gdf_p.plot(column="null_share", cmap=cmap7, ax=ax7, vmin=0, vmax=0.15,
           missing_kwds={"color":"#e0e0e0"}, linewidth=0.15, edgecolor="white")
gdf_dp.boundary.plot(ax=ax7, linewidth=0.7, color="black", alpha=0.7)
sm7 = mpl.cm.ScalarMappable(cmap=cmap7, norm=mpl.colors.Normalize(0, 0.15))
sm7.set_array([])
cb7 = fig7.colorbar(sm7, ax=ax7, shrink=0.45, pad=0.01, aspect=25, location="right")
cb7.set_label("Null vote rate (% of votes cast)", fontsize=9)
cb7.set_ticks([0, 0.05, 0.10, 0.15])
cb7.set_ticklabels(["0%","5%","10%","≥15%"])
cb7.ax.tick_params(labelsize=8)
# title removed — caption in LaTeX
save_fig(fig7, "fig7_map_nullvotes")
plt.close()

# ── FIG 8 — Share of contested mesas by province ─────────────────────────────
print("Fig 8: Share of contested mesas by province")

fig8, ax8 = plt.subplots(figsize=(7, 10))
ax8.set_axis_off()
cmap8 = mpl.colormaps["Oranges"]
gdf_p.plot(column="share_cont", cmap=cmap8, ax=ax8, vmin=0, vmax=0.10,
           missing_kwds={"color":"#e0e0e0"}, linewidth=0.15, edgecolor="white")
gdf_dp.boundary.plot(ax=ax8, linewidth=0.7, color="black", alpha=0.7)
sm8 = mpl.cm.ScalarMappable(cmap=cmap8, norm=mpl.colors.Normalize(0, 0.10))
sm8.set_array([])
cb8 = fig8.colorbar(sm8, ax=ax8, shrink=0.45, pad=0.01, aspect=25, location="right")
cb8.set_label("Share of contested mesas (COMPUTADA RESUELTA)", fontsize=9)
cb8.set_ticks([0, 0.02, 0.05, 0.10])
cb8.set_ticklabels(["0%","2%","5%","≥10%"])
cb8.ax.tick_params(labelsize=8)
# title removed — caption in LaTeX
ax8.annotate("Darker = higher share of actas that were impugned and reviewed by JNE.",
             xy=(0.01,0.01), xycoords="axes fraction", fontsize=7, color="#555", va="bottom")
save_fig(fig8, "fig8_map_contested")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
sep("COMPLETE — Summary")
print()
print("SAMPLE:")
print(f"  Domestic final-status mesas: {len(df):,}  "
      f"(CONTABILIZADA={( df['contested']==0).sum():,}  "
      f"COMPUTADA RESUELTA={(df['contested']==1).sum():,})")
print(f"  Overseas final-status mesas: {len(df_overseas):,}  (reported separately)")
print(f"  Dropped: ANULADA=213  EN PROCESO=20  SIN INSTALAR=6")
print()
print("NATIONAL MARGIN (domestic final-status):")
print(f"  Castillo: {df['CASTILLO'].sum():>10,.0f}")
print(f"  Fujimori: {df['FUJIMORI'].sum():>10,.0f}")
print(f"  Margin:   {df['CASTILLO'].sum()-df['FUJIMORI'].sum():>10,.0f}  "
      f"(official 44,263 — diff {abs(df['CASTILLO'].sum()-df['FUJIMORI'].sum()-44263):,})")
print()
print("FIGURES (PNG 300dpi + PDF):")
for i, nm in enumerate(["fig1_map_castillo","fig2_klimek","fig3_lastdigit",
                         "fig4_primera_vs_segunda","fig5_turnout_dist",
                         "fig6_dept_bars","fig7_map_nullvotes","fig8_map_contested"],1):
    print(f"  Fig {i}: {nm}")
