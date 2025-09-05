import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, shapiro
import math
import os, math, warnings, re
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.weightstats import ttest_ind as sm_ttest_ind


'''
# 1. 데이터 불러오기
df = pd.read_csv('data/questionnaire/final_scores_with_immersion.csv')

# 2. 숫자형 열만 선택
numeric_cols = df.select_dtypes(include='number').columns

# 3. 기초 통계량 저장
summary = df.groupby('Model')[numeric_cols].agg(['mean', 'std', 'min', 'max', 'median', 'count'])
summary.to_csv("processed_output/groupwise_descriptives.csv", encoding="utf-8-sig")
print("📊 기초 통계량 저장 완료")

# 4. 정규성 검사 저장
shapiro_results = []
for col in numeric_cols:
    for model in ['A', 'B']:
        stat, p = shapiro(df[df['Model'] == model][col])
        shapiro_results.append({
            'Variable': col,
            'Model': model,
            'Shapiro-W': round(stat, 4),
            'p-value': round(p, 4)
        })
shapiro_df = pd.DataFrame(shapiro_results)
shapiro_df.to_csv("processed_output/shapiro_normality.csv", index=False, encoding="utf-8-sig")
print("📈 정규성 검사 결과 저장 완료")

# 5. t-test 결과 저장
ttest_results = []
for col in numeric_cols:
    A = df[df['Model'] == 'A'][col]
    B = df[df['Model'] == 'B'][col]
    t, p = ttest_ind(A, B, equal_var=False)
    ttest_results.append({
        'Variable': col,
        't-statistic': round(t, 4),
        'p-value': round(p, 4)
    })
ttest_df = pd.DataFrame(ttest_results)
ttest_df.to_csv("processed_output/ttest_results.csv", index=False, encoding="utf-8-sig")
print("📊 t-test 결과 저장 완료")

# 6. Boxplot 그리드 저장
sns.set(style="whitegrid")

n = len(numeric_cols)
cols = 2
rows = math.ceil(n / cols)

plt.figure(figsize=(cols * 6, rows * 4))

for idx, col in enumerate(numeric_cols, 1):
    plt.subplot(rows, cols, idx)
    sns.boxplot(data=df, x='Model', y=col, palette='Set2')
    sns.stripplot(data=df, x='Model', y=col, color='black', alpha=0.5, jitter=True)
    plt.title(f"{col}")
    plt.xlabel('')
    plt.ylabel('')

plt.tight_layout()
plt.suptitle("📦 변수별 Model 그룹 분포 (Box + Stripplot)", fontsize=16, y=1.02)
plt.savefig("figures/groupwise_boxplots.png", bbox_inches='tight', dpi=300)
plt.show()
print("📦 박스플롯 시각화 저장 완료")

# -*- coding: utf-8 -*-

warnings.filterwarnings("ignore")

# -----------------------------
# 0) I/O setup
# -----------------------------
os.makedirs("processed_output", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# -----------------------------
# 1) Load data
# -----------------------------
df = pd.read_csv("data/questionnaire/final_scores_with_immersion.csv")

# -----------------------------
# 1a) Identify key columns robustly
# -----------------------------
def first_present(cands):
    for c in cands:
        if c in df.columns:
            return c
    return None

PARTIC_COL = first_present(["Participant","participant","ID","Id","id","Subject","subject"])
MODEL_COL    = first_present(["Model","model"])
SCEN_COL     = first_present(["Scenario","scenario"])
ORDER_COL    = first_present(["Order","order","ScenarioOrder","scenario_order"])

required = [PARTIC_COL, MODEL_COL, SCEN_COL]
missing_req = [n for n in ["Participant(ID)","Model","Scenario"] if
               (n=="Participant(ID)" and PARTIC_COL is None) or
               (n=="Model" and MODEL_COL is None) or
               (n=="Scenario" and SCEN_COL is None)]
if missing_req:
    raise ValueError(f"Missing required columns: {missing_req}. "
                     f"Found columns: {list(df.columns)}")

print(f"✅ Using columns -> Participant: '{PARTIC_COL}', Model: '{MODEL_COL}', Scenario: '{SCEN_COL}', Order: '{ORDER_COL}'")

# Coerce categorical
df[MODEL_COL] = df[MODEL_COL].astype("category")
df[SCEN_COL]  = df[SCEN_COL].astype("category")
if ORDER_COL and ORDER_COL in df.columns:
    df[ORDER_COL] = df[ORDER_COL].astype("category")

# -----------------------------
# 2) Identify numeric outcome columns
#    (exclude IDs and obvious non-outcomes)
# -----------------------------
exclude_cols = {PARTIC_COL, MODEL_COL, SCEN_COL}
if ORDER_COL: exclude_cols.add(ORDER_COL)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# Keep only numeric outcomes not obviously IDs (e.g., if ID was imported as number)
numeric_cols = [c for c in numeric_cols if c not in exclude_cols]

if not numeric_cols:
    raise ValueError("No numeric outcome columns detected after exclusions.")

print(f"📌 Outcome variables: {numeric_cols}")

# -----------------------------
# 3) Groupwise descriptives by Model (as you had)
# -----------------------------
summary = df.groupby(MODEL_COL)[numeric_cols].agg(['mean', 'std', 'min', 'max', 'median', 'count'])
summary.to_csv("processed_output/groupwise_descriptives_by_model.csv", encoding="utf-8-sig")
print("📊 기초 통계량(모델별) 저장 완료 → processed_output/groupwise_descriptives_by_model.csv")

# Also by Model × Scenario (useful for the mixed design)
m_s_desc = (df.groupby([MODEL_COL, SCEN_COL])[numeric_cols]
              .agg(['mean','std','median','count'])
              .reset_index())
m_s_desc.to_csv("processed_output/descriptives_by_model_scenario.csv", index=False, encoding="utf-8-sig")
print("📊 기초 통계량(모델×시나리오) 저장 완료 → processed_output/descriptives_by_model_scenario.csv")

# -----------------------------
# 4) (Optional) Normality checks by Model per variable
#     (not strictly needed for mixed models, but kept for continuity)
# -----------------------------
shapiro_results = []
for col in numeric_cols:
    for model in df[MODEL_COL].cat.categories:
        vals = df.loc[df[MODEL_COL] == model, col].dropna()
        if len(vals) >= 3:  # Shapiro needs n>=3
            stat, p = shapiro(vals)
            shapiro_results.append({
                'Variable': col,
                'Model': model,
                'Shapiro-W': round(stat, 4),
                'p-value': round(p, 4),
                'n': len(vals)
            })
shapiro_df = pd.DataFrame(shapiro_results)
shapiro_df.to_csv("processed_output/shapiro_normality_by_model.csv", index=False, encoding="utf-8-sig")
print("📈 정규성 검사 결과 저장 완료 → processed_output/shapiro_normality_by_model.csv")

# -----------------------------
# 5) Mixed-effects models (replaces t-tests)
#     Y ~ Model * Scenario + Order + (1 | Participant)
# -----------------------------
def fit_mixed_for_outcome(y_col, include_order=True):
    fixed = f"{MODEL_COL} * {SCEN_COL}"
    if include_order and ORDER_COL:
        fixed += f" + {ORDER_COL}"
    formula = f"{y_col} ~ {fixed}"

    # MixedLM requires group column (Participant)
    model = smf.mixedlm(formula, data=df, groups=df[PARTIC_COL], re_formula="~1")
    result = model.fit(method="lbfgs", reml=True)
    return result, formula

mixed_rows = []
for col in numeric_cols:
    try:
        res, formula = fit_mixed_for_outcome(col, include_order=True)
        ci = res.conf_int(alpha=0.05)
        # Build a tidy summary row per fixed effect
        for name, coef in res.params.items():
            if name.startswith("Group"):  # skip random effects variance rows
                continue
            se = res.bse[name] if name in res.bse else np.nan
            z  = coef / se if se and se!=0 else np.nan
            p  = res.pvalues[name] if name in res.pvalues else np.nan
            lcl = ci.loc[name, 0] if name in ci.index else np.nan
            ucl = ci.loc[name, 1] if name in ci.index else np.nan
            mixed_rows.append({
                "Outcome": col,
                "Formula": formula,
                "Term": name,
                "Beta": coef,
                "SE": se,
                "z": z,
                "p": p,
                "CI95_L": lcl,
                "CI95_U": ucl,
                "N": df[col].notna().sum()
            })
    except Exception as e:
        mixed_rows.append({
            "Outcome": col,
            "Formula": "ERROR",
            "Term": "ERROR",
            "Beta": np.nan,
            "SE": np.nan,
            "z": np.nan,
            "p": np.nan,
            "CI95_L": np.nan,
            "CI95_U": np.nan,
            "N": df[col].notna().sum(),
            "Error": str(e)
        })

mixed_df = pd.DataFrame(mixed_rows)
mixed_df.to_csv("processed_output/mixedlm_fixed_effects.csv", index=False, encoding="utf-8-sig")
print("🧪 혼합모형 고정효과 결과 저장 완료 → processed_output/mixedlm_fixed_effects.csv")

# -----------------------------
# 6) (Optional) Legacy t-tests (per-scenario snapshots only)
#     Not recommended as the main analysis, but kept for reference
# -----------------------------
ttest_results = []
for col in numeric_cols:
    for scen in df[SCEN_COL].cat.categories:
        sub = df[df[SCEN_COL] == scen]
        A = sub[sub[MODEL_COL] == sub[MODEL_COL].cat.categories[0]][col].dropna()
        B = sub[sub[MODEL_COL] == sub[MODEL_COL].cat.categories[1]][col].dropna()
        if len(A) > 1 and len(B) > 1:
            t, p = ttest_ind(A, B, equal_var=False)
            ttest_results.append({
                'Variable': col,
                'Scenario': scen,
                't-statistic': round(t, 4),
                'p-value': round(p, 4),
                'n_A': len(A),
                'n_B': len(B)
            })
ttest_df = pd.DataFrame(ttest_results)
ttest_df.to_csv("processed_output/ttest_per_scenario_reference_only.csv", index=False, encoding="utf-8-sig")
print("📊 (참고용) 시나리오별 t-test 결과 저장 완료 → processed_output/ttest_per_scenario_reference_only.csv")

# -----------------------------
# 7) Plots: Box+Strip by Model, split by Scenario
#    (helps visualize potential Model×Scenario interaction)
# -----------------------------
sns.set(style="whitegrid")
n = len(numeric_cols)
cols = 2
rows = math.ceil(n / cols)

for scen in df[SCEN_COL].cat.categories:
    sub = df[df[SCEN_COL] == scen]

    plt.figure(figsize=(cols * 6, rows * 4))
    for idx, col in enumerate(numeric_cols, 1):
        plt.subplot(rows, cols, idx)
        sns.boxplot(data=sub, x=MODEL_COL, y=col)
        sns.stripplot(data=sub, x=MODEL_COL, y=col, color='black', alpha=0.5, jitter=True)
        plt.title(f"{col} — {scen}")
        plt.xlabel('')
        plt.ylabel('')
    plt.tight_layout()
    plt.suptitle(f"분포(모델별) — 시나리오: {scen}", fontsize=16, y=1.02)
    outpath = f"figures/box_strip_by_model_{scen}.png".replace(" ", "_")
    plt.savefig(outpath, bbox_inches='tight', dpi=300)
    plt.show()
    print(f"📦 박스플롯 저장 완료 → {outpath}")

print("✅ 완료: 혼합모형 분석 + 요약 테이블 + 시각화")
'''

warnings.filterwarnings("ignore")

os.makedirs("processed_output", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# -----------------------------
# 1) Load data (wide format)
# -----------------------------
df = pd.read_csv('data/questionnaire/final_scores_with_immersion.csv')

# Core columns present in your file
ID_COL = 'ID'
MODEL_COL = 'Model'
if ID_COL not in df.columns or MODEL_COL not in df.columns:
    raise ValueError(f"Expected columns '{ID_COL}' and '{MODEL_COL}' not found. Found: {list(df.columns)}")

# Coerce categorical
df[MODEL_COL] = df[MODEL_COL].astype('category')

# -----------------------------
# 2) Identify scenario-split columns (Hamlet_* vs Venice_* pairs)
#    Pattern: <Scenario>_<Measure>
# -----------------------------
scenario_prefixes = ['Hamlet', 'Venice']
pattern = re.compile(r'^(Hamlet|Venice)_(.+)$')

pairs = {}   # measure -> {'Hamlet': colname, 'Venice': colname}
for col in df.columns:
    m = pattern.match(col)
    if not m:
        continue
    scen, measure = m.group(1), m.group(2)
    pairs.setdefault(measure, {})
    pairs[measure][scen] = col

# Keep only measures that have BOTH scenarios
scenario_measures = [m for m, d in pairs.items() if all(s in d for s in scenario_prefixes)]
scenario_measures.sort()

if not scenario_measures:
    print("⚠️ No Hamlet_/Venice_ pairs detected. "
          "If your columns are named differently, adjust the regex pattern.")

# -----------------------------
# 3) Build LONG dataframe for scenario-split outcomes
# -----------------------------
long_rows = []
for _, row in df.iterrows():
    pid = row[ID_COL]
    mdl = row[MODEL_COL]
    for meas in scenario_measures:
        for scen in scenario_prefixes:
            colname = pairs[meas][scen]
            val = row[colname]
            long_rows.append({
                ID_COL: pid,
                MODEL_COL: mdl,
                'Scenario': scen,
                'Measure': meas,
                'Value': val
            })

long_df = pd.DataFrame(long_rows)
if not long_df.empty:
    long_df['Scenario'] = long_df['Scenario'].astype('category')
    long_df['Measure'] = long_df['Measure'].astype('category')
    long_df.to_csv("processed_output/long_scenario_data.csv", index=False, encoding="utf-8-sig")
    print("✅ LONG format saved → processed_output/long_scenario_data.csv")

# -----------------------------
# 4) Identify global (non-scenario) numeric outcomes
#    e.g., immersion_* columns that are not Hamlet_/Venice_ pairs
# -----------------------------
all_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
# numeric columns used in scenario pairs
paired_numeric_cols = set()
for meas in scenario_measures:
    paired_numeric_cols.update([pairs[meas]['Hamlet'], pairs[meas]['Venice']])

global_numeric = [c for c in all_numeric if c not in paired_numeric_cols]

# Filter out IDs if numeric
global_numeric = [c for c in global_numeric if c not in [ID_COL]]

print(f"📌 Scenario measures (paired): {scenario_measures}")
print(f"📌 Global numeric outcomes: {global_numeric}")

# -----------------------------
# 5) Descriptives
# -----------------------------
# 5a) By Model (global outcomes)
if global_numeric:
    desc_global = df.groupby(MODEL_COL)[global_numeric].agg(['mean','std','median','min','max','count'])
    desc_global.to_csv("processed_output/descriptives_global_by_model.csv", encoding="utf-8-sig")
    print("📊 Global descriptives by Model saved → processed_output/descriptives_global_by_model.csv")

# 5b) By Model × Scenario (scenario outcomes)
if not long_df.empty:
    desc_long = (long_df.groupby([MODEL_COL, 'Scenario', 'Measure'])['Value']
                      .agg(['mean','std','median','min','max','count'])
                      .reset_index())
    desc_long.to_csv("processed_output/descriptives_by_model_scenario_measure.csv",
                     index=False, encoding="utf-8-sig")
    print("📊 Scenario descriptives saved → processed_output/descriptives_by_model_scenario_measure.csv")

# -----------------------------
# 6) Modeling
# -----------------------------
# 6a) Mixed models for scenario-split outcomes:
#     Value ~ Model * Scenario + (1 | ID)
mixed_rows = []
if not long_df.empty:
    for meas in scenario_measures:
        sub = long_df[long_df['Measure'] == meas].dropna(subset=['Value'])
        if sub[MODEL_COL].nunique() < 2 or sub['Scenario'].nunique() < 2:
            continue
        # Fit mixed model
        # Note: using REML; switch to ML if you need model comparisons
        md = smf.mixedlm("Value ~ {} * Scenario".format(MODEL_COL),
                         data=sub,
                         groups=sub[ID_COL],
                         re_formula="~1")
        try:
            mres = md.fit(method="lbfgs", reml=True)
            ci = mres.conf_int(alpha=0.05)
            for name, beta in mres.params.items():
                if name.startswith("Group"):  # skip variance rows
                    continue
                se = mres.bse.get(name, np.nan)
                z  = beta/se if (se and se!=0) else np.nan
                p  = mres.pvalues.get(name, np.nan)
                lcl = ci.loc[name, 0] if name in ci.index else np.nan
                ucl = ci.loc[name, 1] if name in ci.index else np.nan
                mixed_rows.append({
                    "Measure": meas,
                    "Term": name,
                    "Beta": beta,
                    "SE": se,
                    "z": z,
                    "p": p,
                    "CI95_L": lcl,
                    "CI95_U": ucl,
                    "N_rows": len(sub)
                })
        except Exception as e:
            mixed_rows.append({
                "Measure": meas,
                "Term": "ERROR",
                "Beta": np.nan, "SE": np.nan, "z": np.nan, "p": np.nan,
                "CI95_L": np.nan, "CI95_U": np.nan,
                "N_rows": len(sub),
                "Error": str(e)
            })

mixed_df = pd.DataFrame(mixed_rows)
if not mixed_df.empty:
    mixed_df.to_csv("processed_output/mixedlm_scenario_measures.csv", index=False, encoding="utf-8-sig")
    print("🧪 Mixed-model results saved → processed_output/mixedlm_scenario_measures.csv")
else:
    print("ℹ️ No mixed-model results produced (no scenario-split measures or a fitting error).")

# 6b) Between-subjects OLS for global outcomes: Y ~ Model
ols_rows = []
for col in global_numeric:
    sub = df[[MODEL_COL, col]].dropna()
    if sub[MODEL_COL].nunique() < 2:
        continue
    ols = smf.ols(f"{col} ~ C({MODEL_COL})", data=sub).fit()
    ci = ols.conf_int(alpha=0.05)
    for name, beta in ols.params.items():
        se = ols.bse.get(name, np.nan)
        t  = ols.tvalues.get(name, np.nan)
        p  = ols.pvalues.get(name, np.nan)
        lcl = ci.loc[name, 0] if name in ci.index else np.nan
        ucl = ci.loc[name, 1] if name in ci.index else np.nan
        ols_rows.append({
            "Outcome": col,
            "Term": name,
            "Beta": beta,
            "SE": se,
            "t": t,
            "p": p,
            "CI95_L": lcl,
            "CI95_U": ucl,
            "N": len(sub)
        })

ols_df = pd.DataFrame(ols_rows)
if not ols_df.empty:
    ols_df.to_csv("processed_output/ols_global_outcomes.csv", index=False, encoding="utf-8-sig")
    print("🧪 OLS (global outcomes) saved → processed_output/ols_global_outcomes.csv")

# -----------------------------
# 7) Visualization
# -----------------------------
sns.set(style="whitegrid")

# 7a) Scenario-split measures: box/strip by Model, small multiples per Measure × Scenario
if not long_df.empty:
    measures = list(long_df['Measure'].cat.categories)
    n = len(measures)
    cols = 2
    rows = math.ceil(n / cols)
    plt.figure(figsize=(cols * 7, rows * 4.5))
    for idx, meas in enumerate(measures, 1):
        ax = plt.subplot(rows, cols, idx)
        sns.boxplot(data=long_df[long_df['Measure']==meas],
                    x=MODEL_COL, y='Value', hue='Scenario')
        sns.stripplot(data=long_df[long_df['Measure']==meas],
                      x=MODEL_COL, y='Value', hue='Scenario',
                      dodge=True, color='black', alpha=0.4)
        ax.set_title(meas)
        ax.set_xlabel('')
        ax.set_ylabel('')
        # Avoid duplicate legends
        if idx == 1:
            ax.legend(title='Scenario', loc='best')
        else:
            ax.get_legend().remove()
    plt.tight_layout()
    plt.suptitle("Scenario-split outcomes by Model", fontsize=16, y=1.02)
    plt.savefig("figures/scenario_measures_boxstrip.png", bbox_inches='tight', dpi=300)
    plt.show()
    print("📦 Saved → figures/scenario_measures_boxstrip.png")

# 7b) Global outcomes: box/strip by Model
if global_numeric:
    n = len(global_numeric)
    cols = 2
    rows = math.ceil(n / cols)
    plt.figure(figsize=(cols * 7, rows * 4.5))
    for idx, col in enumerate(global_numeric, 1):
        ax = plt.subplot(rows, cols, idx)
        sns.boxplot(data=df, x=MODEL_COL, y=col)
        sns.stripplot(data=df, x=MODEL_COL, y=col, color='black', alpha=0.4, jitter=True)
        ax.set_title(col)
        ax.set_xlabel('')
        ax.set_ylabel('')
    plt.tight_layout()
    plt.suptitle("Global outcomes by Model", fontsize=16, y=1.02)
    plt.savefig("figures/global_measures_boxstrip.png", bbox_inches='tight', dpi=300)
    plt.show()
    print("📦 Saved → figures/global_measures_boxstrip.png")

print("✅ Done.")