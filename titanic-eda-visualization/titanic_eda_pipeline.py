"""
Week 2 Task — Exploratory Data Analysis and Visualization
Dataset : Titanic passenger manifest (same raw source as Week 1)
Source  : Kaggle "Titanic: Machine Learning from Disaster" competition data,
          mirrored at https://github.com/mwaskom/seaborn-data (raw/titanic.csv)

This script:
  1. Rebuilds the cleaned, analysis-ready dataframe using the exact cleaning
     decisions justified in the Week 1 report (kept here as a compact,
     self-contained function so this script runs standalone).
  2. Performs univariate, bivariate, and multivariate exploratory analysis.
  3. Produces annotated Matplotlib/Seaborn visualizations.
  4. Runs light statistical checks (chi-square, t-test) to support the
     interpretations written up in the report.
  5. Exports all figures and a small set of summary tables.

Run: python3 titanic_eda_pipeline.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid", font_scale=0.95)
FIG_DIR = "figures"
DATA_IN = "data/titanic_raw.csv"
pd.set_option("display.width", 120)


def section(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


# ---------------------------------------------------------------------------
# 0. REBUILD THE CLEANED DATASET (decisions justified in the Week 1 report)
# ---------------------------------------------------------------------------
def prepare_data(path):
    df = pd.read_csv(path)

    df["title"] = df["name"].str.extract(r",\s*([^\.]*)\.")
    title_map = {"Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
                 "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
    df["title_group"] = df["title"].map(title_map).fillna("Rare")

    df["has_cabin"] = df["cabin"].notnull().astype(int)
    df["deck"] = df["cabin"].str[0].fillna("Unknown")

    df["family_size"] = df["sibsp"] + df["parch"] + 1
    df["is_alone"] = (df["family_size"] == 1).astype(int)

    df["embarked"] = df["embarked"].fillna("S")
    df["age"] = df.groupby("title_group")["age"].transform(lambda s: s.fillna(s.median()))

    df["fare_is_zero"] = (df["fare"] == 0).astype(int)
    df["fare_log"] = np.log1p(df["fare"])

    # EDA-specific transformations (binning for cleaner categorical plots)
    df["age_group"] = pd.cut(df["age"], bins=[0, 12, 18, 60, 100],
                              labels=["Child (0-12)", "Teen (12-18)", "Adult (18-60)", "Senior (60+)"])
    df["family_group"] = pd.cut(df["family_size"], bins=[0, 1, 4, 11],
                                 labels=["Alone (1)", "Small (2-4)", "Large (5+)"])

    df["survived_label"] = df["survived"].map({0: "Died", 1: "Survived"})
    df = df.drop(columns=["name", "ticket", "cabin", "title"])
    return df


section("0. DATA PREPARATION (recap of Week 1 cleaning)")
df = prepare_data(DATA_IN)
print(f"Analysis-ready shape: {df.shape}")
print(f"Missing cells remaining: {df.isnull().sum().sum()}")
df.to_csv("data/titanic_eda_ready.csv", index=False)

# ---------------------------------------------------------------------------
# 1. INITIAL ANALYSIS — basic statistics & interesting features
# ---------------------------------------------------------------------------
section("1. INITIAL ANALYSIS")
print(df[["age", "fare", "family_size"]].describe().round(2))
print("\nOverall survival rate:", round(df["survived"].mean(), 3))
print("\nSurvival rate by sex:\n", df.groupby("sex")["survived"].mean().round(3))
print("\nSurvival rate by pclass:\n", df.groupby("pclass")["survived"].mean().round(3))
print("\nSurvival rate by title_group:\n", df.groupby("title_group")["survived"].mean().round(3))
print("\nSurvival rate for fare_is_zero passengers:\n", df.groupby("fare_is_zero")["survived"].mean().round(3),
      " (n=", df["fare_is_zero"].sum(), ")")

# ---------------------------------------------------------------------------
# 2. UNIVARIATE ANALYSIS
# ---------------------------------------------------------------------------
section("2. UNIVARIATE ANALYSIS")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
sns.histplot(df["age"], bins=30, kde=True, ax=axes[0], color="#4C72B0")
axes[0].axvline(df["age"].mean(), color="black", ls="--", lw=1, label=f"mean = {df['age'].mean():.1f}")
axes[0].axvline(df["age"].median(), color="red", ls=":", lw=1, label=f"median = {df['age'].median():.1f}")
axes[0].set_title("Age distribution (all passengers)")
axes[0].set_xlabel("Age (years)")
axes[0].legend(fontsize=8)

sns.histplot(df["fare"], bins=40, ax=axes[1], color="#DD8452")
axes[1].set_yscale("log")
axes[1].set_title("Fare distribution (log-scaled y-axis)")
axes[1].set_xlabel("Fare ($)")
axes[1].set_ylabel("Count (log scale)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/01_age_fare_univariate.png", dpi=150)
plt.close()

fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
sns.countplot(data=df, x="sex", hue="sex", legend=False, ax=axes[0, 0], palette="Set2")
axes[0, 0].set_title("Passenger count by sex")
sns.countplot(data=df, x="pclass", hue="pclass", legend=False, ax=axes[0, 1], palette="Set2")
axes[0, 1].set_title("Passenger count by class")
axes[0, 1].set_xlabel("Passenger class")
sns.countplot(data=df, x="embarked", hue="embarked", legend=False, ax=axes[1, 0],
              order=["S", "C", "Q"], palette="Set2")
axes[1, 0].set_title("Passenger count by port of embarkation")
axes[1, 0].set_xlabel("Embarked (S=Southampton, C=Cherbourg, Q=Queenstown)")
sns.countplot(data=df, y="title_group", hue="title_group", legend=False, ax=axes[1, 1],
              order=df["title_group"].value_counts().index, palette="Set2")
axes[1, 1].set_title("Passenger count by title group")
for ax in axes.flat:
    ax.set_ylabel(ax.get_ylabel() or "Count")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/02_categorical_counts.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 3. BIVARIATE ANALYSIS — vs. survival
# ---------------------------------------------------------------------------
section("3. BIVARIATE ANALYSIS")

fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
sns.barplot(data=df, x="sex", y="survived", hue="sex", legend=False, ax=axes[0, 0], palette="Set1")
axes[0, 0].set_title("Survival rate by sex")
axes[0, 0].set_ylabel("Survival rate")
sns.barplot(data=df, x="pclass", y="survived", hue="pclass", legend=False, ax=axes[0, 1], palette="Set1")
axes[0, 1].set_title("Survival rate by class")
axes[0, 1].set_xlabel("Passenger class")
axes[0, 1].set_ylabel("Survival rate")
sns.barplot(data=df, x="embarked", y="survived", hue="embarked", legend=False, ax=axes[1, 0],
            order=["S", "C", "Q"], palette="Set1")
axes[1, 0].set_title("Survival rate by port of embarkation")
axes[1, 0].set_xlabel("Embarked")
axes[1, 0].set_ylabel("Survival rate")
sns.barplot(data=df, y="title_group", x="survived", hue="title_group", legend=False, ax=axes[1, 1],
            order=df.groupby("title_group")["survived"].mean().sort_values(ascending=False).index, palette="Set1")
axes[1, 1].set_title("Survival rate by title group")
axes[1, 1].set_xlabel("Survival rate")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/03_survival_by_category.png", dpi=150)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
sns.violinplot(data=df, x="survived_label", y="age", hue="survived_label", legend=False, ax=axes[0], palette="Set3")
axes[0].set_title("Age distribution by outcome")
axes[0].set_xlabel("")
axes[0].set_ylabel("Age (years)")
sns.boxplot(data=df, x="survived_label", y="fare_log", hue="survived_label", legend=False, ax=axes[1], palette="Set3")
axes[1].set_title("log1p(Fare) distribution by outcome")
axes[1].set_xlabel("")
axes[1].set_ylabel("log1p(Fare)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/04_age_fare_by_outcome.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 4. MULTIVARIATE ANALYSIS
# ---------------------------------------------------------------------------
section("4. MULTIVARIATE ANALYSIS")

pclass_sex = df.pivot_table(index="pclass", columns="sex", values="survived", aggfunc="mean").round(3)
print("Survival rate — pclass x sex:\n", pclass_sex)
plt.figure(figsize=(5.5, 4.2))
sns.heatmap(pclass_sex, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1, cbar_kws={"label": "Survival rate"})
plt.title("Survival rate by class and sex")
plt.ylabel("Passenger class")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/05_heatmap_class_sex.png", dpi=150)
plt.close()

plt.figure(figsize=(6.5, 4.8))
sns.scatterplot(data=df, x="age", y="fare_log", hue="survived_label", style="pclass",
                 palette={"Died": "#C44E52", "Survived": "#55A868"}, alpha=0.7, s=45)
plt.title("Age vs. log1p(Fare), by survival outcome and class")
plt.xlabel("Age (years)")
plt.ylabel("log1p(Fare)")
plt.legend(title="", fontsize=8, loc="upper right")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/06_age_fare_scatter.png", dpi=150)
plt.close()

fam = df.groupby("family_group", observed=True)["survived"].agg(["mean", "count"])
print("\nSurvival rate by family-size group:\n", fam.round(3))
fig, ax1 = plt.subplots(figsize=(6.5, 4.2))
ax1.bar(fam.index.astype(str), fam["mean"], color="#8172B2")
ax1.set_ylabel("Survival rate")
ax1.set_title("Survival rate by family-size group (bars) and group size (labels)")
for i, (m, c) in enumerate(zip(fam["mean"], fam["count"])):
    ax1.text(i, m + 0.02, f"n={c}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/07_family_size_survival.png", dpi=150)
plt.close()

deck_surv = df[df["deck"] != "Unknown"].groupby("deck", observed=True)["survived"].agg(["mean", "count"]).sort_values("mean")
print("\nSurvival rate by known deck:\n", deck_surv.round(3))
plt.figure(figsize=(6.5, 4.2))
sns.barplot(x=deck_surv.index, y=deck_surv["mean"], hue=deck_surv.index, legend=False, palette="crest")
plt.title("Survival rate by deck (passengers with a recorded cabin only)")
plt.xlabel("Deck")
plt.ylabel("Survival rate")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/08_deck_survival.png", dpi=150)
plt.close()

corr_cols = ["survived", "pclass", "sex_num", "age", "fare", "fare_log", "family_size", "is_alone", "has_cabin"]
df["sex_num"] = df["sex"].map({"male": 0, "female": 1})
plt.figure(figsize=(6.5, 5.5))
sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation matrix of numeric / engineered features")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/09_correlation_matrix.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 5. STATISTICAL CHECKS
# ---------------------------------------------------------------------------
section("5. STATISTICAL CHECKS")

ct_sex = pd.crosstab(df["sex"], df["survived"])
chi2, p_sex, _, _ = stats.chi2_contingency(ct_sex)
print(f"Chi-square (sex vs survived): chi2={chi2:.2f}, p={p_sex:.2e}")

ct_class = pd.crosstab(df["pclass"], df["survived"])
chi2c, p_class, _, _ = stats.chi2_contingency(ct_class)
print(f"Chi-square (pclass vs survived): chi2={chi2c:.2f}, p={p_class:.2e}")

t_age, p_age = stats.ttest_ind(df.loc[df.survived == 1, "age"], df.loc[df.survived == 0, "age"], equal_var=False)
print(f"Welch t-test (age, survived vs died): t={t_age:.2f}, p={p_age:.3f}")

t_fare, p_fare = stats.ttest_ind(df.loc[df.survived == 1, "fare_log"], df.loc[df.survived == 0, "fare_log"], equal_var=False)
print(f"Welch t-test (log fare, survived vs died): t={t_fare:.2f}, p={p_fare:.2e}")

print("\nAll figures written to ./figures/")
print("DONE.")
