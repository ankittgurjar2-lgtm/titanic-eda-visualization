# Titanic Dataset — Exploratory Data Analysis & Visualization

Week 2 task for a data science internship: EDA and visualization on a public dataset,
building on the Week 1 cleaning task done on the same data.

## Dataset

Same source as Week 1 — the raw Titanic passenger manifest (891 passengers) from
Kaggle's [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic)
competition, mirrored at
[`mwaskom/seaborn-data`](https://github.com/mwaskom/seaborn-data) (`raw/titanic.csv`).
A local copy is included in `data/titanic_raw.csv`.

## What the pipeline does

`titanic_eda_pipeline.py` is self-contained — it re-applies the cleaning decisions
justified in the Week 1 report (compactly, via `prepare_data()`), then runs:

1. **Initial analysis** — summary statistics, first group-level survival rates
2. **Univariate analysis** — age/fare distributions, categorical counts
3. **Bivariate analysis** — survival rate vs. sex, class, embarked, title, age, fare
4. **Multivariate analysis** — class × sex interaction, age × fare × outcome × class,
   family-size grouping, deck-level survival, a full correlation matrix
5. **Statistical checks** — chi-square tests (sex, class) and Welch's t-tests (age, fare)
   to confirm which visual patterns are statistically significant

Full narrative, code, and all 9 annotated figures are written up in
[`Week2_EDA_Report.docx`](./Week2_EDA_Report.docx).

## Key findings

| Finding | Evidence |
|---|---|
| Sex is the strongest single predictor of survival | 74.2% (female) vs 18.9% (male), χ²=260.72, p≈1.2×10⁻⁵⁸ |
| Class and sex compound rather than acting independently | 1st-class women 96.8% vs 3rd-class women 50.0% |
| Fare/wealth matters as a continuous gradient, not just 3 class tiers | Welch t=10.12 on log-fare, p≈1.8×10⁻²² |
| Family size has a non-linear (inverted-U) relationship with survival | Alone 30.4%, Small(2-4) 57.9%, Large(5+) 16.1% |
| The Week 1 "zero-fare" anomaly reappears as a real survival-rate finding | 6.7% survival vs 38.9% overall (n=15) |

## Running it

```bash
pip install -r requirements.txt
python titanic_eda_pipeline.py
```

Outputs an analysis-ready CSV to `data/titanic_eda_ready.csv` and nine charts to `figures/`.

## Structure

```
titanic_eda_pipeline.py       # full EDA pipeline, run top to bottom
data/titanic_raw.csv          # input (raw, as downloaded)
data/titanic_eda_ready.csv    # output (cleaned, analysis-ready, not one-hot encoded)
figures/                      # nine generated charts
Week2_EDA_Report.docx         # full written report
requirements.txt
```
