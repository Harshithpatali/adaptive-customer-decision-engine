# ACDX Advanced Policy Evaluation — Research Report

## Executive summary

This is the latest successfully completed offline causal-policy research run available in the repository history. It is **research evidence**, not realized production revenue and not a replacement for the frozen production inference stack.

- Dataset: **64,000 customers**
- Cross-fitting: **5 folds**
- Bootstrap decision-stability refits: **50** in this source run
- Policy evaluation: **doubly robust, monetary spend/value scale**
- Source workflow run: **36829950961**
- Production artifacts modified: **No**

## 1. S / T / X learner comparison

The learners were evaluated out-of-fold against the randomized no-email control.

| Treatment | Learner | Qini | AUUC | Uplift @ 20% |
|---|---|---:|---:|---:|
| Mens E-Mail | S-Learner | 133.02 | 0.00566 | 0.369% |
| Mens E-Mail | T-Learner | 126.92 | 0.00563 | 0.711% |
| Mens E-Mail | X-Learner | 138.02 | 0.00687 | 0.708% |
| Womens E-Mail | S-Learner | 69.04 | 0.00419 | 0.395% |
| Womens E-Mail | T-Learner | 73.58 | 0.00292 | 0.370% |
| Womens E-Mail | X-Learner | 77.81 | 0.00423 | 0.618% |

## 2. Doubly robust policy evaluation

- Estimated policy value: **1.25786**
- 95% confidence interval: **1.03337 to 1.47835**
- Standard error: **0.11661**
- Effective sample size: **63,999.8**

This is a cross-fitted offline estimate on randomized data. It should not be described as observed production revenue.

## 3. Causal incremental-value frontier

| Contact fraction | Actual contact rate | Contacts | Mean incremental value | Total net incremental value |
|---:|---:|---:|---:|---:|
| 5% | 5.0% | 3,200 | $8.76 | $28,047.82 |
| 10% | 10.0% | 6,400 | $6.79 | $43,431.64 |
| 20% | 20.0% | 12,800 | $5.03 | $64,079.86 |
| 30% | 30.0% | 19,200 | $4.05 | $77,780.92 |
| 40% | 40.0% | 25,600 | $3.41 | $87,247.39 |
| 50% | 50.0% | 32,000 | $2.93 | $93,770.66 |
| 75% | 75.0% | 48,000 | $2.11 | $101,154.38 |
| 100% evaluated | 81.5% | 52,142 | $1.95 | $101,461.37 |

The final row is not literally 100% contacted: only customers with positive modeled net incremental value are selected, producing an 81.5% actual modeled contact rate.

## 4. Decision stability

The source run included bootstrap decision stability research on a 1,000-customer evaluation sample. Example customer-level stability values observed in that run include 0.86 for a Women's Email decision and 0.74 for a competing-action case. The complete stability artifact is not committed in the current repository state, so this report does not invent an aggregate stability statistic.

## Evidence boundary

**Randomized experimental evidence:** observed treatment/control comparisons.

**Offline research estimates:** S/T/X uplift, cross-fitted potential outcomes, doubly robust policy value, bootstrap decision stability, and the causal value frontier.

**Production:** frozen XGBoost response + T-Learner uplift + LightGBM conditional revenue + value-max policy. The production API does not run this research code.

**Important:** modeled incremental value is not realized revenue. Treatment costs are policy assumptions stored in the production configuration.

## Reproducibility note

This report is seeded from the latest successful workflow run available before the workflow repair. The corrected GitHub Actions workflow now defaults to **5-fold cross-fitting and 200 bootstrap refits** and will regenerate the advanced-policy artifacts when the workflow is run.
