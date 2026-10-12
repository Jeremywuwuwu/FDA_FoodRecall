# Part 1 Spec — Predicting FDA Recall Severity from Structured Data

**Scope assumption:** food enforcement reports only (`/food/enforcement.json`). Tightest supply-chain + food-safety story.

**Deliberate constraint:** Project 1 uses **no text fields**. `reason_for_recall` and `product_description` are held back entirely for Project 2. This is what makes the two projects comparable and gives Project 2 a different ML model using Natural Language Processing by taking different text as features to predict the same outcome variable (class I).

---

## 1. Research question

> Using only structured and firm-level historical features available at the time a recall is initiated, how accurately can we predict whether the FDA will classify it as Class I (reasonable probability of serious health consequences)?

**Framing for the write-up:** classification happens after FDA review. A model that flags likely Class I recalls at intake supports faster triage and earlier public warning. This allows Class I (most severe recalls) to be identified earlier, where if not classified fast enough, might cause larger public health harm as contaminated products remain on shelves and in consumers' homes.

---

## 2. Target variable

**Primary:** `classification == "Class I"` vs. Class II/III combined (Binary)

Rationale: Class II dominates heavily, three-class modeling splits the minority signal, and Class I (serious harm) is the operationally meaningful one. Binary keeps the story and comparison clean and easy to interpret.

**Recording the base rate before modeling anything.** If Class I is under ~5%, I will conduct threshold tuning and PR-AUC as the headline metric from the start.

---

## 3. Leakage audit done first

| Field | Knowable at initiation? | Decision |
|---|---|---|
| `recall_initiation_date` | Yes | Keep — source of temporal features |
| `recalling_firm`, `city`, `state`, `country` | Yes | Keep |
| `voluntary_mandated` | Yes | Keep |
| `initial_firm_notification` | Yes | Keep |
| `distribution_pattern` | Yes | Keep — engineer, don't one-hot |
| `product_quantity` | Yes | Keep — needs parsing |
| `center_classification_date` | **No** | **Drop — post-outcome** |
| `termination_date` | **No** | **Drop — post-outcome** |
| `status` | **No** | **Drop — post-outcome** |
| `report_date` | Partially | **Drop** — gap to initiation reflects FDA processing, which correlates with severity |
| `reason_for_recall`, `product_description` | Yes | **Hold for Project 2** |
| `recall_number`, `event_id` | Yes | IDs only — never features |

The `report_date` call is the subtle one. It's tempting because the initiation-to-report gap is predictive *because* urgent recalls get processed faster, so a shorter report date would likely mean a Class I recall which is the outcome leaking backward. Dropped!

---

## 4. Feature engineering 

Raw structured fields here are thin:

**Firm history (compute causally — this is where leakage sneaks back in)**
- Count of that firm's prior recalls *strictly before* the current initiation date
- Share of those prior recalls that were Class I
- Days since firm's previous recall

> Use an expanding window ordered by date. If I compute firm Class I rate over the whole dataset, I have leaked the test set into training. This is the most likely way this project produces an inflated score.

**Distribution scope**
- Parse `distribution_pattern` for number of distinct states mentioned
- Flag nationwide vs. regional vs. single-state
- Flag international distribution

**Quantity**
- Parse numeric magnitude from `product_quantity` (messy free-form — expect to handle "1,200 cases", "approx. 500 lbs", nulls)
- Log-transform; keep a `quantity_parseable` missingness flag

**Temporal**
- Year, month, day-of-week of initiation
- Firm's recall count in trailing 365 days

---

## 5. Splitting strategy

**Temporal split, not random.** Train on earlier years, test on the most recent. This matches how the model would deploy and avoids the fantasy of predicting 2019 from 2023.

My suggested split on first look: train ≤2019, validation 2020–2021, test 2022+.

**Group check:** one recall event can produce many product rows. Verify no `event_id` spans a split boundary. Use `GroupShuffleSplit` on `event_id` if I supplement with any random splitting.

Verify class balance in each split before proceeding.

---

## 6. Modeling sequence

1. **Baseline:** logistic regression on a handful of features. Record PR-AUC. Every later number is measured against this.
2. **Single decision tree:** shallow, plotted. Gives an interpretable picture of the dominant splits and a figure for the README.
3. **XGBoost, default parameters.** How much does boosting buy over the baseline before tuning?
4. **XGBoost tuned.** `RandomizedSearchCV` over a broad space (~60 iterations), then a small `GridSearchCV` on the narrowed region. Tune on validation only.
5. **Potentially AutoGluon**, same splits, test set untouched until final evaluation. Done only if XGBoost scored higher than the linear logistic regression model.

Handle imbalance with `scale_pos_weight` in XGBoost. Only try resampling if that underperforms, and never resample the validation or test sets.

---

## 7. Evaluation

**Headline metric:** PR-AUC. Report ROC-AUC alongside it, and explain in the README why PR-AUC is the honest one under imbalance.

**Also report:** precision, recall, F1 at a tuned threshold; confusion matrix; accuracy only as context, but never a decision maker.

**Threshold:** tune on validation for the operating point I actually want. A triage system tolerates false positives to avoid missing a Class I — say so explicitly and pick the threshold accordingly.

**Interpretability:** SHAP values on the tuned XGBoost. Global importance plus two or three individual explanations.

**Calibration:** reliability curve. If predicted probabilities are meant to drive triage priority, they need to mean something.

**Error analysis:** pull 20–30 misclassified recalls and evaluate them. Look for patterns by product category, firm size, or year.

---

## 8. Deliverables

- Clean notebook: EDA → leakage audit → feature engineering → modeling → evaluation
- `README.md` opening with the stakeholder framing, not the methods list
- 4–6 figures: class balance, temporal trend, SHAP summary, PR curve, confusion matrix, calibration
- Saved train/val/test indices so Project 2 uses **identical** splits (!High Importance!)

---

## 9. Kill criteria

Stop and reassess if:

- Class I base rate is under 1% → reframe as anomaly detection or switch target
- Tuned XGBoost beats logistic regression by less than ~3 points PR-AUC → structured fields are weak; still a legitimate finding *and* a stronger setup for Project 2 
- Any model exceeds ~0.95 PR-AUC → assume leakage until proven otherwise, and re-run the audit in section 3

---

## 10. Rough sequencing

| Phase | Work |
|---|---|
| 1 | Pull data via openFDA API, EDA, leakage audit table |
| 2 | Feature engineering, especially causal firm-history features |
| 3 | Baseline → tree → XGBoost default → tuned |
| 4 | SHAP, calibration, error analysis |
| 5 | README, figures, repo cleanup |

