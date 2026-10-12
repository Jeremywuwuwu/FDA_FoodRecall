# Predicting FDA Class I Food Recalls from Structured Data (Part 1)

**Business Question:** How accurately can our models use structured data, available on the day
a firm initiates a food recall, to predict whether FDA will later classify that recall as
Class I, its highest-severity designation?

## Why it matters

FDA assigns every recall a class after review, a median of 38 days after the recall starts.
Class I means a reasonable probability of serious health consequences or death [2]. A model
that flags likely Class I recalls at intake could help prioritise review and speed up public
warnings.

Part 1 uses **structured fields only** and compares **logistic regression** with **tuned
XGBoost** [3]. The recall text is held back for Part 2.

## Data

- **Source:** openFDA food enforcement reports [1], snapshot last updated 2026-08-27 (public
  domain, CC0).
- **Size:** 29,316 product rows in 7,836 recall events after cleaning. Class I is 43.9% of
  rows and 41.1% of events, so the target is not rare.
- **Unit:** one recall can cover hundreds of products. FDA grades each product, but a recall
  is one triage decision, so results are reported per recall event.
- **Time-based split** by initiation year: burn-in up to 2013 (builds firm history only),
  train 2014 to 2018, validation 2019 to 2021, and test 2022 to 2026 (scored once).

## Approach

The notebook has four phases, each explained in place next to its code:

1. **Data preprocessing and leakage audit.** Every column gets a keep or drop decision, so
   nothing known only after classification can reach the model [6].
2. **Feature engineering.** 17 features, including firm-history features built only from a
   firm's earlier recalls and checked against a second, independent implementation. A
   univariate screen tests which features carry signal on their own.
3. **Model training and selection.** Logistic regression, a shallow decision tree, default
   XGBoost and tuned XGBoost (222 candidates), compared on validation.
4. **Model evaluation.** Decision thresholds, a calibration check (Brier score), SHAP
   interpretability [4], error analysis, and one held-out test evaluation with a paired
   bootstrap.

The main metric is PR-AUC (average precision) [9], read against each period's no-skill line.

## Results

On the test set (2022 to 2026), event level:

| Model | PR-AUC | ROC-AUC |
|---|---|---|
| No-skill (base rate) | 0.372 | 0.500 |
| **Logistic regression** | **0.479** | **0.600** |
| Tuned XGBoost | 0.461 | 0.583 |

- **Tuned XGBoost is not better than logistic regression on unseen years.** The difference
  is -0.018, with a bootstrap 95% CI of [-0.040, +0.004].
- **The validation advantage did not hold.** XGBoost led by 12.9 points on validation, but
  that came from tuning 222 candidates on a single validation window [7]. The plan's kill
  criterion (XGBoost must win by at least 3 points) fails on test, at -5.1.
- **The structured fields carry modest signal.** Logistic regression scores 1.29 times the
  no-skill line. XGBoost's short list, fixed on validation (p of 0.50 or more), flags 14% of
  test recalls at 52.7% precision, 1.41 times the base rate, but catches only 20% of Class I.
- **The model is nearly blind on first-time recallers.** Recall on Class I is 0.130 without
  firm history against 0.545 with prior recalls on the validation dataset.
- **The probabilities barely beat a constant** (Brier score), so the model is used to rank
  recalls, not to quote probabilities.

![SHAP summary](fig_shap_beeswarm.png)

*What drives the tuned XGBoost (validation): a firm's history of Class I recalls pushes toward
Class I, and notification by letter pushes away.*

### Additional findings and known issues

These came from checks run while writing this README, outside the notebook. None of them
changes the model comparison.

- **Findings.** The grid-search winner sat on the edge of its grid for 6 of 8 parameters. The
  +12.9 validation gap came mostly from a few very large recalls (at event level it was only
  +2.7). Several features with zero SHAP are exact copies of other features, so their
  agreement with the Phase 2 screen is weak evidence.
- **Known issues (Part 2 will fix these if needed).** `prior_class_i_share` sometimes uses
  classifications FDA had not made yet (about 2% of test recalls). "New Mexico" is parsed as
  international (159 rows) instead of state.

## How the original plan changed

`project1_spec.md` is the original plan. Where it turned out wrong:

- **Data assumptions.** It expected a rare Class I target and a 2020 to 2021 validation
  window. The target is 44% of rows, the window was too small (widened to 2019 to 2021),
  and a burn-in period was added for firm history.
- **Not done.** The AutoGluon benchmark. Raw year was dropped as a feature.

## Part 2

Part 2 applies NLP to the recall text (`reason_for_recall`, `product_description`) on the same
`splits.csv` and compares it with this baseline. Lessons it carries over: tune on more than
one validation window, using rolling-origin cross-validation [8]; give every model the same
tuning budget; and learn the text vocabulary on train only.

## Reproducing

Python 3.13 with scikit-learn [5] and the versions below. The zip is the exact snapshot used;
a fresh openFDA download is newer and gives different numbers. The notebook runs in about 100
seconds.

```bash
pip install numpy==2.1.3 pandas==2.2.3 scikit-learn==1.6.1 scipy==1.15.3 \
            statsmodels==0.14.4 xgboost==3.4.1 shap==0.52.0 matplotlib==3.10.0 jupyter
unzip food-enforcement-0001-of-0001.json.zip
jupyter notebook StructuredFields_Part1.ipynb
```

| File | What it is |
|---|---|
| `StructuredFields_Part1.ipynb` | The full analysis, with results explained next to the code |
| `food-enforcement-0001-of-0001.json.zip` | Data snapshot (unzip before running) |
| `splits.csv` | The train, validation and test assignment, shared with Part 2 |
| `fig_shap_beeswarm.png`, `fig_shap_dependence.png` | SHAP figures written by the notebook |
| `project1_spec.md` | The original plan |

## References

1. U.S. Food and Drug Administration. openFDA Food Enforcement Reports API. https://open.fda.gov/apis/food/enforcement/
2. 21 CFR § 7.3(m), Recall classification. https://www.ecfr.gov/current/title-21/chapter-I/subchapter-A/part-7/subpart-A/section-7.3
3. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794. https://doi.org/10.1145/2939672.2939785
4. Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30. https://arxiv.org/abs/1705.07874
5. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830. https://jmlr.org/papers/v12/pedregosa11a.html
6. Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining: Formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data*, 6(4), 1-21. https://doi.org/10.1145/2382577.2382579
7. Cawley, G. C., & Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *Journal of Machine Learning Research*, 11, 2079-2107. https://jmlr.org/papers/v11/cawley10a.html
8. Tashman, L. J. (2000). Out-of-sample tests of forecasting accuracy: an analysis and review. *International Journal of Forecasting*, 16(4), 437-450. https://doi.org/10.1016/S0169-2070(00)00065-0
9. Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE*, 10(3), e0118432. https://doi.org/10.1371/journal.pone.0118432
