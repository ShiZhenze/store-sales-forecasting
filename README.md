# Store Sales Demand Forecasting

16-day-ahead daily demand forecasting for a 54-store grocery retail chain,
covering 1,782 store–product series.

**Kaggle — Store Sales: Time Series Forecasting**
Public leaderboard: **65 / 653 (top 10%)** · RMSLE **0.39683**
[Competition page](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)

Author: Zhenze Shi · 2026

---

## Business problem

Corporación Favorita runs 54 grocery stores across Ecuador, each stocking 33
product families ranging from fresh produce to electronics. Replenishment
decisions for the next two weeks have to be made before demand is observed,
and the two failure modes are asymmetric:

- **Over-forecasting** creates spoilage in perishables and ties up working
  capital in slow-moving categories.
- **Under-forecasting** causes stockouts, which cost both the immediate sale
  and, in grocery retail, some share of the customer's basket.

The operational question this project addresses: *for each store and each
product family, how much demand should be planned for over the coming
sixteen days?*

## Data

| | |
|---|---|
| Source | Kaggle competition dataset (Corporación Favorita, Ecuador) |
| Training period | 2013-01-01 to 2017-08-15 |
| Forecast period | 2017-08-16 to 2017-08-31 (16 days) |
| Scope | 54 stores × 33 product families = 1,782 daily series |
| Training records | 3.0M+ observations |
| Auxiliary data | Store metadata (city, state, cluster, type), daily oil prices, national holiday calendar |


Raw competition data is not redistributed in this repository. To reproduce,
download it from the competition page linked above.

## Key findings

**1. Forecast accuracy degrades monotonically with horizon, and the effect is
large enough to change the modelling design.**

Models predicting one to two days ahead consistently outperformed models
predicting fifteen to sixteen days ahead, with no exceptions across validation
windows. The practical implication is that a single model trained on the full
sixteen-day horizon wastes recent information on the near-term days. Splitting
the horizon into eight two-day buckets, each with its own minimum lag, improved
overall accuracy — and refining from four four-day buckets to eight two-day
buckets improved it again, confirming the pattern held at finer resolution.

**2. Ensemble averaging must be performed in log space when the objective is
RMSLE.**

RMSLE compares log-transformed values. Averaging two models' predictions in
raw sales units and then taking the log produces a systematically higher result
than averaging the logs directly, by Jensen's inequality. Blending in log space
and back-transforming afterwards was measurably better on validation.

**3. A large share of store–product combinations are structurally inactive, and
they are better handled by a rule than by the model.**

Many store–family pairs carry no sales at all for extended periods — products
never stocked in that store, or lines that have been discontinued. Gradient
boosting models emit small positive values for these series rather than exact
zeros, and because RMSLE penalises errors on a log scale, those small values
carry disproportionate cost. Forcing any combination with no sales in the prior
140 days to exactly zero was a meaningful improvement, and it reflects a real
operational distinction: an assortment decision is not a demand-forecasting
problem.

## Approach

**1. Complete series grid.** Train and test were reindexed onto a full
date × store × family grid so that every series has an unbroken daily history.
Missing sales inside the training period were set to zero rather than dropped,
which is what allows lag and rolling features to be computed by simple array
shifts.

**2. Feature engineering (37 features).** Grouped into:

| Group | Examples |
|---|---|
| Calendar | day of week, month, day of year, weekend flag |
| Business cycle | payday flag (15th / month-end), national holiday flag |
| Promotion | same-day promotion count, 14-day rolling promotion mean |
| Sales history | lags at multiple horizons, 364/365-day lags for year-over-year seasonality |
| Rolling statistics | 7/14/28/60/140-day rolling means, 28-day rolling standard deviation, share of zero-sales days over 60 days |
| Cross-sectional | store-level and family-level 28-day means and their ratio to the 365-day mean, capturing recent trend against a long-run base |
| Macro / events | crude oil price and 7-day moving average, days since the April 2016 earthquake |

Lag and rolling features were computed on a wide (dates × series) matrix and
reshaped back, which avoids a per-series groupby loop over 1,782 series.

**3. Horizon bucketing.** The sixteen-day forecast window was split into eight
two-day buckets. Each bucket has its own minimum lag, so the day 1–2 model can
use data up to two days before the forecast date while the day 15–16 model
cannot.

**4. Per-category models.** A separate model was trained for each of the 33
product families. Sales magnitude and seasonality differ by orders of magnitude
across categories; a single pooled model is dominated by the largest ones.

**5. Ensemble.** LightGBM and XGBoost were trained on identical features and
their predictions averaged in log space, then back-transformed.

**6. Zero-sales masking.** Store–family combinations with no sales in the
previous 140 days were forced to zero, as described in finding 3.

## Validation

Two rolling origin windows (2017-07-15 and 2017-07-31) were used, with training
data strictly preceding each origin so no future information leaks into the fit.
Early stopping rounds were tuned per family on the validation windows, then
averaged and reused for the final full-data refit on all data through
2017-08-15.

Each design change was scored against the previous version on identical
validation windows before being submitted, rather than being evaluated on the
leaderboard. The final version improved the submitted score from 0.40269 to
0.39683.

## Results

| | |
|---|---|
| Public leaderboard RMSLE | **0.39683** |
| Rank | **65 / 653 (top 10%)** |
| Submissions | 5 |

## Business implications

1. **Replenishment cadence should follow the pay cycle, not the week.** The
   semi-monthly payroll effect is a stronger driver than the calendar week for
   several categories, which argues for aligning order timing with the 15th and
   month-end rather than a fixed weekly cycle.

2. **Safety stock should scale with lead time.** Because accuracy falls off
   monotonically with horizon, holding a fixed safety-stock level across the
   whole planning window over-protects near-term days and under-protects
   far-term ones.

3. **Assortment and demand are separate decisions.** The zero-inflated tail of
   the product grid should be managed as a range-planning question, not fed
   through a forecasting model that will always return a non-zero number.

## Limitations

- The dataset ends in August 2017; no post-2017 validation is possible.
- Weather, competitor pricing, and local events are not available and likely
  explain part of the residual error.
- Store openings and closures during the period are not modelled explicitly.
- Per-category error decomposition is computed by the notebook but is not
  summarised here; a fuller version of this analysis would report which
  categories carry the highest error and what that implies for automated
  replenishment.
- This is a Getting Started competition with a rolling public leaderboard and no medal system, so the rank reflects relative standing at a point in time rather than a formal competition placement.

## Repository structure

```
store-sales-forecasting/
├── README.md
├── LICENSE
└── sales-final.ipynb      
```

Predictions are not included. 

## Reproducing

The notebook runs end to end on a Kaggle kernel with the competition dataset
attached. Runtime is roughly 1.5–3 hours depending on allocated cores. Reducing
`BUCKETS` from eight to four, or `VAL_STARTS` to a single window, cuts this
substantially at a small cost in accuracy.

```
python >= 3.10
pandas, numpy, lightgbm, xgboost
```

## License

MIT License. See `LICENSE`.
