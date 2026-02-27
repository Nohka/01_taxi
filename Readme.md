# 🚕 Taxi Trip Modeling Report

**Dataset:** `green_tripdata_2021-01.parquet`
**Project:** Regression and Classification Modeling
**Author:** *Tarvo Metspalu*

---

# 1. Introduction

This project analyzes NYC Green Taxi trip data and develops:

* A **regression model** to predict `fare_amount`
* A **classification model** to predict `payment_type`

The workflow includes:

* Exploratory Data Analysis (EDA)
* Feature selection (2–4 features initially)
* Model training and evaluation
* Feature extension analysis (adding one more feature)
* Discussion of robustness, randomness, leakage, and production risks

---

# 2. Exploratory Data Analysis (EDA)

Initial EDA included:

* Dataset shape and structure analysis
* Missing value inspection
* Distribution plots for:

  * `trip_distance`
  * `fare_amount`
  * `trip_duration_min`
* Correlation analysis
* Outlier filtering:

  * `trip_distance` between 0.01 and 100 miles
  * `trip_duration_min` between 0.1 and 300 minutes
  * `fare_amount` between 0.5 and 300 USD
  * `passenger_count` between 1 and 8

A derived feature was created:

```
trip_duration_min = dropoff_time - pickup_time
```

---

# 3. Regression Model

## 3.1 Target

```
fare_amount
```

## 3.2 Initial Features (4)

* `trip_distance`
* `trip_duration_min`
* `passenger_count`
* `RatecodeID`

## 3.3 Initial Regression Results

Rows after cleaning: **37,930**

| Metric | Value |
| ------ | ----- |
| MAE    | 0.656 |
| RMSE   | 3.743 |
| R²     | 0.932 |

### Interpretation

* The model explains **93.2% of variance** in fare.
* Average absolute error ≈ **$0.66**
* Some larger errors exist (reflected in RMSE).

---

## 3.4 Added Feature: `pickup_hour`

Derived as:

```
pickup_hour = lpep_pickup_datetime.hour
```

Updated feature set:

* `trip_distance`
* `trip_duration_min`
* `passenger_count`
* `RatecodeID`
* `pickup_hour`

---

## 3.5 Updated Regression Results

Rows after cleaning: **37,930**

| Metric | Before | After |
| ------ | ------ | ----- |
| MAE    | 0.656  | 0.609 |
| RMSE   | 3.743  | 3.095 |
| R²     | 0.932  | 0.953 |

### Interpretation

* R² improved from **0.932 → 0.953**
* RMSE decreased significantly
* Indicates that **time of day contributes predictive information**

This suggests fare pricing is influenced not only by distance and duration, but also indirectly by traffic and temporal demand patterns.

---

# 4. Classification Model

## 4.1 Target

```
payment_type
```

Due to extreme class imbalance, only the two dominant classes were retained:

* 1 = Credit Card
* 2 = Cash

Class distribution:

* 1: 22,818
* 2: 14,995

Total rows: **37,813**

---

## 4.2 Initial Features (4)

* `trip_distance`
* `trip_duration_min`
* `passenger_count`
* `RatecodeID`

## 4.3 Initial Classification Results

| Metric   | Value  |
| -------- | ------ |
| Accuracy | 0.5746 |
| Macro F1 | 0.5526 |

Confusion Matrix:

```
[[3012 1552]
 [1665 1334]]
```

---

## 4.4 After Adding `pickup_hour`

Updated Features:

* `trip_distance`
* `trip_duration_min`
* `passenger_count`
* `RatecodeID`
* `pickup_hour`

### Updated Results

| Metric   | Before | After  |
| -------- | ------ | ------ |
| Accuracy | 0.5746 | 0.5972 |
| Macro F1 | 0.5526 | 0.5745 |

Confusion Matrix:

```
[[3133 1431]
 [1615 1384]]
```

### Interpretation

* Accuracy improved by ~2.3%
* Macro F1 improved meaningfully
* Time-of-day influences payment behavior

This suggests that payment preference varies across different times of day.

---

# 5. Ensuring Improvement is Not Due to Randomness or Data Leakage

## 5a. Avoiding Randomness

The following steps ensure improvements are not due to randomness:

1. **Fixed random_state**

   * The same `random_state=42` was used in both models.
   * Ensures reproducibility.

2. **Stratified train-test split (classification)**

   * Maintains class balance across splits.

3. **Controlled comparison**

   * The only change between experiments was the addition of `pickup_hour`.
   * All other preprocessing and parameters remained constant.

4. **Optional cross-validation**

   * K-fold cross-validation can confirm consistency across folds.

Therefore, the observed improvement is attributable to the new feature rather than random variation.

---

## 5b. Avoiding Data Leakage

Data leakage occurs when future or target-derived information is used during training.

`pickup_hour`:

* Is derived solely from pickup timestamp.
* Is available at prediction time.
* Does not use payment type or fare information.
* Does not incorporate future data.

Therefore, no target leakage is introduced.

---

# 6. Production Risks

If a feature cannot be reliably generated in production, several risks arise:

## 6.1 Feature Availability Risk

If prediction occurs before the pickup timestamp is recorded, `pickup_hour` would not be available.

More critically:

```
trip_duration_min
```

is only available **after trip completion**.

If predictions are required before the trip ends, this would create:

**Training-serving skew** — a mismatch between training features and production features.

---

## 6.2 Data Quality Risk

* Missing timestamps
* Incorrect timezones
* Clock synchronization issues

These would degrade model reliability.

---

## 6.3 Distribution Shift

Changes in:

* Traffic patterns
* Payment behavior
* Pricing policy

could reduce the predictive value of temporal features.

---

# 7. Conclusion

Adding `pickup_hour` improved both regression and classification performance.

Regression:

* R² increased to 0.953
* RMSE decreased significantly

Classification:

* Accuracy improved from 0.5746 → 0.5972
* Macro F1 improved consistently

This demonstrates that temporal features add meaningful predictive signal.

However, careful consideration must be given to:

* Randomness control
* Data leakage prevention
* Production feature availability
* Training-serving skew

From a modeling perspective, the results are strong.
From an MLOps perspective, production constraints must be carefully evaluated.

---

# End of Report
