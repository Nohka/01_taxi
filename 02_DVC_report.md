# Task 02 — DVC Experiment Report

## Overview

In this task we experimented with different **data versions** while working with the same regression model for taxi fare prediction.

The goal was to understand how model evaluation changes when new data is introduced and how DVC helps track these dataset versions.

Three experiment versions were created:

| Version | Model                         | Data                       |
| ------- | ----------------------------- | -------------------------- |
| **V1**  | Model trained on January data | January dataset            |
| **V2**  | Same V1 model reused          | January + February dataset |
| **V3**  | Model retrained               | January + February dataset |

---

# Version 1 — Initial Model

### Data
Only the **January dataset** was used.
green_tripdata_2021-01.parquet


### Features used

- trip_distance
- trip_duration_min
- passenger_count
- RatecodeID
- pickup_hour

### Data after cleaning

Rows retained:
37,930


### Model Performance

| Metric | Value     |
| ------ | --------- |
| MAE    | **0.609** |
| RMSE   | **3.095** |
| R²     | **0.953** |

The model performs quite well and explains around **95.3% of the variance** in taxi fares.

---

# Version 2 — Evaluation on Expanded Dataset

For version 2, we added **February data** to the dataset.

However, the model itself was **not retrained**.  
Instead, the **existing V1 model** was evaluated on a new dataset split created from **January + February data combined**.

### Data
green_tripdata_2021-01.parquet
green_tripdata_2021-02.parquet


### Data statistics

| Stage           | Rows        |
| --------------- | ----------- |
| Before cleaning | **141,090** |
| After cleaning  | **71,475**  |
| Test set        | **14,295**  |

### Model Performance

| Metric | Value     |
| ------ | --------- |
| MAE    | **0.487** |
| RMSE   | **2.515** |
| R²     | **0.968** |

### Interpretation

Interestingly, the results actually **improved**, even though the model was trained only on January data.

Possible explanations:

- February trips appear to follow similar patterns to January trips.
- The new random train/test split may have produced an **easier test dataset**.
- More data after cleaning can make the evaluation more stable.
- The model seems to **generalize well to similar data from another month**.

However, this does **not mean the model improved**, because the model itself did not change. Only the **evaluation data changed**.

---

# Version 3 — Retrained Model

For version 3 the model was **retrained using the expanded dataset (January + February)**.

This means:

- Data version stayed the same as V2
- Training code changed
- A new model was produced

### Data statistics

| Stage           | Rows        |
| --------------- | ----------- |
| Before cleaning | **141,090** |
| After cleaning  | **71,475**  |
| Training rows   | **57,180**  |
| Test rows       | **14,295**  |

### Model Performance

| Metric | Value     |
| ------ | --------- |
| MAE    | **0.554** |
| RMSE   | **2.805** |
| R²     | **0.961** |

### Interpretation

After retraining on the combined dataset, the performance is still strong but slightly different from version 2.

| Version | MAE   | RMSE  | R²    |
| ------- | ----- | ----- | ----- |
| **V1**  | 0.609 | 3.095 | 0.953 |
| **V2**  | 0.487 | 2.515 | 0.968 |
| **V3**  | 0.554 | 2.805 | 0.961 |

A few observations:

- V2 had the best metrics, but that was only **evaluation of the old model**, not retraining.
- V3 retraining produced a **more realistic performance estimate**.
- The performance is still good and shows the model generalizes reasonably well.

An interesting observation is that Version 2 achieved the best performance metrics even though the model itself was not retrained. This happens because the evaluation dataset changed rather than the model. In Version 2 the model trained on January data was evaluated on a new test split created from the combined January and February dataset. The new test set may have been more representative or slightly easier for the model to predict after cleaning. Therefore the improved results do not necessarily mean the model became better — they reflect differences in the evaluation data rather than improvements in the model itself. Version 3 provides a more realistic estimate of performance because the model was retrained on the expanded dataset.

---

# Role of DVC

DVC was used to track the dataset version independently from the code. When February data was added, the dataset state changed and a new version of the data was recorded. This allowed experiments to be run on different data configurations without committing large data files directly to Git.

| Version | Code                     | Data               | Model           |
| ------- | ------------------------ | ------------------ | --------------- |
| V1      | Original training script | January data       | Trained model   |
| V2      | New evaluation script    | January + February | Same model      |
| V3      | Updated training script  | January + February | Retrained model |

This allows us to easily reproduce different experiment states.

For example:
git checkout <version>
dvc pull
python script.py


---

# Monitoring the Model in Production

Assume **Version 3** is deployed in production and I need to monitor the system to make sure the model continues working properly.

## 1. Data Metric — Feature Distribution Drift

Example:

Monitor the distribution of important features such as:

- trip_distance
- trip_duration_min

If the distribution of these features changes significantly compared to the training data, it could mean the model is seeing **data it was not trained for**.

For example:

- new traffic patterns
- seasonal changes
- new taxi pricing rules

If major data drift is detected, the model should be **retrained with newer data**.

---

## 2. Model Performance Metric — Prediction Error

If real taxi fares become available after prediction, we can measure prediction error over time.

Example metrics:

- MAE
- RMSE

If prediction errors increase consistently, it means the model is becoming less accurate.

Possible reasons:

- behavior patterns changed
- new types of trips appeared
- pricing rules changed

If this happens, retraining the model with newer data would be necessary.

---

## 3. System Metric — Prediction Latency

It is also important to monitor system performance.

Example metrics:

- prediction response time
- CPU usage
- memory usage

If the model becomes too slow or uses too many resources, the system may need scaling or optimization.

---

# When to Retrain or Roll Back

### Retraining

The model should be retrained if:

- prediction errors increase significantly
- feature distributions drift from training data
- large amounts of new data become available

### Rollback

Rolling back to an earlier model may be needed if:

- the newly deployed model performs worse
- unexpected data causes prediction failures
- system stability problems appear

Keeping versions of both **code and data** makes these actions easier and safer.

---

# Conclusion

This task demonstrated how model results can change depending on the dataset used for evaluation. Even without retraining, evaluation results may improve or worsen simply due to differences in data. Using DVC helps keep track of these dataset versions and makes experiments reproducible. 

In a real machine learning workflow, DVC experiments (`dvc exp`) could be used to track these experiment versions automatically. Instead of creating separate scripts for each version, the same training pipeline could be executed multiple times with different data versions or parameters. DVC would then record the results of each run and allow easy comparison of model performance across experiments. In this assignment the experiments were implemented manually through separate scripts to clearly demonstrate the effects of changing the dataset and retraining the model.