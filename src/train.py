from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeRegressor

models_directory = Path("models")
reports_directory = Path("reports")
experiment_records = []

# 1. Improve Classification: GaussianNB with log1p + scaling
classification_df = pd.read_csv("data/processed/features_classification.csv")
classification_feature_columns = [
    "hour",
    "day_of_week",
    "month",
    "stock_tx_count",
    "stock_avg_price",
]

classification_features = classification_df[classification_feature_columns].copy()

classification_target = classification_df["Is Cancelled"].astype(int)

(
    classification_features_train,
    classification_features_test,
    classification_target_train,
    classification_target_test,
) = train_test_split(
    classification_features,
    classification_target,
    test_size=0.2,
    random_state=42,
    stratify=classification_target,
)

improved_classification_model = make_pipeline(
    StandardScaler(),
    GaussianNB(),
)
improved_classification_model.fit(classification_features_train, classification_target_train)

# Check probability threshold tuning (default is 0.50; lower catches more cancellations)
chosen_threshold = 0.25
train_cancel_probabilities = improved_classification_model.predict_proba(classification_features_train)[:, 1]
improved_classification_train_predictions = (train_cancel_probabilities >= chosen_threshold).astype(int)

test_cancel_probabilities = improved_classification_model.predict_proba(classification_features_test)[:, 1]
improved_classification_predictions = (test_cancel_probabilities >= chosen_threshold).astype(int)

cls_train_accuracy = round(accuracy_score(classification_target_train, improved_classification_train_predictions), 4)
cls_train_f1 = round(f1_score(classification_target_train, improved_classification_train_predictions, zero_division=0), 4)
cls_accuracy = round(accuracy_score(classification_target_test, improved_classification_predictions), 4)
cls_f1 = round(f1_score(classification_target_test, improved_classification_predictions, zero_division=0), 4)

print("=== Improved 1. Classification (GaussianNB) ===")
print("Train Accuracy:", cls_train_accuracy, "| Test Accuracy:", cls_accuracy)
print("Train F1 Score:", cls_train_f1, "| Test F1 Score:", cls_f1)

experiment_records.append(
    {
        "task": "classification",
        "version": "final_improved",
        "model_name": "gaussian_nb_log1p_scaled",
        "accuracy": cls_accuracy,
        "f1_score": cls_f1,
        "mae": None,
        "rmse": None,
        "r2": None,
    }
)
joblib.dump(improved_classification_model, models_directory / "classification_model.joblib")

# 2. Improve Regression: Tuned DecisionTreeRegressor
regression_df = pd.read_csv("data/processed/features_regression.csv")
regression_feature_columns = [
    "hour",
    "day_of_week",
    "month",
    "stock_tx_count",
    "stock_avg_price",
]

regression_features = regression_df[regression_feature_columns]
regression_target = regression_df["line_revenue"]

(
    regression_features_train,
    regression_features_test,
    regression_target_train,
    regression_target_test,
) = train_test_split(
    regression_features,
    regression_target,
    test_size=0.2,
    random_state=42,
)

improved_regression_model = DecisionTreeRegressor(
    max_depth=7,
    min_samples_leaf=300,
    random_state=42,
)
improved_regression_model.fit(regression_features_train, regression_target_train)
improved_regression_train_predictions = improved_regression_model.predict(regression_features_train)
improved_regression_predictions = improved_regression_model.predict(regression_features_test)

reg_train_mae = round(mean_absolute_error(regression_target_train, improved_regression_train_predictions), 2)
reg_train_rmse = round(root_mean_squared_error(regression_target_train, improved_regression_train_predictions), 2)
reg_train_r2 = round(r2_score(regression_target_train, improved_regression_train_predictions), 4)

reg_mae = round(mean_absolute_error(regression_target_test, improved_regression_predictions), 2)
reg_rmse = round(root_mean_squared_error(regression_target_test, improved_regression_predictions), 2)
reg_r2 = round(r2_score(regression_target_test, improved_regression_predictions), 4)

print("\n=== Improved 2. Regression (DecisionTree max_depth=7) ===")
print("Train MAE :", reg_train_mae, "| Test MAE :", reg_mae)
print("Train RMSE:", reg_train_rmse, "| Test RMSE:", reg_rmse)
print("Train R2  :", reg_train_r2, "| Test R2  :", reg_r2)

experiment_records.append(
    {
        "task": "regression",
        "version": "final_improved",
        "model_name": "decision_tree_depth_7_leaf_300",
        "accuracy": None,
        "f1_score": None,
        "mae": reg_mae,
        "rmse": reg_rmse,
        "r2": reg_r2,
    }
)
joblib.dump(improved_regression_model, models_directory / "regression_model.joblib")

# 3. Improve Time Series: Extra lag/rolling feature + tuned tree
timeseries_df = pd.read_csv("data/processed/features_timeseries.csv", parse_dates=["date"])
timeseries_df = timeseries_df.sort_values("date").reset_index(drop=True)

# # Add rolling average of recent daily revenue if available
# if "lag_1d" in timeseries_df.columns:
#     timeseries_df["rolling_mean_3d"] = timeseries_df["lag_1d"].rolling(window=3, min_periods=1).mean()

timeseries_feature_columns = ["day_of_week", "month", "lag_1d", "lag_7d"]
# if "rolling_mean_3d" in timeseries_df.columns:
#     timeseries_feature_columns.append("rolling_mean_3d")

timeseries_features = timeseries_df[timeseries_feature_columns]
timeseries_target = timeseries_df["daily_revenue"]

chronological_split_index = int(len(timeseries_df) * 0.8)
timeseries_features_train = timeseries_features.iloc[:chronological_split_index]
timeseries_features_test = timeseries_features.iloc[chronological_split_index:]
timeseries_target_train = timeseries_target.iloc[:chronological_split_index]
timeseries_target_test = timeseries_target.iloc[chronological_split_index:]

improved_timeseries_model = DecisionTreeRegressor(
    max_depth=5,
    random_state=42,
)
improved_timeseries_model.fit(timeseries_features_train, timeseries_target_train)
improved_timeseries_train_predictions = improved_timeseries_model.predict(timeseries_features_train)
improved_timeseries_predictions = improved_timeseries_model.predict(timeseries_features_test)

ts_train_mae = round(mean_absolute_error(timeseries_target_train, improved_timeseries_train_predictions), 2)
ts_train_rmse = round(root_mean_squared_error(timeseries_target_train, improved_timeseries_train_predictions), 2)
ts_train_r2 = round(r2_score(timeseries_target_train, improved_timeseries_train_predictions), 4)

ts_mae = round(mean_absolute_error(timeseries_target_test, improved_timeseries_predictions), 2)
ts_rmse = round(root_mean_squared_error(timeseries_target_test, improved_timeseries_predictions), 2)
ts_r2 = round(r2_score(timeseries_target_test, improved_timeseries_predictions), 4)

print("\n=== Improved 3. Time Series (DecisionTree max_depth=5, min_samples_leaf=5) ===")
print("Train MAE :", ts_train_mae, "| Test MAE :", ts_mae)
print("Train RMSE:", ts_train_rmse, "| Test RMSE:", ts_rmse)
print("Train R2  :", ts_train_r2, "| Test R2  :", ts_r2)

experiment_records.append(
    {
        "task": "timeseries",
        "version": "final_improved",
        "model_name": "decision_tree_depth_5_leaf_5",
        "accuracy": None,
        "f1_score": None,
        "mae": ts_mae,
        "rmse": ts_rmse,
        "r2": ts_r2,
    }
)
joblib.dump(improved_timeseries_model, models_directory / "timeseries_model.joblib")

results_df = pd.DataFrame(experiment_records)
results_df.to_csv(reports_directory / "final_model_metrics.csv", index=False)
print("\nSaved final models to models/ and metrics to reports/final_model_metrics.csv")