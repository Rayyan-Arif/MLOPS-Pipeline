from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
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
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

reports_directory = Path("reports")
reports_directory.mkdir(parents=True, exist_ok=True)
experiment_records = []

# 1. Classification
classification_df = pd.read_csv("data/processed/features_classification.csv")
classification_feature_columns = [
    "hour",
    "day_of_week",
    "month",
    "stock_tx_count",
    "stock_avg_price",
]

classification_features = classification_df[classification_feature_columns]
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

classification_models = {
    "logistic_regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42),
    ),
    "naive_bayes": make_pipeline(
        StandardScaler(),
        GaussianNB(),
    ),
    "decision_tree_depth_5": DecisionTreeClassifier(
        max_depth=5,
        class_weight="balanced",
        random_state=42,
    ),
}

print("=== 1. Classification (Is Cancelled) ===")
for model_name, model in classification_models.items():
    model.fit(classification_features_train, classification_target_train)
    predictions = model.predict(classification_features_test)

    accuracy = round(accuracy_score(classification_target_test, predictions), 4)
    f1 = round(f1_score(classification_target_test, predictions, zero_division=0), 4)
    print(f"{model_name:24s} -> Accuracy: {accuracy} | F1: {f1}")

    experiment_records.append(
        {
            "task": "classification",
            "model_name": model_name,
            "accuracy": accuracy,
            "f1_score": f1,
            "mae": None,
            "rmse": None,
            "r2": None,
        }
    )

# 2. Regression (line_revenue)
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

regression_models = {
    "ridge_regression": make_pipeline(StandardScaler(), Ridge()),
    "decision_tree_depth_5": DecisionTreeRegressor(max_depth=5, random_state=42),
}

print("\n=== 2. Regression (line_revenue) ===")
for model_name, model in regression_models.items():
    model.fit(regression_features_train, regression_target_train)
    predictions = model.predict(regression_features_test)

    mae = round(mean_absolute_error(regression_target_test, predictions), 2)
    rmse = round(root_mean_squared_error(regression_target_test, predictions), 2)
    r2 = round(r2_score(regression_target_test, predictions), 4)
    print(f"{model_name:24s} -> MAE: {mae} | RMSE: {rmse} | R2: {r2}")

    experiment_records.append(
        {
            "task": "regression",
            "model_name": model_name,
            "accuracy": None,
            "f1_score": None,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        }
    )

# 3. Time Series Experiments (daily_revenue)
timeseries_df = pd.read_csv("data/processed/features_timeseries.csv", parse_dates=["date"])
timeseries_df = timeseries_df.sort_values("date").reset_index(drop=True)
timeseries_feature_columns = ["day_of_week", "month", "lag_1d", "lag_7d"]

timeseries_features = timeseries_df[timeseries_feature_columns]
timeseries_target = timeseries_df["daily_revenue"]

chronological_split_index = int(len(timeseries_df) * 0.8)
timeseries_features_train = timeseries_features.iloc[:chronological_split_index]
timeseries_features_test = timeseries_features.iloc[chronological_split_index:]
timeseries_target_train = timeseries_target.iloc[:chronological_split_index]
timeseries_target_test = timeseries_target.iloc[chronological_split_index:]
timeseries_dates_test = timeseries_df["date"].iloc[chronological_split_index:]

timeseries_models = {
    "decision_tree_depth_5": DecisionTreeRegressor(max_depth=5, random_state=42),
    "svr_rbf": make_pipeline(StandardScaler(), SVR(kernel="rbf")),
}

timeseries_test_predictions = {}

print("\n=== 3. Time Series (daily_revenue) ===")
for model_name, model in timeseries_models.items():
    model.fit(timeseries_features_train, timeseries_target_train)
    predictions = model.predict(timeseries_features_test)
    timeseries_test_predictions[model_name] = predictions

    mae = round(mean_absolute_error(timeseries_target_test, predictions), 2)
    rmse = round(root_mean_squared_error(timeseries_target_test, predictions), 2)
    r2 = round(r2_score(timeseries_target_test, predictions), 4)
    print(f"{model_name:24s} -> MAE: {mae} | RMSE: {rmse} | R2: {r2}")

    experiment_records.append(
        {
            "task": "timeseries",
            "model_name": model_name,
            "accuracy": None,
            "f1_score": None,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        }
    )

results_df = pd.DataFrame(experiment_records)
results_df.to_csv(reports_directory / "experiment_results.csv", index=False)
print("\nSaved experiment log to reports/experiment_results.csv")


# Plot 1: Classification metrics
classification_results = results_df[results_df["task"] == "classification"]
classification_results.plot(
    x="model_name",
    y=["accuracy", "f1_score"],
    kind="bar",
    figsize=(8, 4),
    title="Classification Model Comparison (Is Cancelled)",
)
plt.ylabel("Score")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# Plot 2: Regression error metrics
regression_results = results_df[results_df["task"] == "regression"]
regression_results.plot(
    x="model_name",
    y=["mae", "rmse"],
    kind="bar",
    figsize=(8, 4),
    title="Regression Error Comparison (line_revenue)",
)
plt.ylabel("Error")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# Plot 3: Time Series error comparison + Actual vs. Predicted line plot
timeseries_results = results_df[results_df["task"] == "timeseries"]
timeseries_results.plot(
    x="model_name",
    y=["mae", "rmse"],
    kind="bar",
    figsize=(8, 4),
    title="Time Series Error Comparison (daily_revenue)",
)
plt.ylabel("Error")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(timeseries_dates_test, timeseries_target_test.values, label="Actual daily_revenue", marker="o")
for model_name, predictions in timeseries_test_predictions.items():
    plt.plot(timeseries_dates_test, predictions, label=f"Predicted ({model_name})", linestyle="--")
plt.title("Time Series Test Set: Actual vs. Predicted daily_revenue")
plt.xlabel("Date")
plt.ylabel("Daily Revenue")
plt.xticks(rotation=25)
plt.legend()
plt.tight_layout()
plt.show()