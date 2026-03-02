from pathlib import Path
import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


BASE_DIR = Path(__file__).resolve().parent
TMP_DIR = BASE_DIR / "tmp"
DATA_PATH = BASE_DIR / "property database.csv"
METRICS_PATH = BASE_DIR / "svr_performance_metrics.txt"
FIGURE_PATH = BASE_DIR / "SVR_Performance.png"

TMP_DIR.mkdir(exist_ok=True)
os.environ["TMP"] = str(TMP_DIR)
os.environ["TEMP"] = str(TMP_DIR)
tempfile.tempdir = str(TMP_DIR)

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "lines.linewidth": 0.75,
        "font.family": "Times New Roman",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)

data = pd.read_csv(DATA_PATH, header=0)
x = data.iloc[:, 1:26]
y = data.iloc[:, 26]

x_train0, x_test0, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train0)
x_test = scaler.transform(x_test0)

param_grid = {
    "C": [0.1, 1],
    "kernel": ["linear"],
    "gamma": [1e-2, 1e-1],
    "degree": [1, 3, 5],
    "epsilon": [0.01, 0.1],
}

svm = SVR()
cross_validation = KFold(n_splits=10, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    estimator=svm,
    param_grid=param_grid,
    cv=cross_validation,
    n_jobs=8,
    verbose=2,
)
grid_search.fit(x_train, y_train)
svr_model = grid_search.best_estimator_

print(f"Best parameters: {grid_search.best_params_}")

train_pred = svr_model.predict(x_train)
test_pred = svr_model.predict(x_test)

train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
train_r = stats.pearsonr(y_train, train_pred)[0]
train_r2 = r2_score(y_train, train_pred)
train_mae = mean_absolute_error(y_train, train_pred)
train_mape = mean_absolute_percentage_error(y_train, train_pred)
train_mse = mean_squared_error(y_train, train_pred)

test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
test_r = stats.pearsonr(y_test, test_pred)[0]
test_r2 = r2_score(y_test, test_pred)
test_mae = mean_absolute_error(y_test, test_pred)
test_mape = mean_absolute_percentage_error(y_test, test_pred)
test_mse = mean_squared_error(y_test, test_pred)

print(f"Train RMSE: {train_rmse}")
print(f"Train R: {train_r}")
print(f"Train R2: {train_r2}")
print(f"Test RMSE: {test_rmse}")
print(f"Test R: {test_r}")
print(f"Test R2: {test_r2}")

metrics = {
    "Train": {
        "MAE": train_mae,
        "MAPE": train_mape,
        "MSE": train_mse,
        "RMSE": train_rmse,
        "R2": train_r2,
    },
    "Test": {
        "MAE": test_mae,
        "MAPE": test_mape,
        "MSE": test_mse,
        "RMSE": test_rmse,
        "R2": test_r2,
    },
}

with open(METRICS_PATH, "w", encoding="utf-8") as file:
    for split, values in metrics.items():
        file.write(f"{split} Metrics:\n")
        for metric_name, metric_value in values.items():
            file.write(f"  {metric_name}: {metric_value:.6f}\n")
        file.write("\n")

print(f"Metrics saved to {METRICS_PATH.name}")


def compute_regression_ci(actual_values: np.ndarray, predicted_values: np.ndarray):
    design_matrix = sm.add_constant(actual_values)
    regression_model = sm.OLS(predicted_values, design_matrix).fit()
    x_seq = np.linspace(actual_values.min(), actual_values.max(), 200)
    x_seq_design = sm.add_constant(x_seq)
    prediction_res = regression_model.get_prediction(x_seq_design)
    summary = prediction_res.summary_frame(alpha=0.05)
    return (
        x_seq,
        summary["mean"].to_numpy(),
        summary["mean_ci_lower"].to_numpy(),
        summary["mean_ci_upper"].to_numpy(),
    )


train_actual = y_train.to_numpy()
test_actual = y_test.to_numpy()

train_x_seq, train_line_mean, train_ci_lower, train_ci_upper = compute_regression_ci(
    train_actual,
    train_pred,
)
test_x_seq, test_line_mean, test_ci_lower, test_ci_upper = compute_regression_ci(
    test_actual,
    test_pred,
)

plt.figure(dpi=900, figsize=(4, 4))
train_color = "#FF7F00"
test_color = "#377EB8"

plt.scatter(
    train_actual,
    train_pred,
    color=train_color,
    alpha=0.7,
    s=10,
    marker="P",
    label="Train",
)
plt.scatter(
    test_actual,
    test_pred,
    color=test_color,
    alpha=0.7,
    s=10,
    marker="^",
    label="Test",
)

plt.plot(
    train_x_seq,
    train_line_mean,
    color=train_color,
    linewidth=1.0,
    label="Train regression fit",
)
plt.fill_between(
    train_x_seq,
    train_ci_lower,
    train_ci_upper,
    color=train_color,
    alpha=0.15,
    label="Train 95% CI",
)

plt.plot(
    test_x_seq,
    test_line_mean,
    color=test_color,
    linewidth=1.0,
    label="Test regression fit",
)
plt.fill_between(
    test_x_seq,
    test_ci_lower,
    test_ci_upper,
    color=test_color,
    alpha=0.15,
    label="Test 95% CI",
)

plt.text(11, 24.5, f"Train: R2 = {train_r2:.3f}", color="black", fontsize=8)
plt.text(11, 23.5, f"Test: R2 = {test_r2:.3f}", color="black", fontsize=8)

plt.gca().set_aspect("equal")
plt.title("SVR Performance")
plt.minorticks_on()
plt.xlabel("Measured PCE(%)")
plt.ylabel("Predicted PCE(%)")
plt.xlim(10, 28)
plt.ylim(10, 27)
plt.legend()

plt.savefig(
    FIGURE_PATH,
    dpi=900,
    bbox_inches="tight",
    pad_inches=0.05,
    facecolor="white",
)
plt.show()
