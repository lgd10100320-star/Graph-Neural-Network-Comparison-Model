from pathlib import Path
import os
import tempfile
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from catboost import CatBoostRegressor
from matplotlib.ticker import MultipleLocator
from scipy import stats
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
TMP_DIR = BASE_DIR / "tmp"
DATA_PATH = BASE_DIR / "property database.csv"
METRICS_PATH = BASE_DIR / "catboost_performance_metrics.txt"
FIGURE_PATH = BASE_DIR / "CatBoost_Performance.png"
MODEL_PATH = BASE_DIR / "catboost_model.pkl"
SCALER_PATH = BASE_DIR / "catboost_scaler.pkl"

TMP_DIR.mkdir(exist_ok=True)
os.environ["TMP"] = str(TMP_DIR)
os.environ["TEMP"] = str(TMP_DIR)
tempfile.tempdir = str(TMP_DIR)

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "lines.linewidth": 1.0,
        "font.family": "Times New Roman",
        "font.weight": "bold",
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
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

warnings.filterwarnings("ignore")

param_grid = {
    "max_depth": [2, 4, 6],
    "learning_rate": [0.1, 0.05, 0.001],
    "l2_leaf_reg": [1e-3, 1e-2],
    "iterations": [50, 30, 10],
}

cat = CatBoostRegressor(random_state=42, verbose=0)
grid_search = GridSearchCV(
    cat,
    param_grid=param_grid,
    scoring="neg_mean_squared_error",
    cv=10,
    n_jobs=8,
)
grid_search.fit(x_train, y_train)
cat = grid_search.best_estimator_

print(f"Best parameters: {grid_search.best_params_}")

train_pred = cat.predict(x_train)
test_pred = cat.predict(x_test)

train_mae = mean_absolute_error(y_train, train_pred)
train_mape = mean_absolute_percentage_error(y_train, train_pred)
train_mse = mean_squared_error(y_train, train_pred)
train_rmse = np.sqrt(train_mse)
train_r = stats.pearsonr(y_train, train_pred)[0]
train_r2 = r2_score(y_train, train_pred)

test_mae = mean_absolute_error(y_test, test_pred)
test_mape = mean_absolute_percentage_error(y_test, test_pred)
test_mse = mean_squared_error(y_test, test_pred)
test_rmse = np.sqrt(test_mse)
test_r = stats.pearsonr(y_test, test_pred)[0]
test_r2 = r2_score(y_test, test_pred)

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

width_cm = 8
height_cm = 6
width_inch = width_cm / 2.54
height_inch = height_cm / 2.54

plt.figure(dpi=900, figsize=(width_inch, height_inch))
train_color = "#4FBDFF"
test_color = "#FB6F6F"

plt.scatter(
    train_actual,
    train_pred,
    color=train_color,
    alpha=0.7,
    s=8,
    marker="P",
    label="Train",
)
plt.scatter(
    test_actual,
    test_pred,
    color=test_color,
    alpha=0.7,
    s=8,
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

plt.text(11, 26, f"Train: R2 = {train_r2:.3f}", color="black", fontsize=5.5)
plt.text(11, 25, f"Test: R2 = {test_r2:.3f}", color="black", fontsize=5.5)

plt.gca().set_aspect("equal")
plt.minorticks_on()

ax = plt.gca()
ax.xaxis.set_major_locator(MultipleLocator(2))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(2))
ax.yaxis.set_minor_locator(MultipleLocator(1))

plt.xlabel("Measured PCE(%)", fontsize=9, fontweight="bold", labelpad=5)
plt.ylabel("Predicted PCE(%)", fontsize=9, fontweight="bold", labelpad=5)
plt.xlim(10, 28)
plt.ylim(10, 28)
plt.grid(True, which="major", linestyle="--", linewidth=0.3, alpha=0.7)
plt.grid(False, which="minor")
plt.legend(
    loc="lower right",
    fontsize=5.5,
    frameon=True,
    fancybox=True,
    shadow=True,
    markerscale=0.7,
    handlelength=1.2,
    borderpad=0.3,
    labelspacing=0.2,
)
plt.tight_layout(pad=1.0)

plt.savefig(
    FIGURE_PATH,
    dpi=900,
    bbox_inches="tight",
    pad_inches=0.05,
    facecolor="white",
)
plt.show()

joblib.dump(cat, MODEL_PATH)
print(f"CatBoost model saved as {MODEL_PATH.name}")

joblib.dump(scaler, SCALER_PATH)
print(f"Scaler saved as {SCALER_PATH.name}")
