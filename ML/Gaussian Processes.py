import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import ShuffleSplit, train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
import matplotlib.ticker as mticker
# from matplotlib.backends import backend_pdf
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    Matern,
    RationalQuadratic,
    DotProduct,
    ConstantKernel,
)
import statsmodels.api as sm

# 设置临时文件夹
import os, tempfile
os.environ['TMP'] = r'C:\Temp'
os.environ['TEMP'] = r'C:\Temp'
tempfile.tempdir = r'C:\Temp'

# 可视化配置
plt.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'lines.linewidth': 0.75,
    'font.family': 'Times New Roman',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none'
})

# 导入文件
data = pd.read_csv('Molecular descriptors - dataset.csv', header=0)

# 特征与目标变量分离
x = data.iloc[:, 1:26]
y = data.iloc[:, 26]

# 交叉验证设置
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

# 划分训练集和测试集
x_train0, x_test0, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# 标准化处理
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train0)
x_test = scaler.transform(x_test0)

# 高斯过程回归
# 定义核函数组合
kernels = [
    1.0 * RBF(length_scale=1.0),
    1.0 * Matern(length_scale=1.0, nu=1.5),
    1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1),
    1.0 * DotProduct(sigma_0=1.0),
    ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
]

param_grid = {
    'kernel': kernels,
    'alpha': [1e-5, 1e-3, 1e-1],
    'n_restarts_optimizer': [3, 5]
}

gpr = GaussianProcessRegressor()
cross_valid = KFold(n_splits=5, shuffle=True, random_state=42)  # 减少折数以加速计算
gs = GridSearchCV(estimator=gpr,
                 param_grid=param_grid,
                 cv=cross_valid,
                 n_jobs=8,
                 verbose=2,
                 scoring='neg_mean_squared_error')
gs.fit(x_train, y_train)
GPR_model = gs.best_estimator_

print("Best parameters:", gs.best_params_)

train_pred = GPR_model.predict(x_train)
test_pred = GPR_model.predict(x_test)

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

print(f"RMSE_train: {train_rmse}")
print(f"R_train: {train_r}")
print(f"R²_train: {train_r2}")
print(f"RMSE_test: {test_rmse}")
print(f"R_test: {test_r}")
print(f"R²_test: {test_r2}")

metrics = {
    'Train': {
        'MAE': train_mae,
        'MAPE': train_mape,
        'MSE': train_mse,
        'RMSE': train_rmse,
        'R2': train_r2,
    },
    'Test': {
        'MAE': test_mae,
        'MAPE': test_mape,
        'MSE': test_mse,
        'RMSE': test_rmse,
        'R2': test_r2,
    },
}

metrics_path = 'gpr_performance_metrics.txt'
with open(metrics_path, 'w', encoding='utf-8') as f:
    for split, values in metrics.items():
        f.write(f'{split} Metrics:\n')
        for metric_name, metric_value in values.items():
            f.write(f'  {metric_name}: {metric_value:.6f}\n')
        f.write('\n')

print(f'Metrics saved to {metrics_path}')


def compute_regression_ci(actual_values: np.ndarray, predicted_values: np.ndarray):
    design_matrix = sm.add_constant(actual_values)
    regression_model = sm.OLS(predicted_values, design_matrix).fit()
    x_seq = np.linspace(actual_values.min(), actual_values.max(), 200)
    x_seq_design = sm.add_constant(x_seq)
    prediction_res = regression_model.get_prediction(x_seq_design)
    summary = prediction_res.summary_frame(alpha=0.05)
    return (
        x_seq,
        summary['mean'].to_numpy(),
        summary['mean_ci_lower'].to_numpy(),
        summary['mean_ci_upper'].to_numpy(),
    )


train_actual = y_train.to_numpy()
test_actual = y_test.to_numpy()

(train_x_seq, train_line_mean, train_ci_lower, train_ci_upper) = compute_regression_ci(
    train_actual,
    train_pred,
)
(test_x_seq, test_line_mean, test_ci_lower, test_ci_upper) = compute_regression_ci(
    test_actual,
    test_pred,
)

plt.figure(dpi=900, figsize=(4, 4))
train_color = '#FF7F00'
test_color = '#377EB8'

plt.scatter(
    train_actual,
    train_pred,
    color=train_color,
    alpha=0.7,
    s=10,
    marker='P',
    label='Train',
)
plt.scatter(
    test_actual,
    test_pred,
    color=test_color,
    alpha=0.7,
    s=10,
    marker='^',
    label='Test',
)

plt.plot(
    train_x_seq,
    train_line_mean,
    color=train_color,
    linewidth=1.0,
    label='Train regerss fit',
)
plt.fill_between(
    train_x_seq,
    train_ci_lower,
    train_ci_upper,
    color=train_color,
    alpha=0.15,
    label='Train 95% conf.bounds',
)

plt.plot(
    test_x_seq,
    test_line_mean,
    color=test_color,
    linewidth=1.0,
    label='Test regerss fit',
)
plt.fill_between(
    test_x_seq,
    test_ci_lower,
    test_ci_upper,
    color=test_color,
    alpha=0.15,
    label='Test 95% conf.bounds',
)

plt.text(
    11,
    24.5,
    f'Train: R² = {train_r2:.3f}',
    color='black',
    fontsize=8,
)
plt.text(
    11,
    23.5,
    f'Test: R² = {test_r2:.3f}',
    color='black',
    fontsize=8,
)

plt.gca().set_aspect('equal')
plt.title('Gaussian Process Performance')
plt.minorticks_on()
plt.xlabel('Measured PCE(%)')
plt.ylabel('Predicted PCE(%)')
plt.xlim(10, 28)
plt.ylim(10, 27)
plt.legend()

plt.savefig(
    'GPR_Performance.png',
    dpi=900,
    bbox_inches='tight',
    pad_inches=0.05,
    facecolor='white',
)

plt.show()

