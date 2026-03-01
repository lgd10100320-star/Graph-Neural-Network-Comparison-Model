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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import matplotlib.ticker as mticker
import joblib
import xgboost as xgb
import statsmodels.api as sm
import os
import tempfile

# 设置临时文件夹
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
    'svg.fonttype': 'none',
})

# 导入数据
data = pd.read_csv(
    'Molecular descriptors - dataset.csv',
    header=0,
)
x = data.iloc[:, 1:26]
y = data.iloc[:, 26]

# 划分训练测试集
x_train0, x_test0, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0
)

# 标准化
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train0)
x_test = scaler.transform(x_test0)

# XGBoost 参数网格
param_grid = {
    'learning_rate': [0.05, 0.075],
    'n_estimators': [150],
    'max_depth': [2, 3, 5],
    'min_child_weight': [2],
    'subsample': [0.6],
    'colsample_bytree': [0.6],
    'gamma': [0.1],
    'reg_alpha': [0.05],
    'reg_lambda': [0.01],
}

# 创建并训练模型
model = xgb.XGBRegressor(objective='reg:squarederror', seed=0)
gsearch = GridSearchCV(
    model,
    param_grid,
    scoring='neg_mean_squared_error',
    cv=10,
    n_jobs=8,
    verbose=4,
)
gsearch.fit(x_train, y_train)
xgb_model = gsearch.best_estimator_

# 评估模型
train_pred = xgb_model.predict(x_train)
test_pred = xgb_model.predict(x_test)

# 保存模型和标准化器
joblib.dump(xgb_model, 'xgboost_model.pkl')
joblib.dump(scaler, 'standard_scaler.pkl')
print('模型和标准化器已保存')

# 计算更多评价指标
train_mae = mean_absolute_error(y_train, train_pred)
train_mape = mean_absolute_percentage_error(y_train, train_pred)
train_mse = mean_squared_error(y_train, train_pred)
train_rmse = np.sqrt(train_mse)
train_r2 = r2_score(y_train, train_pred)

test_mae = mean_absolute_error(y_test, test_pred)
test_mape = mean_absolute_percentage_error(y_test, test_pred)
test_mse = mean_squared_error(y_test, test_pred)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, test_pred)

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

metrics_path = 'xgboost_performance_metrics.txt'
with open(metrics_path, 'w', encoding='utf-8') as f:
    for split, values in metrics.items():
        f.write(f'{split}集指标:\n')
        for metric_name, metric_value in values.items():
            f.write(f'  {metric_name}: {metric_value:.6f}\n')
        f.write('\n')

print(f'评估指标已保存至 {metrics_path}')

# 构造回归直线及其 95% 置信区间
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

# 绘制结果图
plt.figure(dpi=900, figsize=(4, 4))
train_color = '#4FBDFF'
test_color = '#FB6F6F'


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
plt.title('XGBoost Performance')
plt.minorticks_on()
plt.xlabel('Measured PCE(%)')
plt.ylabel('Predicted PCE(%)')
plt.xlim(10, 28)
plt.ylim(10, 26)
plt.legend()

plt.savefig(
    'XGBoost_Performance.png',
    dpi=900,
    bbox_inches='tight',
    pad_inches=0.05,
    facecolor='white',
)
plt.show()
