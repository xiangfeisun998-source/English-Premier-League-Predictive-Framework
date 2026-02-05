# -*- coding: utf-8 -*-
"""
大小球 (Over/Under) 预测模型 V3 - 深度优化版
目标：寻找正 ROI 的"价值孤岛"
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 85)
print("大小球 (Over/Under) 预测模型 V3 - 深度优化版")
print("目标：寻找正 ROI 的价值孤岛")
print("=" * 85)

# =============================================================================
# 1. 盈亏计算逻辑 (完全重写，确保正确)
# =============================================================================

def calculate_ou_profit_v3(total_goals, goal_line, bet_type, odds, stake=1.0):
    """
    计算大小球投注的盈亏 V3版本
    
    正确的赔率计算:
    - 全赢: profit = stake * (odds - 1)  # 例如赔率0.95，净赢0.95-1=-0.05? 不对
    - 实际: 赔率0.95表示投1赢0.95，净利润=0.95*stake (不是odds-1)
    
    修正说明:
    - bet365的大小球赔率通常显示为 0.85-1.05 区间
    - 这是"港式赔率"，表示投1单位赢多少
    - 全赢: 净利润 = odds * stake
    - 全输: 净亏损 = -stake
    - 赢半: 净利润 = odds * stake / 2
    - 输半: 净亏损 = -stake / 2
    - 走水: 返还本金，净利润 = 0
    """
    # 判断盘口类型
    decimal_part = round(goal_line - int(goal_line), 2)
    
    if decimal_part == 0.25:
        # x.25盘口: 分解为 x 和 x.5 各半注
        line1 = int(goal_line)       # 整数线
        line2 = int(goal_line) + 0.5 # 半球线
        profit1 = _calc_single_profit(total_goals, line1, bet_type, odds, stake/2)
        profit2 = _calc_single_profit(total_goals, line2, bet_type, odds, stake/2)
        return profit1 + profit2
    
    elif decimal_part == 0.75:
        # x.75盘口: 分解为 x.5 和 x+1 各半注
        line1 = int(goal_line) + 0.5  # 半球线
        line2 = int(goal_line) + 1    # 整数线
        profit1 = _calc_single_profit(total_goals, line1, bet_type, odds, stake/2)
        profit2 = _calc_single_profit(total_goals, line2, bet_type, odds, stake/2)
        return profit1 + profit2
    
    else:
        # 标准盘口 (0.0 或 0.5)
        return _calc_single_profit(total_goals, goal_line, bet_type, odds, stake)


def _calc_single_profit(total_goals, goal_line, bet_type, odds, stake):
    """单一盘口线的盈亏计算"""
    if bet_type == 'over':
        if total_goals > goal_line:
            return odds * stake  # 全赢
        elif total_goals < goal_line:
            return -stake  # 全输
        else:
            return 0  # 走水
    else:  # under
        if total_goals < goal_line:
            return odds * stake  # 全赢
        elif total_goals > goal_line:
            return -stake  # 全输
        else:
            return 0  # 走水


def analyze_profit_result(total_goals, goal_line, bet_type, odds, stake=1.0):
    """分析投注结果类型"""
    decimal_part = round(goal_line - int(goal_line), 2)
    profit = calculate_ou_profit_v3(total_goals, goal_line, bet_type, odds, stake)
    
    # 判断结果类型
    full_win_profit = odds * stake
    half_win_profit = odds * stake / 2
    
    if abs(profit - full_win_profit) < 0.001:
        return 'full_win', profit
    elif abs(profit - half_win_profit) < 0.001:
        return 'half_win', profit
    elif abs(profit + stake) < 0.001:
        return 'full_loss', profit
    elif abs(profit + stake/2) < 0.001:
        return 'half_loss', profit
    elif abs(profit) < 0.001:
        return 'push', profit
    else:
        # 复合盘口可能产生其他结果
        if profit > 0:
            if profit > half_win_profit:
                return 'full_win', profit
            else:
                return 'half_win', profit
        elif profit < 0:
            if profit < -stake/2:
                return 'full_loss', profit
            else:
                return 'half_loss', profit
        else:
            return 'push', profit


# =============================================================================
# 2. 加载数据
# =============================================================================

print("\n[1] 加载数据...")
# 尝试读取数据文件（优先使用完整数据，如果没有则使用样本数据）
try:
    df = pd.read_csv('data/all_seasons_features.csv')
except FileNotFoundError:
    df = pd.read_csv('data/sample_data.csv')
df_ou = df.copy()
print(f"原始数据: {len(df_ou)} 场")

# =============================================================================
# 3. 特征工程
# =============================================================================

print("\n[2] 特征工程...")

# 排序
df_ou = df_ou.sort_values(['Season', 'Date']).reset_index(drop=True)

# 基础特征
df_ou['Total_Exp_Goals'] = (df_ou['Home_Avg_GS'] + df_ou['Away_Avg_GA']) + \
                            (df_ou['Away_Avg_GS'] + df_ou['Home_Avg_GA'])

# (1) 使用标准盘口线（2.5）计算Goal_Gap（用于特征工程，不依赖实际盘口数据）
standard_goal_line = 2.5
df_ou['Goal_Gap'] = df_ou['Total_Exp_Goals'] - standard_goal_line
df_ou['Goal_Gap_Percentage'] = df_ou['Goal_Gap'] / standard_goal_line * 100

# (2) 泊松分布计算（使用标准盘口线2.5）
def calc_poisson_over_prob(row):
    home_exp = (row['Home_Avg_GS'] + row['Away_Avg_GA']) / 2
    away_exp = (row['Away_Avg_GS'] + row['Home_Avg_GA']) / 2
    total_exp = home_exp + away_exp
    # 使用标准盘口线2.5
    goal_line = 2.5
    
    if goal_line == int(goal_line):
        over_prob = 1 - poisson.cdf(int(goal_line), total_exp)
    else:
        threshold = int(np.ceil(goal_line))
        over_prob = 1 - poisson.cdf(threshold - 1, total_exp)
    
    return over_prob

df_ou['Poisson_Over_Prob'] = df_ou.apply(calc_poisson_over_prob, axis=1)

# 泊松偏离（移除，因为需要赔率数据）
# df_ou['Poisson_Deviation'] = 0  # 设置为0或移除

# (3) BTTS概率
def calc_btts(row):
    home_exp = (row['Home_Avg_GS'] + row['Away_Avg_GA']) / 2
    away_exp = (row['Away_Avg_GS'] + row['Home_Avg_GA']) / 2
    home_score = 1 - poisson.pmf(0, home_exp)
    away_score = 1 - poisson.pmf(0, away_exp)
    return home_score * away_score

df_ou['BTTS_Rate'] = df_ou.apply(calc_btts, axis=1)

# (4) Sentiment_Clash: 移除，因为需要盘口变动数据
# df_ou['Sentiment_Clash_Over'] = 0
# df_ou['Sentiment_Clash_Under'] = 0

# (5) 进攻防守
df_ou['Attack_Power'] = df_ou['Home_Avg_GS'] + df_ou['Away_Avg_GS']
df_ou['Defense_Solid'] = df_ou['Home_Avg_GA'] + df_ou['Away_Avg_GA']

# (6) 月份提取
df_ou['Month'] = pd.to_datetime(df_ou['Date']).dt.month
df_ou['Month_Name'] = pd.to_datetime(df_ou['Date']).dt.strftime('%Y-%m')

# =============================================================================
# 4. 剔除走水，准备训练
# =============================================================================

print("\n[3] 准备训练数据...")
df_train = df_ou[df_ou['OU_Result'] != 0].copy()
print(f"有效数据: {len(df_train)} 场")

# 特征列表（移除所有依赖盘口线的特征）
features = [
    'Home_Avg_GS', 'Away_Avg_GS', 'Home_Avg_GA', 'Away_Avg_GA',
    'Home_GS_AtHome', 'Away_GS_AtAway', 'Attack_Power', 'Defense_Solid',
    'Total_Exp_Goals', 'Goal_Gap', 'Goal_Gap_Percentage',
    'Poisson_Over_Prob', 'BTTS_Rate',
    'Elo_Diff', 'Form_Diff'
]

# 填充缺失值
for col in features:
    if col in df_train.columns:
        df_train[col] = df_train[col].fillna(df_train[col].median())

available_features = [f for f in features if f in df_train.columns]
print(f"可用特征: {len(available_features)}")

# 划分
train_df = df_train[df_train['Season'] != '2024-2025'].copy()
test_df = df_train[df_train['Season'] == '2024-2025'].copy()

train_df['Target'] = (train_df['OU_Result'] == 1).astype(int)
test_df['Target'] = (test_df['OU_Result'] == 1).astype(int)

X_train = train_df[available_features]
y_train = train_df['Target']
X_test = test_df[available_features]
y_test = test_df['Target']

print(f"训练集: {len(train_df)}, 测试集: {len(test_df)}")

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =============================================================================
# 5. 训练模型
# =============================================================================

print("\n[4] 训练模型...")

lgb_model = lgb.LGBMClassifier(
    n_estimators=100, max_depth=4, learning_rate=0.02,
    num_leaves=15, min_child_samples=30,
    reg_alpha=0.3, reg_lambda=0.3,
    random_state=42, verbose=-1
)

xgb_model = xgb.XGBClassifier(
    n_estimators=100, max_depth=3, learning_rate=0.02,
    min_child_weight=8, subsample=0.6, colsample_bytree=0.6,
    random_state=42, eval_metric='logloss', verbosity=0
)

rf_model = RandomForestClassifier(
    n_estimators=100, max_depth=5, min_samples_split=30,
    min_samples_leaf=15, random_state=42
)

ensemble = VotingClassifier(
    estimators=[('lgb', lgb_model), ('xgb', xgb_model), ('rf', rf_model)],
    voting='soft'
)

calibrated = CalibratedClassifierCV(ensemble, method='isotonic', cv=3)
calibrated.fit(X_train_scaled, y_train)

y_pred_proba = calibrated.predict_proba(X_test_scaled)[:, 1]
y_pred = (y_pred_proba > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)
print(f"测试集准确率: {accuracy*100:.2f}%")
print(f"AUC: {auc:.4f}")

test_df['Pred_Over_Prob'] = y_pred_proba
test_df['Pred_Under_Prob'] = 1 - y_pred_proba

print("\n" + "=" * 85)
print("V3 深度优化总结报告")
print("=" * 85)

print(f"\n【模型性能】")
print(f"  准确率: {accuracy*100:.2f}%")
print(f"  AUC: {auc:.4f}")

print("\n" + "=" * 85)
