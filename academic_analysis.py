# -*- coding: utf-8 -*-
"""
学术评估分析脚本
用于提取项目的学术成果描述
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 90)
print("学术评估分析 - 英超预测模型")
print("=" * 90)

# =============================================================================
# 1. 加载数据
# =============================================================================

print("\n[1] 加载数据...")
# 尝试读取数据文件（优先使用完整数据，如果没有则使用样本数据）
try:
    df = pd.read_csv('data/all_seasons_features.csv')
except FileNotFoundError:
    df = pd.read_csv('data/sample_data.csv')

df_ou = df.copy()
df_ou = df_ou.sort_values(['Season', 'Date']).reset_index(drop=True)

# 特征工程（与主模型一致，移除盘口线依赖）
df_ou['Total_Exp_Goals'] = (df_ou['Home_Avg_GS'] + df_ou['Away_Avg_GA']) + \
                            (df_ou['Away_Avg_GS'] + df_ou['Home_Avg_GA'])
# 使用标准盘口线2.5计算Goal_Gap
standard_goal_line = 2.5
df_ou['Goal_Gap'] = df_ou['Total_Exp_Goals'] - standard_goal_line
df_ou['Goal_Gap_Percentage'] = df_ou['Goal_Gap'] / standard_goal_line * 100

# 准备训练数据
df_train = df_ou[df_ou['OU_Result'] != 0].copy()

features = [
    'Home_Avg_GS', 'Away_Avg_GS', 'Home_Avg_GA', 'Away_Avg_GA',
    'Home_GS_AtHome', 'Away_GS_AtAway', 
    'Total_Exp_Goals', 'Goal_Gap', 'Goal_Gap_Percentage',
    'Elo_Diff', 'Form_Diff'
]

# 填充缺失值
for col in features:
    if col in df_train.columns:
        df_train[col] = df_train[col].fillna(df_train[col].median())

available_features = [f for f in features if f in df_train.columns]

# 划分训练集和测试集
train_df = df_train[df_train['Season'] != '2024-2025'].copy()
test_df = df_train[df_train['Season'] == '2024-2025'].copy()

train_df['Target'] = (train_df['OU_Result'] == 1).astype(int)
test_df['Target'] = (test_df['OU_Result'] == 1).astype(int)

X_train = train_df[available_features]
y_train = train_df['Target']
X_test = test_df[available_features]
y_test = test_df['Target']

print(f"训练集: {len(train_df)} 场")
print(f"测试集: {len(test_df)} 场")
print(f"特征数: {len(available_features)}")

# 检查类别分布
print(f"\n类别分布:")
print(f"  训练集 - 大球: {y_train.sum()}, 小球: {len(y_train) - y_train.sum()}, 比例: {y_train.mean():.3f}")
print(f"  测试集 - 大球: {y_test.sum()}, 小球: {len(y_test) - y_test.sum()}, 比例: {y_test.mean():.3f}")

# =============================================================================
# 2. 训练模型
# =============================================================================

print("\n[2] 训练模型...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

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

# 使用交叉验证进行概率校准
calibrated = CalibratedClassifierCV(ensemble, method='isotonic', cv=3)
calibrated.fit(X_train_scaled, y_train)

# =============================================================================
# 3. 评估指标计算
# =============================================================================

print("\n[3] 计算评估指标...")

y_pred_proba = calibrated.predict_proba(X_test_scaled)[:, 1]
y_pred = (y_pred_proba > 0.5).astype(int)

# 基础指标
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

# Log Loss (对数损失)
logloss = log_loss(y_test, y_pred_proba)

# Brier Score (概率预测的均方误差)
brier = brier_score_loss(y_test, y_pred_proba)

print("\n" + "=" * 90)
print("评估指标结果")
print("=" * 90)
print(f"\n准确率 (Accuracy):           {accuracy*100:.2f}%")
print(f"ROC-AUC:                     {auc:.4f}")
print(f"Log Loss (对数损失):         {logloss:.4f}")
print(f"Brier Score (概率校准分数):  {brier:.4f}")

# =============================================================================
# 4. 特征重要性分析
# =============================================================================

print("\n[4] 特征重要性分析...")

# 从各个模型中提取特征重要性
feature_importance_dict = {}

# LightGBM特征重要性
lgb_model.fit(X_train_scaled, y_train)
lgb_importance = lgb_model.feature_importances_
for i, feat in enumerate(available_features):
    feature_importance_dict[feat] = feature_importance_dict.get(feat, 0) + lgb_importance[i]

# XGBoost特征重要性
xgb_model.fit(X_train_scaled, y_train)
xgb_importance = xgb_model.feature_importances_
for i, feat in enumerate(available_features):
    feature_importance_dict[feat] = feature_importance_dict.get(feat, 0) + xgb_importance[i]

# Random Forest特征重要性
rf_model.fit(X_train_scaled, y_train)
rf_importance = rf_model.feature_importances_
for i, feat in enumerate(available_features):
    feature_importance_dict[feat] = feature_importance_dict.get(feat, 0) + rf_importance[i]

# 归一化重要性分数
total_importance = sum(feature_importance_dict.values())
feature_importance_normalized = {k: v/total_importance*100 for k, v in feature_importance_dict.items()}

# 按重要性排序
sorted_features = sorted(feature_importance_normalized.items(), key=lambda x: x[1], reverse=True)

print("\n" + "=" * 90)
print("特征重要性排名 (Top 10)")
print("=" * 90)
print(f"\n{'排名':<6} {'特征名称':<30} {'重要性 (%)':>15}")
print("-" * 90)

for idx, (feat, importance) in enumerate(sorted_features[:10], 1):
    print(f"{idx:<6} {feat:<30} {importance:>15.2f}")

# =============================================================================
# 5. 技术亮点总结
# =============================================================================

print("\n" + "=" * 90)
print("技术亮点总结")
print("=" * 90)

print("\n1. 过拟合控制:")
print("   - L1/L2正则化: LightGBM (reg_alpha=0.3, reg_lambda=0.3)")
print("   - 树结构限制: max_depth=4, num_leaves=15, min_child_samples=30")
print("   - 特征采样: XGBoost (subsample=0.6, colsample_bytree=0.6)")
print("   - 交叉验证: CalibratedClassifierCV (cv=3)")

print("\n2. 概率校准:")
print("   - 方法: Isotonic Calibration (保序回归)")
print("   - 目的: 提高概率预测的可靠性，减少概率偏差")
print("   - 效果: Brier Score = {:.4f}".format(brier))

print("\n3. 集成学习:")
print("   - 方法: Voting Classifier (Soft Voting)")
print("   - 模型: LightGBM + XGBoost + Random Forest")
print("   - 优势: 降低方差，提高泛化能力")

print("\n4. 数据不平衡处理:")
class_ratio = y_train.mean()
if class_ratio < 0.4 or class_ratio > 0.6:
    print(f"   - 类别比例: {class_ratio:.3f} (存在轻微不平衡)")
    print("   - 处理方式: 使用概率预测而非硬分类，通过校准提高少数类预测")
else:
    print(f"   - 类别比例: {class_ratio:.3f} (相对平衡)")

print("\n5. 特征工程:")
print("   - 滚动平均: 基于最近N场比赛的进攻/防守统计")
print("   - 预期进球: Total_Exp_Goals = (主队进攻+客队防守) + (客队进攻+主队防守)")
print("   - 预期偏离: Goal_Gap_Percentage = (预期-标准线2.5)/标准线 * 100")
print("   - ELO评分: 基于历史表现的动态评分系统")

# =============================================================================
# 6. 保存结果
# =============================================================================

results = {
    'Metric': ['Accuracy', 'ROC-AUC', 'Log Loss', 'Brier Score'],
    'Value': [accuracy, auc, logloss, brier]
}
results_df = pd.DataFrame(results)
results_df.to_csv('evaluation_metrics.csv', index=False, encoding='utf-8-sig')

importance_df = pd.DataFrame(sorted_features, columns=['Feature', 'Importance_Percentage'])
importance_df.to_csv('feature_importance.csv', index=False, encoding='utf-8-sig')

print("\n结果已保存:")
print("  - evaluation_metrics.csv")
print("  - feature_importance.csv")

print("\n" + "=" * 90)
print("分析完成！")
print("=" * 90)
