# -*- coding: utf-8 -*-
"""
综合预测脚本 - 同时预测亚盘和大小球
自动从历史数据提取球队统计，用户只需提供：
- 球队名称

使用方法：
1. 填写 data/future_matches_template.csv（只需球队名）
2. 运行 python predict_all.py
3. 查看预测结果

注意：本脚本使用纯统计特征进行预测，不依赖任何盘口线或赔率数据。
"""

import pandas as pd
import numpy as np
from scipy.stats import poisson
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

print("=" * 90)
print("英超综合预测系统 - 亚盘 & 大小球")
print("自动从历史数据提取球队统计")
print("=" * 90)


def calculate_confidence(prob_diff, bankroll=1000):
    """
    根据概率差距计算置信度和建议投注金额
    prob_diff: 概率差距（0-1之间）
    bankroll: 本金（默认1000）
    返回: (等级, 星级, 描述, 建议投注金额)
    
    3档置信度：
    - 3星（高）：概率差距≥10%，投注50元
    - 2星（中）：概率差距5-10%，投注30元
    - 1星（低）：概率差距<5%，不投注
    """
    diff_pct = abs(prob_diff) * 100  # 转换为百分比
    
    if diff_pct >= 10:
        return 3, "★★★", "高", 50
    elif diff_pct >= 5:
        return 2, "★★☆", "中", 30
    else:
        return 1, "★☆☆", "低", 0


def show_available_teams(hist_df):
    """显示所有可用的球队名称"""
    all_teams = set(hist_df['Home_Team'].unique()) | set(hist_df['Away_Team'].unique())
    # 按最近出场排序
    recent = hist_df.sort_values('Date', ascending=False)
    recent_teams = []
    for _, row in recent.iterrows():
        if row['Home_Team'] not in recent_teams:
            recent_teams.append(row['Home_Team'])
        if row['Away_Team'] not in recent_teams:
            recent_teams.append(row['Away_Team'])
        if len(recent_teams) >= 25:
            break
    
    print("\n可用球队名称（按最近出场排序）：")
    print("-" * 50)
    for i, team in enumerate(recent_teams[:20], 1):
        print(f"  {i:2}. {team}")
    print(f"\n共 {len(all_teams)} 支球队有历史数据")

# =============================================================================
# 1. 从历史数据提取球队最新统计
# =============================================================================

def get_team_stats(hist_df, n_matches=5):
    """
    从历史数据中提取每支球队的最新统计
    基于最近 n_matches 场比赛计算
    """
    print(f"\n[提取球队统计] 基于最近{n_matches}场比赛...")
    
    # 按日期排序
    hist_df = hist_df.sort_values('Date').reset_index(drop=True)
    
    # 存储每支球队的比赛记录
    team_home_matches = defaultdict(list)  # 主场比赛
    team_away_matches = defaultdict(list)  # 客场比赛
    team_all_matches = defaultdict(list)   # 所有比赛
    
    for _, row in hist_df.iterrows():
        home = row['Home_Team']
        away = row['Away_Team']
        
        # 主场记录
        team_home_matches[home].append({
            'goals_scored': row['FT_Home_Goals'],
            'goals_conceded': row['FT_Away_Goals'],
            'date': row['Date']
        })
        
        # 客场记录
        team_away_matches[away].append({
            'goals_scored': row['FT_Away_Goals'],
            'goals_conceded': row['FT_Home_Goals'],
            'date': row['Date']
        })
        
        # 所有比赛记录
        team_all_matches[home].append({
            'goals_scored': row['FT_Home_Goals'],
            'goals_conceded': row['FT_Away_Goals'],
            'is_home': True,
            'date': row['Date']
        })
        team_all_matches[away].append({
            'goals_scored': row['FT_Away_Goals'],
            'goals_conceded': row['FT_Home_Goals'],
            'is_home': False,
            'date': row['Date']
        })
    
    # 计算每支球队的统计
    team_stats = {}
    
    for team in set(list(team_home_matches.keys()) + list(team_away_matches.keys())):
        # 最近n场总体统计
        recent_all = team_all_matches[team][-n_matches:] if team in team_all_matches else []
        
        # 最近n场主场统计
        recent_home = team_home_matches[team][-n_matches:] if team in team_home_matches else []
        
        # 最近n场客场统计
        recent_away = team_away_matches[team][-n_matches:] if team in team_away_matches else []
        
        stats = {
            # 场均进球/失球（所有比赛）
            'Avg_GS': np.mean([m['goals_scored'] for m in recent_all]) if recent_all else 1.3,
            'Avg_GA': np.mean([m['goals_conceded'] for m in recent_all]) if recent_all else 1.3,
            
            # 主场进球/失球
            'GS_AtHome': np.mean([m['goals_scored'] for m in recent_home]) if recent_home else 1.5,
            'GA_AtHome': np.mean([m['goals_conceded'] for m in recent_home]) if recent_home else 1.0,
            
            # 客场进球/失球
            'GS_AtAway': np.mean([m['goals_scored'] for m in recent_away]) if recent_away else 1.2,
            'GA_AtAway': np.mean([m['goals_conceded'] for m in recent_away]) if recent_away else 1.5,
            
            # 近期状态（最近5场净胜球）
            'Form': sum([m['goals_scored'] - m['goals_conceded'] for m in recent_all]) if recent_all else 0
        }
        
        team_stats[team] = stats
    
    print(f"  已提取 {len(team_stats)} 支球队的统计数据")
    return team_stats


def get_team_elo(hist_df):
    """从历史数据中提取每支球队的最新Elo评分"""
    # 按日期排序，取最后一场比赛的Elo
    hist_df = hist_df.sort_values('Date').reset_index(drop=True)
    
    team_elo = {}
    
    for _, row in hist_df.iterrows():
        if 'Home_Elo' in row and pd.notna(row['Home_Elo']):
            team_elo[row['Home_Team']] = row['Home_Elo']
        if 'Away_Elo' in row and pd.notna(row['Away_Elo']):
            team_elo[row['Away_Team']] = row['Away_Elo']
    
    return team_elo


# =============================================================================
# 2. 数据加载
# =============================================================================

def load_data():
    """加载历史数据和未来比赛数据"""
    data_dir = Path('data')
    
    # 历史特征数据
    hist_file = data_dir / 'all_seasons_features.csv'
    if not hist_file.exists():
        # 尝试样本数据
        hist_file = data_dir / 'sample_data.csv'
    if not hist_file.exists():
        raise FileNotFoundError(f"找不到历史数据: {data_dir / 'all_seasons_features.csv'} 或 {data_dir / 'sample_data.csv'}")
    
    hist_df = pd.read_csv(hist_file)
    print(f"历史数据: {len(hist_df)} 场")
    
    # 未来比赛（简化版）
    future_file = data_dir / 'future_matches_template.csv'
    if not future_file.exists():
        raise FileNotFoundError(f"找不到未来比赛数据: {future_file}")
    
    # 自动检测编码并读取
    future_df = None
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']
    
    for encoding in encodings:
        try:
            temp_df = pd.read_csv(future_file, encoding=encoding)
            # 验证是否能正确读取中文（检查是否包含乱码）
            first_team = str(temp_df['Home_Team'].iloc[0]) if len(temp_df) > 0 else ''
            # 如果包含常见中文字符，说明编码正确
            if any(c in first_team for c in ['阿', '曼', '利', '切', '热', '维', '诺', '布', '水', '富', '狼', '森', '托', '纳', '城', '联', '顿']):
                future_df = temp_df
                break
            # 如果是纯英文或数字也接受
            elif first_team.replace(' ', '').replace('-', '').isalnum():
                future_df = temp_df
                break
        except:
            continue
    
    if future_df is None:
        # 最后尝试gbk（Excel默认中文编码）
        future_df = pd.read_csv(future_file, encoding='gbk')
    
    # 删除空行
    future_df = future_df.dropna(subset=['Home_Team', 'Away_Team'], how='all')
    future_df = future_df[future_df['Home_Team'].notna() & (future_df['Home_Team'] != '')]
    
    print(f"待预测比赛: {len(future_df)} 场")
    
    return hist_df, future_df


# =============================================================================
# 3. 为未来比赛填充统计数据
# =============================================================================

def enrich_future_matches(future_df, team_stats, team_elo):
    """为未来比赛自动填充球队统计数据"""
    
    enriched = future_df.copy()
    
    for idx, row in enriched.iterrows():
        home = row['Home_Team']
        away = row['Away_Team']
        
        # 获取主队统计
        home_stats = team_stats.get(home, {
            'Avg_GS': 1.3, 'Avg_GA': 1.3, 'GS_AtHome': 1.5, 
            'GA_AtHome': 1.0, 'GS_AtAway': 1.2, 'GA_AtAway': 1.5, 'Form': 0
        })
        
        # 获取客队统计
        away_stats = team_stats.get(away, {
            'Avg_GS': 1.3, 'Avg_GA': 1.3, 'GS_AtHome': 1.5,
            'GA_AtHome': 1.0, 'GS_AtAway': 1.2, 'GA_AtAway': 1.5, 'Form': 0
        })
        
        # 填充主队数据
        enriched.loc[idx, 'Home_Avg_GS'] = home_stats['Avg_GS']
        enriched.loc[idx, 'Home_Avg_GA'] = home_stats['Avg_GA']
        enriched.loc[idx, 'Home_GS_AtHome'] = home_stats['GS_AtHome']
        enriched.loc[idx, 'Home_GA_AtHome'] = home_stats['GA_AtHome']
        
        # 填充客队数据
        enriched.loc[idx, 'Away_Avg_GS'] = away_stats['Avg_GS']
        enriched.loc[idx, 'Away_Avg_GA'] = away_stats['Avg_GA']
        enriched.loc[idx, 'Away_GS_AtAway'] = away_stats['GS_AtAway']
        enriched.loc[idx, 'Away_GA_AtAway'] = away_stats['GA_AtAway']
        
        # Elo评分
        enriched.loc[idx, 'Home_Elo'] = team_elo.get(home, 1500)
        enriched.loc[idx, 'Away_Elo'] = team_elo.get(away, 1500)
        
        # 状态差
        enriched.loc[idx, 'Form_Diff'] = home_stats['Form'] - away_stats['Form']
    
    return enriched


# =============================================================================
# 4. 特征工程
# =============================================================================

def add_features(df):
    """添加预测所需特征（移除所有盘口线和赔率相关特征）"""
    df = df.copy()
    
    # Elo差值
    if 'Home_Elo' in df.columns and 'Away_Elo' in df.columns:
        df['Elo_Diff'] = df['Home_Elo'] - df['Away_Elo']
    
    # 进攻防守差异
    if 'Home_Avg_GS' in df.columns:
        df['Home_Strength'] = df['Home_Avg_GS'] - df['Home_Avg_GA']
        df['Away_Strength'] = df['Away_Avg_GS'] - df['Away_Avg_GA']
        df['Strength_Diff'] = df['Home_Strength'] - df['Away_Strength']
        df['Attack_Diff'] = df['Home_Avg_GS'] - df['Away_Avg_GS']
        df['Defense_Diff'] = df['Away_Avg_GA'] - df['Home_Avg_GA']
    
    # 预期总进球
    if 'Home_Avg_GS' in df.columns:
        df['Total_Exp_Goals'] = (df['Home_Avg_GS'] + df['Away_Avg_GA']) + \
                                 (df['Away_Avg_GS'] + df['Home_Avg_GA'])
    
    # 使用标准盘口线2.5计算Goal_Gap（静态化处理）
    standard_goal_line = 2.5
    df['Goal_Gap'] = df['Total_Exp_Goals'] - standard_goal_line
    df['Goal_Gap_Percentage'] = df['Goal_Gap'] / standard_goal_line * 100
    
    # Elo隐含概率（基于Elo差值）
    if 'Elo_Diff' in df.columns:
        df['Elo_Implied_Prob_Home'] = 1 / (1 + 10 ** (-df['Elo_Diff'] / 400))
    
    return df


def calc_poisson_probs(df):
    """计算泊松分布概率"""
    probs = []
    for _, row in df.iterrows():
        home_exp = (row['Home_Avg_GS'] + row['Away_Avg_GA']) / 2
        away_exp = (row['Away_Avg_GS'] + row['Home_Avg_GA']) / 2
        total_exp = home_exp + away_exp
        
        p_ge_2 = 1 - poisson.cdf(1, total_exp)
        p_ge_3 = 1 - poisson.cdf(2, total_exp)
        p_ge_4 = 1 - poisson.cdf(3, total_exp)
        p_le_2 = poisson.cdf(2, total_exp)
        p_le_3 = poisson.cdf(3, total_exp)
        
        btts = (1 - poisson.pmf(0, home_exp)) * (1 - poisson.pmf(0, away_exp))
        
        probs.append({
            'P_GE_2': p_ge_2,
            'P_GE_3': p_ge_3,
            'P_GE_4': p_ge_4,
            'P_LE_2': p_le_2,
            'P_LE_3': p_le_3,
            'BTTS_Rate': btts
        })
    
    return pd.DataFrame(probs)


# =============================================================================
# 5. 模型训练
# =============================================================================

def train_ah_model(hist_df):
    """训练亚盘预测模型（移除盘口线相关特征）"""
    print("\n[训练亚盘模型...]")
    
    df = add_features(hist_df)
    
    # 移除所有盘口线和赔率相关特征
    feature_cols = [
        'Elo_Diff', 'Home_Strength', 'Away_Strength', 'Strength_Diff',
        'Attack_Diff', 'Defense_Diff', 'Form_Diff',
        'Elo_Implied_Prob_Home',
        'Home_Avg_GS', 'Home_Avg_GA', 'Away_Avg_GS', 'Away_Avg_GA',
        'Home_GS_AtHome', 'Home_GA_AtHome', 'Away_GS_AtAway', 'Away_GA_AtAway'
    ]
    
    available_cols = [c for c in feature_cols if c in df.columns]
    
    train_df = df[df['AH_Result'] != 0].copy()
    for col in available_cols:
        train_df[col] = train_df[col].fillna(train_df[col].median())
    
    X_train = train_df[available_cols]
    y_train = (train_df['AH_Result'] + 1) // 2
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    model = lgb.LGBMClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.03,
        num_leaves=20, min_child_samples=25,
        random_state=42, verbose=-1
    )
    
    calibrated = CalibratedClassifierCV(model, method='isotonic', cv=3)
    calibrated.fit(X_scaled, y_train)
    
    print(f"  特征数: {len(available_cols)}")
    
    return calibrated, scaler, available_cols


def train_ou_model(hist_df):
    """训练大小球预测模型（移除盘口线相关特征）"""
    print("\n[训练大小球模型...]")
    
    df = add_features(hist_df)
    prob_df = calc_poisson_probs(df)
    df = pd.concat([df, prob_df], axis=1)
    
    # 移除所有盘口线相关特征
    feature_cols = [
        'Home_Avg_GS', 'Away_Avg_GS', 'Home_Avg_GA', 'Away_Avg_GA',
        'Total_Exp_Goals', 'Goal_Gap', 'Goal_Gap_Percentage',
        'P_GE_2', 'P_GE_3', 'P_GE_4', 'P_LE_2', 'P_LE_3', 'BTTS_Rate',
        'Elo_Diff', 'Form_Diff'
    ]
    
    available_cols = [c for c in feature_cols if c in df.columns]
    
    train_df = df[df['OU_Result'] != 0].copy()
    for col in available_cols:
        train_df[col] = train_df[col].fillna(train_df[col].median())
    
    X_train = train_df[available_cols]
    y_train = (train_df['OU_Result'] == 1).astype(int)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    lgb_model = lgb.LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.02,
                                    num_leaves=15, random_state=42, verbose=-1)
    xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.02,
                                   random_state=42, verbosity=0)
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    
    ensemble = VotingClassifier(
        estimators=[('lgb', lgb_model), ('xgb', xgb_model), ('rf', rf_model)],
        voting='soft'
    )
    
    calibrated = CalibratedClassifierCV(ensemble, method='isotonic', cv=3)
    calibrated.fit(X_scaled, y_train)
    
    print(f"  特征数: {len(available_cols)}")
    
    return calibrated, scaler, available_cols


# =============================================================================
# 6. 预测函数
# =============================================================================

def predict_matches(future_df, ah_model, ah_scaler, ah_features, 
                    ou_model, ou_scaler, ou_features):
    """预测未来比赛"""
    
    df = add_features(future_df)
    prob_df = calc_poisson_probs(df)
    df = pd.concat([df.reset_index(drop=True), prob_df.reset_index(drop=True)], axis=1)
    
    predictions = []
    
    for idx, row in df.iterrows():
        pred = {
            'Home': row['Home_Team'],
            'Away': row['Away_Team'],
            'Date': row.get('Date', 'N/A')
        }
        
        # ============ 亚盘预测 ============
        try:
            ah_data = {col: row.get(col, 0) for col in ah_features}
            ah_X = pd.DataFrame([ah_data])[ah_features].fillna(0)
            ah_X_scaled = ah_scaler.transform(ah_X)
            
            ah_prob = ah_model.predict_proba(ah_X_scaled)[0][1]
            pred['AH_Home_Prob'] = ah_prob
            pred['AH_Away_Prob'] = 1 - ah_prob
            
            # 基于概率的建议（移除盘口线相关逻辑）
            if ah_prob > 0.55:
                pred['AH_Recommendation'] = '主队'
                pred['AH_Strategy'] = '概率优势'
            elif ah_prob < 0.45:
                pred['AH_Recommendation'] = '客队'
                pred['AH_Strategy'] = '概率优势'
            else:
                pred['AH_Recommendation'] = '观望'
                pred['AH_Strategy'] = '-'
            
            pred['AH_Value'] = '-'  # 移除价值分析
            
            # 计算亚盘置信度（基于1000本金）
            ah_prob_diff = abs(ah_prob - 0.5)
            ah_conf_level, ah_conf_stars, ah_conf_desc, ah_bet_amount = calculate_confidence(ah_prob_diff * 2, bankroll=1000)
            pred['AH_Confidence_Level'] = ah_conf_level
            pred['AH_Confidence_Stars'] = ah_conf_stars
            pred['AH_Confidence_Desc'] = ah_conf_desc
            pred['AH_Bet_Amount'] = ah_bet_amount
                
        except Exception as e:
            pred['AH_Recommendation'] = f'错误'
            pred['AH_Home_Prob'] = 0.5
            pred['AH_Away_Prob'] = 0.5
            pred['AH_Value'] = '-'
            pred['AH_Confidence_Level'] = 1
            pred['AH_Confidence_Stars'] = '★☆☆☆☆'
            pred['AH_Confidence_Desc'] = '观望'
            pred['AH_Bet_Amount'] = 0
        
        # ============ 大小球预测 ============
        try:
            ou_data = {col: row.get(col, 0) for col in ou_features}
            ou_X = pd.DataFrame([ou_data])[ou_features].fillna(0)
            ou_X_scaled = ou_scaler.transform(ou_X)
            
            ou_prob = ou_model.predict_proba(ou_X_scaled)[0][1]
            
            # 使用标准盘口线2.5（静态化）
            standard_goal_line = 2.5
            pred['OU_Line'] = standard_goal_line
            pred['OU_Over_Prob'] = ou_prob
            pred['OU_Under_Prob'] = 1 - ou_prob
            pred['Exp_Goals'] = row.get('Total_Exp_Goals', 2.5)
            pred['P_GE_2'] = row['P_GE_2']
            pred['P_GE_3'] = row['P_GE_3']
            pred['BTTS'] = row['BTTS_Rate']
            
            # 移除等盘建议（因为不再有盘口变动）
            pred['OU_Wait_Rec'] = '-'
            
            # 综合建议（基于概率，移除盘口线相关策略）
            if ou_prob > 0.55:
                pred['OU_Recommendation'] = '考虑大球'
                pred['OU_Strategy'] = '概率优势'
            elif ou_prob < 0.45:
                pred['OU_Recommendation'] = '考虑小球'
                pred['OU_Strategy'] = '概率优势'
            else:
                pred['OU_Recommendation'] = '观望'
                pred['OU_Strategy'] = '-'
            
            # 计算大小球置信度（基于1000本金）
            ou_prob_diff = abs(ou_prob - 0.5)
            ou_conf_level, ou_conf_stars, ou_conf_desc, ou_bet_amount = calculate_confidence(ou_prob_diff * 2, bankroll=1000)
            pred['OU_Confidence_Level'] = ou_conf_level
            pred['OU_Confidence_Stars'] = ou_conf_stars
            pred['OU_Confidence_Desc'] = ou_conf_desc
            pred['OU_Bet_Amount'] = ou_bet_amount
            
        except Exception as e:
            pred['OU_Recommendation'] = f'错误'
            pred['OU_Over_Prob'] = 0.5
            pred['OU_Under_Prob'] = 0.5
            pred['OU_Wait_Rec'] = '-'
            pred['OU_Confidence_Level'] = 1
            pred['OU_Confidence_Stars'] = '★☆☆☆☆'
            pred['OU_Confidence_Desc'] = '观望'
            pred['OU_Bet_Amount'] = 0
        
        predictions.append(pred)
    
    return predictions


# =============================================================================
# 7. 输出结果
# =============================================================================

def print_predictions(predictions):
    """打印预测结果"""
    
    print("\n" + "=" * 90)
    print("预测结果")
    print("=" * 90)
    
    for pred in predictions:
        print(f"\n{'='*70}")
        print(f"  {pred['Home']} vs {pred['Away']}  ({pred['Date']})")
        print(f"{'='*70}")
        
        print(f"\n  【亚盘预测】")
        print(f"    主队概率: {pred['AH_Home_Prob']*100:.1f}%")
        print(f"    客队概率: {pred['AH_Away_Prob']*100:.1f}%")
        print(f"    建议: {pred['AH_Recommendation']}")
        if pred.get('AH_Strategy', '-') != '-':
            print(f"    策略: {pred.get('AH_Strategy', '-')}")
        ah_bet = pred.get('AH_Bet_Amount', 0)
        if ah_bet > 0:
            print(f"    置信度: {pred.get('AH_Confidence_Stars', '★☆☆☆☆')} ({pred.get('AH_Confidence_Desc', '观望')}) - 建议投注: {ah_bet}元")
        else:
            print(f"    置信度: {pred.get('AH_Confidence_Stars', '★☆☆☆☆')} ({pred.get('AH_Confidence_Desc', '观望')}) - 不建议投注")
        
        print(f"\n  【大小球预测】")
        print(f"    预期进球: {pred.get('Exp_Goals', 'N/A'):.2f}")
        print(f"    大球概率: {pred['OU_Over_Prob']*100:.1f}%")
        print(f"    小球概率: {pred['OU_Under_Prob']*100:.1f}%")
        print(f"    P(>=2球): {pred.get('P_GE_2', 0)*100:.1f}%")
        print(f"    P(>=3球): {pred.get('P_GE_3', 0)*100:.1f}%")
        print(f"    BTTS: {pred.get('BTTS', 0)*100:.1f}%")
        print(f"    建议: {pred['OU_Recommendation']}")
        print(f"    策略: {pred['OU_Strategy']}")
        ou_bet = pred.get('OU_Bet_Amount', 0)
        if ou_bet > 0:
            print(f"    置信度: {pred.get('OU_Confidence_Stars', '★☆☆☆☆')} ({pred.get('OU_Confidence_Desc', '观望')}) - 建议投注: {ou_bet}元")
        else:
            print(f"    置信度: {pred.get('OU_Confidence_Stars', '★☆☆☆☆')} ({pred.get('OU_Confidence_Desc', '观望')}) - 不建议投注")
    
    # 汇总表
    print("\n" + "=" * 100)
    print("预测汇总")
    print("=" * 100)
    print(f"\n{'对阵':<18} {'亚盘建议':>8} {'置信度':>6} {'投注':>5} {'大小球建议':>10} {'置信度':>6} {'投注':>5}")
    print("-" * 100)
    for pred in predictions:
        match = f"{pred['Home'][:6]}vs{pred['Away'][:6]}"
        ah_level = pred.get('AH_Confidence_Level', 1)
        ou_level = pred.get('OU_Confidence_Level', 1)
        ah_bet = pred.get('AH_Bet_Amount', 0)
        ou_bet = pred.get('OU_Bet_Amount', 0)
        ah_bet_str = f"{ah_bet}元" if ah_bet > 0 else "不投"
        ou_bet_str = f"{ou_bet}元" if ou_bet > 0 else "不投"
        print(f"{match:<18} {pred['AH_Recommendation']:>8} {ah_level}星 {ah_bet_str:>6} "
              f"{pred['OU_Recommendation']:>10} {ou_level}星 {ou_bet_str:>6}")


def save_predictions(predictions):
    """保存预测结果"""
    df = pd.DataFrame(predictions)
    df.to_csv('prediction_results.csv', index=False, encoding='utf-8-sig')
    print(f"\n预测结果已保存: prediction_results.csv")


# =============================================================================
# 主程序
# =============================================================================

if __name__ == '__main__':
    try:
        # 1. 加载数据
        hist_df, future_df = load_data()
        
        # 显示可用球队名称
        show_available_teams(hist_df)
        
        # 检查球队名称是否匹配
        all_teams = set(hist_df['Home_Team'].unique()) | set(hist_df['Away_Team'].unique())
        for _, row in future_df.iterrows():
            if row['Home_Team'] not in all_teams:
                print(f"\n[警告] 主队名称不匹配: '{row['Home_Team']}' - 将使用默认统计")
            if row['Away_Team'] not in all_teams:
                print(f"\n[警告] 客队名称不匹配: '{row['Away_Team']}' - 将使用默认统计")
        
        # 2. 提取球队统计
        team_stats = get_team_stats(hist_df, n_matches=5)
        team_elo = get_team_elo(hist_df)
        
        # 3. 为未来比赛填充统计数据
        print("\n[填充球队统计数据...]")
        future_enriched = enrich_future_matches(future_df, team_stats, team_elo)
        
        # 显示填充后的数据
        print("\n填充后的球队统计:")
        print("-" * 60)
        for _, row in future_enriched.iterrows():
            print(f"{row['Home_Team']}: 进{row['Home_Avg_GS']:.2f}/失{row['Home_Avg_GA']:.2f} (Elo:{row['Home_Elo']:.0f})")
            print(f"{row['Away_Team']}: 进{row['Away_Avg_GS']:.2f}/失{row['Away_Avg_GA']:.2f} (Elo:{row['Away_Elo']:.0f})")
            print()
        
        # 4. 训练模型
        ah_model, ah_scaler, ah_features = train_ah_model(hist_df)
        ou_model, ou_scaler, ou_features = train_ou_model(hist_df)
        
        # 5. 预测
        predictions = predict_matches(future_enriched, ah_model, ah_scaler, ah_features,
                                       ou_model, ou_scaler, ou_features)
        
        # 6. 输出结果
        print_predictions(predictions)
        save_predictions(predictions)
        
        print("\n" + "=" * 90)
        print("预测完成！")
        print("=" * 90)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
