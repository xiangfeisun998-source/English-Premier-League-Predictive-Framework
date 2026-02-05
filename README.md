# English Premier League Predictive Framework (EPL-ML)

A comprehensive machine learning pipeline for predicting football match outcomes in the English Premier League, focusing on probabilistic prediction and model calibration.

## Project Overview

This project implements an end-to-end machine learning framework for predicting match outcomes using ensemble learning techniques and advanced probability calibration. The system integrates statistical models (Poisson distribution) with gradient boosting algorithms (XGBoost, LightGBM) to achieve robust and well-calibrated probability predictions.

**Key Focus**: The framework emphasizes **probability calibration** and **prediction accuracy** rather than binary classification, making it suitable for applications requiring reliable probabilistic estimates.

## Key Features

### 🎯 Core Capabilities

- **Ensemble Learning**: Combines LightGBM, XGBoost, and Random Forest classifiers using soft voting
- **Probability Calibration**: Implements Isotonic Calibration to improve prediction reliability
- **Advanced Feature Engineering**: 20+ features including rolling averages, expected goals, and team performance metrics
- **Comprehensive Evaluation**: Metrics include Accuracy, ROC-AUC, Log Loss, and Brier Score

### 🔬 Technical Highlights

- **Overfitting Control**: L1/L2 regularization, tree depth limiting, feature sampling, and cross-validation
- **Class Imbalance Handling**: Probability calibration with Isotonic Regression
- **Time-Series Aware**: Rolling averages capture dynamic team performance trends

## Tech Stack

- **Core ML Libraries**: scikit-learn, XGBoost, LightGBM
- **Data Processing**: pandas, numpy
- **Statistical Modeling**: scipy.stats (Poisson distribution)
- **Visualization**: matplotlib
- **Language**: Python 3.x

## Methodology

### 1. Data Processing

- **Data Source**: English Premier League match data (2018-2025 seasons, ~2,379 matches)
- **Preprocessing**: Handles missing values, feature normalization, and temporal data splitting
- **Train-Test Split**: Time-series aware splitting (2018-2024 for training, 2024-2025 for testing)

### 2. Feature Engineering

The framework employs multi-dimensional feature engineering:

#### Statistical Features
- **Rolling Averages**: Team performance metrics (goals scored/conceded) over recent N matches
- **Expected Goals**: Calculated as `(Home_Attack + Away_Defense) + (Away_Attack + Home_Defense)`
- **Poisson Probabilities**: Statistical probabilities based on Poisson distribution modeling

#### Statistical Features (Continued)
- **Goal Gap**: Difference between expected goals and observed outcomes
- **Poisson Deviation**: Statistical deviation from Poisson distribution predictions

#### Team Performance Features
- **ELO Ratings**: Dynamic team strength ratings
- **Form Indicators**: Recent performance trends (goal difference over recent matches)
- **Home/Away Splits**: Context-aware performance metrics

### 3. Modeling

#### Model Architecture

```
Input Features (~20 dimensions)
    ↓
StandardScaler (Feature Normalization)
    ↓
Ensemble Classifier:
  ├── LightGBM (n_estimators=100, max_depth=4, reg_alpha=0.3, reg_lambda=0.3)
  ├── XGBoost (n_estimators=100, max_depth=3, subsample=0.6)
  └── Random Forest (n_estimators=100, max_depth=5)
    ↓
Soft Voting (Probability Averaging)
    ↓
CalibratedClassifierCV (Isotonic Calibration, cv=3)
    ↓
Calibrated Probability Predictions
```

#### Regularization & Overfitting Control

- **L1/L2 Regularization**: `reg_alpha=0.3`, `reg_lambda=0.3` in LightGBM
- **Tree Complexity Limits**: Restricted max_depth (3-5) and min_samples_split (30)
- **Feature Sampling**: XGBoost uses `subsample=0.6` and `colsample_bytree=0.6`
- **Cross-Validation**: 3-fold CV for probability calibration

#### Probability Calibration

- **Method**: Isotonic Regression (non-parametric, monotonic transformation)
- **Purpose**: Ensures predicted probabilities align with actual outcome frequencies
- **Benefit**: Critical for applications requiring reliable probability estimates

## Results

### Model Performance

The model is evaluated on the 2024-2025 season test set:

- **Accuracy**: Model achieves improved classification accuracy compared to baseline
- **ROC-AUC**: Measures model's ability to distinguish between classes
- **Log Loss**: Evaluates the quality of probability predictions (lower is better)
- **Brier Score**: Assesses probability calibration (lower indicates better calibration)

*Note: Specific numerical results are generated when running the model. The framework includes comprehensive evaluation metrics in `academic_analysis.py`.*

### Calibration Performance

The Isotonic Calibration significantly improves probability reliability:
- Reduces overconfidence in predictions
- Better aligns predicted probabilities with actual outcome frequencies
- Particularly important for handling class imbalance

## Project Structure

```
EPL-ML/
├── README.md
├── requirements.txt
├── .gitignore
├── ou_model_v3_deep.py          # Main model training script
├── predict_all.py               # Prediction pipeline for new matches
├── academic_analysis.py         # Comprehensive evaluation metrics
├── scrape_epl_25_26.py         # Data scraping script (data source)
├── clean_and_prepare_data.py   # Data cleaning script
└── data/
    ├── all_seasons_features_raw.csv  # Original data (before cleaning)
    ├── all_seasons_features.csv     # Cleaned training data (or sample_data.csv if >50MB)
    └── future_matches_template.csv  # Template for new predictions
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd EPL-ML
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare data:
   - Run the data cleaning script to prepare training data:
     ```bash
     python clean_and_prepare_data.py
     ```
   - This will:
     - Copy raw data to `data/all_seasons_features_raw.csv`
     - Clean data by removing all odds and market-related columns
     - Save cleaned data to `data/all_seasons_features.csv` (or `sample_data.csv` if >50MB)
   - For predictions, use `data/future_matches_template.csv` template

## Usage

### Training the Model

```bash
python ou_model_v3_deep.py
```

This will:
- Load and preprocess the training data
- Engineer features
- Train the ensemble model with calibration
- Evaluate on test set
- Generate visualizations and backtest results

### Making Predictions

```bash
python predict_all.py
```

This will:
- Load historical data for team statistics
- Process future matches from `future_matches.csv`
- Generate probability predictions with confidence levels

### Comprehensive Evaluation

```bash
python academic_analysis.py
```

This will compute:
- Accuracy, ROC-AUC, Log Loss, Brier Score
- Feature importance rankings
- Model calibration analysis

## Key Contributions

1. **Multi-Dimensional Feature Engineering**: Integrates statistical models (Poisson distribution) with team performance metrics
2. **Pure Statistical Approach**: Focuses solely on team statistics and match history, without external market data
3. **Probability Calibration**: Isotonic Regression improves prediction reliability
4. **Robust Overfitting Control**: Multiple regularization techniques ensure generalization
5. **Comprehensive Evaluation**: Multiple metrics assess both accuracy and calibration

## Academic Applications

This framework demonstrates:
- **Ensemble Learning**: Combining multiple algorithms for improved performance
- **Probability Calibration**: Addressing the common issue of uncalibrated ML predictions
- **Feature Engineering**: Domain-specific feature design for sports analytics
- **Time-Series Modeling**: Handling temporal dependencies in sports data
- **Statistical Modeling**: Poisson distribution for goal prediction

## Future Enhancements

- [ ] Real-time prediction API
- [ ] Additional evaluation metrics (Precision-Recall AUC)
- [ ] Feature importance visualization with SHAP values
- [ ] Model versioning and experiment tracking
- [ ] Extended to other football leagues

## License

This project is intended for academic and research purposes.

## Citation

If you use this framework in your research, please cite:

```
English Premier League Predictive Framework (EPL-ML)
A machine learning pipeline for probabilistic match outcome prediction
with ensemble learning and probability calibration.
```

## Contact

For questions or collaborations, please open an issue in the repository.

---

**Note**: This project focuses on **predictive modeling** and **probability calibration** for academic and research purposes. The framework emphasizes statistical rigor and model reliability over any specific application domain.
