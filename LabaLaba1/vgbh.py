import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Union, Optional
import warnings

# Sklearn (для сравнения и вспомогательных функций)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

# ==============================================
# КОНСТАНТЫ
# ==============================================

EPSILON = 1e-8
TARGET_COL = "RiskScore"
ID_COL = "ID"
SEED = 42

# Оптимизированные диапазоны для Grid Search
# Фокус на низкой регуляризации и агрессивном отборе признаков.
ALPHAS_TO_TEST = [0.001, 0.01, 0.1, 0.5, 1.0, 2.5]
PERCENTILES_TO_TEST = [30, 50, 70, 90]


# ==============================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==============================================

def safe_log_transform(x: np.ndarray) -> np.ndarray:
    """Безопасное логарифмическое преобразование."""
    return np.sign(x) * np.log1p(np.abs(x))


def robust_clip(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Отсечение выбросов."""
    return np.clip(values, lower, upper)


# ==============================================
# КАСТОМНЫЕ МЕТРИКИ (0.5p x 4)
# ==============================================

class ModelMetrics:
    @staticmethod
    def compute_mse(actual: np.ndarray, predicted: np.ndarray) -> float:
        return np.mean((actual - predicted) ** 2)

    @staticmethod
    def compute_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
        return np.mean(np.abs(actual - predicted))

    @staticmethod
    def compute_r2(actual: np.ndarray, predicted: np.ndarray) -> float:
        ss_residual = np.sum((actual - predicted) ** 2)
        ss_total = np.sum((actual - np.mean(actual)) ** 2)
        return 1 - ss_residual / (ss_total + EPSILON)

    @staticmethod
    def compute_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
        denominator = np.abs(actual) + EPSILON
        return np.mean(np.abs((actual - predicted) / denominator)) * 100


# ==============================================
# КАСТОМНАЯ РЕГРЕССИЯ (3p + Доп. баллы за L2)
# ==============================================

class CustomRegressor:
    """
    Линейная регрессия с L2-регуляризацией (Ridge) через аналитическое решение.
    """

    def __init__(self, reg_strength: float = 1.0):
        self.reg_strength = reg_strength
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        # Добавляем bias (единичный столбец)
        X_b = np.c_[np.ones((n_samples, 1)), X]

        # Матрица регуляризации (L2)
        I = np.eye(n_features + 1)
        I[0, 0] = 0  # Не штрафуем bias

        # Аналитическое решение: w = (X^T X + alpha*I)^-1 X^T y
        try:
            theta = np.linalg.solve(
                X_b.T @ X_b + self.reg_strength * I,
                X_b.T @ y
            )
        except np.linalg.LinAlgError:
            theta = np.linalg.pinv(X_b.T @ X_b + self.reg_strength * I) @ X_b.T @ y

        self.bias = theta[0]
        self.weights = theta[1:]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights + self.bias


# ==============================================
# ИНЖЕНЕРИЯ ПРИЗНАКОВ
# ==============================================

class FeatureEngineer:
    @staticmethod
    def add_bins(df: pd.DataFrame, column: str, bins: list) -> None:
        if column not in df.columns:
            return
        labels = [f"{i}" for i in range(len(bins) - 1)]
        df[f"{column}_Bin"] = pd.cut(
            df[column], bins=bins, labels=labels, include_lowest=True
        ).astype(str)

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # 1. Даты и Образование
        if "ApplicationDate" in df.columns:
            parsed = pd.to_datetime(df["ApplicationDate"], errors="coerce")
            df["App_Year"] = parsed.dt.year
            df["App_Month"] = parsed.dt.month
            df.drop(columns=["ApplicationDate"], inplace=True)

        edu_map = {'High School': 1, 'Associate': 2, 'Bachelor': 3, 'Master': 4, 'Doctorate': 5}
        if 'EducationLevel' in df.columns:
            df['EducationLevel'] = df['EducationLevel'].map(edu_map).fillna(1)

        # 2. Финансовые фичи (Усиленные)
        df["InterestRateSpread"] = df["InterestRate"] - df["BaseInterestRate"]
        df["LoanToIncome"] = df["LoanAmount"] / (df["AnnualIncome"] + EPSILON)

        total_debt = df["MonthlyLoanPayment"] + df["MonthlyDebtPayments"]
        df["DebtServiceRatio"] = total_debt / (df["MonthlyIncome"] + EPSILON)
        df["DisposableIncome"] = df["MonthlyIncome"] - total_debt
        df["LiabilityGap"] = df["TotalLiabilities"] - df["TotalAssets"]
        df["UtilizationPerLine"] = df["CreditCardUtilizationRate"] / (
                    df["NumberOfOpenCreditLines"] + 1)

        # 3. Биннинг
        self.add_bins(df, "CreditScore", [0, 580, 670, 740, 800, 850, 999])
        self.add_bins(df, "Age", [0, 25, 35, 45, 60, 100])
        self.add_bins(df, "AnnualIncome", [0, 40000, 80000, 120000, 1e9])
        self.add_bins(df, "LoanAmount", [0, 10000, 30000, 60000, 1e9])

        return df


# ==============================================
# ПАЙПЛАЙН
# ==============================================

class RegressionPipeline:
    def __init__(self, alpha=1.0, percentile=90):
        self.alpha = alpha
        self.percentile = percentile

        self.imputer_vals = None
        self.cat_modes = None
        self.scaler = None
        self.poly = None
        self.ohe = None
        self.selector = None
        self.model = None
        self.numeric_cols = None
        self.cat_cols = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

        # 1. Импутация
        self.imputer_vals = X[self.numeric_cols].median()
        self.cat_modes = X[self.cat_cols].mode().iloc[0]

        X_filled = X.copy()
        X_filled[self.numeric_cols] = X_filled[self.numeric_cols].fillna(self.imputer_vals)
        X_filled[self.cat_cols] = X_filled[self.cat_cols].fillna(self.cat_modes)

        # 2. Лог-трансформация
        X_log = X_filled.copy()
        X_log[self.numeric_cols] = safe_log_transform(X_log[self.numeric_cols].values)

        # 3. Полиномиальные признаки
        self.poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = self.poly.fit_transform(X_log[self.numeric_cols])

        # Стандартизация (Z-score)
        self.scaler = StandardScaler()
        X_poly_scaled = self.scaler.fit_transform(X_poly)

        # 4. One-Hot Encoding
        self.ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        X_cat_encoded = self.ohe.fit_transform(X_log[self.cat_cols])

        # Сборка
        X_final = np.hstack([X_poly_scaled, X_cat_encoded])

        # 5. Отбор признаков (SelectPercentile)
        self.selector = SelectPercentile(score_func=f_regression, percentile=self.percentile)
        X_selected = self.selector.fit_transform(X_final, y)

        # 6. Обучение
        self.model = CustomRegressor(reg_strength=self.alpha)
        self.model.fit(X_selected, y.values)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Повторяем те же трансформации
        X_filled = X.copy()
        X_filled[self.numeric_cols] = X_filled[self.numeric_cols].fillna(self.imputer_vals)
        X_filled[self.cat_cols] = X_filled[self.cat_cols].fillna(self.cat_modes)

        X_log = X_filled.copy()
        X_log[self.numeric_cols] = safe_log_transform(X_log[self.numeric_cols].values)

        X_poly = self.poly.transform(X_log[self.numeric_cols])
        X_poly_scaled = self.scaler.transform(X_poly)

        X_cat_encoded = self.ohe.transform(X_log[self.cat_cols])
        X_final = np.hstack([X_poly_scaled, X_cat_encoded])
        X_selected = self.selector.transform(X_final)

        return self.model.predict(X_selected)


# ==============================================
# MAIN (С ГРИД-СЕРЧЕМ и CV)
# ==============================================

def main():
    # --- НАСТРОЙКИ ПУТЕЙ (Проверь их перед запуском!) ---
    TRAIN_PATH = "D:/Laba ML/LabaLaba1/train.csv"
    TEST_PATH = "D:/Laba ML/LabaLaba1/test.csv"
    OUTPUT_PATH = "D:/Laba ML/LabaLaba1/submission.csv"

    print("1. Loading Data...")
    try:
        train_df = pd.read_csv(TRAIN_PATH)
        test_df = pd.read_csv(TEST_PATH)
    except FileNotFoundError:
        print("ОШИБКА: Файлы не найдены. Проверь пути TRAIN_PATH и TEST_PATH.")
        return

    # --- ПРЕПРОЦЕССИНГ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ ---
    print("2. Cleaning Target Variable...")
    train_df = train_df.dropna(subset=[TARGET_COL])

    # !!! КРИТИЧЕСКОЕ ИЗМЕНЕНИЕ ДЛЯ MSE < 25.0 !!!
    # Строго фильтруем целевую переменную (RiskScore = 0-100)
    initial_len = len(train_df)
    train_df = train_df[(train_df[TARGET_COL] >= 0) & (train_df[TARGET_COL] <= 100)]
    print(f"   Removed {initial_len - len(train_df)} outlier rows from training data.")

    # --- ИНЖЕНЕРИЯ ---
    print("3. Engineering Features...")
    engineer = FeatureEngineer()
    train_df = engineer.process(train_df)
    test_df = engineer.process(test_df)

    # Разделение
    X = train_df.drop(columns=[TARGET_COL, ID_COL], errors='ignore')
    y = train_df[TARGET_COL]

    if ID_COL in test_df.columns:
        test_ids = test_df[ID_COL]
        X_test = test_df.drop(columns=[ID_COL], errors='ignore')
    else:
        test_ids = pd.Series(range(len(test_df)))
        X_test = test_df

    # --- GRID SEARCH (КРОСС-ВАЛИДАЦИЯ И ПОИСК ПАРАМЕТРОВ) ---
    print("\n" + "=" * 50)
    print("4. STARTING GRID SEARCH (K-FOLD) FOR BEST PARAMETERS")
    print("=" * 50)

    best_mse = float('inf')
    best_params = {}

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    # Перебор параметров
    for alpha in ALPHAS_TO_TEST:
        for pct in PERCENTILES_TO_TEST:
            mse_scores = []

            # Кросс-валидация
            for train_idx, val_idx in kf.split(X):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

                pipeline = RegressionPipeline(alpha=alpha, percentile=pct)
                pipeline.fit(X_tr, y_tr)

                preds = pipeline.predict(X_val)
                preds = robust_clip(preds, 0, 100)

                mse_scores.append(ModelMetrics.compute_mse(y_val.values, preds))

            mean_mse = np.mean(mse_scores)
            print(f"Alpha: {alpha:<5} | Percentile: {pct:<4} | MSE: {mean_mse:.4f}")

            if mean_mse < best_mse:
                best_mse = mean_mse
                best_params = {'alpha': alpha, 'percentile': pct}

    print("\n" + "=" * 50)
    print(f"BEST RESULT: MSE {best_mse:.4f}")
    print(f"BEST PARAMS: {best_params}")
    print("=" * 50)

    # --- МЕТРИКИ (ДЛЯ ОТЧЕТА) ---
    print("\n5. Calculating All Metrics (on validation split)...")
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)

    final_pipeline = RegressionPipeline(**best_params)
    final_pipeline.fit(X_tr, y_tr)
    val_preds = robust_clip(final_pipeline.predict(X_val), 0, 100)

    print(f"MSE  (Custom): {ModelMetrics.compute_mse(y_val.values, val_preds):.4f}")
    print(f"MAE  (Custom): {ModelMetrics.compute_mae(y_val.values, val_preds):.4f}")
    print(f"R2   (Custom): {ModelMetrics.compute_r2(y_val.values, val_preds):.4f}")
    print(f"MAPE (Custom): {ModelMetrics.compute_mape(y_val.values, val_preds):.4f}")

    # --- ФИНАЛЬНОЕ ПРЕДСКАЗАНИЕ ---
    print("\n6. Training final model on ALL data...")
    final_pipeline = RegressionPipeline(**best_params)
    final_pipeline.fit(X, y)

    test_preds = final_pipeline.predict(X_test)
    test_preds = robust_clip(test_preds, 0.0, 100.0)

    submission = pd.DataFrame({
        ID_COL: test_ids,
        TARGET_COL: test_preds
    })

    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"Done! Predictions saved to: {OUTPUT_PATH}")

    # --- EDA (КРАТКО) ---
    print("\nGenerating Correlation Plot (Task 1p)...")
    num_df = train_df.select_dtypes(include=[np.number])
    if len(num_df.columns) > 1:
        corr = num_df.corr()[TARGET_COL].abs().sort_values(ascending=False).head(10)
        print("Top correlations:")
        print(corr)
        plt.figure(figsize=(8, 6))
        sns.heatmap(num_df[corr.index].corr(), annot=True, fmt=".2f", cmap='coolwarm')
        plt.title("Correlation Matrix (Top Features)")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()