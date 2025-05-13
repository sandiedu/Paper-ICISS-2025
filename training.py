import os
import joblib
import pandas as pd
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Constants
DATASET_PATH = os.path.join(os.getcwd(), "dataset_scored.csv")
MODEL_DIR = os.path.join(os.getcwd(), "models")
EVALUATION_PATH = os.path.join(os.getcwd(), "evaluation.csv")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = [
    'Layer Count', 'Avg Imports per File',
    'Architecture Score', 'Avg Cyclomatic', 'Avg Volume',
    'Avg Difficulty', 'Avg Effort', 'Score'
]

# Helper functions


def preprocess_data(df):
    """Preprocess the dataset."""
    df = df[FEATURES].astype(float)
    df.rename(columns={
        'Avg Cyclomatic': 'cyclomatic',
        'Avg Volume': 'volume',
        'Avg Difficulty': 'difficulty',
        'Avg Effort': 'effort',
        'Architecture Score': 'architecture',
        'Layer Count': 'layer',
        'Avg Imports per File': 'import',
        'Score': 'score'
    }, inplace=True)
    scaler = MinMaxScaler()
    norm = scaler.fit_transform(df)
    norm_df = pd.DataFrame(norm, columns=df.columns)
    return norm_df


def train_model(name, model, params, X_train, X_test, y_train, y_test, scaler=None):
    """Train a model and evaluate its performance."""
    print(f"Training {name}...")
    if scaler:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    grid = GridSearchCV(model, params, cv=3, scoring='r2',
                        n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    joblib.dump(best_model, os.path.join(
        MODEL_DIR, f"{name.replace(' ', '_').lower()}_model.pkl"))

    y_pred = best_model.predict(X_test).clip(0, 100)

    if hasattr(best_model, "feature_importances_"):
        print("Feature Importances:")
        for i, importance in enumerate(best_model.feature_importances_):
            print(f"Feature {i}: {importance}")

    return {
        "Model": name,
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred)
    }


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train and evaluate multiple regression models."""
    results = []
    scaler = StandardScaler(with_mean=False)

    model_params = {
        "Random Forest": (RandomForestRegressor(), {
            "n_estimators": [100, 250, 500, 750, 1000],
            "max_depth": [10, 25, 50, 75, 100, None]
        }),
        "Linear Regression": (LinearRegression(), {}),
        "Gradient Boosting": (GradientBoostingRegressor(n_iter_no_change=10, validation_fraction=0.2), {
            "n_estimators": [100, 250, 500, 750, 1000],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [10, 25, 50, 75, 100, None]
        }),
        "Support Vector Regression": (SVR(), {
            "C": [0.1, 1, 10, 100],
            "kernel": ["rbf"],
            "gamma": [0.001, 0.01, 0.1, 1]
        }),
        "XGBoost Regressor": (XGBRegressor(objective="reg:squarederror", eval_metric="rmse", verbosity=1, tree_method="auto"), {
            "n_estimators": [100, 250, 500],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [3, 5, 7, 10],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0]
        })
    }

    for name, (model, params) in model_params.items():
        scaler_to_use = scaler if name in [
            "Support Vector Regression", "Linear Regression"] else None
        results.append(train_model(name, model, params, X_train,
                       X_test, y_train, y_test, scaler_to_use))

    return results


# Main script
if __name__ == "__main__":
    # Load and preprocess data
    df = pd.read_csv(DATASET_PATH)
    norm_df = preprocess_data(df)

    # Feature extraction
    X = norm_df[['cyclomatic', 'volume', 'difficulty',
                 'effort', 'architecture', 'layer', 'import']]
    y = norm_df['score']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Train and evaluate models
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    pd.DataFrame(results).to_csv(EVALUATION_PATH, index=False)
