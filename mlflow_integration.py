import pandas as pd
import mlflow
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score


class PricePredictor:
    def __init__(self, csv_path, model_type="random_forest", test_size=0.25, random_state=42, tracking_uri="http://127.0.0.1:5000"):
        self.csv_path = csv_path
        self.model_type = model_type
        self.test_size = test_size
        self.random_state = random_state
        self.tracking_uri = tracking_uri
        self.model = None
        self.grid_search = None
        self.signature = None
        self.r2 = None

    def load_data(self):
        df = pd.read_csv(self.csv_path)
        X = df.drop('Price', axis=1)
        y = df['Price']
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

    def get_model_and_grid(self):
        if self.model_type == "random_forest":
            model = RandomForestRegressor()
            param_grid = {
                "n_estimators": [50, 100],
                "max_depth": [2, 5, 10, None],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2]
            }
        elif self.model_type == "xgboost":
            model = XGBRegressor(objective="reg:squarederror", n_jobs=-1, verbosity=0)
            param_grid = {
                "n_estimators": [50, 100],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.05, 0.1],
                "subsample": [0.8, 1.0]
            }
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        return model, param_grid

    def tune_hyperparameters(self, X_train, y_train):
        model, param_grid = self.get_model_and_grid()
        self.grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=3,
            scoring='r2',
            n_jobs=-1
        )
        self.grid_search.fit(X_train, y_train)
        self.model = self.grid_search.best_estimator_

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        self.r2 = r2_score(y_test, y_pred)
        return self.r2

    def log_experiment(self, X_train, y_train):
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment("Airlines Price Prediction")
        self.signature = infer_signature(X_train, y_train)

        with mlflow.start_run():
            mlflow.set_tag("model_type", self.model_type)

            for param, value in self.grid_search.best_params_.items():
                mlflow.log_param(f"best_{param}", value)

            mlflow.log_metric("r2_score", self.r2)

            mlflow.sklearn.log_model(
                sk_model=self.model,
                artifact_path=f"{self.model_type}_model",
                signature=self.signature
            )


def main():
    # Change model_type to "xgboost" or "random_forest"
    predictor = PricePredictor(csv_path="Data MLFlow.csv", model_type="random_forest")

    X_train, X_test, y_train, y_test = predictor.load_data()
    predictor.tune_hyperparameters(X_train, y_train)
    predictor.evaluate(X_test, y_test)
    predictor.log_experiment(X_train, y_train)


if __name__ == "__main__":
    main()
