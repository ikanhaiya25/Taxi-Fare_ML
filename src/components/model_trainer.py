from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from src.utils.common import save_object
from src.logger import logging

class ModelTrainer:

    def __init__(self, config):
        self.config = config

    def initiate_model_trainer(self, X_train, X_test, y_train, y_test, preprocessor):
        logging.info("Model training started")

        model = DecisionTreeRegressor(
            random_state=self.config.random_state
        )

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)
        logging.info("Pipeline fitted successfully")

        preds = pipeline.predict(X_test)

        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)

        logging.info(f"DecisionTreeRegressor Metrics: R2={r2}, MAE={mae}, MSE={mse}")
        print(f"DecisionTreeRegressor Metrics:\nR2 Score: {r2}\nMAE: {mae}\nMSE: {mse}")

        save_object(self.config.model_path, pipeline)
        logging.info("Pipeline saved successfully")