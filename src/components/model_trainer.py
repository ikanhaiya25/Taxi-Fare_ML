from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from src.utils.common import save_object

class ModelTrainer:

    def __init__(self,config):

        self.config=config


    def initiate_model_trainer(self,X_train,X_test,y_train,y_test):

        model=DecisionTreeRegressor(
            random_state=self.config.random_state
        )

        model.fit(X_train,y_train)

        preds=model.predict(X_test)

        r2 = r2_score(y_test,preds)
        mae = mean_absolute_error(y_test,preds)
        mse = mean_squared_error(y_test,preds)

        print(f"DecisionTreeReressor Metrics:\nR2 Score: {r2}\nMAE: {mae}\nMSE: {mse}")

        save_object(self.config.model_path,model)