from src.config.configuration import ConfigurationManager
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainingPipeline:

    def start_pipeline(self):

        config_manager=ConfigurationManager()

        ingestion_config=config_manager.get_data_ingestion_config()
        trainer_config=config_manager.get_model_trainer_config()

        ingestion=DataIngestion(ingestion_config)

        train_path,test_path=ingestion.initiate_data_ingestion()

        transformation=DataTransformation()

        X_train,X_test,y_train,y_test=transformation.initiate_data_transformation(
            train_path,test_path
        )

        trainer=ModelTrainer(trainer_config)

        trainer.initiate_model_trainer(
            X_train,X_test,y_train,y_test
        )