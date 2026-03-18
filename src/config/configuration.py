import yaml
from src.entity.config_entity import DataIngestionConfig, ModelTrainerConfig

class ConfigurationManager:

    def __init__(self):

        with open("config/config.yaml") as file:
            self.config = yaml.safe_load(file)


    def get_data_ingestion_config(self):

        cfg = self.config["data_ingestion"]

        return DataIngestionConfig(
        dataset_path=cfg["dataset_path"],
        train_path=cfg["train_path"],
        test_path=cfg["test_path"],
        test_size=cfg["test_size"]
)


    def get_model_trainer_config(self):

        cfg = self.config["model_trainer"]

        return ModelTrainerConfig(
            model_path=cfg["model_path"],
            random_state=cfg["random_state"]
        )