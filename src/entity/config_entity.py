from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    dataset_path: str
    train_path: str
    test_path: str
    test_size: float


@dataclass
class ModelTrainerConfig:
    model_path: str
    random_state:int