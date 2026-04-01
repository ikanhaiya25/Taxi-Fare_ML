import pandas as pd
import os
from sklearn.model_selection import train_test_split
from src.logger import logging

class DataIngestion:

    def __init__(self, config):
        self.config = config

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion started")

        df = pd.read_csv(self.config.dataset_path, low_memory=False)
        logging.info(f"Dataset loaded with shape {df.shape}")

        os.makedirs("artifacts", exist_ok=True)

        train, test = train_test_split(
            df,
            test_size=self.config.test_size,
            random_state=42,
            shuffle=False
        )
        logging.info(f"Train-test split done: train={train.shape}, test={test.shape}")

        train.to_csv(self.config.train_path, index=False)
        test.to_csv(self.config.test_path, index=False)
        logging.info("Data Ingestion completed")

        return self.config.train_path, self.config.test_path