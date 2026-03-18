import pandas as pd
import os
from sklearn.model_selection import train_test_split

class DataIngestion:

    def __init__(self, config):
        self.config = config

    def initiate_data_ingestion(self):

        df = pd.read_csv(self.config.dataset_path, low_memory=False)

        os.makedirs("artifacts", exist_ok=True)

        train, test = train_test_split(
            df,
            test_size=self.config.test_size,
            random_state=42,
            shuffle=False
        )

        train.to_csv(self.config.train_path, index=False)
        test.to_csv(self.config.test_path, index=False)

        return self.config.train_path, self.config.test_path