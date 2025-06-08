from src.data.dataset import BaseDataset
import pandas as pd
import os

class OAIDataset(BaseDataset):
    def _load_dataframe(self):
        df = pd.read_csv(self.csv_path)
        df["path"] = df["filename"].apply(lambda x: os.path.join(self.image_dir, x))
        df["label"] = df["JSN_score"].astype(int)  # example
        return df
