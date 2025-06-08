from src.data.dataset import BaseDataset
import pandas as pd
import os

class OMERACTDataset(BaseDataset):
    def _load_dataframe(self):
        df = pd.read_csv(self.csv_path)
        df["path"] = df["image_id"].apply(lambda x: os.path.join(self.image_dir, f"{x}.jpg"))
        df["label"] = df["SharpScore_bin"]  # already pre-binned?
        return df
