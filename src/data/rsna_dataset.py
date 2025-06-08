from src.data.dataset import BaseDataset
import pandas as pd
import os

class RSNABoneAgeDataset(BaseDataset):
    def _bin_boneage(self, age):
        # Example: 3 bins (you can change this logic)
        if age < 72:
            return 0
        elif age < 144:
            return 1
        else:
            return 2

    def _load_dataframe(self):
        df = pd.read_csv(self.csv_path)
        df["path"] = df["id"].apply(lambda x: os.path.join(self.image_dir, f"{x}.png"))
        df["label"] = df["boneage"].apply(self._bin_boneage)
        return df
