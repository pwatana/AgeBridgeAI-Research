import yaml
from src.data.rsna_dataset import RSNABoneAgeDataset

with open("config.yaml") as f:
    config = yaml.safe_load(f)

dataset = RSNABoneAgeDataset(
    csv_path=config["data"]["train_csv"],
    image_dir=config["data"]["image_dir"],
    config=config["training"]
)

train_ds, val_ds = dataset.split_datasets()
