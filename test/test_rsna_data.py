import yaml
import matplotlib.pyplot as plt
import tensorflow as tf
from src.data.rsna_dataset import RSNABoneAgeDataset

def show_sample(dataset, class_names=None):
    for images, labels in dataset.take(1):
        for i in range(5):
            plt.imshow(tf.squeeze(images[i]), cmap="gray")
            label = tf.argmax(labels[i]).numpy()
            title = f"Label: {label}"
            if class_names:
                title += f" ({class_names[label]})"
            plt.title(title)
            plt.axis("off")
            plt.show()

def test_rsna_loader():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    dataset = RSNABoneAgeDataset(
        csv_path=config["data"]["train_csv"],
        image_dir=config["data"]["image_dir"],
        config=config["training"]
    )

    train_ds, val_ds = dataset.split_datasets()

    print("Dataset loaded successfully.")
    print(f"Train batches: {len(train_ds)} | Val batches: {len(val_ds)}")

    show_sample(train_ds, class_names=["Mild", "Moderate", "Severe"])

if __name__ == "__main__":
    test_rsna_loader()
