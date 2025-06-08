import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

class BaseDataset:
    def __init__(self, csv_path, image_dir, config):
        self.csv_path = csv_path
        self.image_dir = image_dir
        self.img_size = tuple(config["image_size"])
        self.batch_size = config["batch_size"]
        self.seed = config["seed"]
        self.num_classes = config["num_classes"]
        self.config = config
        self.df = self._load_dataframe()

    def _load_dataframe(self):
        raise NotImplementedError("Subclasses must implement _load_dataframe")

    def _preprocess(self, image_path, label):
        image = tf.io.read_file(image_path)
        image = tf.io.decode_png(image, channels=1 if self.config["grayscale"] else 3)
        image = tf.image.resize(image, self.img_size)
        image = tf.cast(image, tf.float32) / 255.0
        label = tf.one_hot(label, self.num_classes)
        return image, label

    def _create_tf_dataset(self, df, training=True):
        paths = df["path"].values
        labels = df["label"].values
        dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
        dataset = dataset.map(self._preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        if training:
            dataset = dataset.shuffle(buffer_size=1000, seed=self.seed)
        return dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

    def split_datasets(self, val_split=0.2):
        train_df, val_df = train_test_split(
            self.df, test_size=val_split, stratify=self.df["label"], random_state=self.seed
        )
        train_ds = self._create_tf_dataset(train_df, training=True)
        val_ds = self._create_tf_dataset(val_df, training=False)
        return train_ds, val_ds
