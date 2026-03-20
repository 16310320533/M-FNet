import numpy as np
import tensorflow as tf


class MultimodalDataGenerator(tf.keras.utils.Sequence):
    """
    Robust data demo generator for feeding synchronized multi-modal data demo
    (NCCT images and clinical tabular features) into M2FNet.
    """

    def __init__(self, image_paths, tabular_data, labels, batch_size=16, dim=(224, 224), n_channels=3, shuffle=True):
        self.image_paths = np.array(image_paths)
        self.tabular_data = np.array(tabular_data)
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.indices = np.arange(len(self.image_paths))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_paths = self.image_paths[indexes]
        batch_tabular = self.tabular_data[indexes]
        y = self.labels[indexes]

        X_img = np.empty((self.batch_size, *self.dim, self.n_channels))

        # Simulating image loading and preprocessing (triplication for grayscale)
        for i, path in enumerate(batch_paths):
            # Placeholder for actual image decoding logic
            # img = load_and_preprocess_dicom(path, self.dim)
            img = np.random.rand(*self.dim, self.n_channels)  # Replace with actual loader
            X_img[i,] = img

        return [X_img, batch_tabular], y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)