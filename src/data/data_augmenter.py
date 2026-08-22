import tensorflow as tf
from config import factor

class DataAugmenter:
    def __init__(self):
        self.factor= factor
        self.augmentation= self._data_augmentation()

    def geometric_augmentation(self):
        data_geometric_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip(mode= "horizontal"),
            tf.keras.layers.RandomRotation(factor=self.factor),
            tf.keras.layers.RandomZoom(height_factor=self.factor)
        ])
        return data_geometric_augmentation

    def color_space_augmentation(self):
        data_color_space_augmentation= tf.keras.Sequential([
            tf.keras.layers.RandomBrightness(factor=self.factor,value_range=(0,255))
        ])
        return data_color_space_augmentation

    def _data_augmentation(self):
        return tf.keras.Sequential([
            self.geometric_augmentation(),
            self.color_space_augmentation()
        ])
        
