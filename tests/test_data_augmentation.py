from src.data.data_augmenter import DataAugmenter
import tensorflow as tf




def test_data_augmentation():
    image= tf.keras.utils.load_img('/Users/arshianaveed/FlowerRecognition/flower.avif')
    image = tf.keras.utils.img_to_array(image)
    image= tf.expand_dims(image, axis=0)
    augmenter =  DataAugmenter()
    results = augmenter.augmentation(image)
    assert results.shape == image.shape


