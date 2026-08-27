import numpy as np
from keras.utils import load_img, img_to_array


class FlowerPredictor:
    def __init__(self, model, class_names, img_height, img_width):
        self.model = model
        self.class_names = class_names
        self.img_height = img_height
        self.img_width = img_width

    def load_and_preprocess(self, image_path):
        img = load_img(image_path, target_size=(self.img_height, self.img_width))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict(self, image_array):
        predictions = self.model.predict(image_array, verbose=0)
        predicted_index = int(np.argmax(predictions[0]))
        predicted_class = self.class_names[predicted_index]
        confidence = float(predictions[0][predicted_index])
        return predicted_class, confidence

    def predict_from_path(self, image_path):
        image_array = self.load_and_preprocess(image_path)
        return self.predict(image_array)