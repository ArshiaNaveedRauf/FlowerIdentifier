from keras import Sequential, layers
from src.models.base_model import BaseClassifer
from config import num_class, rescale, dropout_rate

class ConvolutionalNeuralNetwork(BaseClassifer):
    def __init__(self,augmenter):
    
        super().__init__(name= 'CNN', num_class= num_class, augmenter= augmenter , dropout_rate=dropout_rate)


    def _build_model(self,augmenter):
        model = Sequential()

        if augmenter is not None:
            model.add(augmenter.augmentation)

        model.add(layers.Rescaling(rescale))

        model.add(layers.Conv2D(32,(3,3), activation='relu'))
        model.add(layers.MaxPooling2D((2,2)))

        model.add(layers.Conv2D(64,(3,3), activation='relu'))
        model.add(layers.MaxPooling2D((2,2)))

        model.add(layers.Conv2D(128,(3,3), activation='relu'))
        model.add(layers.MaxPooling2D((2,2)))

        # global spatial reduction
        model.add(layers.GlobalAveragePooling2D())

        # dropout layer
        model.add(layers.Dropout(rate= dropout_rate))

        # output classifer 
        model.add(layers.Dense(num_class, activation='softmax'))

        return model 


        