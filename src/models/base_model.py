from abc import ABC, abstractmethod
from keras.callbacks import EarlyStopping,  ReduceLROnPlateau

class BaseClassifer(ABC):
    def __init__(self,name, num_class,augmenter,dropout_rate):
        self.name= name 
        self.num_class= num_class
        self.dropout_rate= dropout_rate
        self.model= self._build_model(augmenter)
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics= ['accuracy']
        )

    @abstractmethod
    def _build_model(self,augmenter):
        pass

    def train_model(self,training_data,validation_data,epochs):
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True)

        reduce_learning_rate= ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience= 4,
            min_lr= 1e-6

        )
        return self.model.fit(training_data,validation_data=validation_data, epochs=epochs, callbacks=[early_stopping,reduce_learning_rate])
    

    def predict(self, x):
        return self.model.predict(x)