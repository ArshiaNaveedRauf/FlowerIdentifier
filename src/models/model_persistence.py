import tensorflow as tf

class ModelPersistence:
    def save_model(self,model,path):
        model.save(path)
        print(f'model saved to {path}')

    def load(self, path):
        return tf.keras.models.load_model(path)
        