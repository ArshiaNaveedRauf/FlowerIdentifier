from src.data.data_loader import DataLoader
from src.visualization.raw_data_visualizer import RawDataVisualizer
from src.data.data_augmenter import DataAugmenter
from src.models.convolution_neural_network import ConvolutionalNeuralNetwork
from src.models.model_persistence import ModelPersistence
from config import validation_spl, batch_size,img_width,img_height, epochs


class FlowerRecognitionPipeline:
    def __init__(self):
        self.loader = DataLoader(validation_spl,img_height,img_width,batch_size)
        self.raw_visualizer= RawDataVisualizer()
        self.augmenter= DataAugmenter()
        self.model = ConvolutionalNeuralNetwork(self.augmenter)
        self.persistence= ModelPersistence()



    def run_pipeline(self):
        training_data= self.loader.training_dataset_loader()
        validation_data= self.loader.validation_dataset_loader()
        class_names= training_data.class_names
        self.raw_visualizer.raw_data_visualizer(training_data)
        self.model.train_model(training_data,validation_data,epochs)
        self.persistence.save_model(self.model.model,'models_saved/cnn_model.keras')

