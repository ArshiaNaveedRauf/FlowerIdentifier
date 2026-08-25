from src.data.data_loader import DataLoader
from src.visualization.raw_data_visualizer import RawDataVisualizer
from src.visualization.augmented_data_visualizer import AugmentedDataVisulaizer
from src.data.data_augmenter import DataAugmenter
from config import validation_spl, batch_size,img_width,img_height


class FlowerRecognitionPipeline:
    def __init__(self):
        self.loader = DataLoader(validation_spl,img_height,img_width,batch_size)
        self.raw_visualizer= RawDataVisualizer()
        self.augmenter= DataAugmenter()
        self.augmentation_visulizer= AugmentedDataVisulaizer()



    def run_pipeline(self):
        training_data= self.loader.training_dataset_loader()
        validation_data= self.loader.validation_dataset_loader()
        self.raw_visualizer.raw_data_visualizer(training_data)
        self.augmentation_visulizer.augmented_data_visualizer(training_data,self.augmenter.augmentation)
        