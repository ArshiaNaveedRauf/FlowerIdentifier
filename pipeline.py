from src.data.data_loader import DataLoader
from src.visualization.raw_data_visualizer import RawDataVisualizer
from config import validation_spl, batch_size,img_width,img_height


class FlowerRecognitionPipeline:
    def __init__(self):
        self.loader = DataLoader(validation_spl,img_height,img_width,batch_size)
        self.raw_visualizer= RawDataVisualizer()



    def run_pipeline(self):
        training_data= self.loader.training_dataset_loader()
        validation_data= self.loader.validation_dataset_loader()
        self.raw_visualizer.raw_data_visualizer(training_data)
        