from src.models.model_evaluator import ModelEvaluator
from src.data.data_loader import DataLoader
from src.models.model_persistence import ModelPersistence
from src.inference.flower_predictor import FlowerPredictor
from config import validation_spl,img_height,img_width,batch_size




class SavedModelPipeline:
    def __init__(self):
        self.persistence= ModelPersistence()
        self.evaluator= ModelEvaluator()


    def run_saved_model_pipeline(self):
        model= self.persistence.load("models_saved/cnn_model.keras")
        loader= DataLoader(validation_spl,img_height,img_width,batch_size)
        validation_data= loader.validation_dataset_loader()
        class_name= validation_data.class_names

        self.evaluator.model_evaluator(class_name,validation_data,model)

        predictor = FlowerPredictor(model, ["daisy", "dandelion", "rose", "sunflower", "tulip"], img_height, img_width)
        flower, confidence = predictor.predict_from_path("/Users/arshianaveed/FlowerRecognition/test_image.jpg")
        print(f"{flower} ({confidence * 100:.1f}% confidence)")


run= SavedModelPipeline()
run.run_saved_model_pipeline()

