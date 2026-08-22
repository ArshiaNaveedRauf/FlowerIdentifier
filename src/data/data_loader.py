import tensorflow as tf
from config import data_path


class DataLoader:
    def __init__(self,validation_spl,img_height,img_width,batch_size):
        self.validation_spl= validation_spl
        self.img_height= img_height
        self.img_width= img_width
        self.batch_size= batch_size
        self.data_dir= data_path

    def training_dataset_loader(self):
        # training dataset
        train_dataset = tf.keras.utils.image_dataset_from_directory(
            self.data_dir,
            validation_split=self.validation_spl,
            subset= "training",
            seed= 123,
            image_size=(self.img_height,self.img_width),
            batch_size= self.batch_size
        )
        return train_dataset


    def validation_dataset_loader(self):
        validation_dataset= tf.keras.utils.image_dataset_from_directory(
            self.data_dir,
            validation_split=self.validation_spl,
            subset= "validation",
            seed= 123,
            image_size=(self.img_height,self.img_width),
            batch_size= self.batch_size
        )
        return validation_dataset