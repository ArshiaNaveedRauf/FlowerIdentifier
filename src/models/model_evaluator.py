
import numpy as np 
from sklearn.metrics import classification_report,confusion_matrix
import seaborn as sns


class ModelEvaluator:
    def __init__(self):
        pass

    def true_labels_and_predictions(self,validation_data,model):
        true_lables=[]
        prediction_labels=[]
        for images, labels in validation_data:
            predictions= model.predict(images, verbose=0)
            prediction_labels.append(np.argmax(predictions,axis=1))
            true_lables.append(labels.numpy())
        true_lables= np.concatenate(true_lables,axis=0)
        prediction_labels= np.concatenate(prediction_labels,axis=0) 
        return true_lables, prediction_labels

    def accuracy_report(self,true_lables,prediction_labels):
        print(classification_report(true_lables,prediction_labels))

    def confusion_matrix_display(self,true_lables,prediction_labels,class_name):
        cm = confusion_matrix(true_lables,prediction_labels)
        cm_plot= sns.heatmap(cm, xticklabels=class_name,yticklabels=class_name,annot=True)
        cm_plot.figure.savefig('outputs/figures/confusion_matrix.png')

    def model_evaluator(self, class_name,validation_data,model):
        true_label,prediction_labels= self.true_labels_and_predictions(validation_data,model)
        self.accuracy_report(true_label,prediction_labels)
        self.confusion_matrix_display(true_label,prediction_labels,class_name)



