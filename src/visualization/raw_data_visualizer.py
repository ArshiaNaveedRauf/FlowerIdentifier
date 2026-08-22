import matplotlib.pyplot as plt 

class RawDataVisualizer:
    def raw_data_visualizer(self,training_data):
        class_names= training_data.class_names 
        for image, labels in training_data.take(1):
            for i in range(9):
                plt.subplot(3,3,i+1)
                plt.imshow(image[i].numpy().astype("uint8"))
                plt.title(class_names[labels[i]])
                plt.axis("off")
        plt.savefig("outputs/figures/raw_data_visualization.png")
        plt.close()
        
