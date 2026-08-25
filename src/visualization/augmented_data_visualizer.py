import matplotlib.pyplot as plt


class AugmentedDataVisulaizer:
    def augmented_data_visualizer(self, training_data, augmenter):
        class_name= training_data.class_names
        i=0
        plt.figure(figsize=(10,10))
        for images,labels in training_data.take(1):
            for i in range (9):
                images = augmenter(images)
                plt.subplot(3,3,i+1)
                plt.imshow(images[0].numpy().astype('uint8'))
                plt.axis('off')
                plt.savefig('outputs/figures/augmented_data_visualized.png')
            plt.close()