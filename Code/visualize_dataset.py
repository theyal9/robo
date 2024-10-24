# visualize_dataset.py
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def visualize_data(dataset_path):
    # Create an image data generator with rescaling
    train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    
    # Load the dataset to get class names
    train_data = train_datagen.flow_from_directory(
        dataset_path, target_size=(128, 128), batch_size=32, class_mode='categorical'
    )

    # Get class names and their corresponding indices
    class_names = train_data.class_indices
    print("Class Names and Indices:", class_names)
    
    # Plot 5 samples from each class
    fig, axes = plt.subplots(len(class_names), 5, figsize=(15, len(class_names) * 3))
    axes = axes.flatten()

    for class_name, idx in class_names.items():
        # Go through a batch of images and display them
        train_data.reset()  # Reset to the beginning of the dataset
        count = 0
        for batch in train_data:
            images, labels = batch
            # Select images of the current class
            for i in range(images.shape[0]):
                if np.argmax(labels[i]) == idx and count < 5:
                    axes[idx * 5 + count].imshow(images[i])
                    axes[idx * 5 + count].axis('off')
                    axes[idx * 5 + count].set_title(class_name)
                    count += 1
                if count == 5:
                    break
            if count == 5:
                break

    plt.tight_layout()
    plt.show()

# Call the function with your dataset path
if __name__ == "__main__":
    visualize_data(r'C:\Users\HP\Desktop\Uno_Card_Recognition\Dataset')
