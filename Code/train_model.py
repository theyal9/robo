# train_model.py
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import json

def create_cnn_model(num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    # Create an image data generator for training
    train_datagen = ImageDataGenerator(rescale=1.0/255.0)
    
    # Provide the path to your dataset folder
    dataset_path = r'C:\Users\HP\Desktop\Uno_Card_Recognition\Dataset'

    # Load the training data
    train_data = train_datagen.flow_from_directory(
        dataset_path, target_size=(128, 128), batch_size=32, class_mode='categorical')

    # Number of classes based on the number of folders in the dataset
    num_classes = len(train_data.class_indices)

    # Create the CNN model with the right number of classes
    model = create_cnn_model(num_classes)
    
    # Train the model
    model.fit(train_data, epochs=10)
    
    # Save the trained model
    model.save('uno_card_model.h5')

    # After model.save('uno_card_model.h5')
    with open('class_indices.json', 'w') as f:
        json.dump(train_data.class_indices, f)

if __name__ == "__main__":
    train_model()
