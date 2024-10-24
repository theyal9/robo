# recognize_card.py
import cv2
import numpy as np
import tensorflow as tf
import json
from preprocess import preprocess_image, find_card_contours

def load_class_names(file_path):
    with open(file_path, 'r') as f:
        class_indices = json.load(f)
    # Reverse the dictionary to get class names by index
    class_names = {v: k for k, v in class_indices.items()}
    return class_names

def recognize_card(model_path, img_path, class_names_path):
    # Load pre-trained model
    model = tf.keras.models.load_model(model_path)
    
    # Load class names
    class_names = load_class_names(class_names_path)
    
    # Preprocess and find contours of the card
    card_img = preprocess_image(img_path)
    if card_img is not None:
        # Reshape image to match the model input
        card_img = card_img.reshape(1, 128, 128, 3)
        # Predict card
        prediction = model.predict(card_img)
        
        # Get the index of the predicted class
        predicted_class_index = np.argmax(prediction)
        
        # Get the class name from the class index
        card_name = class_names.get(predicted_class_index, "Unknown")
        return card_name

if __name__ == "__main__":
    model_path = 'uno_card_model.h5'
    img_path = r'C:\Users\HP\Desktop\Uno_Card_Recognition\my.jpg'
    class_names_path = 'class_indices.json'
    
    recognized_card = recognize_card(model_path, img_path, class_names_path)
    print(f"Recognized Card: {recognized_card}")



# # recognize_card.py
# import cv2
# import numpy as np
# import tensorflow as tf
# from preprocess import preprocess_image, find_card_contours

# def recognize_card(model_path, img_path):
#     # Load pre-trained model
#     model = tf.keras.models.load_model(model_path)
    
#     # Preprocess and find contours of the card
#     card_img = preprocess_image(img_path)
#     if card_img is not None:
#         # Reshape image to match the model input
#         card_img = card_img.reshape(1, 128, 128, 3)
#         # Predict card
#         prediction = model.predict(card_img)
#         return np.argmax(prediction)

# if __name__ == "__main__":
#     model_path = 'uno_card_model.h5'
#     # img_path = 'path/to/your/card/image.jpg'
#     # img_path = r'C:\Users\HP\Desktop\Uno_Card_Recognition\WIN_20230430_04_28_49_Pro - Copy.jpg'
#     img_path = r'C:\Users\HP\Desktop\Uno_Card_Recognition\WIN_20230430_04_37_06_Pro - Copy.jpg'
#     card_type = recognize_card(model_path, img_path)
#     print(f"Recognized Card: {card_type}")
