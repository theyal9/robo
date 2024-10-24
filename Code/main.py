# main.py
import cv2
import numpy as np
import tensorflow as tf
import json

def load_class_names(file_path):
    with open(file_path, 'r') as f:
        class_indices = json.load(f)
    # Reverse the dictionary to get class names by index
    class_names = {v: k for k, v in class_indices.items()}
    return class_names

def recognize_card(model, frame, class_names):
    # Resize and preprocess the frame directly without saving it as a file
    resized_frame = cv2.resize(frame, (128, 128))
    normalized_frame = resized_frame / 255.0  # Normalize
    card_img = normalized_frame.reshape(1, 128, 128, 3)  # Reshape to match model input
    
    # Predict card
    prediction = model.predict(card_img)
    predicted_class_index = np.argmax(prediction)
    card_name = class_names.get(predicted_class_index, "Unknown")
    
    return card_name

def main():
    # Path to the saved model and class indices
    model_path = 'uno_card_model.h5'
    class_names_path = 'class_indices.json'

    # Load the pre-trained model
    model = tf.keras.models.load_model(model_path)

    # Load class names from the JSON file
    class_names = load_class_names(class_names_path)

    # Capture video from the external USB webcam
    cap = cv2.VideoCapture(1)  # Change to 1 for external USB webcam

    if not cap.isOpened():
        print("Error: Could not open external webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


    frame_counter = 0
    frame_interval = 10  # Process every 10th frame to avoid too much lag

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image. Exiting...")
            break

        # Display the current frame
        cv2.imshow("Card Recognition - Press 'q' to exit", frame)

        # Only process every nth frame to reduce lag
        if frame_counter % frame_interval == 0:
            recognized_card = recognize_card(model, frame, class_names)
            print(f"Recognized Card: {recognized_card}")

        frame_counter += 1

        # Press 'q' to quit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()



# # main.py
# import cv2
# import numpy as np
# import tensorflow as tf
# import json
# import os
# import time
# from preprocess import preprocess_image

# def load_class_names(file_path):
#     with open(file_path, 'r') as f:
#         class_indices = json.load(f)
#     # Reverse the dictionary to get class names by index
#     class_names = {v: k for k, v in class_indices.items()}
#     return class_names

# def recognize_card(model, img_path, class_names):
#     # Preprocess and find contours of the card
#     card_img = preprocess_image(img_path)
#     if card_img is not None:
#         # Reshape image to match the model input
#         card_img = card_img.reshape(1, 128, 128, 3)
#         # Predict card
#         prediction = model.predict(card_img)
        
#         # Get the index of the predicted class
#         predicted_class_index = np.argmax(prediction)
        
#         # Get the class name from the class index
#         card_name = class_names.get(predicted_class_index, "Unknown")
#         return card_name
#     return "No card detected"

# def main():
#     # Path to the saved model and class indices
#     model_path = 'uno_card_model.h5'
#     class_names_path = 'class_indices.json'
#     temp_image_path = 'temp_frame.jpg'  # Temporary file path for webcam frames

#     # Load the pre-trained model
#     model = tf.keras.models.load_model(model_path)

#     # Load class names from the JSON file
#     class_names = load_class_names(class_names_path)

#     # Capture an image using the webcam (or specify an image path)
#     cap = cv2.VideoCapture(0)  # 0 is usually the default webcam index

#     frame_counter = 0
#     frame_interval = 30  # Process every 30th frame, can be adjusted as needed
#     delay_seconds = 0.5  # Add a half-second delay between frames

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to capture image. Exiting...")
#             break

#         # Display the frame
#         cv2.imshow("Card Recognition - Press 'q' to exit", frame)

#         # Only process every nth frame
#         if frame_counter % frame_interval == 0:
#             # Save the current frame as a temporary image file
#             cv2.imwrite(temp_image_path, frame)

#             # Recognize the card in the saved image
#             recognized_card = recognize_card(model, temp_image_path, class_names)
#             print(f"Recognized Card: {recognized_card}")

#         frame_counter += 1

#         # Press 'q' to quit the loop
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#         # Introduce a delay between frames
#         time.sleep(delay_seconds)

#     # Release the webcam and close windows
#     cap.release()
#     cv2.destroyAllWindows()

#     # Clean up: remove the temporary image file if it exists
#     if os.path.exists(temp_image_path):
#         os.remove(temp_image_path)

# if __name__ == "__main__":
#     main()





# # main.py
# import cv2
# import numpy as np
# import tensorflow as tf
# import json
# import os
# from preprocess import preprocess_image

# def load_class_names(file_path):
#     with open(file_path, 'r') as f:
#         class_indices = json.load(f)
#     # Reverse the dictionary to get class names by index
#     class_names = {v: k for k, v in class_indices.items()}
#     return class_names

# def recognize_card(model, img_path, class_names):
#     # Preprocess and find contours of the card
#     card_img = preprocess_image(img_path)
#     if card_img is not None:
#         # Reshape image to match the model input
#         card_img = card_img.reshape(1, 128, 128, 3)
#         # Predict card
#         prediction = model.predict(card_img)
        
#         # Get the index of the predicted class
#         predicted_class_index = np.argmax(prediction)
        
#         # Get the class name from the class index
#         card_name = class_names.get(predicted_class_index, "Unknown")
#         return card_name
#     return "No card detected"

# def main():
#     # Path to the saved model and class indices
#     model_path = 'uno_card_model.h5'
#     class_names_path = 'class_indices.json'
#     temp_image_path = 'temp_frame.jpg'  # Temporary file path for webcam frames

#     # Load the pre-trained model
#     model = tf.keras.models.load_model(model_path)

#     # Load class names from the JSON file
#     class_names = load_class_names(class_names_path)

#     # Capture an image using the webcam (or specify an image path)
#     cap = cv2.VideoCapture(0)  # 0 is usually the default webcam index

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to capture image. Exiting...")
#             break

#         # Display the frame
#         cv2.imshow("Card Recognition - Press 'q' to exit", frame)

#         # Save the current frame as a temporary image file
#         cv2.imwrite(temp_image_path, frame)

#         # Recognize the card in the saved image
#         recognized_card = recognize_card(model, temp_image_path, class_names)
#         print(f"Recognized Card: {recognized_card}")

#         # Press 'q' to quit the loop
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release the webcam and close windows
#     cap.release()
#     cv2.destroyAllWindows()

#     # Clean up: remove the temporary image file if it exists
#     if os.path.exists(temp_image_path):
#         os.remove(temp_image_path)

# if __name__ == "__main__":
#     main()
