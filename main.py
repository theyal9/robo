# main.py
import cv2
from recognize_card import recognize_card
from preprocess import find_card_contours

def capture_card_image(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        card_img = find_card_contours(frame)
        if card_img is not None:
            # Display detected card region
            cv2.imshow('Detected Card', card_img)
            
            # Wait for user to press 's' to save image and predict
            if cv2.waitKey(1) & 0xFF == ord('s'):
                cv2.imwrite('captured_card.jpg', card_img)
                break

    cap.release()
    cv2.destroyAllWindows()
    return 'captured_card.jpg'

if __name__ == "__main__":
    # Capture image from the camera
    img_path = capture_card_image()
    
    # Recognize the card from the captured image
    model_path = 'uno_card_model.h5'
    class_names_path = 'class_indices.json'
    recognized_card = recognize_card(model_path, img_path, class_names_path)
    print(f"Recognized Card: {recognized_card}")










# # main.py
# import cv2
# from recognize_card import recognize_card
# from preprocess import find_card_contours

# def capture_card_image(camera_index=0):
#     cap = cv2.VideoCapture(camera_index)
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         card_img = find_card_contours(frame)
#         if card_img is not None:
#             # Display detected card region
#             cv2.imshow('Detected Card', card_img)
            
#             # Wait for user to press 's' to save image and predict
#             if cv2.waitKey(1) & 0xFF == ord('s'):
#                 cv2.imwrite('captured_card.jpg', card_img)
#                 break

#     cap.release()
#     cv2.destroyAllWindows()
#     return 'captured_card.jpg'

# if __name__ == "__main__":
#     # Capture image from the camera
#     img_path = capture_card_image()
    
#     # Recognize the card from the captured image
#     recognized_card = recognize_card('uno_card_model.h5', img_path)
#     print(f"Recognized Card: {recognized_card}")
