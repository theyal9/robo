import cv2
import os

# List of folder names (UNO cards)
card_names = [
    'BLUE0', 'BLUE1', 'BLUE2', 'BLUE3', 'BLUE4', 'BLUE5', 'BLUE6', 'BLUE7', 'BLUE8', 'BLUE9',
    'BLUEDRAW2', 'BLUEREVERSE', 'BLUESKIP', 'DRAW4',
    'GREEN0', 'GREEN1', 'GREEN2', 'GREEN3', 'GREEN4', 'GREEN5', 'GREEN6', 'GREEN7', 'GREEN8', 'GREEN9',
    'GREENDRAW2', 'GREENREVERSE', 'GREENESKIP',
    'RED0', 'RED1', 'RED2', 'RED3', 'RED4', 'RED5', 'RED6', 'RED7', 'RED8', 'RED9',
    'REDDRAW2', 'REDREVERSE', 'REDSKIP',
    'WILDCARD',
    'YELLOW0', 'YELLOW1', 'YELLOW2', 'YELLOW3', 'YELLOW4', 'YELLOW5', 'YELLOW6', 'YELLOW7', 'YELLOW8', 'YELLOW9',
    'YELLOWDRAW2', 'YELLOWREVERSE', 'YELLOWSKIP'
]

# Function to save an image with the proper name in the correct folder
def save_image(folder_name, img_name, image):
    folder_path = os.path.join(os.getcwd(), folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    img_path = os.path.join(folder_path, img_name)
    cv2.imwrite(img_path, image)
    print(f"Saved {img_name} in {folder_name}")

def main():
    # Initialize the webcam (use 1 for external webcam, 0 for internal)
    cap = cv2.VideoCapture(1)  # Change 1 to 0 if internal webcam is to be used

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    card_index = 0
    image_count = 1  # Image count starts at 1

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            break

        # Display the frame
        cv2.imshow('Webcam', frame)

        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):  # 'c' key to capture and save the image
            if card_index < len(card_names):
                folder_name = f"folder{image_count}"
                img_name = f"{card_names[card_index]}.png"
                save_image(folder_name, img_name, frame)
                card_index += 1

                if card_index == len(card_names):
                    card_index = 0  # Reset after the last card
                    image_count += 1  # Move to next folder for subsequent images
            else:
                print("All cards have been saved.")
                break

        elif key == ord('q'):  # 'q' key to quit the application
            print("Exiting...")
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
