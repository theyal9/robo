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
    os.makedirs(folder_path, exist_ok=True)  # Ensure the folder is created
    img_path = os.path.join(folder_path, img_name)
    if cv2.imwrite(img_path, image):
        print(f"Saved {img_name} in {folder_path}")
    else:
        print(f"Error: Could not save {img_name} in {folder_path}")

def main():
    # Initialize the webcam (try changing to 0 if using an internal webcam)
    cap = cv2.VideoCapture(1)  # Change to 0 if you are using an internal webcam

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
        if key == ord('c'):  # Press 'c' to capture and save the image
            if card_index < len(card_names):
                folder_name = f"{card_names[card_index]}"
                img_name = f"{card_names[card_index]}124.png"
                print(f"Attempting to save {img_name} in folder {folder_name}...")
                
                # Attempt to save the image
                save_image(folder_name, img_name, frame)
                
                card_index += 1

                if card_index == len(card_names):
                    card_index = 0  # Reset after the last card
                    image_count += 1  # Move to the next folder for subsequent images
            else:
                print("All cards have been saved.")
                break

        elif key == ord('q'):  # Press 'q' to quit the application
            print("Exiting...")
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()