from face_recognizer import FaceRecognitionSystem
import cv2
import os

def main():
    # Define the directory containing training images
    data_path = "training_data"
    
    # Create directory if it doesn't exist
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        print(f"Created directory: {data_path}")
        print("Please add person folders with training images before running again.")
        return
        
    try:
        # Initialize the face recognition system
        face_system = FaceRecognitionSystem(data_path)
        
        # Train the recognizer
        face_system.train_recognizer()
        
        # Start recognition
        face_system.start_recognition()
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()