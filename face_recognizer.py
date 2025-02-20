import cv2
import os
import numpy as np

class FaceRecognitionSystem:
    def __init__(self, data_path):
        self.data_path = data_path
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Store mappings between person names and labels
        self.name_to_id = {}
        self.id_to_name = {}

        # Assign unique IDs to names
        for person_id, person_name in enumerate(sorted(os.listdir(self.data_path))):
            person_dir = os.path.join(self.data_path, person_name)
            if os.path.isdir(person_dir):
                self.name_to_id[person_name] = person_id
                self.id_to_name[person_id] = person_name

    def prepare_training_data(self):
        """Prepare training data from images"""
        faces = []
        labels = []
        
        print("Preparing training data...")
        
        for person_name, person_id in self.name_to_id.items():
            person_dir = os.path.join(self.data_path, person_name)
            
            # Loop through all images of the person
            for image_name in os.listdir(person_dir):
                if image_name.endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    image_path = os.path.join(person_dir, image_name)
                    
                    # Read and convert image to grayscale
                    image = cv2.imread(image_path)
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    gray = cv2.equalizeHist(gray)  # Improve contrast
                    
                    # Detect faces
                    detected_faces = self.face_cascade.detectMultiScale(
                        gray, scaleFactor=1.05, minNeighbors=4, minSize=(30, 30)
                    )

                    if len(detected_faces) > 0:
                        (x, y, w, h) = detected_faces[0]  # Take the first face
                        print(f"✅ Face detected in {image_name}: x={x}, y={y}, w={w}, h={h}")
                        face = gray[y:y+h, x:x+w]
                        face = cv2.resize(face, (100, 100))  # Normalize size
                        faces.append(face)
                        labels.append(person_id)
                        print(f"✔ Face detected in {image_name}")
                    else:
                        print(f"❌ No face detected in {image_name}")
        
        print(f"Total faces detected for training: {len(faces)}")
        return faces, labels
    
    def train_recognizer(self):
        """Train the face recognizer"""
        print("Training face recognizer...")
        faces, labels = self.prepare_training_data()
        
        if len(faces) == 0:
            raise Exception("No faces found in training data!")

        self.recognizer.train(faces, np.array(labels))
        print("Training completed successfully! 🎯")
    
    def start_recognition(self):
        """Start live face recognition"""
        print("Starting face recognition...")
        video_capture = cv2.VideoCapture(0)
        
        if not video_capture.isOpened():
            print("Error: Could not open webcam")
            return
            
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Error: Could not read frame")
                break

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)  # Improve contrast
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=4, minSize=(30, 30)
            )

            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (100, 100))
                
                # Predict the face
                label, confidence = self.recognizer.predict(face_roi)
                print(label)
                print(confidence)
                # Confidence threshold
                if confidence < 80:  # Lower confidence is better
                    name = self.id_to_name.get(label, "Unknown")
                    confidence_text = f"{confidence:.1f}%"
                else:
                    name = "Unknown"
                    confidence_text = ""

                # Draw rectangle and label
                color = (0, 255, 0)  # Green
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, name, (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                cv2.putText(frame, confidence_text, (x, y+h+25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Display the frame
            cv2.imshow('Face Recognition', frame)

            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        video_capture.release()
        cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    data_path = "training_data"  # Make sure this folder exists
    frs = FaceRecognitionSystem(data_path)
    frs.train_recognizer()
    frs.start_recognition()
