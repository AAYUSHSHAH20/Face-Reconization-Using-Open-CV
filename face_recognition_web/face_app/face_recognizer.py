import cv2
import os
import numpy as np
import base64
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        
        # Check if training data exists
        if os.path.exists(self.data_path):
            # Assign unique IDs to names
            for person_id, person_name in enumerate(sorted(os.listdir(self.data_path))):
                person_dir = os.path.join(self.data_path, person_name)
                if os.path.isdir(person_dir):
                    actual_name = person_name.split("_", 1)[-1]  # Extracts the real name after "personX_"
                    self.name_to_id[actual_name] = person_id
                    self.id_to_name[person_id] = actual_name  # ✅ Stores just "elonmusk" or "messi"
            
            # Train if data exists
            if self.name_to_id:
                try:
                    self.train_recognizer()
                except Exception as e:
                    logger.error(f"Error training recognizer: {str(e)}")

    def prepare_training_data(self):
        """Prepare training data from images"""
        faces = []
        labels = []
        
        logger.info("Preparing training data...")
        
        for person_name, person_id in self.name_to_id.items():
            person_dir = os.path.join(self.data_path, person_name)
            
            # Loop through all images of the person
            for image_name in os.listdir(person_dir):
                if image_name.endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    image_path = os.path.join(person_dir, image_name)
                    
                    # Read and convert image to grayscale
                    image = cv2.imread(image_path)
                    if image is None:
                        logger.warning(f"Could not read image: {image_path}")
                        continue
                        
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    gray = cv2.equalizeHist(gray)  # Improve contrast
                    
                    # Detect faces
                    detected_faces = self.face_cascade.detectMultiScale(
                        gray, scaleFactor=1.05, minNeighbors=4, minSize=(30, 30)
                    )

                    if len(detected_faces) > 0:
                        (x, y, w, h) = detected_faces[0]  # Take the first face
                        logger.info(f"Face detected in {image_name}: x={x}, y={y}, w={w}, h={h}")
                        face = gray[y:y+h, x:x+w]
                        face = cv2.resize(face, (100, 100))  # Normalize size
                        faces.append(face)
                        labels.append(person_id)
                    else:
                        logger.warning(f"No face detected in {image_name}")
        
        logger.info(f"Total faces detected for training: {len(faces)}")
        return faces, labels
    
    def train_recognizer(self):
        """Train the face recognizer"""
        logger.info("Training face recognizer...")
        faces, labels = self.prepare_training_data()
        
        if len(faces) == 0:
            raise Exception("No faces found in training data!")

        self.recognizer.train(faces, np.array(labels))
        logger.info("Training completed successfully! 🎯")
    
    def process_frame(self, frame_data):
        """Process a frame from the webcam (base64 encoded)"""
        try:
            # Decode base64 image
            image_data = base64.b64decode(frame_data.split(',')[1])
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return {"success": False, "error": "Invalid frame data"}

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)  # Improve contrast
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=4, minSize=(30, 30)
            )
            
            results = []
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (100, 100))
                
                if hasattr(self, 'recognizer') and len(self.id_to_name) > 0:
                    # Predict the face
                    label, confidence = self.recognizer.predict(face_roi)
                    
                    # Convert confidence (lower is better in LBPH) to percentage (higher is better)
                    confidence_score = 100 - min(confidence, 100)
                    print(confidence_score)
                    print(confidence)
                    # Confidence threshold
                    if confidence < 80:  # Lower confidence is better in LBPH
                        name = self.id_to_name.get(label, "Unknown")
                        confidence_text = f"{confidence_score:.1f}%"
                    else:
                        name = "Unknown"
                        confidence_text = f"{confidence_score:.1f}%"
                else:
                    name = "Unknown"
                    confidence_text = "N/A"
                
                results.append({
                    "box": [int(x), int(y), int(w), int(h)],
                    "name": name,
                    "confidence": confidence_text
                })
            
            return {
                "success": True,
                "faces": results
            }
        
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return {"success": False, "error": str(e)}