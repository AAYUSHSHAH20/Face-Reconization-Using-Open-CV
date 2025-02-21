import json
import os
import base64
import cv2
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from .face_recognizer import FaceRecognitionSystem
import numpy as np

# Initialize face recognition system
DATA_PATH = os.path.join(settings.MEDIA_ROOT, 'training_data')
# Create directories if they don't exist
os.makedirs(DATA_PATH, exist_ok=True)
face_system = FaceRecognitionSystem(DATA_PATH)
face_system.train_recognizer()


print("DATA_PATH:", DATA_PATH)
print("Contents:", os.listdir(DATA_PATH))

for folder in os.listdir(DATA_PATH):
    person_dir = os.path.join(DATA_PATH, folder)
    if os.path.isdir(person_dir):
        print(f"{folder}: {os.listdir(person_dir)}")



def index(request):
    """Home page view"""
    return render(request, 'face_app/index.html')

def live_detection(request):
    """Live face detection page"""
    return render(request, 'face_app/live_detection.html')

@csrf_exempt
def process_frame(request):
    """API endpoint to process frames sent from the client"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            frame_data = data.get('frame')
            
            if not frame_data:
                return JsonResponse({"success": False, "error": "No frame data provided"})
            
            # Process the frame
            result = face_system.process_frame(frame_data)
            print(result)
            return JsonResponse(result)
            
        except Exception as e:
            return JsonResponse({"success": False, "error": str(e)})
    
    return JsonResponse({"success": False, "error": "Invalid request method"})


@csrf_exempt
def upload_image(request):
    """Handle image upload and perform face recognition"""
    if request.method == 'POST' and request.FILES.get('photo'):
        image_file = request.FILES['photo']
        image_path = os.path.join(settings.MEDIA_ROOT, 'uploads', image_file.name)
        
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        
        with open(image_path, 'wb') as f:
            for chunk in image_file.chunks():
                f.write(chunk)

        # Read the uploaded image
        image = cv2.imread(image_path)
        if image is None:
            return JsonResponse({"success": False, "error": "Invalid image"})

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_system.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        print(faces)
        results = []
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (100, 100))
            print(face_system.id_to_name)
            if hasattr(face_system, 'recognizer') and len(face_system.id_to_name) > 0:
                label, confidence = face_system.recognizer.predict(face_roi)
                confidence_score = 100 - min(confidence, 100)
                
                if confidence < 100:
                    name = face_system.id_to_name.get(label, "Unknown")
                    confidence_text = f"{confidence_score:.1f}%"
                else:
                    name = "Unknown"
                    confidence_text = "N/A"
            else:
                name = "Unknown"
                confidence_text = "N/A"

            results.append({
                "box": [int(x), int(y), int(w), int(h)],
                "name": name,
                "confidence": confidence_text
            })

        return JsonResponse({"success": True, "faces": results})
    
    return JsonResponse({"success": False, "error": "No image uploaded"})