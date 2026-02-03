import cv2
import numpy as np
import os
import urllib.request
import sys

# Try to import tensorflow and handle potential version issues
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import img_to_array
except ImportError:
    print("\n" + "="*50)
    print("ERROR: TensorFlow not found!")
    print(f"You are currently running Python {sys.version.split()[0]}")
    print("TensorFlow does not support Python 3.14 yet.")
    print("Please run this script using Python 3.8 by copying this command:")
    print("& \"C:\\Users\\ASUS\\AppData\\Local\\Microsoft\\WindowsApps\\PythonSoftwareFoundation.Python.3.8_qbz5n2kfra8p0\\python.exe\" face_emotion_detection.py")
    print("="*50 + "\n")
    sys.exit(1)

# Constants
MODEL_URL = "https://github.com/oarriaga/face_classification/raw/master/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5"
MODEL_PATH = "fer2013_mini_XCEPTION.hdf5"
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def download_model():
    """Download the pre-trained model if it doesn't exist."""
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading pre-trained model from {MODEL_URL}...")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("Model downloaded successfully.")
        except Exception as e:
            print(f"Error downloading model: {e}")
            sys.exit(1)

def load_emotion_model():
    """Load the Keras model."""
    print("Loading emotion recognition model...")
    if not os.path.exists(MODEL_PATH):
        download_model()
    
    try:
        model = load_model(MODEL_PATH, compile=False)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def get_face_classifier():
    """Initialize OpenCV's Haar Cascade face detector."""
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if not os.path.exists(cascade_path):
        # Fallback for some environments where cv2.data.haarcascades might not be set correctly
        print("Warning: Haar cascade path not found in cv2.data.haarcascades.")
    
    classifier = cv2.CascadeClassifier(cascade_path)
    if classifier.empty():
        print("Error: Could not load Haar cascade classifier.")
        sys.exit(1)
    return classifier

def detect_emotions(frame, face_classifier, model):
    """Detect faces and predict emotions."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Preprocess face ROI
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (64, 64), interpolation=cv2.INTER_AREA)
        roi = roi_gray.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # Predict emotion
        preds = model.predict(roi, verbose=0)[0]
        label = EMOTION_LABELS[preds.argmax()]
        
        # UI Elements
        # Ensure text is visible even if face is at the top edge
        label_y = y - 10 if y - 10 > 10 else y + 20
        label_position = (x, label_y)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return frame

def main():
    # Initialization
    model = load_emotion_model()
    face_classifier = get_face_classifier()

    # Start Video Capture
    print("Accessing webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam. Please check your camera connection.")
        return

    print("Webcam started. Press 'q' to quit.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            # Process frame
            frame = detect_emotions(frame, face_classifier, model)
            
            # Display output
            cv2.imshow('Facial Emotion Recognition', frame)

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam released and windows closed.")

if __name__ == "__main__":
    main()
