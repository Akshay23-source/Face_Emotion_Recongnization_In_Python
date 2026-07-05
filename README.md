

# 😊 Face Emotion Recognition using Python

## 📌 Project Overview

Face Emotion Recognition is a Computer Vision and Deep Learning application that detects human faces in real time using a webcam and predicts the emotion displayed by each detected face.

The project combines **OpenCV**, **TensorFlow/Keras**, and a pre-trained **Mini-XCEPTION** deep learning model to recognize facial expressions such as:

* 😀 Happy
* 😢 Sad
* 😠 Angry
* 😨 Fear
* 😮 Surprise
* 😐 Neutral


The application automatically downloads the required pre-trained model during its first execution and then performs live emotion detection from the webcam.

---

# 🎯 Objectives

The main objectives of this project are:

* Detect human faces in real time.
* Classify facial emotions using Deep Learning.
* Demonstrate practical use of Computer Vision.
* Learn image preprocessing techniques.
* Build a real-world AI application using Python.

---

# 🚀 Features

* Real-time webcam emotion detection
* Automatic face detection
* Emotion classification
* Automatic model download
* Lightweight implementation
* Easy to understand code
* Beginner-friendly project structure

---

# 🛠 Technologies Used

| Technology    | Purpose                           |
| ------------- | --------------------------------- |
| Python        | Programming Language              |
| OpenCV        | Image Processing & Face Detection |
| TensorFlow    | Deep Learning Framework           |
| Keras         | Loading the trained model         |
| NumPy         | Numerical Operations              |
| Haar Cascade  | Face Detection                    |
| Mini-XCEPTION | Emotion Classification Model      |

---

# 📂 Project Structure

```
Face_Emotion_Recongnization_In_Python/
│
├── face_emotion_detection.py
├── requirements.txt
├── run.bat
├── README.md
└── fer2013_mini_XCEPTION.hdf5
      (Downloaded Automatically)
```

---

# 📁 Files Explanation

## 1. face_emotion_detection.py

This is the main application file.

Responsibilities:

* Loads the emotion recognition model.
* Downloads the model if missing.
* Opens webcam.
* Detects faces.
* Preprocesses detected faces.
* Predicts emotions.
* Displays prediction on screen.

Main functions include:

### download_model()

Downloads the pretrained Mini-XCEPTION model automatically.

---

### load_emotion_model()

Loads the downloaded Keras model into memory.

---

### get_face_classifier()

Loads OpenCV Haar Cascade classifier for face detection.

---

### detect_emotions()

Processes each webcam frame by:

* Converting image to grayscale
* Detecting faces
* Resizing face
* Normalizing pixels
* Running prediction
* Drawing bounding box
* Displaying predicted emotion

---

### main()

Controls the complete workflow:

* Load model
* Load Haar Cascade
* Start webcam
* Detect emotions continuously
* Exit when **Q** is pressed

---

## 2. requirements.txt

Contains all required Python packages.

```
opencv-python
numpy
tensorflow
```

Install using:

```bash
pip install -r requirements.txt
```

---

## 3. run.bat

A Windows batch file that starts the application automatically without typing Python commands manually.

---


# ⚙ Working Flow

```
Start Program
      │
      ▼
Load Emotion Model
      │
      ▼
Load Haar Cascade
      │
      ▼
Open Webcam
      │
      ▼
Capture Video Frame
      │
      ▼
Convert to Grayscale
      │
      ▼
Detect Face
      │
      ▼
Resize Face Image
      │
      ▼
Normalize Pixels
      │
      ▼
Predict Emotion
      │
      ▼
Display Result
      │
      ▼
Repeat Until User Presses Q
```

---

# 🧠 Machine Learning Model

The project uses the **Mini-XCEPTION** Convolutional Neural Network trained on the **FER2013 (Facial Expression Recognition 2013)** dataset.

Recognized emotions:

* Angry
* Fear
* Happy
* Sad
* Surprise
* Neutral



# 📋 Requirements

* Python 3.10 or 3.11 (recommended)
* Webcam
* Internet connection (first run only)
* Windows/Linux/macOS

---

# 📊 Output

The application:

* Detects faces
* Draws a green bounding box
* Predicts emotion
* Displays emotion label above the face
* Updates continuously in real time



# 📚 Learning Outcomes

Through this project, I learned:

* Computer Vision fundamentals
* Face Detection using OpenCV
* Deep Learning inference using TensorFlow
* Image preprocessing
* Real-time webcam processing
* Model loading and prediction
* Python project organization

---

# 👨‍💻 Author

**Akshay Gabrieal R**

Final Year Computer Science Engineering Student




