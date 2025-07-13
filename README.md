# American-Sign-Language-Detection-Model
This project is a real-time hand gesture recognition system that detects hand gestures from the webcam and classifies them into alphabets (A–Z) using a trained machine learning model.

It has two parts:

1. Data Collection for training custom gestures.

2. Prediction & Classification using a Flask web application or a simple OpenCV window.

✨ Features
1. Hand detection using cvzone.HandTrackingModule

2. Gesture cropping and resizing for model input

3. Alphabet classification (A–Z) using a pre-trained Keras model

4. Real-time webcam support

5. Flask web app to serve gesture recognition in browser

6. Toggle video stream and prediction in the browser interface

🗂️ Project Structure
├── DataCollection.py      # Script for capturing gesture images
├── test.py                # Script for testing the model in OpenCV
├── app.py                 # Flask web app for real-time recognition
├── Model/
│   ├── keras_model.h5     # Trained Keras model
│   └── labels.txt         # Label mapping for model output
├── templates/
│   └── index.html         # Web interface for the app
└── Data/
    └── eg/                # Captured gesture images

🔧 Requirements
Install dependencies using pip:
pip install opencv-python cvzone numpy flask tensorflow
Ensure you have a webcam connected.


🚀 How to Run
1. Data Collection
To collect gesture images:

python DataCollection.py
Press 's' to save a frame.
Press 'q' to quit.
Images will be saved in Data/eg/.

2. Model Training (Optional)
You can train your own model using the captured images. The current model is stored at Model/keras_model.h5.
(visit TeachableMachine website by Google to train your model)

3. Run Local Test (OpenCV)
To test prediction in a window:
python test.py

4. Run Web App
To launch the Flask-based web app:
python app.py
Visit http://127.0.0.1:5000/ in your browser.
Toggle video and prediction using buttons.

📷 Sample Output
The app displays real-time camera feed with bounding box and predicted label on top of the hand.

📌 Notes
1. Model and label files must be inside the Model/ folder.

2. Captured data should be cleaned and labeled before training.

3. You can enhance it further by supporting numbers, dynamic gestures, or sign language.

🧑‍💻 Author
Dharm Rathod
BTech CSE-AIML | PPSU College


