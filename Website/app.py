from flask import Flask, render_template, Response
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import math

app = Flask(__name__)

# Global variables
cap = cv2.VideoCapture(0)
video_enabled = True
prediction_enabled = True


def gen_frames():
    global cap, video_enabled, prediction_enabled

    detector = HandDetector(maxHands=1)
    classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

    offset = 20
    imgSize = 300

    labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
              "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]


    while True:
        if video_enabled:
            if cap is None:
                cap = cv2.VideoCapture(0)  # Create capture if not already created
            success, img = cap.read()
            if not success:
                continue  # Skip to the next iteration if reading from camera fails
            imgOutput = img.copy()
            hands, img = detector.findHands(img)
            if hands and prediction_enabled:
                hand = hands[0]
                x, y, w, h = hand['bbox']

                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

                aspectRatio = h / w

                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    imgResizeShape = imgResize.shape
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)


                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)

                cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 90, y - offset - 50 + 50),
                              (255, 0, 255), cv2.FILLED)
                cv2.putText(imgOutput, labels[index], (x, y - 25), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
                cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 3)

            ret, buffer = cv2.imencode('.jpg', imgOutput)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            if cap is not None:
                cap.release()  # Release the camera capture if video is disabled
                cap = None  # Reset cap object to None
            # If video is disabled, yield an empty frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/toggle_video')
def toggle_video():
    global video_enabled
    video_enabled = not video_enabled
    return 'Success'


@app.route('/toggle_prediction')
def toggle_prediction():
    global prediction_enabled
    prediction_enabled = not prediction_enabled
    return 'Success'


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
