from flask import Flask, request, jsonify
import cv2
import numpy as np
import pickle

app = Flask(__name__)

positions_path = "carPos-2"
with open(positions_path, "rb") as f:
    posList = pickle.load(f)

labels = [f"Spot-{i+1}" for i in range(len(posList))]

width, height = 210, 158

def check_parking_spots(img):
    results = []
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    _, imgThreshold = cv2.threshold(imgBlur, 110, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    imgClosing = cv2.morphologyEx(imgThreshold, cv2.MORPH_CLOSE, kernel)
    imgDilate = cv2.dilate(imgClosing, kernel, iterations=1)

    for idx, pos in enumerate(posList):
        x, y = pos
        imgCrop = imgDilate[y:y + height, x:x + width]
        countBlack = cv2.countNonZero(cv2.bitwise_not(imgCrop))
        if countBlack < 10000:
            results.append({"label": labels[idx], "status": "empty"})
        else:
            results.append({"label": labels[idx], "status": "occupied"})
    return results

@app.route("/process-frame", methods=["POST"])
def process_frame():
    nparr = np.frombuffer(request.data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    results = check_parking_spots(img)
    return jsonify(results)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
