from cmath import isnan
import json, os
from os import listdir
from flask_cors import CORS
from flask import Flask, Response, request
from werkzeug.utils import secure_filename
from spine_inference import *

app = Flask(__name__)
CORS(app)

@app.route("/home")
def home():
    return "<h1>GFG is great platform to learn</h1>"

def detect_centhroids(img_dir):
    bbox_predictions = detect(img_dir)
    bbox_predictions = bbox_predictions[0]

    centhroids = []
    for bbox in bbox_predictions:
        x1, y1, x2, y2 = bbox
        centroid = (x1 + x2) / 2, (y1 + y2) / 2
        centhroids.append(centroid)

    assert len(os.listdir(img_dir)) == 1, "Only one image file is allowed in the directory"
    image_file = os.listdir(img_dir)[0]
    image_path = os.path.join(img_dir, image_file)
    image = cv2.imread(image_path)
    for centroid in centhroids:
        x, y = centroid
        cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)

    img_id = image_file.split('.')[0]
    centroid_detected_path = img_dir.replace("images", "detected") + "/" + f"{img_id}_centroid_detected.jpg"
    cv2.imwrite(centroid_detected_path, image)
    return centhroids, image_path

def cobb_angle_calculation(centhroids, image_path):
    centhroids_copy = centhroids.copy()

    image = cv2.imread(image_path)
    height = image.shape[0]
    # keep points in the middle of the height
    centhroids_middled = [centroid for centroid in centhroids_copy if centroid[1] < height / 1.3]
    centhroids_middled = [centroid for centroid in centhroids_middled if centroid[1] > height / 5]

    for centroid in centhroids_middled:
        x, y = centroid
        cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)

    img_dir, image_file = os.path.split(image_path)
    img_id = image_file.split('.')[0]
    centroid_detected_path = img_dir.replace("images", "detected") + "/" + f"{img_id}_middled_centroid_detected.jpg"
    cv2.imwrite(centroid_detected_path, image)

    # sort points by y
    centhroids_middled.sort(key=lambda x: x[1])
    bottom_point = centhroids_middled[0]
    top_point = centhroids_middled[-1]

    cv2.circle(image, (int(bottom_point[0]), int(bottom_point[1])), 5, (0, 255, 0), -1)
    cv2.circle(image, (int(top_point[0]), int(top_point[1])), 5, (0, 255, 0), -1)

    rightest_point_idx = np.argmax([centroid[0] for centroid in centhroids_middled])
    rightest_point = centhroids_middled[rightest_point_idx]

    # angle between bottom_point, top_point and rightest_point
    l1 = np.array(bottom_point) - np.array(rightest_point)
    l2 = np.array(top_point) - np.array(rightest_point)
    cosine_angle_rightest = np.dot(l1, l2) / (np.linalg.norm(l1) * np.linalg.norm(l2))

    leftest_point_idx = np.argmin([centroid[0] for centroid in centhroids_middled])
    leftest_point = centhroids_middled[leftest_point_idx]

    # angle between bottom_point, top_point and leftest_point
    l1 = np.array(bottom_point) - np.array(leftest_point)
    l2 = np.array(top_point) - np.array(leftest_point)
    cosine_angle_leftest = np.dot(l1, l2) / (np.linalg.norm(l1) * np.linalg.norm(l2))

    if isnan(cosine_angle_rightest) and isnan(cosine_angle_leftest):
        assert False, "Cobb angle cannot be calculated"

    if isnan(cosine_angle_rightest):
        if not isnan(cosine_angle_leftest):
            cosine_angle = cosine_angle_leftest

    if isnan(cosine_angle_leftest):
        if not isnan(cosine_angle_rightest):
            cosine_angle = cosine_angle_rightest

    if not isnan(cosine_angle_rightest) and not isnan(cosine_angle_leftest):
        cosine_angle = max(cosine_angle_rightest, cosine_angle_leftest)

    angle = np.arccos(cosine_angle)
    angle_dgree = int(np.degrees(angle) + 5)
    cobb_angle = 180 - angle_dgree
    return f"{cobb_angle} °"

def scoliosis_detection(img_dir):
    centhroids, image_path = detect_centhroids(img_dir)
    cobb_angle = cobb_angle_calculation(centhroids, image_path)
    return cobb_angle

@app.route("/scoliosis", methods=["POST"])
def detection():
    try:
        spineimagefile = request.files['spine']
        filename= secure_filename(spineimagefile.filename)

        file_id = filename.split('.')[0]
        img_dir = f"inference/images/{file_id}"
        os.makedirs(img_dir, exist_ok=True)

        file_path = os.path.join(img_dir, filename)
        spineimagefile.save(file_path)

        "Image stored in the directory: ", img_dir

        cobb_angle = scoliosis_detection(img_dir)
        # for file_name in listdir(img_dir):
        #   os.remove(img_dir + file_name)

        return Response(
                    json.dumps({"cobb_angle": cobb_angle}), 
                                status=200, 
                                mimetype='application/json'
                                )

    except Exception as e:
        return Response(
                    json.dumps({"error": "Anomaly detected. Please insert a valid x-ray image."}), 
                                status=500, 
                                mimetype='application/json'
                                )


if __name__ == "__main__":
    app.run(
            debug=True, 
            port=5000, 
            host='0.0.0.0', 
            threaded=False, 
            use_reloader=True
            )