import torch
import dlib
import cv2
import imutils
import torchvision
import json
import sys
import urllib
import numpy as np

from torch.autograd import Variable

from model import CNN

""" Preprocess """
from fastapi import FastAPI, Request
from skimage import io

app = FastAPI()


def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    imageBGR = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image


def getData(url):
    # image_path = sys.argv[1]
    img = imutils.url_to_image(url)
    # img = cv2.imread(image_path)
    # # img=url_to_image('https://media.geeksforgeeks.org/wp-content/uploads/20211003151646/geeks14.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Face Crop
    face_cascade = cv2.CascadeClassifier(
        "data/haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.2, 4)
    if len(faces) <= 0:
        return {}
    height, width = img.shape[:2]

    x, y, w, h = faces[0]
    r = max(w, h) / 2
    centerx = x + w / 2
    centery = y + h / 2
    nx = int(centerx - r)
    ny = int(centery - r)
    nr = int(r * 2)

    img = img[ny:ny+nr, nx:nx+nr]
    img = cv2.resize(img, (200, 200))

    # Convert to Tensor
    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    img = preprocess(img)
    img.unsqueeze_(0)

    # Load model
    eye_model = CNN(3)
    eye_model.load_state_dict(torch.load("models/eye_model.pt"))

    eyebrow_model = CNN(3)
    eyebrow_model.load_state_dict(torch.load("models/eyebrow_model.pt"))

    jaw_model = CNN(4)
    jaw_model.load_state_dict(torch.load("models/jaw_model.pt"))

    mouth_model = CNN(3)
    mouth_model.load_state_dict(torch.load("models/mouth_model.pt"))

    nose_model = CNN(3)
    nose_model.load_state_dict(torch.load("models/nose_model.pt"))

    """ Predict """
    # Inference
    eyebrow = eyebrow_model(Variable(img))
    eye = eye_model(Variable(img))
    nose = nose_model(Variable(img))
    mouth = mouth_model(Variable(img))
    jaw = jaw_model(Variable(img))

    # Analysis
    types = dict()
    types["eyebrow"] = ["Arch", "Circle", "Straight"]
    types["eye"] = ["Big", "Silt", "Small"]
    types["nose"] = ["Long", "Small", "Wide"]
    types["mouth"] = ["Medium", "Small", "Thick"]
    types["jaw"] = ["Circle", "Oval", "Square", "Triangle"]

    eyebrow_type = types["eyebrow"][torch.argmax(eyebrow).item()]
    eye_type = types["eye"][torch.argmax(eye).item()]
    nose_type = types["nose"][torch.argmax(nose).item()]
    mouth_type = types["mouth"][torch.argmax(mouth).item()]
    jaw_type = types["jaw"][torch.argmax(jaw).item()]
    lst = [('eyebrows', eyebrow_type), ('eyes', eye_type),
           ('nose', nose_type), ('mouth', mouth_type), ('face', jaw_type)]
    results = {k: {'type': t, 'analysis': None} for k, t in lst}

    with open('data/analysis.json') as json_file:
        data = json.load(json_file)
        for region_name, region_type in lst:
            for region in data["face_regions"]:
                if region["name"] == region_name:
                    for feature in region["features"]:
                        if feature["name"] == region_type:
                            results[region_name]['analysis'] = feature["analysis"]

    """ Json """
    # results = json.dumps(results)
    # print(results)
    return results


@app.post("/face_analysis")
async def root(request: Request):
    data = await request.json()
    return getData(data['url'])
