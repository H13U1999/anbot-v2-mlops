
import requests
from flask import Flask, request, jsonify
from NTS_model import DiscordBotNeuralStyleTransfer
from dotenv import load_dotenv
import pyrebase
import os
load_dotenv()


firebaseConfig = {
    "apiKey": os.getenv('API_KEY'),
    "authDomain": os.getenv('AUTH_DOMAIN'),
    "projectId": os.getenv('PROJECT_ID'),
    "storageBucket": os.getenv('STORAGE_BUCKET'),
    "messagingSenderId": os.getenv('MESSAGING_SENDER_ID'),
    "appId": os.getenv('APP_ID'),
    "measurementId": os.getenv('MEASUREMENT_ID'),
    "databaseURL": ""
}

predictor = DiscordBotNeuralStyleTransfer()
firebase = pyrebase.initialize_app(firebaseConfig)
storage = firebase.storage()

auth = firebase.auth()


app = Flask(__name__)
PORT = os.getenv('PORT')


@app.route("/combine-NTS", methods=["POST"])
def return_style_transfer():
    img_url_1 = str(request.json['img_1'])
    img_url_2 = str(request.json['img_2'])

    stylized_image = predictor.combine_style(img_url_1, img_url_2)
    storage.child("NTS-images/"+stylized_image).put(stylized_image)
    os.remove(stylized_image)
    return {"url": storage.child("NTS-images/"+stylized_image).get_url(None)}


@app.route("/health", methods=["GET"])
def health_check():
    return "ok"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
