from flask import Flask, request, jsonify, send_file
import os
import torch
import librosa
import numpy as np
from model_def import Convautoenc
from utils import signal2pytorch
import soundfile as sf


app = Flask(__name__)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = Convautoenc().to(device)
checkpoint = torch.load("audio_autoenc.torch", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route("/", methods=["GET"])
def home():
    return "Audio Denoising API. Use /upload to upload an audio file."

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Process the audio file
    output_path = os.path.join(PROCESSED_FOLDER, "denoised_" + file.filename)
    denoise_audio(filepath, output_path)

    return send_file(output_path, as_attachment=True)

def denoise_audio(input_path, output_path):
    # Load and process the audio file
    audio, sr = librosa.load(input_path, mono=True, sr=None)
    audio = audio / np.abs(audio).max()  # Normalize
    X_input = signal2pytorch(audio).to(device)

    # Denoise
    with torch.no_grad():
        output_audio = model(X_input).cpu().numpy().flatten()

    # Save output
    sf.write(output_path, output_audio, sr)

if __name__ == "__main__":
    app.run(debug=True)
