import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import torch
from torchvision import models, transforms
from PIL import Image

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 14)  # 14 classes
model.load_state_dict(torch.load("mobilenet_v2_crime_detection.pth", map_location=device))
model.to(device)
model.eval()

# Class labels
class_names = [
    "Abuse", "Arrest", "Arson","Assault", "Burglary", "Explosion",
    "Fighting", "Normal","Road accidents", "Robbery", "Shooting", "Shoplifting",
    "Stealing", "Vandalism", "Weapon Threat"
]


# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return class_names[predicted.item()]



@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_url = None
    if request.method == "POST":
        file = request.files["image"]
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            prediction = predict_image(file_path)
            image_url = file_path
    return render_template("index.html", prediction=prediction, image_url=image_url)

if __name__ == "__main__":
    app.run(debug=True)
