import sys
import json
from PIL import Image
import torch
import torchvision.transforms as transforms
from model_definition import PlantDiseaseModel

MODEL_PATH = 'models/best_model4.pth'

# Class names list
class_names = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___healthy",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___healthy",
    "Potato___Late_blight",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___healthy",
    "Strawberry___Leaf_scorch",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___healthy",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
]

def load_model(path):
    try:
        print("Loading model and weights…", file=sys.stderr)
        model = PlantDiseaseModel(num_classes=38)
        state_dict = torch.load(path, map_location='cpu')
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.eval()
        print("Model loaded successfully.", file=sys.stderr)
        return model
    except Exception as e:
        err = {'error': f'Model initialization failed: {str(e)}'}
        print(json.dumps(err), file=sys.stderr)
        sys.exit(1)

model = load_model(MODEL_PATH)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def infer(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            output = model(tensor)
        pred_class_idx = int(output.argmax(dim=1).item())  # Get the predicted class index
        confidence = float(output.softmax(dim=1)[0][pred_class_idx].item())  # Get the confidence score
        class_label = class_names[pred_class_idx]  # Get the class name using the index

        # Return both the class number (index) and the class name
        return {'class_number': pred_class_idx, 'class_name': class_label, 'confidence': confidence}
    except Exception as e:
        err = {'error': f'Inference failed: {str(e)}'}
        print(json.dumps(err), file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        err = {'error': 'No image path provided.'}
        print(json.dumps(err), file=sys.stderr)
        sys.exit(1)

    image_path = sys.argv[1]
    result = infer(image_path)
    sys.stdout.write(json.dumps(result))
    sys.stdout.flush()
