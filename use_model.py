import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import os
import json
from collections import OrderedDict

# --- Correct setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
mapping_file = os.path.join('models', 'class_mappings.json')
with open(mapping_file, 'r') as f:
    maps = json.load(f)
class_to_idx = maps['class_to_idx']
idx_to_class = maps['idx_to_class']
print(f"Loaded mappings for {len(idx_to_class)} classes.")

# --- THE GUARANTEED FIX ---

# 1. Build the model architecture EXACTLY as it was before compiling in training
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
num_classes = len(class_to_idx)
num_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(num_features, 512),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5, inplace=True),
    nn.Linear(512, num_classes)
)

# 2. Load the state_dict from your compiled model file
model_path = os.path.join('models', '92accuracyflag.pth')
state_dict = torch.load(model_path, map_location=device)

# 3. Create a new, clean dictionary and strip the `_orig_mod.module.` prefix
cleaned_state_dict = OrderedDict()
prefix_to_strip = '_orig_mod.'
for k, v in state_dict.items():
    print(f"Processing key: {k}")
    if k.startswith(prefix_to_strip):
        name = k[len(prefix_to_strip):]
        cleaned_state_dict[name] = v
    else:
        # If for some reason a key doesn't have the prefix, keep it as is
        cleaned_state_dict[k] = v

# 4. Load the cleaned dictionary into the WHOLE model.
#    Since the file contains weights for the whole model, we load it into model,
#    not just model.classifier.
model.load_state_dict(cleaned_state_dict)

# 5. Prepare model for inference
model.to(device)
model.eval()

print("Model loaded successfully!")

def predict_flag(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    try:
        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        predicted_class = idx_to_class[str(predicted_idx.item())]
        confidence_score = confidence.item()
        return predicted_class, confidence_score
    except Exception as e:
        print(f"Error predicting image: {e}")
        return None, 0.0

if __name__ == '__main__':
    test_image_path = "image.png"
    image = Image.open(test_image_path).convert('RGB')
    if os.path.exists(test_image_path):
        predicted_country, confidence = predict_flag(image)
        if predicted_country:
            print(f"Predicted country: {predicted_country} (Confidence: {confidence:.4f})")
    else:
        print(f"Test image not found at: {test_image_path}")
