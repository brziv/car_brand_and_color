import gradio as gr
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np

# ------------------ Model ------------------
num_classes = 22

model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)

# Load trained weights
model.load_state_dict(torch.load("enetv2s.pth", map_location="cpu"))
model.eval()

# class names
with open("annot/brand.txt", "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f if line.strip()]

# ------------------ Hooks for Grad-CAM ------------------
gradients, features = [], []

def save_gradient(grad):
    gradients.append(grad)

def forward_hook(module, inp, out):
    features.append(out)

layer = model.features[-1]
layer.register_forward_hook(forward_hook)
layer.register_full_backward_hook(lambda m, gi, go: save_gradient(go[0]))

# ------------------ Preprocess ------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ------------------ Predict (top-5 like your ONNX) ------------------
def predict(img: Image.Image):
    x = transform(img).unsqueeze(0)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze().detach().numpy()

    top5_idx = probs.argsort()[-5:][::-1]
    return {class_names[i]: float(probs[i]) for i in top5_idx}

# ------------------ Grad-CAM ------------------
def grad_cam(img: Image.Image):
    gradients.clear()
    features.clear()

    x = transform(img).unsqueeze(0)
    output = model(x)
    pred = output.argmax()

    model.zero_grad()
    output[0, pred].backward()

    grad = gradients[-1]
    feat = features[-1]

    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * feat).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=(img.size[1], img.size[0]), mode='bilinear', align_corners=False)

    cam = cam.squeeze().detach().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min())

    heatmap = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(img_cv, 0.5, heatmap, 0.5, 0)

    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    top5 = predict(img)  # dùng predict ở trên
    return overlay, top5

# ------------------ Gradio UI ------------------
demo = gr.Interface(
    fn=grad_cam,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Image(label="Grad-CAM Heatmap"), gr.Label(label="Top-5 Prediction")],
    title="EfficientNet-V2-S Grad-CAM + Classification"
)

demo.launch()
