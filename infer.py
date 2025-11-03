import gradio as gr
import onnxruntime as ort
import numpy as np
import cv2
import torch

# model & config
model_path = "enetv2s.onnx"
input_size = 224

with open("annot/brand.txt", "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f if line.strip()]

# load onnx
session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_size, input_size))
    img = img.astype(np.float32) / 255.0
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0).astype(np.float32)
    return img

def predict(image):
    img = preprocess(image)
    logits = session.run([output_name], {input_name: img})[0]
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[0]
    
    top5_idx = probs.argsort()[-5:][::-1]
    return {class_names[i]: float(probs[i]) for i in top5_idx}

ui = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Upload Image"),
    outputs=[
        gr.Label(label="Prediction")
    ],
    title="Car brand classification"
)

if __name__ == "__main__":
    ui.launch(server_name="0.0.0.0", server_port=7860)
