import cv2
import torch
import torchvision.transforms as transforms
from transformers import AutoModelForImageClassification
from PIL import Image
import pytesseract
import numpy as np

# Preprocessing function from your ocr_script.py
def preprocess_image(image):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Apply adaptive thresholding
    thresh_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
    
    # Apply morphological transformations to enhance text
    kernel = np.ones((3, 3), np.uint8)
    processed_image = cv2.morphologyEx(thresh_image, cv2.MORPH_CLOSE, kernel)
    
    return processed_image

# Function for OCR text detection
def detect_text(image):
    # Preprocess the image using the function from ocr_script.py
    processed_image = preprocess_image(image)
    
    # Use Tesseract to detect text
    text = pytesseract.image_to_string(processed_image)
    return text

# Function for Fruits/Vegetables Detection (HFT Model)
def classify_fruit_vegetable(image):
    model = AutoModelForImageClassification.from_pretrained("jazzmacedo/fruits-and-vegetables-detector-36")
    labels = list(model.config.id2label.values())

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    pil_image = Image.fromarray(image)
    input_tensor = preprocess(pil_image).unsqueeze(0)

    # Run the image through the model
    outputs = model(input_tensor)

    # Get the predicted label index
    predicted_idx = torch.argmax(outputs.logits, dim=1).item()

    # Get the predicted label text
    predicted_label = labels[predicted_idx]
    return predicted_label

# Main function to automatically decide between OCR and classification
def main():
    # Hardcoded image path (replace this with your image path)
    image_path = r"C:\Users\91824\Desktop\springboard\sample_images\orange.jpg"

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load the image from {image_path}")
        return

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Step 1: Preprocess and detect text to determine if it's a packet
    detected_text = detect_text(image)

    # Step 2: Decide whether to run OCR or Fruit/Vegetable classification
    if len(detected_text.strip()) > 10:  # Arbitrary threshold for "significant text"
        print("Significant text detected. Running OCR...")

        print("\nTesseract OCR Output:")
        print(detected_text)

    else:
        print("Minimal text detected. Running fruit/vegetable classification...")
        predicted_label = classify_fruit_vegetable(image_rgb)
        print(f"Detected label: {predicted_label}")

if __name__ == "__main__":
    main()
