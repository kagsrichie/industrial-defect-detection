import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import load_model

# Import from our modules
import sys
sys.path.append('..')
from utils.visualization import grad_cam

def predict_single_image(model, image_path, img_size=(224, 224)):
    """
    Make a prediction on a single image
    """
    # Load and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    img = cv2.resize(img, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_normalized = img / 255.0
    
    # Make prediction
    prediction = model.predict(np.expand_dims(img_normalized, axis=0))[0]
    is_defect = prediction > 0.5
    
    # Generate Grad-CAM
    if is_defect:
        cam = grad_cam(model, img_normalized)
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Superimpose heatmap on original image
        superimposed = heatmap * 0.4 + img * 0.6
        superimposed = superimposed / np.max(superimposed)
    else:
        superimposed = None
    
    return img, prediction[0], is_defect, superimposed

def main():
    parser = argparse.ArgumentParser(description='Detect defects in a single image')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the image to analyze')
    parser.add_argument('--output_path', type=str, default='detection_result.png', help='Path to save the output visualization')
    args = parser.parse_args()
    
    # Load the model
    model = load_model(args.model_path)
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process the image
    img, confidence, is_defect, heatmap = predict_single_image(model, args.image_path)
    
    # Display and save the results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2 if is_defect else 1, 1)
    plt.imshow(img)
    status = "DEFECT" if is_defect else "NORMAL"
    plt.title(f"Status: {status} (Confidence: {confidence:.2f})")
    plt.axis('off')
    
    if is_defect and heatmap is not None:
        plt.subplot(1, 2, 2)
        plt.imshow(heatmap)
        plt.title("Defect Localization (Grad-CAM)")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(args.output_path)
    
    print(f"Prediction: {status} with {confidence:.2f} confidence")
    print(f"Result saved to {args.output_path}")
    
    if is_defect:
        print("Defect detected! Please check the highlighted areas in the output image.")
    else:
        print("No defect detected. The product appears normal.")

if __name__ == "__main__":
    main()
