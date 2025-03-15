import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model

def plot_training_history(history):
    """
    Plot training and validation accuracy/loss
    """
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='lower right')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper right')

    plt.tight_layout()
    plt.savefig('images/training_history.png')
    plt.show()

def visualize_predictions(X, y_true, y_pred, num_samples=5):
    """
    Visualize predictions alongside ground truth
    """
    indices = np.random.choice(range(len(X)), num_samples, replace=False)
    
    plt.figure(figsize=(15, 4*num_samples))
    for i, idx in enumerate(indices):
        plt.subplot(num_samples, 2, 2*i+1)
        plt.imshow(X[idx])
        plt.title(f"True: {'Defect' if y_true[idx] == 1 else 'Normal'}")
        plt.axis('off')
        
        plt.subplot(num_samples, 2, 2*i+2)
        plt.imshow(X[idx])
        plt.title(f"Pred: {'Defect' if y_pred[idx] > 0.5 else 'Normal'} ({y_pred[idx][0]:.2f})")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('images/prediction_samples.png')
    plt.show()

def grad_cam(model, img, layer_name='conv2d_5'):
    """
    Generate Grad-CAM visualization to highlight important regions for prediction
    """
    grad_model = Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.array([img]))
        loss = predictions[:, 0]
    
    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]
    
    # Global average pooling
    weights = tf.reduce_mean(grads, axis=(0, 1))
    
    # Create a weighted combination of feature maps
    cam = np.zeros(output.shape[0:2], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]
    
    # Apply ReLU
    cam = np.maximum(cam, 0)
    
    # Resize and normalize
    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    
    return cam

def visualize_grad_cam(model, X, y_true, y_pred, num_samples=3):
    """
    Visualize Grad-CAM for defects
    """
    # Find indices of correctly predicted defects
    defect_indices = np.where((y_true == 1) & (y_pred.reshape(-1) > 0.5))[0]
    
    if len(defect_indices) == 0:
        print("No correctly predicted defects found.")
        return
    
    indices = np.random.choice(defect_indices, min(num_samples, len(defect_indices)), replace=False)
    
    plt.figure(figsize=(15, 4*len(indices)))
    for i, idx in enumerate(indices):
        img = X[idx]
        
        # Generate Grad-CAM
        cam = grad_cam(model, img)
        
        # Overlay heatmap on image
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        superimposed_img = heatmap * 0.4 + img * 0.6
        
        # Normalize
        superimposed_img = superimposed_img / np.max(superimposed_img)
        
        # Display original image
        plt.subplot(len(indices), 3, 3*i+1)
        plt.imshow(img)
        plt.title("Original Image")
        plt.axis('off')
        
        # Display heatmap
        plt.subplot(len(indices), 3, 3*i+2)
        plt.imshow(cam, cmap='jet')
        plt.title("Grad-CAM Heatmap")
        plt.axis('off')
        
        # Display superimposed image
        plt.subplot(len(indices), 3, 3*i+3)
        plt.imshow(superimposed_img)
        plt.title("Superimposed")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('images/grad_cam_visualization.png')
    plt.show()
