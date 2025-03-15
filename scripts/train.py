import os
import argparse
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Import from our modules
import sys
sys.path.append('..')
from models.cnn_model import build_defect_detection_model
from data.data_loader import get_train_test_data
from utils.visualization import plot_training_history, visualize_predictions, visualize_grad_cam

def main():
    parser = argparse.ArgumentParser(description='Train a defect detection CNN model')
    parser.add_argument('--category', type=str, default='bottle', help='MVTec dataset category to use')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--img_size', type=str, default='224,224', help='Image size as width,height')
    args = parser.parse_args()
    
    # Parse image size
    width, height = map(int, args.img_size.split(','))
    img_size = (width, height)
    
    # Define paths
    BASE_DIR = "mvtec_anomaly_detection"
    CATEGORY = args.category
    NORMAL_DIR = os.path.join(BASE_DIR, CATEGORY, "train", "good")
    ANOMALY_DIR = os.path.join(BASE_DIR, CATEGORY, "test")
    
    # Create images directory if it doesn't exist
    os.makedirs('images', exist_ok=True)
    
    # Load and prepare the data
    print(f"Loading and preparing data for category: {CATEGORY}...")
    X_train, X_test, y_train, y_test = get_train_test_data(NORMAL_DIR, ANOMALY_DIR)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    # Build the model
    model = build_defect_detection_model(input_shape=(img_size[0], img_size[1], 3))
    model.summary()
    
    # Data augmentation for training
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Train the model
    print("Training the model...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=args.batch_size),
        epochs=args.epochs,
        validation_data=(X_test, y_test),
        steps_per_epoch=len(X_train) // args.batch_size
    )
    
    # Evaluate the model
    print("Evaluating the model...")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Plot training history
    plot_training_history(history)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Visualize predictions
    visualize_predictions(X_test, y_test, y_pred)
    
    # Visualize Grad-CAM
    visualize_grad_cam(model, X_test, y_test, y_pred)
    
    # Save the model
    model_filename = f'defect_detection_{CATEGORY}_model.h5'
    model.save(model_filename)
    print(f"Model saved to '{model_filename}'")

if __name__ == "__main__":
    main()
