import os
import glob
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

def load_and_prepare_data(normal_dir, anomaly_dir, img_size=(224, 224)):
    """
    Load and prepare data from normal and anomaly directories
    """
    # Create empty lists for data and labels
    images = []
    labels = []
    
    # Load normal images (label 0)
    normal_files = glob.glob(os.path.join(normal_dir, "*.png"))
    for img_path in normal_files:
        img = cv2.imread(img_path)
        img = cv2.resize(img, img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0  # Normalize
        images.append(img)
        labels.append(0)  # 0 for normal
    
    # Load anomaly images (label 1)
    anomaly_categories = [d for d in os.listdir(anomaly_dir) if d != "good"]
    for category in anomaly_categories:
        anomaly_files = glob.glob(os.path.join(anomaly_dir, category, "*.png"))
        for img_path in anomaly_files:
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0  # Normalize
            images.append(img)
            labels.append(1)  # 1 for anomaly
    
    return np.array(images), np.array(labels)

def get_train_test_data(normal_dir, anomaly_dir, test_size=0.2, random_state=42):
    """
    Load and split data into training and testing sets
    """
    X, y = load_and_prepare_data(normal_dir, anomaly_dir)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test
