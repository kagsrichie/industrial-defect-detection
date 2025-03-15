import unittest
import numpy as np
import os
import sys

from pathlib import Path
import tensorflow as tf

# Add the parent directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from models.cnn_model import build_defect_detection_model
from data.data_loader import load_and_prepare_data
from utils.visualization import grad_cam

class TestDefectDetection(unittest.TestCase):
    """
    Unit tests for the defect detection system
    """
    
    def setUp(self):
        """
        Set up for tests
        """
        # Create a tiny model for testing
        self.test_model = build_defect_detection_model(input_shape=(32, 32, 3))
        
        # Create dummy data for testing
        self.dummy_normal = np.random.random((5, 32, 32, 3))
        self.dummy_defect = np.random.random((5, 32, 32, 3))
        
    def test_model_creation(self):
        """
        Test if the model is created correctly
        """
        # Check model type
        self.assertIsInstance(self.test_model, tf.keras.Model)
        
        # Check input shape
        self.assertEqual(self.test_model.input_shape, (None, 32, 32, 3))
        
        # Check output shape
        self.assertEqual(self.test_model.output_shape, (None, 1))
    
    def test_model_prediction(self):
        """
        Test if the model can make predictions
        """
        # Make a prediction on a single image
        dummy_image = np.random.random((1, 32, 32, 3))
        prediction = self.test_model.predict(dummy_image)
        
        # Check prediction shape and type
        self.assertEqual(prediction.shape, (1, 1))
        self.assertTrue(0 <= prediction[0][0] <= 1)
        
        # Make predictions on a batch
        dummy_batch = np.random.random((10, 32, 32, 3))
        predictions = self.test_model.predict(dummy_batch)
        
        # Check batch prediction shape
        self.assertEqual(predictions.shape, (10, 1))
        
    def test_grad_cam(self):
        """
        Test if Grad-CAM visualization works
        """
        # Requires a trained model, so we'll just test the function signature
        # This is more of an integration test that would be run after training
        
        try:
            # Get a reference to the function without calling it
            self.assertTrue(callable(grad_cam))
        except ImportError:
            self.fail("Failed to import grad_cam function")
    
    def test_model_save_load(self):
        """
        Test if model can be saved and loaded
        """
        # Save the model to a temporary file
        temp_model_path = "temp_test_model.h5"
        self.test_model.save(temp_model_path)
        
        # Check if the file exists
        self.assertTrue(os.path.exists(temp_model_path))
        
        # Load the model
        loaded_model = tf.keras.models.load_model(temp_model_path)
        
        # Check if the loaded model has the same architecture
        self.assertEqual(
            len(self.test_model.layers),
            len(loaded_model.layers)
        )
        
        # Clean up
        os.remove(temp_model_path)


class TestDataLoading(unittest.TestCase):
    """
    Test data loading functionality
    """
    
    def test_data_loader_function(self):
        """
        Test if the data loader function has the correct signature
        """
        try:
            # Get a reference to the function without calling it
            self.assertTrue(callable(load_and_prepare_data))
        except ImportError:
            self.fail("Failed to import load_and_prepare_data function")
    
    # This would be an integration test requiring actual data
    # def test_data_loading_with_real_files(self):
    #     """
    #     Test loading actual files
    #     """
    #     # This test requires actual data files
    #     pass


if __name__ == '__main__':
    unittest.main()
