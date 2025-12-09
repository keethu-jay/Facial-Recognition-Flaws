"""
FaceNet Model Integration
Loads the pre-trained FaceNet model for use in adversarial attack generation.

This script handles loading the FaceNet model weights without needing to copy
the entire David Sandberg repository into the project.
"""

import os
import tensorflow as tf
from tensorflow import keras

# Path to the models directory
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
WEIGHTS_PATH = os.path.join(MODELS_DIR, 'facenet_weights.h5')

# Alternative: If using frozen graph format (.pb file)
FROZEN_GRAPH_PATH = os.path.join(MODELS_DIR, '20180402-114759.pb')


def get_facenet_model():
    """
    Loads the pre-trained FaceNet model.
    
    This function attempts to load FaceNet using different methods:
    1. Try loading from H5 weights file
    2. Try loading from frozen graph (.pb file)
    3. Try using keras-facenet library if installed
    
    Returns:
        tf.keras.Model: The loaded FaceNet model, or None if loading fails
    """
    # Method 1: Try loading from H5 file (if you have converted weights)
    if os.path.exists(WEIGHTS_PATH):
        try:
            model = keras.models.load_model(WEIGHTS_PATH, compile=False)
            print("✓ FaceNet model loaded successfully from H5 file.")
            return model
        except Exception as e:
            print(f"Warning: Could not load from H5 file: {e}")
    
    # Method 2: Try using keras-facenet library (recommended)
    try:
        from keras_facenet import FaceNet
        model = FaceNet()
        print("✓ FaceNet model loaded successfully using keras-facenet library.")
        return model
    except ImportError:
        print("Note: keras-facenet not installed. Install with: pip install keras-facenet")
    except Exception as e:
        print(f"Warning: Could not load using keras-facenet: {e}")
    
    # Method 3: Try loading from frozen graph (original David Sandberg format)
    if os.path.exists(FROZEN_GRAPH_PATH):
        try:
            # Load frozen graph
            with tf.io.gfile.GFile(FROZEN_GRAPH_PATH, 'rb') as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
            
            # Create a new graph and import the frozen graph
            with tf.Graph().as_default() as graph:
                tf.import_graph_def(graph_def, name='')
            
            print("✓ FaceNet frozen graph loaded successfully.")
            print("Note: You'll need to use tf.compat.v1.Session to run this model.")
            return graph
        except Exception as e:
            print(f"Warning: Could not load frozen graph: {e}")
    
    # If all methods fail
    print("\n✗ Error: Could not load FaceNet model.")
    print("\nPlease ensure one of the following:")
    print("1. Install keras-facenet: pip install keras-facenet")
    print("2. Download FaceNet weights and place in models/ directory")
    print("3. Download frozen graph from David Sandberg repository")
    print("\nModel download link: https://github.com/davidsandberg/facenet")
    return None


def get_face_embedding(model, image_tensor):
    """
    Get face embedding from FaceNet model.
    
    Args:
        model: FaceNet model
        image_tensor: Preprocessed image tensor (batch_size, height, width, channels)
    
    Returns:
        Tensor: Face embedding vector
    """
    # FaceNet outputs a 512-dimensional embedding vector
    # This represents the face in the embedding space
    embedding = model(image_tensor)
    return embedding

