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
    # Helper function to safely print Unicode characters on Windows
    def safe_print(msg):
        try:
            print(msg)
        except UnicodeEncodeError:
            # Replace Unicode characters with ASCII equivalents for Windows
            msg_ascii = msg.replace('✓', '[OK]').replace('✗', '[ERROR]')
            print(msg_ascii)
    
    # Method 1: Try loading from H5 file (if you have converted weights)
    if os.path.exists(WEIGHTS_PATH):
        try:
            model = keras.models.load_model(WEIGHTS_PATH, compile=False)
            safe_print("✓ FaceNet model loaded successfully from H5 file.")
            return model
        except Exception as e:
            print(f"Warning: Could not load from H5 file: {e}")
    
    # Method 2: Try using keras-facenet library (recommended)
    try:
        from keras_facenet import FaceNet
        model = FaceNet()
        safe_print("✓ FaceNet model loaded successfully using keras-facenet library.")
        # Test that it works
        import numpy as np
        test_input = np.random.rand(1, 160, 160, 3).astype(np.float32)
        _ = model.embeddings(test_input)
        return model
    except ImportError:
        print("Note: keras-facenet not installed. Install with: pip install keras-facenet")
    except Exception as e:
        print(f"Warning: Could not load using keras-facenet: {e}")
        import traceback
        traceback.print_exc()
    
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
            
            safe_print("✓ FaceNet frozen graph loaded successfully.")
            print("Note: You'll need to use tf.compat.v1.Session to run this model.")
            return graph
        except Exception as e:
            print(f"Warning: Could not load frozen graph: {e}")
    
    # If all methods fail
    safe_print("\n✗ Error: Could not load FaceNet model.")
    print("\nPlease ensure one of the following:")
    print("1. Install keras-facenet: pip install keras-facenet")
    print("2. Download FaceNet weights and place in models/ directory")
    print("3. Download frozen graph from David Sandberg repository")
    print("\nModel download link: https://github.com/davidsandberg/facenet")
    return None


def get_face_embedding(model, image_input):
    """
    Get face embedding from FaceNet model.
    
    Args:
        model: FaceNet model (keras-facenet FaceNet object or Keras model)
        image_input: PIL Image, numpy array, or tensor
    
    Returns:
        Tensor/Array: Face embedding vector
    """
    import numpy as np
    from PIL import Image as PILImage
    
    # Check if it's a keras-facenet FaceNet object
    if hasattr(model, 'embeddings'):
        # For tensors (needed for gradient computation in attacks), use the underlying model directly
        if isinstance(image_input, tf.Tensor) or isinstance(image_input, tf.Variable):
            # Use the underlying Keras model for gradient computation
            # The underlying model expects normalized input [0, 1] with shape (batch, 160, 160, 3)
            if hasattr(model, 'model'):
                # Ensure tensor is in correct format [0, 1] range
                # The tensor from preprocess_image is already in [0, 1] range with batch dimension
                try:
                    if len(image_input.shape) == 4:
                        # Already has batch dimension - use directly
                        embedding = model.model(image_input)
                        # Keep batch dimension for now - will be handled in loss calculation
                        # Shape will be (1, 512) or (batch, 512)
                    elif len(image_input.shape) == 3:
                        # Add batch dimension
                        embedding = model.model(tf.expand_dims(image_input, 0))
                        # Shape will be (1, 512)
                    else:
                        raise ValueError(f"Unexpected tensor shape: {image_input.shape}")
                    
                    # Validate embedding
                    if embedding is None:
                        raise ValueError("Model returned None embedding")
                    
                except Exception as e:
                    raise ValueError(f"Failed to get embedding from underlying model: {e}")
            else:
                raise ValueError("keras-facenet model does not have underlying 'model' attribute")
        else:
            # For PIL Images or numpy arrays (no gradients needed), use embeddings() method
            # keras-facenet expects PIL Images or numpy arrays (not tensors)
            # It handles face detection, alignment, and preprocessing internally
            
            # Convert to numpy array if needed
            if isinstance(image_input, PILImage.Image):
                # PIL Image - convert to numpy array (RGB format)
                img_array = np.array(image_input.convert('RGB'))
            elif isinstance(image_input, np.ndarray):
                # Already numpy array - ensure it's uint8
                img_array = image_input.copy()
                if img_array.dtype != np.uint8:
                    if img_array.max() <= 1.0:
                        img_array = (img_array * 255).astype(np.uint8)
                    else:
                        img_array = img_array.astype(np.uint8)
            else:
                # Try to convert to numpy
                img_array = np.array(image_input)
                if img_array.dtype != np.uint8:
                    img_array = img_array.astype(np.uint8)
            
            # Validate and fix image array before passing to keras-facenet
            if img_array.size == 0:
                raise ValueError("Image array is empty")
            
            # Remove batch dimension if present (keras-facenet expects (H, W, 3))
            if len(img_array.shape) == 4:
                if img_array.shape[0] == 1:
                    img_array = img_array[0]  # Remove batch dimension
                else:
                    raise ValueError(f"Batch dimension present but batch size > 1: {img_array.shape}")
            
            # Validate final shape
            if len(img_array.shape) != 3:
                raise ValueError(f"Invalid image shape: {img_array.shape}. Expected (H, W, 3) after processing")
            if img_array.shape[2] != 3:
                raise ValueError(f"Invalid number of channels: {img_array.shape[2]}. Expected 3 (RGB)")
            if img_array.shape[0] == 0 or img_array.shape[1] == 0:
                raise ValueError(f"Image has zero width or height: {img_array.shape}")
            
            # keras-facenet expects a list of images (numpy arrays)
            # It handles face detection internally, which may fail for some images
            try:
                embedding = model.embeddings([img_array])
            except Exception as e:
                # If face detection fails, try resizing the image first
                # Sometimes keras-facenet has issues with certain image sizes
                if "resize" in str(e).lower() or "empty" in str(e).lower():
                    # Try resizing to a standard size that keras-facenet expects
                    from PIL import Image as PILImage
                    pil_img = PILImage.fromarray(img_array)
                    # Resize to a reasonable size if too small or too large
                    if pil_img.size[0] < 160 or pil_img.size[1] < 160:
                        pil_img = pil_img.resize((224, 224), PILImage.LANCZOS)
                    elif pil_img.size[0] > 512 or pil_img.size[1] > 512:
                        pil_img = pil_img.resize((224, 224), PILImage.LANCZOS)
                    img_array = np.array(pil_img)
                    try:
                        embedding = model.embeddings([img_array])
                    except Exception as e2:
                        raise ValueError(f"Failed to get embedding after resize: {e2}. Original error: {e}")
                else:
                    raise
            
            # Remove list dimension (keras-facenet returns list)
            if isinstance(embedding, list):
                if len(embedding) == 0:
                    raise ValueError("Face detection failed: no embedding returned")
                embedding = embedding[0]
            
            # Check if embedding is None (face detection may have failed)
            if embedding is None:
                raise ValueError("Face detection failed: embedding is None. The image may not contain a detectable face.")
    else:
        # Assume it's a standard Keras model - needs tensor input
        if not isinstance(image_input, tf.Tensor):
            # Convert to tensor
            if isinstance(image_input, PILImage.Image):
                img_array = np.array(image_input)
            else:
                img_array = np.array(image_input)
            image_input = tf.convert_to_tensor(img_array)
        embedding = model(image_input)
    
    # For tensors (from model.model()), keep as tensor to preserve gradients
    # For numpy arrays (from model.embeddings()), convert to numpy if needed
    # Only convert to numpy if it's not needed for gradient computation
    # We'll check the input type to decide
    
    # If input was a tensor/Variable, keep output as tensor (needed for gradients)
    if isinstance(image_input, tf.Tensor) or isinstance(image_input, tf.Variable):
        # Keep as tensor - don't convert to numpy
        return embedding
    else:
        # Input was PIL/numpy - convert to numpy for consistency
        if isinstance(embedding, tf.Tensor):
            embedding = embedding.numpy()
        elif hasattr(embedding, 'numpy'):
            embedding = embedding.numpy()
    
    return embedding

