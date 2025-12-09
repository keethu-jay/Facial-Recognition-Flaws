# FaceNet Model Documentation

## Model Source and Attribution

This project uses the **FaceNet** model implementation from the open-source `keras-facenet` library.

### Model Details

- **Library**: `keras-facenet` (version 0.3.2)
- **Repository**: https://github.com/SergeyDmitriev/keras-facenet
- **Original FaceNet Paper**: 
  - Schroff, F., Kalenichenko, D., & Philbin, J. (2015). FaceNet: A unified embedding for face recognition and clustering. CVPR 2015.
  - Paper: https://arxiv.org/abs/1503.03832

### Pre-trained Weights

The `keras-facenet` library automatically downloads pre-trained FaceNet weights when first used. The weights are based on the Inception ResNet v1 architecture trained on the VGGFace2 dataset.

- **Architecture**: Inception ResNet v1
- **Training Dataset**: VGGFace2
- **Embedding Dimension**: 512
- **Input Size**: 160x160 pixels
- **Model Size**: ~90-100 MB (downloaded automatically)

### How We Use It

We use the FaceNet model to:
1. **Generate embeddings** for each face image (512-dimensional vectors)
2. **Calculate distances** between embeddings to determine if two faces are the same person
3. **Compute gradients** for white-box adversarial attacks (FGSM, PGD, C&W)

### Installation

```bash
pip install keras-facenet
pip install opencv-python  # Required dependency
```

### Model Loading

The model is loaded using:
```python
from keras_facenet import FaceNet
model = FaceNet()
```

On first use, the library automatically downloads the pre-trained weights to a cache directory (typically `~/.keras/models/`).

### Modifications Made

**No modifications were made to the FaceNet model itself.** We use the model as-is from the `keras-facenet` library.

The only customizations are:
1. **Wrapper functions** (`FaceNet_Model.py`) to standardize the interface
2. **Preprocessing functions** to convert PIL Images to the format expected by keras-facenet
3. **Embedding extraction** wrapper to handle the model's API

### License and Attribution

- **keras-facenet**: MIT License (check repository for details)
- **FaceNet**: Original implementation by Google (see paper for details)
- **VGGFace2**: Dataset used for training (see original FaceNet repository)

### References

1. keras-facenet GitHub: https://github.com/SergeyDmitriev/keras-facenet
2. Original FaceNet Paper: https://arxiv.org/abs/1503.03832
3. David Sandberg's FaceNet Implementation: https://github.com/davidsandberg/facenet

