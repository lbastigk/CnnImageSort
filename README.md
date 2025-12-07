# CNN Image Sorter

This project uses a configurable Convolutional Neural Network (CNN) to sort images into categories. It uses a JSON-defined model architecture for easy configuration.

## Directory Structure
```
.
├── categories/         # Training images, organized by category
├── model_cnn.py        # Shared CNN model class
├── model.jsonc         # Model architecture/configuration
├── requirements.txt    # Python dependencies
├── sort.py             # Script to sort images
├── suggestions/        # Output folder for sorted images
├── tosort/             # Images to be sorted
├── train.py            # Script to train the model
```

## Installation
1. Clone the repository and navigate to the project folder.
2. Run `./install.sh`

## Usage
### Training
Place your training images in `categories/<category>/`. Each subfolder is a class.
Run:
```bash
python train.py
```
This will train the CNN and save the model to `trained_model.pth`.

### Sorting
Place images to be sorted in the `tosort/` folder.
Run:
```bash
python sort.py
```
Images will be copied into `suggestions/<predicted_category>/` based on the model's predictions.

## Notes
- Images are automatically resized for training and sorting (no modification of originals).
- Model architecture is defined in `model.jsonc`.
- Corrupted or unreadable images are skipped automatically.

## Requirements
- Python 3.8+
- See `requirements.txt` for required packages (PyTorch, torchvision, Pillow, etc.)
