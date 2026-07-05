# Vision Transformer-Based Pneumonia Detection System

This project builds a pneumonia classification pipeline using a pretrained Vision Transformer (ViT) model fine-tuned on chest X-ray images. It includes a Jupyter notebook for exploration and modular Python scripts for data loading, training, evaluation, and inference.

## What this project does

The system classifies chest X-ray images into two categories:

- NORMAL
- PNEUMONIA

It uses a Hugging Face pretrained ViT backbone and adds a classification head for binary image classification.

## Project structure

- ViT_PNEUMONIA_DETECTION.ipynb: End-to-end notebook containing data download, preprocessing, training, and evaluation steps.
- config.yaml: Configuration file for dataset paths, model settings, training hyperparameters, and output paths.
- src/model.py: ViT model architecture and model save/load utilities.
- src/data_loader.py: Dataset loading, image transformations, train/validation/test splitting, and DataLoader creation.
- src/train.py: Training loop, validation, metrics, early stopping, and model checkpointing.
- src/predict.py: Command-line inference for predicting labels for new images.
- train/: Training dataset directory with class subfolders.
- test/: Test dataset directory with class subfolders.

## Dataset

The project expects image folders organized as:

- train/NORMAL
- train/PNEUMONIA
- test/NORMAL
- test/PNEUMONIA

The notebook includes steps to download a public chest X-ray pneumonia dataset from Kaggle.

## Requirements

Install the required Python packages:

```bash
pip install torch torchvision transformers pillow matplotlib seaborn scikit-learn pyyaml tqdm
```

Recommended:

- Python 3.9+ or 3.10+
- CUDA-enabled GPU for faster training
- Google Colab for notebook-based experimentation

## Setup

1. Clone the repository:

```bash
git clone https://github.com/your-username/Vision-transformer-based-pneumonia-detection-system.git
cd Vision-transformer-based-pneumonia-detection-system
```

2. Place the dataset in the appropriate folders or update the paths in config.yaml.

3. Review config.yaml and adjust values such as:

- data_dir
- test_dir
- image_size
- batch_size
- num_epochs
- learning_rate
- model_path

## Running the project

### Option 1: Use the notebook

Open ViT_PNEUMONIA_DETECTION.ipynb in Jupyter or VS Code and run the cells in order.

### Option 2: Run the training script

```bash
python src/train.py
```

This will:

- load the data
- create train/validation/test DataLoaders
- initialize the ViT model
- train the model
- save the best model checkpoint
- generate evaluation metrics and visual reports

### Option 3: Run inference on a new image

```bash
python src/predict.py
```

The script will prompt you for an image path and print the predicted class along with the confidence score.

## Output files

Training and evaluation outputs may include:

- model checkpoints in the path defined by model_path in config.yaml
- confusion matrix images in the reports/ folder
- classification report text files in the reports/ folder

## Notes

- The scripts are designed to work with a pretrained ViT from Hugging Face, so internet access may be required for the first run.
- GPU usage is strongly recommended for faster training.
- If you run the code in Google Colab, you may need to update the file paths used in the scripts to match your mounted Drive location.

## License

This project is intended for educational and research purposes.
