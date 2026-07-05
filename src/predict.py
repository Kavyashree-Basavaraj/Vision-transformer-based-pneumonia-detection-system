# predict.py
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
import yaml
import os
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, Any

# Assuming model.py containing ViTForImageClassification and load_model is in the same directory or accessible
from model import ViTForImageClassification, load_model

def predict_image(
    model: ViTForImageClassification,
    image_path: str,
    config: Dict[str, Any],
    device: torch.device
) -> Optional[Tuple[str, float]]:
    """
    Predicts the class of a single image using the trained ViT model.

    Args:
        model (ViTForImageClassification): The loaded and trained model instance.
        image_path (str): Path to the input image file.
        config (Dict[str, Any]): Configuration dictionary (used for image_size).
        device (torch.device): The device the model is on ('cuda' or 'cpu').

    Returns:
        Optional[Tuple[str, float]]: A tuple containing the predicted class name (str)
                                     and its probability (float), or None if prediction fails.
    """
    model.eval()  # Set model to evaluation mode

    # Define the transformations - MUST match the validation/test transformations used during training
    try:
        transform = transforms.Compose([
            transforms.Resize((config['image_size'], config['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet stats
        ])
    except KeyError as e:
        print(f"Error: Missing key {e} in configuration for transformations.")
        return None

    # Load and process the image
    try:
        image = Image.open(image_path).convert('RGB') # Ensure image is RGB
    except FileNotFoundError:
        print(f"Error: Image file not found at path: {image_path}")
        return None
    except UnidentifiedImageError:
        print(f"Error: Cannot identify image file (possibly corrupt): {image_path}")
        return None
    except Exception as e:
         print(f"Error opening or processing image {image_path}: {e}")
         return None

    # Apply transformations, add batch dimension, and move to the correct device
    try:
        image_tensor = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error applying transformations to image {image_path}: {e}")
        return None

    # Perform inference
    with torch.no_grad(): # Disable gradient calculations for inference
        try:
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            prediction_idx = torch.argmax(output, dim=1).item() # Get predicted class index (0 or 1)
        except Exception as e:
            print(f"Error during model inference: {e}")
            return None

    # Map index to class name (NOTE: Assumes 0: NORMAL, 1: PNEUMONIA - Hardcoded)
    # Ideally, this mapping should be loaded from where it was defined during training.
    class_names = {0: 'NORMAL', 1: 'PNEUMONIA'}
    if prediction_idx not in class_names:
        print(f"Error: Predicted index {prediction_idx} is not in class_names map {class_names.keys()}")
        return None

    predicted_class = class_names[prediction_idx]
    probability = probabilities[0][prediction_idx].item() # Get probability of the predicted class

    return predicted_class, probability


def main():
    # --- Configuration Loading ---
    config_path = '/content/drive/MyDrive/data/config.yaml' # Primary config path
    fallback_config_path = 'config.yaml' # Fallback if primary not found

    if not os.path.exists(config_path):
        print(f"Warning: Config file not found at {config_path}. Trying fallback: {fallback_config_path}")
        config_path = fallback_config_path

    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at either primary or fallback path. Exiting.")
        return

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            print("Configuration loaded successfully.")
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        return

    # --- Device Setup ---
    device_name = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    if device_name == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA specified in config but not available. Using CPU.")
        device_name = 'cpu'
    device = torch.device(device_name)
    print(f"Using device: {device}")

    # --- Model Loading ---
    model_path = config.get('model_path')
    if not model_path:
        print("Error: 'model_path' not found in configuration file. Exiting.")
        return

    # Instantiate and load model using the load_model function
    # No need to create ViTForImageClassification or call .to(device) here, load_model handles it.
    model = load_model(model_path, device, config)

    if model is None:
        print(f"Failed to load model from {model_path}. Exiting.")
        return
    print(f"Model loaded from {model_path} and ready for prediction.")

    # --- Prediction Loop ---
    while True:
        try:
            image_path_input = input("Enter the path to the image (or type 'quit' to exit): ").strip()
            if image_path_input.lower() == 'quit':
                break
            if not image_path_input: # Handle empty input
                 continue

            # Validate if image path exists
            if not os.path.exists(image_path_input):
                print(f"Error: Image path '{image_path_input}' not found. Please provide a valid path.")
                continue

            # Run prediction
            predicted_class, probability = predict_image(model, image_path_input, config, device)

            if predicted_class is not None:
                print(f"\nPrediction for '{os.path.basename(image_path_input)}':")
                print(f"  Class: {predicted_class}")
                print(f"  Probability: {probability:.4f}\n")

                # Display the image with prediction
                try:
                    img = Image.open(image_path_input)
                    plt.imshow(img, cmap='gray' if img.mode == 'L' else None) # Handle grayscale images
                    plt.title(f"Prediction: {predicted_class} ({probability:.2f})")
                    plt.axis('off') # Hide axes
                    plt.show()
                except Exception as e:
                    print(f"Error displaying image: {e}")
            else:
                print("Could not generate prediction for this image.")

        except EOFError: # Handle Ctrl+D or unexpected end of input
             print("\nInput stream closed. Exiting.")
             break
        except KeyboardInterrupt: # Handle Ctrl+C
             print("\nPrediction interrupted by user. Exiting.")
             break
        except Exception as e:
            print(f"An unexpected error occurred in the input loop: {e}")
            # Depending on severity, you might want to break or continue
            # break

    print("Prediction script finished.")


if __name__ == "__main__":
    main()