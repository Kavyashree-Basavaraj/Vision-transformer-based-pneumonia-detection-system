# model.py
import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig # ViTConfig might be useful for custom heads
import yaml
import os
from typing import Optional, Dict, Any # For type hinting

class ViTForImageClassification(nn.Module):
    """
    Vision Transformer (ViT) model for image classification.

    Loads a pre-trained ViT backbone from Hugging Face and adds a classification head.

    Args:
        config (Dict[str, Any]): Configuration dictionary containing model settings.
            Expected keys:
            - 'model_name' (str): The Hugging Face identifier for the pre-trained ViT model.
            - 'num_classes' (int): The number of output classes for the classifier.
            - 'dropout_rate' (float, optional): Dropout probability for the classifier head. Defaults to 0.0.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        # Validate required config keys
        required_keys = ['model_name', 'num_classes']
        if not all(key in config for key in required_keys):
            raise ValueError(f"Config dictionary must contain keys: {required_keys}")

        self.model_name = config['model_name']
        self.num_classes = config['num_classes']
        self.dropout_rate = config.get('dropout_rate', 0.0) # Get dropout rate, default to 0.0

        # Load the pre-trained ViT model backbone
        try:
            self.vit = ViTModel.from_pretrained(self.model_name)
        except OSError as e:
            print(f"Error loading pre-trained model '{self.model_name}'. Check model name and internet connection.")
            raise e # Re-raise after printing message

        # Define the classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout_rate), # Add dropout before the linear layer
            nn.Linear(self.vit.config.hidden_size, self.num_classes) # Linear layer for classification
        )

        print(f"Initialized ViTForImageClassification with backbone: {self.model_name}")
        print(f"Number of classes: {self.num_classes}, Classifier Dropout: {self.dropout_rate}")


    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            pixel_values (torch.Tensor): Input tensor containing pixel values for images.
                                         Shape: (batch_size, num_channels, height, width).

        Returns:
            torch.Tensor: Output logits from the classifier head.
                          Shape: (batch_size, num_classes).
        """
        # Pass input through the ViT backbone
        # We are interested in the output of the [CLS] token, which is used for classification tasks
        outputs = self.vit(pixel_values=pixel_values)

        # Extract the last hidden state of the [CLS] token (index 0)
        cls_token_output = outputs.last_hidden_state[:, 0]

        # Pass the [CLS] token output through the classifier head
        logits = self.classifier(cls_token_output)

        return logits

# --- Model Loading and Saving Functions ---

def load_model(model_path: str, device: torch.device, config: Dict[str, Any]) -> Optional[ViTForImageClassification]:
    """
    Loads the ViTForImageClassification model state dict from a specified path.

    Args:
        model_path (str): The path to the saved model state_dict (.pth file).
        device (torch.device): The device to load the model onto ('cuda' or 'cpu').
        config (Dict[str, Any]): Configuration dictionary used to instantiate the model architecture.

    Returns:
        Optional[ViTForImageClassification]: The loaded model instance, or None if loading fails.
    """
    try:
        print(f"Attempting to load model from: {model_path}")
        # Instantiate the model architecture first
        model = ViTForImageClassification(config)

        # Load the state dictionary
        # weights_only=True is recommended for security unless you saved more than just weights
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)

        model.to(device) # Ensure the model is on the correct device
        model.eval() # Set to evaluation mode after loading
        print(f"Model loaded successfully from {model_path} and moved to {device}")
        return model

    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}. Cannot load model.")
        return None
    except Exception as e:
        print(f"Error loading model state_dict from {model_path}: {e}")
        print("Ensure the saved state_dict matches the current model architecture defined by the config.")
        return None


def save_model(model: ViTForImageClassification, model_path: str):
    """
    Saves the model's state_dict to the specified path.

    Args:
        model (ViTForImageClassification): The model instance to save.
        model_path (str): The path where the model state_dict will be saved (.pth file).
    """
    try:
        # Ensure the directory exists
        save_dir = os.path.dirname(model_path)
        if save_dir: # Only create if path includes a directory
            os.makedirs(save_dir, exist_ok=True)

        # Save the model state dictionary
        torch.save(model.state_dict(), model_path)
        print(f"Model state_dict saved successfully to {model_path}")

    except Exception as e:
        print(f"Error saving model to {model_path}: {e}")


# --- Example Usage / Basic Test ---
if __name__ == '__main__':
    # Load config (adjust path as needed)
    config_path = '/content/drive/MyDrive/data/config.yaml'
    if not os.path.exists(config_path):
        config_path = '/content/drive/MyDrive/data/config.yaml' # Fallback

    if not os.path.exists(config_path):
         print(f"ERROR: Config file not found at {config_path} or ./config.yaml")
         exit()

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            print("Configuration loaded.")
    except Exception as e:
        print(f"Error loading config: {e}")
        exit()

    # Ensure essential keys are in config for the test
    if 'num_classes' not in config:
        print("Warning: 'num_classes' not found in config, defaulting to 2 for test.")
        config['num_classes'] = 2
    if 'model_name' not in config:
        print("Warning: 'model_name' not found in config, defaulting to 'google/vit-base-patch16-224-in21k' for test.")
        config['model_name'] = 'google/vit-base-patch16-224-in21k'

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Test Model Instantiation ---
    try:
        print("\nInstantiating model...")
        model_instance = ViTForImageClassification(config)
        model_instance.to(device) # Move model to device
        print("Model instantiated successfully.")
        # (Optional) Print model architecture summary
        # print(model_instance)
    except Exception as e:
        print(f"Error during model instantiation: {e}")
        exit()


    # --- Test Saving and Loading ---
    test_model_path = config.get('model_path', 'models/test_ViT_pneumonia_model.pth') # Use path from config or a default test path
    print(f"\nTesting model saving to: {test_model_path}")
    save_model(model_instance, test_model_path)

    print(f"\nTesting model loading from: {test_model_path}")
    loaded_model = load_model(test_model_path, device, config)

    if loaded_model:
        print("\nModel saving and loading test completed successfully.")
        # Optional: Test forward pass with dummy data
        try:
            print("Testing forward pass with dummy data...")
            dummy_input = torch.randn(config['batch_size'], 3, config['image_size'], config['image_size']).to(device)
            with torch.no_grad():
                output = loaded_model(dummy_input)
            print(f"Output shape: {output.shape}") # Should be [batch_size, num_classes]
            assert output.shape == (config['batch_size'], config['num_classes'])
            print("Forward pass successful.")
        except Exception as e:
            print(f"Error during dummy forward pass: {e}")
    else:
        print("\nModel loading failed during test.")