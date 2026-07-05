# data_loader.py
import os
import random # For potential seed setting
from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split
import yaml
import numpy as np
from tqdm import tqdm # Optional: for progress during loading if very large

# Set seed for reproducibility (optional, but good practice)
# random.seed(42)
# np.random.seed(42)
# torch.manual_seed(42)

class PneumoniaDataset(Dataset):
    """Custom Dataset for Pneumonia images."""
    def __init__(self, image_paths, labels, transform=None, class_to_idx=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        # Store class mapping if provided (useful for debugging/consistency)
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()} if class_to_idx else None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            # Open image and ensure it's RGB
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}. Skipping.")
            # Return None or a dummy tensor. Needs careful handling in collate_fn or filtering beforehand.
            # Returning None might require a custom collate_fn. Easiest is to filter bad paths before creating Dataset.
            return None, None # Indicate failure
        except UnidentifiedImageError:
             print(f"Error: Cannot identify image file {image_path}. Might be corrupt. Skipping.")
             return None, None
        except Exception as e:
            print(f"Error loading image {image_path}: {e}. Skipping.")
            return None, None

        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)

        # Return image and label as tensors
        # Ensure label is long type for CrossEntropyLoss
        return image, torch.tensor(label, dtype=torch.long)

def load_data(data_dir):
    """
    Loads image paths and numerical labels from a directory structured as data_dir/class_name/image.jpg.
    Returns:
        list: List of image file paths.
        list: List of corresponding integer labels.
        dict: Mapping from class names to integer labels.
    """
    image_paths = []
    labels = []

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found or is not a directory: {data_dir}")

    # Dynamically find classes based on subdirectories
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    if not class_names:
        raise ValueError(f"No subdirectories (classes) found in {data_dir}.")

    class_to_idx = {name: i for i, name in enumerate(class_names)}
    print(f"Found classes: {class_to_idx}")

    num_skipped = 0
    print(f"Scanning directory: {data_dir}...")
    # Iterate through classes and images
    for class_name, class_idx in class_to_idx.items():
        class_dir = os.path.join(data_dir, class_name)
        print(f" Loading images from class '{class_name}'...") # Progress indication
        # Use tqdm here if loading takes a long time for a single class
        for image_name in os.listdir(class_dir):
            # Check for common image file extensions
            if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                image_path = os.path.join(class_dir, image_name)
                # Basic check for non-empty file (doesn't guarantee valid image)
                try:
                    if os.path.getsize(image_path) > 0:
                        image_paths.append(image_path)
                        labels.append(class_idx)
                    else:
                        print(f"Warning: Skipping empty file: {image_path}")
                        num_skipped += 1
                except OSError:
                     print(f"Warning: Cannot access file (possible permission issue?): {image_path}")
                     num_skipped += 1
            # else: # Debugging: print non-image files found
            #     print(f"Skipping non-image file: {os.path.join(class_dir, image_name)}")

    if not image_paths:
        raise ValueError(f"No valid image files found in {data_dir} or its subdirectories.")

    print(f"Loaded {len(image_paths)} image paths.")
    if num_skipped > 0:
         print(f"Skipped {num_skipped} potentially problematic files (empty or inaccessible).")

    return image_paths, labels, class_to_idx

def create_data_loaders(config):
    """
    Creates training, validation, and test DataLoaders with appropriate transformations and sampling.

    Args:
        config (dict): Configuration dictionary containing parameters like data paths, image size, batch size, etc.

    Returns:
        tuple: (train_dataloader, val_dataloader, test_dataloader, class_to_idx)
    """

    # --- Configurable Parameters ---
    data_dir = config['data_dir']
    test_dir = config['test_dir']
    img_size = config['image_size']
    batch_size = config['batch_size']
    num_workers = config.get('num_workers', 0) # Default to 0 if not specified
    test_size = config['test_size']
    random_state = config['random_state']
    pin_memory = config.get('device', 'cpu') == 'cuda' # Use pin_memory if using CUDA

    print(f"--- Data Loader Configuration ---")
    print(f"Train Dir: {data_dir}, Test Dir: {test_dir}")
    print(f"Image Size: {img_size}, Batch Size: {batch_size}")
    print(f"Num Workers: {num_workers}, Pin Memory: {pin_memory}")
    print(f"Validation Split: {test_size*100}%, Random State: {random_state}")
    print(f"--------------------------------")


    # --- Load Data (Train/Val combined first, then Test) ---
    print("Loading training/validation dataset info...")
    # Assuming class_to_idx is consistent between train and test is usually safe, but derived from train is primary.
    train_val_image_paths, train_val_labels, class_to_idx = load_data(data_dir)

    print("\nLoading test dataset info...")
    test_image_paths, test_labels, test_class_to_idx = load_data(test_dir)

    # Sanity check: Ensure test set uses the same class mapping or is compatible
    if class_to_idx != test_class_to_idx:
         print("Warning: Class-to-index mapping differs between train/val and test directories!")
         # Decide how to handle: error, remap test labels, or proceed with caution.
         # For simplicity, we'll proceed assuming the structure implies the same classes.

    # --- Split Training and Validation Data ---
    print("\nSplitting training and validation data...")
    try:
        train_image_paths, val_image_paths, train_labels, val_labels = train_test_split(
            train_val_image_paths, train_val_labels,
            test_size=test_size,
            random_state=random_state,
            stratify=train_val_labels # Stratify to maintain class proportions
        )
    except ValueError as e:
         if 'must be greater than the number of classes' in str(e):
             print(f"ERROR: Cannot stratify split. Not enough samples per class in the dataset ({len(train_val_image_paths)} samples total). "
                   f"Try reducing test_size or acquiring more data. Error: {e}")
         else:
             print(f"Error during train/val split: {e}")
         raise e # Re-raise the error

    print(f" Training samples: {len(train_image_paths)}, Validation samples: {len(val_image_paths)}")
    print(f" Test samples: {len(test_image_paths)}")


    # --- Define Transformations ---
    # Common normalization for ImageNet pre-trained models
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Training transformations (with augmentation)
    # Consider using Albumentations library for more complex/efficient augmentations
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)), # Crop a random part and resize
        transforms.RandomHorizontalFlip(p=0.5), # Standard augmentation
        transforms.RandomRotation(degrees=15), # Slight rotation
        # transforms.ColorJitter(brightness=0.2, contrast=0.2), # Adjust brightness/contrast
        transforms.ToTensor(), # Convert PIL image to Tensor
        normalize, # Normalize tensor
    ])

    # Validation/Test transformations (no augmentation, just resize and normalize)
    val_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)), # Resize to fixed size
        transforms.ToTensor(),
        normalize,
    ])


    # --- Create Datasets ---
    print("\nCreating PyTorch Datasets...")
    train_dataset = PneumoniaDataset(train_image_paths, train_labels, transform=train_transform, class_to_idx=class_to_idx)
    val_dataset = PneumoniaDataset(val_image_paths, val_labels, transform=val_test_transform, class_to_idx=class_to_idx)
    test_dataset = PneumoniaDataset(test_image_paths, test_labels, transform=val_test_transform, class_to_idx=class_to_idx) # Use train class_to_idx

    # Filter out None samples potentially returned by __getitem__ due to loading errors
    # Note: This requires iterating through the dataset once, could be slow for huge datasets.
    # Alternatively, implement a robust custom collate_fn.
    # Or filter paths *before* creating the dataset.
    # train_dataset = [(img, lbl) for img, lbl in train_dataset if img is not None]
    # val_dataset = [(img, lbl) for img, lbl in val_dataset if img is not None]
    # test_dataset = [(img, lbl) for img, lbl in test_dataset if img is not None]
    # If filtering, length needs recalculation, complicates WeightedRandomSampler.
    # Recommendation: Ensure data is clean beforehand or handle errors robustly in __getitem__ / collate_fn


    # --- Handle Class Imbalance (WeightedRandomSampler for Training Loader) ---
    print("\nSetting up sampler for training data (addressing class imbalance)...")
    class_counts = np.bincount(train_labels)
    num_classes_found = len(class_counts)

    # Check if number of classes found matches expected
    if num_classes_found != len(class_to_idx):
        print(f"Warning: Number of classes in train labels ({num_classes_found}) doesn't match directories ({len(class_to_idx)}). "
              f"Counts: {class_counts}. Mapping: {class_to_idx}. Check data integrity.")
        # Decide how to proceed: maybe error, maybe continue if counts cover expected indices.

    print(f" Class counts in training set: {class_counts}")

    # Calculate weights: Inverse frequency. Add small epsilon for stability if a class count is 0 (shouldn't happen with stratify if data is ok).
    class_weights = 1. / np.where(class_counts > 0, class_counts, 1e-6) # Avoid division by zero

    # Assign weight to each sample in the training set based on its class
    sample_weights = np.array([class_weights[label] for label in train_labels])
    sample_weights = torch.from_numpy(sample_weights).double() # Sampler expects double type

    # Create the sampler
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    print(f" Calculated class weights for sampler: {class_weights}")
    print(f" WeightedRandomSampler created for training loader.")


    # --- Create DataLoaders ---
    print("\nCreating DataLoaders...")
    # Training DataLoader: Use sampler, shuffle MUST be False.
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler, # Use the weighted sampler
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False # Sampler handles shuffling based on weights
        # collate_fn=custom_collate_fn # Use if __getitem__ can return None
    )

    # Validation DataLoader: No sampler, shuffle=False for consistent evaluation.
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
        # collate_fn=custom_collate_fn
    )

    # Test DataLoader: No sampler, shuffle=False.
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
        # collate_fn=custom_collate_fn
    )

    print("DataLoaders created successfully.")

    # Return dataloaders and the class mapping
    return train_dataloader, val_dataloader, test_dataloader, class_to_idx


if __name__ == '__main__':
    # Example Usage & Testing
    config_path = '/content/drive/MyDrive/data/config.yaml' # Adjust path as needed
    # Fallback path (optional)
    if not os.path.exists(config_path):
        config_path = 'config.yaml'

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            print("Configuration loaded successfully.")
    except FileNotFoundError:
        print(f"ERROR: Configuration file not found at {config_path} or ./config.yaml")
        exit() # Exit if config is missing
    except Exception as e:
        print(f"Error loading configuration: {e}")
        exit()

    # Add default num_workers if missing (optional, but good practice)
    if 'num_workers' not in config:
        config['num_workers'] = 0 # Default to 0 for broader compatibility if not set
        print("Warning: 'num_workers' not found in config, defaulting to 0.")
    if 'device' not in config: # Set device for pin_memory check
         config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'


    try:
        # Create data loaders
        train_loader, val_loader, test_loader, class_map = create_data_loaders(config)

        # --- Basic Test: Iterate through one batch of each loader ---
        print("\n--- Testing DataLoaders (First Batch) ---")

        # Test Train Loader
        print("Train Loader:")
        try:
            train_images, train_labels = next(iter(train_loader))
            print(f"  Image batch shape: {train_images.shape}") # Should be [batch_size, 3, image_size, image_size]
            print(f"  Label batch shape: {train_labels.shape}") # Should be [batch_size]
            print(f"  Sample labels: {train_labels[:5]}...") # Show a few labels
        except StopIteration:
            print("  Train Loader is empty!")
        except Exception as e:
            print(f"  Error iterating train_loader: {e}")


        # Test Validation Loader
        print("\nValidation Loader:")
        try:
            val_images, val_labels = next(iter(val_loader))
            print(f"  Image batch shape: {val_images.shape}")
            print(f"  Label batch shape: {val_labels.shape}")
        except StopIteration:
            print("  Validation Loader is empty!")
        except Exception as e:
            print(f"  Error iterating val_loader: {e}")

        # Test Test Loader
        print("\nTest Loader:")
        try:
            test_images, test_labels = next(iter(test_loader))
            print(f"  Image batch shape: {test_images.shape}")
            print(f"  Label batch shape: {test_labels.shape}")
        except StopIteration:
            print("  Test Loader is empty!")
        except Exception as e:
            print(f"  Error iterating test_loader: {e}")


        print("\nClass mapping returned:", class_map)
        print("-----------------------------------------")
        print("Data loader script finished basic tests successfully.")

    except FileNotFoundError as e:
        print(f"\nERROR: Data directory issue - {e}")
        print("Please ensure the data directories specified in config.yaml exist and are structured correctly.")
    except ValueError as e:
        print(f"\nERROR: Data loading issue - {e}")
        print("Please check the data content (e.g., enough samples per class for split, valid image files).")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        # You might want to print the full traceback in development:
        # import traceback
        # traceback.print_exc()