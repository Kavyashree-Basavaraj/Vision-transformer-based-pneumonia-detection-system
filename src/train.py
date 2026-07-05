import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import yaml
import os
from transformers import get_linear_schedule_with_warmup # Recommended scheduler

# Assuming these imports are from your files
from model import ViTForImageClassification, load_model, save_model
from data_loader import create_data_loaders # Assuming create_data_loaders returns class_to_idx now

def train_model(model, train_dataloader, val_dataloader, config, device):
    """Trains the model with early stopping and saves the best version."""

    # --- Configuration ---
    num_epochs = config['num_epochs']
    learning_rate = config['learning_rate']
    accumulation_steps = config.get('accumulation_steps', 1) # Default to 1 if not specified
    patience = config.get('early_stopping_patience', 10) # Default patience
    early_stopping_metric = config.get('early_stopping_metric', 'loss').lower() # 'loss', 'f1', or 'accuracy'
    warmup_steps = config.get('warmup_steps', 0) # Number of warmup steps for scheduler
    weight_decay = config.get('weight_decay', 0.01) # AdamW weight decay
    model_path = config['model_path']
    freeze_epochs = config.get('freeze_backbone_epochs', 0) # Epochs to train only classifier

    print(f"--- Training Configuration ---")
    print(f"Epochs: {num_epochs}, LR: {learning_rate}, Accumulation: {accumulation_steps}")
    print(f"Patience: {patience}, Stop Metric: {early_stopping_metric.upper()}")
    print(f"Warmup Steps: {warmup_steps}, Weight Decay: {weight_decay}")
    print(f"Freeze Backbone Epochs: {freeze_epochs}")
    print(f"Device: {device}")
    print(f"-----------------------------")

    # Ensure save directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # --- Initial Setup ---
    criterion = nn.CrossEntropyLoss() # Add weights here if NOT using WeightedRandomSampler in dataloader

    # --- Layer Freezing (Optional) ---
    if freeze_epochs > 0:
        print(f"Freezing backbone for the first {freeze_epochs} epochs.")
        for param in model.vit.parameters():
            param.requires_grad = False
        # Only optimize the classifier parameters initially
        optimizer = optim.AdamW(model.classifier.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


    # --- Scheduler ---
    num_training_steps = (len(train_dataloader) // accumulation_steps) * num_epochs
    print(f"Total optimization steps: {num_training_steps}")
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )

    # --- Tracking Variables ---
    best_metric_value = float('inf') if early_stopping_metric == 'loss' else -float('inf')
    epochs_no_improve = 0
    best_epoch = 0

    # --- Training Loop ---
    for epoch in range(num_epochs):
        # --- Unfreezing Logic (Optional) ---
        if freeze_epochs > 0 and epoch == freeze_epochs:
            print(f"\nUnfreezing backbone at epoch {epoch+1}. Resetting optimizer and scheduler.")
            for param in model.vit.parameters():
                param.requires_grad = True
            # Re-initialize optimizer with all parameters and potentially a lower LR
            unfreeze_lr = config.get('unfreeze_learning_rate', learning_rate / 10) # Example: Lower LR for fine-tuning
            print(f"Using learning rate: {unfreeze_lr} for full model fine-tuning.")
            optimizer = optim.AdamW(model.parameters(), lr=unfreeze_lr, weight_decay=weight_decay)
            # Re-initialize scheduler (adjust total steps if needed, though often okay to continue)
            remaining_epochs = num_epochs - epoch
            num_training_steps = (len(train_dataloader) // accumulation_steps) * remaining_epochs
            scheduler = get_linear_schedule_with_warmup(
                 optimizer,
                 num_warmup_steps=config.get('unfreeze_warmup_steps', 0), # Optionally different warmup
                 num_training_steps=num_training_steps
            )
            print(f"Scheduler reset. New total optimization steps: {num_training_steps}")


        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        optimizer.zero_grad() # Initialize gradients to zero at the start of the epoch

        loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}/{num_epochs} [Training]")
        for i, (images, labels) in loop:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            unscaled_loss = criterion(outputs, labels) # Calculate loss BEFORE scaling

            # Scale loss for accumulation
            scaled_loss = unscaled_loss / accumulation_steps

            # Backward pass (accumulates gradients)
            scaled_loss.backward()

            # Track training accuracy and loss (use unscaled loss for meaningful reporting)
            train_loss += unscaled_loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # Optimizer step (update weights)
            if (i + 1) % accumulation_steps == 0:
                # Gradient clipping (optional but often helpful for transformers)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step() # Perform optimization step
                scheduler.step() # Update learning rate
                optimizer.zero_grad() # Reset gradients for the next accumulation cycle

            loop.set_postfix({
                'Loss': f"{unscaled_loss.item():.4f}", # Show current batch unscaled loss
                'Acc': f"{100 * train_correct / train_total:.2f}%",
                'LR': f"{optimizer.param_groups[0]['lr']:.1e}" # Show current learning rate
                })

        # Handle final partial batch if dataloader size not divisible by accumulation_steps
        # Note: Typically not needed if optimizer.zero_grad() is outside the `if`
        # If you put zero_grad inside the if, uncomment below
        # if len(train_dataloader) % accumulation_steps != 0:
        #      optimizer.step()
        #      optimizer.zero_grad()

        avg_train_loss = train_loss / len(train_dataloader) # Average loss over batches
        train_accuracy = 100 * train_correct / train_total

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        all_val_preds = []
        all_val_labels = []
        with torch.no_grad():
            loop_val = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]")
            for images, labels in loop_val:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
                loop_val.set_postfix({'Val Loss': f"{loss.item():.4f}"})

        avg_val_loss = val_loss / len(val_dataloader)
        # Calculate validation metrics from sklearn
        val_accuracy = accuracy_score(all_val_labels, all_val_preds) * 100
        # Use 'weighted' or 'macro' for multi-class, 'binary' for binary
        # Ensure your labels are 0 and 1 for 'binary'
        val_f1 = f1_score(all_val_labels, all_val_preds, average='binary', zero_division=0)


        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}%")
        print(f"  Val Loss  : {avg_val_loss:.4f} | Val Acc  : {val_accuracy:.2f}% | Val F1: {val_f1:.4f}")
        print(f"  Current LR: {optimizer.param_groups[0]['lr']:.8f}")

        # --- Early Stopping & Model Saving ---
        metric_to_check = 0.0
        if early_stopping_metric == 'loss':
            metric_to_check = avg_val_loss
            improved = metric_to_check < best_metric_value
        elif early_stopping_metric == 'f1':
            metric_to_check = val_f1
            improved = metric_to_check > best_metric_value
        else: # Default or 'accuracy'
            metric_to_check = val_accuracy
            improved = metric_to_check > best_metric_value

        if improved:
            print(f"Validation {early_stopping_metric.upper()} improved ({best_metric_value:.4f} --> {metric_to_check:.4f}). Saving model...")
            best_metric_value = metric_to_check
            save_model(model, model_path) # Save the best model
            epochs_no_improve = 0
            best_epoch = epoch + 1
        else:
            epochs_no_improve += 1
            print(f"Validation {early_stopping_metric.upper()} did not improve for {epochs_no_improve} epoch(s). Best was {best_metric_value:.4f} at epoch {best_epoch}.")

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs!")
            break # Exit training loop

    # --- End of Training ---
    print(f"\nTraining finished after {epoch+1} epochs.")
    print(f"Best validation {early_stopping_metric.upper()}: {best_metric_value:.4f} achieved at epoch {best_epoch}.")

    # Load the best model weights before returning
    print(f"Loading best model from {model_path} for final evaluation...")
    model = load_model(model_path, device, config)
    if model is None:
         raise RuntimeError("Could not load the best saved model for evaluation.")

    return model # Return the best model loaded


def evaluate_model(model, test_dataloader, criterion, device, class_to_idx):
    """Evaluates the final model on the test set."""
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []

    # Inverse mapping for classification report
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    target_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())]

    with torch.no_grad():
        loop = tqdm(test_dataloader, desc="Evaluating on Test Set")
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            loop.set_postfix({'Loss': f"{loss.item():.4f}"})

    avg_test_loss = test_loss / len(test_dataloader)
    test_accuracy = 100 * test_correct / test_total

    print("\n--- Test Set Evaluation ---")
    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    # Classification Report
    print("\nClassification Report:")
    report = classification_report(all_labels, all_preds, target_names=target_names, zero_division=0)
    print(report)

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)

    # Plot and save confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('Test Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    report_dir = "reports"
    os.makedirs(report_dir, exist_ok=True)
    cm_path = os.path.join(report_dir, "test_confusion_matrix.png")
    report_path = os.path.join(report_dir, "test_classification_report.txt")

    try:
        plt.savefig(cm_path)
        print(f"Confusion matrix saved to {cm_path}")
        plt.close()
        with open(report_path, 'w') as f:
             f.write(f"Test Accuracy: {test_accuracy:.2f}%\n")
             f.write(f"Test Loss: {avg_test_loss:.4f}\n\n")
             f.write("Classification Report:\n")
             f.write(report)
             f.write("\n\nConfusion Matrix:\n")
             f.write(np.array2string(cm))
        print(f"Classification report saved to {report_path}")

    except Exception as e:
        print(f"Error saving report files: {e}")


    return test_accuracy

def main():
    # Load config
    config_path = '/content/drive/MyDrive/data/config.yaml' # Adjust if necessary
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"ERROR: Configuration file not found at {config_path}")
        return
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return

    # Set device
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA specified but not available. Falling back to CPU.")
        device = "cpu"
    print(f"Using device: {device}")

    # Create DataLoaders
    try:
        # Assume create_data_loaders now returns class_to_idx mapping
        train_dataloader, val_dataloader, test_dataloader, class_to_idx = create_data_loaders(config)
        print(f"Class mapping from data loader: {class_to_idx}")
    except Exception as e:
        print(f"Error creating data loaders: {e}")
        return

    # Create Model
    try:
        model = ViTForImageClassification(config)
        model = model.to(device)
    except Exception as e:
        print(f"Error creating model: {e}")
        return

    # Train the Model (returns the best loaded model)
    print("\n>>> Starting training...")
    try:
        best_model = train_model(model, train_dataloader, val_dataloader, config, device)
    except Exception as e:
        print(f"Error during training: {e}")
        # Optionally try loading last saved model if training failed mid-way
        print("Attempting to load previously saved best model for evaluation...")
        best_model = load_model(config['model_path'], device, config)
        if best_model is None:
             print("Could not load any model. Exiting.")
             return
        else:
             print("Loaded previously saved model successfully.")

    # Evaluate the Model on the Test Set
    print("\n>>> Starting evaluation on the test set...")
    try:
        # Define criterion again just for evaluation loss calculation
        criterion_eval = nn.CrossEntropyLoss()
        test_accuracy = evaluate_model(best_model, test_dataloader, criterion_eval, device, class_to_idx)
        print(f"\nFinal Test Accuracy: {test_accuracy:.2f}%")
    except Exception as e:
        print(f"Error during evaluation: {e}")

    print("\nScript finished.")


if __name__ == "__main__":
    main()