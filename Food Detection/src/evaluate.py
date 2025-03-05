"""
This code is used to evaluate our trained model against a number of datasets.
"""
"""
This code is used to evaluate our trained model against a dataset.
"""
import argparse
import os
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers import Layer

# =============== CONFIGURATION ===============
IMAGE_SIZE = 299
BATCH_SIZE = 16  # Must match training batch size

class CustomScaleLayer(Layer):
    """A custom scaling layer (example implementation)."""
    def __init__(self, scale_factor=1.0, **kwargs):
        super(CustomScaleLayer, self).__init__(**kwargs)
        self.scale_factor = scale_factor

    def call(self, inputs):
        return inputs * self.scale_factor

    def get_config(self):
        config = super().get_config()
        config.update({"scale_factor": self.scale_factor})
        return config

def load_dataset(dataset_dir):
    """Loads validation dataset using tf.data API."""
    dataset = image_dataset_from_directory(
        dataset_dir,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False  # No need to shuffle for evaluation
    )
    
    class_names = dataset.class_names
    num_classes = len(class_names)
    
    return dataset, class_names, num_classes

def evaluate(model_dir, dataset_dir):
    """
    Evaluates a trained model against a dataset.
    """
    model_path = os.path.join(model_dir, "final_model.h5")  # Ensure .h5 format

    if not os.path.exists(model_path):
        raise Exception(f"Model file {model_path} does not exist. Train the model first.")

    print(f"Loading model from {model_path}...")

    # Use custom_objects to load the model correctly
    with tf.keras.utils.custom_object_scope({'CustomScaleLayer': CustomScaleLayer}):
        model = tf.keras.models.load_model(model_path)

    # Load dataset
    dataset, class_names, num_classes = load_dataset(dataset_dir)

    # Get predictions
    predictions = model.predict(dataset)
    predicted_classes = np.argmax(predictions, axis=1)

    # Save images with predicted class names
    save_results(dataset, predicted_classes, class_names, model_dir)

    # Evaluate model
    loss, accuracy = model.evaluate(dataset)
    print(f"Evaluation Results - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

def save_results(dataset, predicted_classes, class_names, model_dir):
    """
    Saves validation images with their predicted class labels.
    """
    validation_results_dir = os.path.join(model_dir, "Validations")
    os.makedirs(validation_results_dir, exist_ok=True)

    print("Saving evaluation results...")

    for i, (image_batch, label_batch) in enumerate(dataset):
        for j in range(len(image_batch)):
            img_array = np.array(image_batch[j] * 255, dtype=np.uint8)  # Convert back to 0-255 range
            predicted_class = class_names[predicted_classes[i * BATCH_SIZE + j]]
            
            # Save the image with predicted class name
            save_path = os.path.join(validation_results_dir, f"{predicted_class}_{i * BATCH_SIZE + j}.png")
            cv2.imwrite(save_path, img_array)

    print(f"Evaluation images saved to {validation_results_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a trained model against a dataset.')

    parser.add_argument('--model', required=True, help='Path to the trained model directory.')
    parser.add_argument('--dataset', required=True, help='Path to the dataset directory.')

    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise Exception(f"Path {args.model} doesn't exist.")

    if not os.path.exists(args.dataset):
        raise Exception(f"Path {args.dataset} doesn't exist.")

    evaluate(args.model, args.dataset)
