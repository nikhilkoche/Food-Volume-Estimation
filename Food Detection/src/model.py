import argparse
import os
import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing import image_dataset_from_directory

# =============== CONFIGURATION ===============
IMAGE_SIZE = 299
BATCH_SIZE = 16  # Adjusted for efficiency
EPOCHS = 5
LEARNING_RATE = 0.005
WEIGHT_DECAY = 2e-4
GPU_COUNT = tf.config.experimental.list_physical_devices('GPU')

# Enable Mirrored Strategy for Multi-GPU Training
strategy = tf.distribute.MirroredStrategy()

def load_dataset(dataset_dir, batch_size=16, image_size=299):
    """Loads images from a local dataset directory using tf.data."""
    dataset = image_dataset_from_directory(
        dataset_dir,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=True
    )
    
    # Get the number of classes from the dataset
    class_names = dataset.class_names
    num_classes = len(class_names)
    
    return dataset, num_classes

def build_model(num_classes):
    """Creates a Keras model based on InceptionResNetV2."""
    with strategy.scope():
        base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
        base_model.trainable = False  # Freeze pre-trained weights

        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(1024, activation='relu')(x)
        x = Dense(num_classes, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=x)
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    return model

def train(model_dir, dataset_dir):
    """Trains or resumes training the Keras model using a local dataset directory."""
    train_data, num_classes = load_dataset(dataset_dir)

    model_path = os.path.join(model_dir, "model.h5")

    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}...")
        model = tf.keras.models.load_model(model_path)
    else:
        print("No existing model found. Creating a new model...")
        model = build_model(num_classes)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(model_path, save_best_only=True),
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    ]

    model.fit(
        train_data,
        epochs=5,
        callbacks=callbacks
    )

    model.save(os.path.join(model_dir, "final_model.h5"))
    print("Model training complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model against a dataset.')

    parser.add_argument('--model', required=True, help='Path to save the trained model.')
    parser.add_argument('--dataset', required=True, help='Dataset directory.')

    args = parser.parse_args()

    if not os.path.exists(args.model):
        os.makedirs(args.model)

    train(args.model, args.dataset)
