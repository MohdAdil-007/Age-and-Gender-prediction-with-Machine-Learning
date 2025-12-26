import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, 
    GlobalAveragePooling2D
)
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# --- 1. Configuration ---
# Your dataset path is set here
dataset_path = r"C:\Users\adilv\Downloads\utk-face-cropped\UTKFace\utkcropped"
img_width, img_height = 96, 96
batch_size = 64
epochs = 25  # Start with 25, you may need more

# --- 2. Data Loading and Preprocessing ---

def load_data(path):
    """
    Loads the UTKFace dataset and parses labels from filenames.
    Filename format: [age]_[gender]_[race]_[date&time].jpg
    """
    images = []
    ages = []
    genders = []
    
    print(f"Loading images from: {path}")
    
    # Check if the path exists
    if not os.path.exists(path):
        print(f"Error: Path not found: {path}")
        print("Please double-check the 'dataset_path' variable.")
        return None, None, None

    for filename in os.listdir(path):
        if not filename.endswith(".jpg"):
            continue
            
        try:
            # Parse filename
            parts = filename.split("_")
            age = int(parts[0])
            gender = int(parts[1]) # 0 = Male, 1 = Female
            
            # Load and preprocess image
            img_path = os.path.join(path, filename)
            image = cv2.imread(img_path)
            
            # Check if image loaded successfully
            if image is None:
                # print(f"Skipping {filename}: Could not read image.")
                continue

            # Note: UTKFace is already cropped, so we just resize
            image = cv2.resize(image, (img_width, img_height))
            image = image.astype('float32') / 255.0  # Normalize
            
            images.append(image)
            ages.append(age)
            genders.append(gender)
            
        except Exception as e:
            # Skip files with parsing errors (e.g., '.DS_Store' or bad filenames)
            # print(f"Skipping {filename}: {e}")
            pass
            
    if not images:
        print(f"No images loaded. Check the path and file structure in {path}.")
        return None, None, None
        
    print(f"Loaded {len(images)} images.")
    
    # Convert lists to NumPy arrays
    images = np.array(images)
    ages = np.array(ages)
    genders = np.array(genders)
    
    return images, ages, genders

# Load the data
images, ages, genders = load_data(dataset_path)

# Proceed only if data was loaded successfully
if images is not None:
    # Split data into training and validation sets
    (X_train, X_val, 
     y_age_train, y_age_val, 
     y_gender_train, y_gender_val) = train_test_split(
        images, ages, genders, test_size=0.2, random_state=42
    )

    # --- 3. Build the Multi-Task Model ---

    def build_model(input_shape=(96, 96, 3)):
        """
        Builds the multi-task CNN model using MobileNetV2.
        """
        
        # Define the input
        inputs = Input(shape=input_shape)
        
        # Load the MobileNetV2 backbone (pre-trained on ImageNet)
        # We exclude the top (classification) layer
        base_model = MobileNetV2(
            weights='imagenet', 
            include_top=False, 
            input_tensor=inputs
        )
        
        # Freeze the layers of the base model so we only train our new "heads"
        base_model.trainable = False
        
        # --- Shared Layers ---
        # We take the output of the base model and add our own shared layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        
        # --- Output Head 1: Gender (Classification) ---
        gender_output = Dense(128, activation='relu')(x)
        gender_output = Dropout(0.5)(gender_output)
        gender_output = Dense(1, activation='sigmoid', name='gender_output')(gender_output)
        
        # --- Output Head 2: Age (Regression) ---
        age_output = Dense(128, activation='relu')(x)
        age_output = Dropout(0.5)(age_output)
        age_output = Dense(1, activation='linear', name='age_output')(age_output)
        
        # --- Create the Model ---
        # The model has one input and two outputs
        model = Model(inputs=inputs, outputs=[gender_output, age_output])
        
        return model

    # Build the model
    model = build_model(input_shape=(img_width, img_height, 3))

    # --- 4. Compile the Model ---

    # We need to define a loss for *each* output head
    losses = {
        'gender_output': 'binary_crossentropy',  # For binary classification
        'age_output': 'mean_absolute_error'    # For regression
    }

    # We can also define different metrics for each head
    metrics = {
        'gender_output': 'accuracy',
        'age_output': 'mae'  # Mean Absolute Error
    }

    # Use Adam optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer,
        loss=losses,
        metrics=metrics
    )

    # Print a summary of the model architecture
    model.summary()

    # --- 5. Train the Model ---

    # We must prepare the labels as a dictionary that matches the output names
    y_train = {
        'gender_output': y_gender_train,
        'age_output': y_age_train
    }

    y_val = {
        'gender_output': y_gender_val,
        'age_output': y_age_val
    }

    print("\nStarting model training...")

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        verbose=1
    )

    print("Training complete.")

    # --- 6. Save the Model ---
    model.save("age_gender_model.h5")
    print("Model saved as 'age_gender_model.h5'")

else:
    print("Data loading failed. Please check the 'dataset_path' variable and ensure the UTKFace folder is correct.")