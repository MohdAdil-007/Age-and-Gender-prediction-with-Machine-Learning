import cv2
import numpy as np
import tensorflow as tf
import os

# --- 1. Configuration ---
img_width, img_height = 96, 96
model_path = "age_gender_model.h5"
window_name = "Age & Gender Prediction" # Define the window name once

# This is your image path
# NOTE: Update this path to your current image location!
image_path = r"C:\Users\adilv\Downloads\Screenshot 2025-11-12 012005.png"

# Path to the OpenCV Haar Cascade for face detection
try:
    face_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    if face_cascade.empty():
        raise IOError(f"Could not load Haar cascade from {face_cascade_path}")
except Exception as e:
    print(f"Error loading face detector: {e}")
    print("Please ensure OpenCV is installed correctly ('pip install opencv-python')")
    exit()

# --- 2. Load Model ---
# We compile=False because we are only doing inference (prediction)
try:
    model = tf.keras.models.load_model(model_path, compile=False)
except IOError:
    print(f"Error: Could not load model from {model_path}")
    print("Make sure 'age_gender_model.h5' is in the same directory as this script.")
    exit()

# --- 3. Load and Process the Image ---
image = cv2.imread(image_path)
if image is None:
    print(f"Error: Could not read image from {image_path}")
    exit()

# Convert to grayscale for the face detector
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(50, 50)  # Look for faces at least 50x50 pixels
)

if len(faces) == 0:
    print("No faces found in the image.")
    exit()

print(f"Found {len(faces)} face(s). Running prediction on the first one.")

# Get coordinates for the first face
(x, y, w, h) = faces[0]

# Crop the face (Region of Interest - ROI) from the *color* image
face_roi = image[y:y+h, x:x+w]

# --- 4. Prepare Cropped Face for Model ---
# Preprocess the cropped face EXACTLY as we did the training data
processed_face = cv2.resize(face_roi, (img_width, img_height))
processed_face = processed_face.astype('float32') / 255.0
processed_face = np.expand_dims(processed_face, axis=0)  # Create a batch of 1

# --- 5. Make Prediction ---
predictions = model.predict(processed_face)

gender_pred = predictions[0]  # First output
age_pred = predictions[1]      # Second output

# --- 6. Interpret and Display Results (with moveWindow) ---
gender = "Female" if gender_pred[0][0] > 0.5 else "Male"
age = int(age_pred[0][0])

print(f"\n--- Prediction ---")
print(f"Gender: {gender}")
print(f"Age:    {age}")

# Draw a rectangle and text on the original image to show the result
cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
label = f"{gender}, {age}"
cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Create the named window
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE) 

# Move the window to the desired position (e.g., 100 pixels right, 100 pixels down)
cv2.moveWindow(window_name, 100, 100) 

# Show the image in the new window
cv2.imshow(window_name, image)

print("\nPress any key to close the image window...")
cv2.waitKey(0)
cv2.destroyAllWindows()