import tensorflow as tf
import numpy as np
import sys
import os

# ==============================
# CONFIGURATION
# ==============================

MODEL_PATH = "models/best_model.keras"
IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "Apple_Fresh",
    "Apple_Rotten",
    "Banana_Fresh",
    "Banana_Rotten",
    "Bellpepper_Fresh",
    "Bellpepper_Rotten",
    "Bittergourd_Fresh",
    "Bittergourd_Rotten",
    "Capsicum_Fresh",
    "Capsicum_Rotten",
    "Carrot_Fresh",
    "Carrot_Rotten",
    "Cucumber_Fresh",
    "Cucumber_Rotten",
    "Mango_Fresh",
    "Mango_Rotten",
    "Okra_Fresh",
    "Okra_Rotten",
    "Orange_Fresh",
    "Orange_Rotten",
    "Potato_Fresh",
    "Potato_Rotten",
    "Strawberry_Fresh",
    "Strawberry_Rotten",
    "Tomato_Fresh",
    "Tomato_Rotten"
]


# ==============================
# LOAD MODEL
# ==============================

print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# ==============================
# PREDICTION FUNCTION
# ==============================

def predict_image(image_path):

    if not os.path.exists(image_path):
        print(f"\nError: Image not found -> {image_path}")
        return

    # Load image
    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMG_SIZE
    )

    # Convert image to array
    image_array = tf.keras.utils.img_to_array(image)

    # Normalize exactly like training
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    predictions = model.predict(image_array, verbose=0)

    predicted_index = np.argmax(predictions[0])

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = predictions[0][predicted_index] * 100

    # Split Food + Quality
    food_name, quality = predicted_class.rsplit("_", 1)

    # ==============================
    # RESULT
    # ==============================

    print("\n================================")
    print("      FOOD QUALITY ANALYZER")
    print("================================")

    print(f"Food       : {food_name}")
    print(f"Quality    : {quality}")
    print(f"Confidence : {confidence:.2f}%")

    print("================================")


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("\nUsage:")
        print("python src\\predict.py <image_path>")
        print("\nExample:")
        print('python src\\predict.py "test_images\\tomato.jpg"')
    else:
        image_path = sys.argv[1]
        predict_image(image_path)