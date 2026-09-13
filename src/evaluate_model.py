import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import (
    test_dataset,
    test_df,
    class_names
)

MODEL_PATH = "models/best_model.keras"

print("\nLoading best model...")
model = tf.keras.models.load_model(MODEL_PATH)

# Test accuracy
test_loss, test_accuracy = model.evaluate(test_dataset)

print("\n==============================")
print("TEST RESULTS")
print("==============================")
print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")

# Predictions
y_true = test_df["label"].map(
    {name: i for i, name in enumerate(class_names)}
).values

predictions = model.predict(test_dataset)
y_pred = np.argmax(predictions, axis=1)

# Classification report
print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(16, 14))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Food Quality Analyzer - Confusion Matrix")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()

plt.savefig("models/confusion_matrix.png")

print("\nConfusion matrix saved:")
print("models/confusion_matrix.png")