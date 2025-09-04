import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def evaluate_model(model_path="model.h5"):
    # Base directory = current src folder
    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(base_dir, "test_set")
    model_path = os.path.join(base_dir, model_path)

    # Load model
    model = tf.keras.models.load_model(model_path)

    # Test data generator
    test_datagen = ImageDataGenerator(rescale=1.0/255)

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode="binary",
        shuffle=False
    )

    # Evaluate
    loss, accuracy = model.evaluate(test_generator)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    evaluate_model()
