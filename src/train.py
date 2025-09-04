import config
from data_loader import load_data
from model import build_model
import matplotlib.pyplot as plt
import os


def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Ensure outputs folder exists
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    plt.savefig(os.path.join(output_dir, "accuracy_loss.png"))
    plt.show()

def train():
    train_generator, val_generator = load_data()
    model = build_model(input_shape=(150,150,3))

    history = model.fit(
        train_generator,
        steps_per_epoch=100,
        epochs=config.EPOCHS,
        validation_data=val_generator,
        validation_steps=20
    )

    model.save(config.MODEL_PATH)
    plot_history(history)
    print("Model saved at", config.MODEL_PATH)

if __name__ == "__main__":
    train()
