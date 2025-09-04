import config
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np

def predict_image(img_path):
    model = load_model(config.MODEL_PATH)

    img = image.load_img(img_path, target_size=config.IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    if prediction > 0.5:
        print(f"{img_path} -> Dog")
    else:
        print(f"{img_path} -> Cat")

if __name__ == "__main__":
    predict_image("cat.4002.jpg")
