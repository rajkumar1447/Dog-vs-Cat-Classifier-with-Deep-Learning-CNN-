# Dog-vs-Cat-Classifier-with-Deep-Learning-CNN-


This project is a convolutional neural network (CNN) built to classify images into **cats** or **dogs** using Keras and TensorFlow.

---

##  Dataset

You can download the **Cat vs Dog dataset** from Kaggle:

[Cat and Dog Dataset by tongpython](https://www.kaggle.com/datasets/tongpython/cat-and-dog) :contentReference[oaicite:0]{index=0}

Collected images should be organized into two folders inside the `src/` directory:


1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/Cat-vs-Dog-Classifier.git
   cd Cat-vs-Dog-Classifier


Install Python 3.9+ if not already installed.

2. **Create & activate a virtual environment:**

Windows (PowerShell):

python -m venv venv
.\venv\Scripts\activate


macOS/Linux:

python3 -m venv venv
source venv/bin/activate


Select the venv interpreter in VS Code:

Open Command Palette (⇧⌘P or Ctrl+Shift+P)

Choose Python: Select Interpreter

Pick your project's venv

3. **Install dependencies:**

pip install -r requirements.txt





1. **Training the Model**
cd src
python train.py


Trains the CNN with early stopping and dropout

Saves the best model and accuracy/loss plot to outputs/

2. **Evaluating the Model**
cd src
python evaluate.py


Evaluates saved model on test_set/

Prints test loss & accuracy

3. **Predicting on a New Image**
cd src
python predict.py path/to/image.jpg


Predicts cat or dog for the provided image file