🖌️ MNIST Handwritten Digit Classifier

A deep learning project that trains a Convolutional Neural Network (CNN) on the MNIST dataset and deploys it with a Gradio interface.
You can draw digits (0–9) on a canvas, and the model will predict the digit in real time.

📂 Project Structure

MNIST/
│── README.md
│── requirements.txt
│── app.py                # Gradio app for demo
│── train.py              # Training script
│── visualize.py          # Data visualization script
│── digits_model.h5       # Trained model  
│── mnist_training.ipynb  #  Jupyter notebook


⚡ Features

📊 Data Visualization – View sample digits & class distribution

🧠 CNN Model – Trained with TensorFlow/Keras

🔄 Data Augmentation – Improves generalization with rotations/shifts/zoom

📈 Evaluation – Accuracy, classification report, and confusion matrix

🎨 Interactive Demo – Draw digits in a sketchpad and get predictions instantly


📊 Results

Accuracy: ~99% on MNIST test set

Confusion Matrix Example:<img width="820" height="674" alt="image" src="https://github.com/user-attachments/assets/38c57354-5431-46db-9a2f-7899cbaa9148" />

