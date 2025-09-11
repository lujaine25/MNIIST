# 🖌️ MNIST Handwritten Digit Classifier  

A deep learning project that trains a **Convolutional Neural Network (CNN)** on the **MNIST dataset** and deploys it with a **Gradio interface**.  
You can draw digits (0–9) on a canvas, and the model will predict the digit in real time.  






## 📂 Project Structure  

MNIST/

│── README.md

│── requirements.txt

│── app.py              # Gradio app for demo

│── train.py            # Training script

│── visualize.py        # Data visualization script

│── digits_model.h5     # Trained model

│── mnist_training.ipynb # Jupyter notebook






## ⚡ Features  

- 📊 Data Visualization – View sample digits & class distribution  
- 🧠 CNN Model – Trained with TensorFlow/Keras  
- 🔄 Data Augmentation – Rotations, shifts, and zoom for better generalization  
- 📈 Evaluation – Accuracy, classification report, and confusion matrix  
- 🎨 Interactive Demo – Draw digits in a sketchpad and get predictions instantly




 

 ## 🏋️ Training the Model  

Run the training script:  

python train.py  

This will:  
✅ Train a CNN on MNIST  
✅ Show evaluation metrics & confusion matrix  
✅ Save the model as digits_model.h5  





## 🏋️ Training the Model  

Run the training script:  

python train.py  

This will:  
✅ Train a CNN on MNIST  
✅ Show evaluation metrics & confusion matrix  
✅ Save the model as digits_model.h5  






## 🎨 Running the App  

Launch the Gradio app:  

python app.py  

This will open a browser window where you can draw digits and get predictions in real-time.  






## 📊 Results  

- Accuracy: ~99% on MNIST test set  
- Confusion Matrix Example:  



<img width="820" height="674" alt="Screenshot 2025-09-09 192226" src="https://github.com/user-attachments/assets/7aea9f0c-2548-4ab5-b7af-a52af5d3024b" />

















[📑 View the Presentation](https://drive.google.com/file/d/1OIF8jTp7xhPs1VIgarvbuD3XbYOlPLKL/view?usp=sharing)



