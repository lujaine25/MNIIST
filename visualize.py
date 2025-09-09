import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tensorflow.keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Show samples
fig, axes = plt.subplots(1, 10, figsize=(12, 4))
for i, ax in enumerate(axes):
    ax.imshow(X_train[i], cmap="gray")
    ax.set_title(f"Label: {y_train[i]}")
    ax.axis("off")
plt.suptitle("Sample MNIST Digits", fontsize=16)
plt.show()

# Distribution
plt.figure(figsize=(8, 5))
sns.countplot(x=y_train, palette="tab10")
plt.title("Digit Distribution in Training Set")
plt.show()