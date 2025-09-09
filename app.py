from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageOps
import gradio as gr

# Load trained model
model = load_model("model/digits_model.h5")

def predict_digit(sketchpad):
    if sketchpad is not None:
        image_data = sketchpad['composite']
        pil_image = Image.fromarray(image_data)

        # Fill transparent background with white
        if pil_image.mode == "RGBA":
            white_background = Image.new("RGBA", pil_image.size, (255, 255, 255, 255))
            pil_image = Image.alpha_composite(white_background, pil_image)
            pil_image = pil_image.convert("RGB")

        grayscale_image = pil_image.convert("L")
        img = ImageOps.invert(grayscale_image)
        img = img.resize((28, 28))
        img = np.array(img) / 255.0
        img = img.reshape(-1, 28, 28, 1)

        pred = model.predict(img)
        predicted_digit = np.argmax(pred)

        return f"Predicted Digit: {predicted_digit}"

# Gradio interface
iface = gr.Interface(
    fn=predict_digit,
    inputs="sketchpad",
    outputs="text",
    live=True
)

if _name_ == "_main_":
    iface.launch()