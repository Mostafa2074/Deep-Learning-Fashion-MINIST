import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
import numpy as np
import tensorflow as tf
import os


class ModelHandler:
    def __init__(self, model_path=None):
        import os
        import tensorflow as tf

        if model_path is None:
            model_path = "D:/DEPI ASSIGN 14/Trial/models/fashion_ann.keras"

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model not found at: {model_path}")

        self.model = tf.keras.models.load_model(model_path)
        print(f"✅ Model loaded: {os.path.basename(model_path)}")

        # Fashion-MNIST labels
        self.class_names = [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        ]

    def preprocess(self, image_path):
        """Convert real image to Fashion-MNIST style (grayscale, inverted, binary, flattened)."""
        from PIL import Image
        import numpy as np

        image = Image.open(image_path).convert("L")  # Grayscale
        image = image.resize((28, 28))

        # Convert to NumPy array
        arr = np.array(image).astype("float32")

        # Invert white backgrounds
        if np.mean(arr) > 127:
            arr = 255 - arr

        # Normalize & threshold
        arr = arr / 255.0
        arr = (arr > 0.5).astype("float32")

        arr = arr.reshape(1, -1)  # Flatten to (1, 784)
        return arr

    def predict(self, image_path):
        """Run prediction using the loaded model."""
        import numpy as np

        arr = self.preprocess(image_path)
        preds = self.model.predict(arr)
        label_idx = np.argmax(preds)
        confidence = float(np.max(preds))
        return self.class_names[label_idx], confidence



# ---------- GUI ----------
class FashionClassifierApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("👗 Fashion-MNIST Image Classifier")
        self.geometry("720x600")
        self.resizable(False, False)
        self.image_path = None

        # Load model
        try:
            self.model_handler = ModelHandler()
        except FileNotFoundError as e:
            ctk.CTkLabel(self, text=str(e), font=("Arial", 16)).pack(pady=20)
            return

        # Title
        self.title_label = ctk.CTkLabel(
            self,
            text="🧠 Fashion-MNIST Image Classifier",
            font=("Arial", 24, "bold")
        )
        self.title_label.pack(pady=20)

        # Image display
        self.image_frame = ctk.CTkFrame(self, width=400, height=300)
        self.image_frame.pack(pady=10)
        self.image_label = ctk.CTkLabel(
            self.image_frame, text="No image uploaded", width=400, height=300
        )
        self.image_label.pack()

        # Buttons
        self.upload_button = ctk.CTkButton(
            self, text="Upload Image", command=self.upload_image
        )
        self.upload_button.pack(pady=15)

        self.predict_button = ctk.CTkButton(
            self, text="Predict", command=self.predict
        )
        self.predict_button.pack(pady=10)

        # Results
        self.result_label = ctk.CTkLabel(self, text="", font=("Arial", 18))
        self.result_label.pack(pady=10)
        self.prob_label = ctk.CTkLabel(self, text="", font=("Arial", 16))
        self.prob_label.pack(pady=5)

    def upload_image(self):
        """Upload and display an image."""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")]
        )
        if file_path:
            self.image_path = file_path
            image = Image.open(file_path)
            ctk_image = ctk.CTkImage(light_image=image, size=(300, 300))
            self.image_label.configure(image=ctk_image, text="")
            self.image_label.image = ctk_image
            self.result_label.configure(text="")
            self.prob_label.configure(text="")

    def predict(self):
        """Run prediction."""
        if not self.image_path:
            self.result_label.configure(text="⚠️ Please upload an image first.")
            return

        label, conf = self.model_handler.predict(self.image_path)
        self.result_label.configure(text=f"Prediction: {label}")
        self.prob_label.configure(text=f"Confidence: {conf * 100:.2f}%")


# ---------- Run ----------
if __name__ == "__main__":
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("blue")
    app = FashionClassifierApp()
    app.mainloop()
