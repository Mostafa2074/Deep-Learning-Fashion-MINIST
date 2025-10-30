import sys
import os

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Import and run GUI
from gui import FashionClassifierApp
import customtkinter as ctk

if __name__ == "__main__":
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("blue")
    app = FashionClassifierApp()
    app.mainloop()
