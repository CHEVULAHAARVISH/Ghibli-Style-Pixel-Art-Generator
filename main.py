import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


class PixelArtWebcam:
    def __init__(self, root):
        self.root = root
        self.root.title("Ghibli-Style Pixel Art Generator")

        # Set up the UI
        self.frame = ttk.Frame(root, padding="10")
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Create a label to display the webcam feed
        self.video_label = ttk.Label(self.frame)
        self.video_label.pack(padx=10, pady=10)

        # Create control frame
        self.control_frame = ttk.Frame(self.frame)
        self.control_frame.pack(fill=tk.X, padx=10, pady=5)

        # Create pixel size slider
        ttk.Label(self.control_frame, text="Pixel Size:").pack(side=tk.LEFT)
        self.pixel_size_var = tk.IntVar(value=16)
        self.pixel_slider = ttk.Scale(
            self.control_frame,
            from_=4,
            to=32,
            orient=tk.HORIZONTAL,
            variable=self.pixel_size_var,
            length=200
        )
        self.pixel_slider.pack(side=tk.LEFT, padx=5)

        # Create color palette dropdown
        ttk.Label(self.control_frame, text="Color Palette:").pack(side=tk.LEFT, padx=(20, 5))
        self.palette_var = tk.StringVar(value="ghibli")
        self.palette_dropdown = ttk.Combobox(
            self.control_frame,
            textvariable=self.palette_var,
            values=["ghibli", "retro", "pastel", "monochrome"]
        )
        self.palette_dropdown.pack(side=tk.LEFT, padx=5)

        # Create buttons
        self.button_frame = ttk.Frame(self.frame)
        self.button_frame.pack(fill=tk.X, padx=10, pady=10)

        self.capture_btn = ttk.Button(
            self.button_frame,
            text="Create Pixel Art",
            command=self.capture_pixel_art
        )
        self.capture_btn.pack(side=tk.LEFT, padx=5)

        self.save_btn = ttk.Button(
            self.button_frame,
            text="Save Image",
            command=self.save_image
        )
        self.save_btn.pack(side=tk.LEFT, padx=5)

        self.toggle_btn = ttk.Button(
            self.button_frame,
            text="Toggle Normal/Pixel",
            command=self.toggle_view
        )
        self.toggle_btn.pack(side=tk.LEFT, padx=5)

        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            error_label = ttk.Label(self.frame, text="Error: Could not open webcam!")
            error_label.pack()

        # Set initial values
        self.pixelated = False
        self.pixel_art_image = None
        self.current_frame = None

        # Start video stream
        self.update_frame()

        # Color palettes
        self.palettes = {
            "ghibli": [
                [124, 176, 203], [188, 215, 200], [230, 216, 176],
                [220, 161, 140], [190, 128, 128], [122, 143, 122],
                [237, 177, 131], [170, 216, 232], [255, 232, 190]
            ],
            "retro": [
                [56, 32, 38], [126, 37, 83], [171, 82, 54],
                [234, 137, 70], [250, 199, 76], [169, 231, 146],
                [0, 149, 233], [171, 81, 164], [64, 53, 83]
            ],
            "pastel": [
                [255, 209, 220], [207, 232, 255], [255, 254, 196],
                [204, 255, 218], [234, 195, 255], [255, 231, 186],
                [170, 222, 167], [255, 174, 174], [189, 178, 255]
            ],
            "monochrome": [
                [0, 0, 0], [37, 37, 37], [73, 73, 73],
                [109, 109, 109], [146, 146, 146], [182, 182, 182],
                [219, 219, 219], [255, 255, 255]
            ]
        }

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Flip the frame horizontally for a mirror effect
            frame = cv2.flip(frame, 1)
            self.current_frame = frame

            # If pixelated mode is on, show the pixelated version
            if self.pixelated and self.pixel_art_image is not None:
                display_image = self.pixel_art_image
            else:
                # Convert frame color from BGR to RGB for tkinter
                display_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert to uint8 to ensure compatibility with PIL
            display_image = np.array(display_image, dtype=np.uint8)

            # Convert to PIL Image
            pil_img = Image.fromarray(display_image)
            # Resize the image to fit in the window
            width, height = 640, 480
            pil_img = pil_img.resize((width, height), Image.LANCZOS)
            # Convert to tkinter-compatible photo image
            tk_img = ImageTk.PhotoImage(image=pil_img)

            # Update the video label
            self.video_label.config(image=tk_img)
            self.video_label.image = tk_img  # Keep a reference

        # Call this function again after 10ms
        self.root.after(10, self.update_frame)

    def create_pixel_art(self, frame):
        # Get the pixel size from the slider
        pixel_size = self.pixel_size_var.get()

        # Resize to create pixelation effect
        height, width = frame.shape[:2]
        small_frame = cv2.resize(frame, (width // pixel_size, height // pixel_size),
                                 interpolation=cv2.INTER_LINEAR)
        # Scale back up using nearest neighbor
        pixel_frame = cv2.resize(small_frame, (width, height),
                                 interpolation=cv2.INTER_NEAREST)

        # Apply color palette
        palette_name = self.palette_var.get()
        palette = np.array(self.palettes.get(palette_name, self.palettes["ghibli"]), dtype=np.uint8)

        # Convert to RGB for easier color mapping
        rgb_pixel_frame = cv2.cvtColor(pixel_frame, cv2.COLOR_BGR2RGB)

        # Apply color palette (map each pixel to closest palette color)
        # This is a simplified version - you could use k-means for better results
        h, w, _ = rgb_pixel_frame.shape

        # Vectorized version (faster)
        reshaped_frame = rgb_pixel_frame.reshape((-1, 3))
        distances = np.sqrt(((reshaped_frame[:, np.newaxis, :].astype(np.float32) -
                              palette[np.newaxis, :, :].astype(np.float32)) ** 2).sum(axis=2))
        indices = np.argmin(distances, axis=1)
        result = palette[indices].reshape(h, w, 3)

        # Ensure the result is uint8
        result = np.array(result, dtype=np.uint8)

        return result

    def capture_pixel_art(self):
        if self.current_frame is not None:
            self.pixel_art_image = self.create_pixel_art(self.current_frame)
            self.pixelated = True

    def toggle_view(self):
        self.pixelated = not self.pixelated

    def save_image(self):
        if self.pixel_art_image is not None:
            # Save the image
            filename = f"pixel_art_{self.palette_var.get()}_{self.pixel_size_var.get()}.png"
            save_img = cv2.cvtColor(self.pixel_art_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename, save_img)
            save_label = ttk.Label(self.frame, text=f"Image saved as {filename}")
            save_label.pack()
            self.root.after(2000, save_label.destroy)

    def __del__(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()


if __name__ == "__main__":
    root = tk.Tk()
    app = PixelArtWebcam(root)
    root.mainloop()

