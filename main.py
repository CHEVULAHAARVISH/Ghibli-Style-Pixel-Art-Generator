import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import os
import time


class PixelArtWebcam:
    def __init__(self, root):
        self.root = root
        self.root.title("Animated Pixel Art Generator")

        # Set up the UI with tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Camera tab
        self.camera_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.camera_tab, text="Camera")

        # Image upload tab
        self.upload_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.upload_tab, text="Upload Image")

        # Create a label to display the video/image
        self.video_label = ttk.Label(self.camera_tab)
        self.video_label.pack(padx=10, pady=10)

        # Create a label for uploaded image
        self.upload_label = ttk.Label(self.upload_tab)
        self.upload_label.pack(padx=10, pady=10)

        # Setup controls for camera tab
        self.setup_camera_controls()

        # Setup controls for upload tab
        self.setup_upload_controls()

        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            error_label = ttk.Label(self.camera_tab, text="Error: Could not open webcam!")
            error_label.pack()

        # Set initial values
        self.pixelated = False
        self.pixel_art_image = None
        self.current_frame = None
        self.uploaded_image = None
        self.pixelated_upload = None
        self.animation_style = "none"
        self.animation_frame = 0
        self.last_animation_time = time.time()

        # Extended color palettes
        self.setup_palettes()

        # Start video stream
        self.update_frame()

    def setup_camera_controls(self):
        # Create control frames
        self.camera_control_frame = ttk.Frame(self.camera_tab)
        self.camera_control_frame.pack(fill=tk.X, padx=10, pady=5)

        # Create pixel size slider
        ttk.Label(self.camera_control_frame, text="Pixel Size:").pack(side=tk.LEFT)
        self.pixel_size_var = tk.IntVar(value=16)
        self.pixel_slider = ttk.Scale(
            self.camera_control_frame,
            from_=4,
            to=32,
            orient=tk.HORIZONTAL,
            variable=self.pixel_size_var,
            length=200
        )
        self.pixel_slider.pack(side=tk.LEFT, padx=5)

        # Create second row for palette and animation controls
        self.camera_control_frame2 = ttk.Frame(self.camera_tab)
        self.camera_control_frame2.pack(fill=tk.X, padx=10, pady=5)

        # Create color palette dropdown
        ttk.Label(self.camera_control_frame2, text="Color Palette:").pack(side=tk.LEFT)
        self.palette_var = tk.StringVar(value="ghibli")
        self.palette_dropdown = ttk.Combobox(
            self.camera_control_frame2,
            textvariable=self.palette_var,
            values=["ghibli", "retro", "pastel", "monochrome", "cyberpunk",
                    "vaporwave", "forest", "sunset", "ocean", "candy"]
        )
        self.palette_dropdown.pack(side=tk.LEFT, padx=5)

        # Create animation style dropdown
        ttk.Label(self.camera_control_frame2, text="Animation:").pack(side=tk.LEFT, padx=(20, 5))
        self.animation_var = tk.StringVar(value="none")
        self.animation_dropdown = ttk.Combobox(
            self.camera_control_frame2,
            textvariable=self.animation_var,
            values=["none", "pulse", "shimmer", "wave", "scan", "glitch"]
        )
        self.animation_dropdown.pack(side=tk.LEFT, padx=5)
        self.animation_dropdown.bind("<<ComboboxSelected>>", self.update_animation)

        # Create buttons
        self.camera_button_frame = ttk.Frame(self.camera_tab)
        self.camera_button_frame.pack(fill=tk.X, padx=10, pady=10)

        self.capture_btn = ttk.Button(
            self.camera_button_frame,
            text="Create Pixel Art",
            command=self.capture_pixel_art
        )
        self.capture_btn.pack(side=tk.LEFT, padx=5)

        self.save_btn = ttk.Button(
            self.camera_button_frame,
            text="Save Image",
            command=lambda: self.save_image(from_camera=True)
        )
        self.save_btn.pack(side=tk.LEFT, padx=5)

        self.toggle_btn = ttk.Button(
            self.camera_button_frame,
            text="Toggle Normal/Pixel",
            command=self.toggle_view
        )
        self.toggle_btn.pack(side=tk.LEFT, padx=5)

    def setup_upload_controls(self):
        # Create control frames
        self.upload_control_frame = ttk.Frame(self.upload_tab)
        self.upload_control_frame.pack(fill=tk.X, padx=10, pady=5)

        # Create pixel size slider for uploads
        ttk.Label(self.upload_control_frame, text="Pixel Size:").pack(side=tk.LEFT)
        self.upload_pixel_size_var = tk.IntVar(value=16)
        self.upload_pixel_slider = ttk.Scale(
            self.upload_control_frame,
            from_=4,
            to=32,
            orient=tk.HORIZONTAL,
            variable=self.upload_pixel_size_var,
            length=200
        )
        self.upload_pixel_slider.pack(side=tk.LEFT, padx=5)

        # Create second row for palette and animation controls
        self.upload_control_frame2 = ttk.Frame(self.upload_tab)
        self.upload_control_frame2.pack(fill=tk.X, padx=10, pady=5)

        # Create color palette dropdown for uploads
        ttk.Label(self.upload_control_frame2, text="Color Palette:").pack(side=tk.LEFT)
        self.upload_palette_var = tk.StringVar(value="ghibli")
        self.upload_palette_dropdown = ttk.Combobox(
            self.upload_control_frame2,
            textvariable=self.upload_palette_var,
            values=["ghibli", "retro", "pastel", "monochrome", "cyberpunk",
                    "vaporwave", "forest", "sunset", "ocean", "candy"]
        )
        self.upload_palette_dropdown.pack(side=tk.LEFT, padx=5)

        # Create animation style dropdown for uploads
        ttk.Label(self.upload_control_frame2, text="Animation:").pack(side=tk.LEFT, padx=(20, 5))
        self.upload_animation_var = tk.StringVar(value="none")
        self.upload_animation_dropdown = ttk.Combobox(
            self.upload_control_frame2,
            textvariable=self.upload_animation_var,
            values=["none", "pulse", "shimmer", "wave", "scan", "glitch"]
        )
        self.upload_animation_dropdown.pack(side=tk.LEFT, padx=5)
        self.upload_animation_dropdown.bind("<<ComboboxSelected>>", self.update_animation)

        # Create buttons for uploads
        self.upload_button_frame = ttk.Frame(self.upload_tab)
        self.upload_button_frame.pack(fill=tk.X, padx=10, pady=10)

        self.browse_btn = ttk.Button(
            self.upload_button_frame,
            text="Browse Image",
            command=self.browse_image
        )
        self.browse_btn.pack(side=tk.LEFT, padx=5)

        self.process_btn = ttk.Button(
            self.upload_button_frame,
            text="Process Image",
            command=self.process_uploaded_image
        )
        self.process_btn.pack(side=tk.LEFT, padx=5)

        self.save_upload_btn = ttk.Button(
            self.upload_button_frame,
            text="Save Processed Image",
            command=lambda: self.save_image(from_camera=False)
        )
        self.save_upload_btn.pack(side=tk.LEFT, padx=5)

    def setup_palettes(self):
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
            ],
            "cyberpunk": [
                [0, 24, 72], [8, 5, 45], [44, 9, 60], [92, 9, 91],
                [0, 135, 170], [194, 37, 92], [255, 56, 56], [255, 117, 255],
                [9, 245, 245], [108, 230, 174], [0, 187, 63]
            ],
            "vaporwave": [
                [44, 25, 59], [96, 42, 104], [176, 38, 255], [228, 0, 128],
                [255, 77, 195], [127, 179, 255], [0, 223, 252], [57, 255, 215]
            ],
            "forest": [
                [11, 61, 12], [21, 84, 28], [42, 126, 25], [68, 157, 68],
                [116, 170, 62], [172, 196, 108], [223, 230, 153], [247, 245, 231]
            ],
            "sunset": [
                [18, 18, 55], [38, 30, 82], [61, 27, 95], [127, 25, 98],
                [183, 45, 96], [232, 77, 91], [246, 137, 83], [255, 196, 108]
            ],
            "ocean": [
                [9, 11, 39], [11, 40, 70], [23, 65, 90], [30, 102, 119],
                [61, 143, 157], [93, 189, 194], [148, 216, 216], [217, 240, 240]
            ],
            "candy": [
                [255, 92, 159], [255, 148, 193], [255, 186, 216], [255, 223, 238],
                [214, 173, 255], [187, 148, 255], [148, 110, 255], [110, 73, 255]
            ]
        }

    def update_animation(self, event=None):
        self.animation_style = self.animation_var.get()
        # Reset animation counter
        self.animation_frame = 0
        self.last_animation_time = time.time()

    def update_frame(self):
        current_tab = self.notebook.index(self.notebook.select())

        if current_tab == 0:  # Camera tab
            ret, frame = self.cap.read()
            if ret:
                # Flip the frame horizontally for a mirror effect
                frame = cv2.flip(frame, 1)
                self.current_frame = frame

                # If pixelated mode is on, show the pixelated version
                if self.pixelated and self.pixel_art_image is not None:
                    display_image = self.apply_animation(self.pixel_art_image.copy(), self.animation_var.get())
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

        elif current_tab == 1:  # Upload tab
            if self.uploaded_image is not None:
                if self.pixelated_upload is not None:
                    display_image = self.apply_animation(self.pixelated_upload.copy(), self.upload_animation_var.get())
                else:
                    display_image = self.uploaded_image

                # Convert to uint8 to ensure compatibility with PIL
                display_image = np.array(display_image, dtype=np.uint8)

                # Convert to PIL Image
                pil_img = Image.fromarray(display_image)
                # Resize the image to fit in the window
                width, height = 640, 480
                pil_img = pil_img.resize((width, height), Image.LANCZOS)
                # Convert to tkinter-compatible photo image
                tk_img = ImageTk.PhotoImage(image=pil_img)

                # Update the upload label
                self.upload_label.config(image=tk_img)
                self.upload_label.image = tk_img  # Keep a reference

        # Call this function again after 20ms (50 FPS)
        self.root.after(20, self.update_frame)

    def apply_animation(self, image, animation_style):
        # Check if we should update animation frame
        current_time = time.time()
        if current_time - self.last_animation_time > 0.05:  # 50ms = 20fps for animation
            self.animation_frame = (self.animation_frame + 1) % 60  # 60 frame cycle
            self.last_animation_time = current_time

        if animation_style == "none":
            return image

        h, w = image.shape[:2]

        if animation_style == "pulse":
            # Pulse effect: Modulate brightness
            factor = 0.2 * np.sin(self.animation_frame * 0.1) + 1.0
            result = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
            return result

        elif animation_style == "shimmer":
            # Shimmer effect: Add random sparkles to some pixels
            mask = np.random.random((h, w)) > 0.97  # 3% of pixels
            shimmer = np.zeros_like(image)
            shimmer[mask] = [255, 255, 255]
            result = np.clip(image.astype(np.float32) + shimmer, 0, 255).astype(np.uint8)
            return result

        elif animation_style == "wave":
            # Wave effect: Displace pixels horizontally
            result = np.zeros_like(image)
            for y in range(h):
                offset = int(10 * np.sin((y / 30) + (self.animation_frame * 0.1)))
                for x in range(w):
                    src_x = (x - offset) % w
                    result[y, x] = image[y, src_x]
            return result

        elif animation_style == "scan":
            # Scanning line effect
            result = image.copy()
            scan_line = int((h * (self.animation_frame % 60)) / 60)
            scan_width = 20
            mask = np.zeros((h, w), dtype=bool)
            mask[max(0, scan_line - scan_width):min(h, scan_line + scan_width), :] = True

            # Brighten the scan line
            result[mask] = np.clip(result[mask].astype(np.float32) * 1.5, 0, 255).astype(np.uint8)
            return result

        elif animation_style == "glitch":
            # Glitch effect: Randomly offset horizontal chunks
            if self.animation_frame % 10 == 0:  # Change glitch every 10 frames
                result = image.copy()
                num_glitches = 3
                for _ in range(num_glitches):
                    # Random position and size
                    y_start = np.random.randint(0, h - 20)
                    height = np.random.randint(10, 50)
                    x_offset = np.random.randint(-20, 20)

                    # Copy and shift the slice
                    slice_height = min(y_start + height, h) - y_start
                    x_start = max(0, x_offset) if x_offset < 0 else 0
                    x_end = w if x_offset < 0 else min(w - x_offset, w)

                    if x_offset < 0:
                        result[y_start:y_start + slice_height, x_start:x_end] = image[
                                                                                y_start:y_start + slice_height,
                                                                                -x_offset:w]
                    else:
                        result[y_start:y_start + slice_height, x_offset:x_end + x_offset] = image[
                                                                                            y_start:y_start + slice_height,
                                                                                            0:x_end]
                return result
            else:
                return image

        return image

    def create_pixel_art(self, frame, pixel_size, palette_name):
        # Resize to create pixelation effect
        height, width = frame.shape[:2]
        small_frame = cv2.resize(frame, (width // pixel_size, height // pixel_size),
                                 interpolation=cv2.INTER_LINEAR)
        # Scale back up using nearest neighbor
        pixel_frame = cv2.resize(small_frame, (width, height),
                                 interpolation=cv2.INTER_NEAREST)

        # Apply color palette
        palette = np.array(self.palettes.get(palette_name, self.palettes["ghibli"]), dtype=np.uint8)

        # Convert to RGB for easier color mapping
        if frame.shape[2] == 3:  # If RGB already
            rgb_pixel_frame = pixel_frame
        else:  # If BGR (from OpenCV)
            rgb_pixel_frame = cv2.cvtColor(pixel_frame, cv2.COLOR_BGR2RGB)

        # Apply color palette (map each pixel to closest palette color)
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
            self.pixel_art_image = self.create_pixel_art(
                self.current_frame,
                self.pixel_size_var.get(),
                self.palette_var.get()
            )
            self.pixelated = True

    def browse_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"), ("All files", "*.*")]
        )
        if file_path:
            try:
                # Read the image
                img = cv2.imread(file_path)
                if img is None:
                    raise Exception("Could not read image")

                # Convert to RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Resize to fit display
                h, w = img_rgb.shape[:2]
                aspect_ratio = w / h

                if aspect_ratio > 640 / 480:  # Width is limiting factor
                    new_w = 640
                    new_h = int(640 / aspect_ratio)
                else:  # Height is limiting factor
                    new_h = 480
                    new_w = int(480 * aspect_ratio)

                img_resized = cv2.resize(img_rgb, (new_w, new_h))

                self.uploaded_image = img_resized
                self.pixelated_upload = None

                # Display the image
                pil_img = Image.fromarray(img_resized)
                tk_img = ImageTk.PhotoImage(image=pil_img)
                self.upload_label.config(image=tk_img)
                self.upload_label.image = tk_img
            except Exception as e:
                error_msg = f"Error loading image: {str(e)}"
                self.upload_label.config(text=error_msg, image='')

    def process_uploaded_image(self):
        if self.uploaded_image is not None:
            self.pixelated_upload = self.create_pixel_art(
                self.uploaded_image,
                self.upload_pixel_size_var.get(),
                self.upload_palette_var.get()
            )

            # Display the pixelated image
            pil_img = Image.fromarray(self.pixelated_upload)
            tk_img = ImageTk.PhotoImage(image=pil_img)
            self.upload_label.config(image=tk_img)
            self.upload_label.image = tk_img

    def toggle_view(self):
        self.pixelated = not self.pixelated

    def save_image(self, from_camera=True):
        if from_camera and self.pixel_art_image is not None:
            img_to_save = self.pixel_art_image
            palette = self.palette_var.get()
            pixel_size = self.pixel_size_var.get()
        elif not from_camera and self.pixelated_upload is not None:
            img_to_save = self.pixelated_upload
            palette = self.upload_palette_var.get()
            pixel_size = self.upload_pixel_size_var.get()
        else:
            return

        # Create default file name
        default_name = f"pixel_art_{palette}_{pixel_size}.png"

        # Ask for file location
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")],
            initialfile=default_name
        )

        if file_path:
            # Convert from RGB to BGR for cv2.imwrite
            save_img = cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR)
            cv2.imwrite(file_path, save_img)

            # Show confirmation
            parent_frame = self.camera_tab if from_camera else self.upload_tab
            save_label = ttk.Label(parent_frame, text=f"Image saved as {os.path.basename(file_path)}")
            save_label.pack()
            self.root.after(2000, save_label.destroy)

    def __del__(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()


if __name__ == "__main__":
    root = tk.Tk()
    app = PixelArtWebcam(root)
    root.mainloop()