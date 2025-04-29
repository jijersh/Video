import json
import logging
import os
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans

# GUI-specific imports (optional for telephone=False)
try:
    from PIL import Image, ImageTk
    import tkinter as tk
    from tkinter import filedialog, messagebox
    import ttkbootstrap as ttk
    from ttkbootstrap.constants import *
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('media_processing.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Configuration class
@dataclass
class ProcessingConfig:
    """Configuration for media processing."""
    resolution: Tuple[int, int] = (16, 16)
    max_frames: int = 50
    fps: int = 24
    n_colors: int = 8
    cpu_cores: int = os.cpu_count() or 4
    output_path: str = "render_output.build"
    media_type: str = "video"  # 'video' or 'photo'

class MediaProcessor:
    """Handles media processing for video or photo to Roblox animation."""

    def __init__(self, config: ProcessingConfig):
        self.config = config

    def extract_frame(self, args: Tuple[List[int], str, Tuple[int, int]]) -> List[Optional[Tuple[List[int], np.ndarray]]]:
        """Extract and process a batch of video frames."""
        frame_indices, video_path, resolution = args
        results: List[Optional[Tuple[List[int], np.ndarray]]] = []
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            logger.error(f"Failed to open video for frames {frame_indices}")
            return [None] * len(frame_indices)

        try:
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Failed to read frame {idx}")
                    results.append(None)
                    continue

                frame = cv2.resize(frame, resolution)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_flat = frame_rgb.reshape(-1, 3).tolist()

                if not all(0 <= v <= 255 for pixel in frame_flat for v in pixel):
                    logger.error(f"Invalid RGB values in frame {idx}")
                    results.append(None)
                    continue

                results.append((frame_flat, frame_rgb))
            return results
        except Exception as e:
            logger.error(f"Error processing frames {frame_indices}: {e}")
            return [None] * len(frame_indices)
        finally:
            cap.release()

    def process_image(self, image_path: str, resolution: Tuple[int, int]) -> Optional[Tuple[List[int], np.ndarray]]:
        """Process a single image."""
        try:
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Failed to load image: {image_path}")
                return None

            img = cv2.resize(img, resolution)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_flat = img_rgb.reshape(-1, 3).tolist()

            if not all(0 <= v <= 255 for pixel in img_flat for v in pixel):
                logger.error("Invalid RGB values in image")
                return None

            return img_flat, img_rgb
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return None

    @staticmethod
    def apply_rle(frame: List[int]) -> List[Union[str, int]]:
        """Apply Run-Length Encoding to a frame."""
        if not frame:
            return []

        compressed: List[Union[str, int]] = []
        count = 1
        current = frame[0]

        for pixel in frame[1:]:
            if pixel == current and count < 65535:
                count += 1
            else:
                compressed.append(f"{current}x{count}" if count > 1 else current)
                current = pixel
                count = 1

        compressed.append(f"{current}x{count}" if count > 1 else current)
        return compressed

    def extract_dominant_colors(self, sample_frames: List[np.ndarray], n_colors: int) -> Tuple[Dict[str, List[int]], Optional[MiniBatchKMeans]]:
        """Extract dominant colors using MiniBatchKMeans."""
        if not sample_frames:
            logger.warning("No sample frames for color extraction")
            return {str(i): [0, 0, 0] for i in range(n_colors)}, None

        pixels = np.concatenate([frame.reshape(-1, 3) for frame in sample_frames])
        sample_size = min(50000, len(pixels))

        if sample_size < len(pixels):
            indices = np.random.choice(len(pixels), sample_size, replace=False)
            pixels = pixels[indices]

        try:
            kmeans = MiniBatchKMeans(n_clusters=n_colors, random_state=42, batch_size=1000)
            kmeans.fit(pixels)
            colors = kmeans.cluster_centers_.astype(int).clip(0, 255)
            color_map = {str(i): colors[i].tolist() for i in range(n_colors)}
            logger.info(f"Extracted {n_colors} colors: {color_map}")
            return color_map, kmeans
        except Exception as e:
            logger.error(f"Color quantization failed: {e}")
            return {str(i): [0, 0, 0] for i in range(n_colors)}, None

    def quantize_frame(self, frame_data: List[int], kmeans: Optional[MiniBatchKMeans]) -> Optional[List[int]]:
        """Quantize frame to color indices."""
        if not frame_data or kmeans is None:
            logger.warning("Invalid frame data or kmeans model")
            return None

        pixels = np.array(frame_data).reshape(-1, 3)
        labels = kmeans.predict(pixels)
        return labels.tolist()

    def process_media(self, config: ProcessingConfig, media_path: Path) -> int:
        """Process media with parallel processing and RLE compression."""
        frame_data: List[List[int]] = []
        sample_frames: List[np.ndarray] = []
        processed_frames = 0

        if config.media_type == "video":
            cap = cv2.VideoCapture(str(media_path))
            if not cap.isOpened():
                raise ValueError("Could not open video file")

            try:
                video_fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if frame_count <= 0 or video_fps <= 0:
                    raise ValueError("Invalid video: No frames or invalid FPS")

                frames_to_extract = min(config.max_frames, frame_count)
                step = max(1, int(video_fps / config.fps))
                frame_indices = list(range(0, frame_count, step))[:frames_to_extract]
                logger.info(f"Processing {frames_to_extract} frames, step={step}")
            finally:
                cap.release()

            batch_size = max(1, frames_to_extract // config.cpu_cores)
            frame_batches = [frame_indices[i:i + batch_size] for i in range(0, len(frame_indices), batch_size)]
            frame_args = [(batch, str(media_path), config.resolution) for batch in frame_batches]

            with Pool(processes=config.cpu_cores) as pool:
                results = pool.map(self.extract_frame, frame_args)
                for batch_results in results:
                    for frame_flat, frame_rgb in (batch_results or []):
                        if frame_flat is None:
                            logger.warning("Skipping invalid frame")
                            continue

                        frame_data.append(frame_flat)
                        if frame_rgb is not None and processed_frames % 10 == 0:
                            sample_frames.append(frame_rgb)

                        processed_frames += 1
                        print(f"Processed frame {processed_frames}/{frames_to_extract}")
        else:
            result = self.process_image(str(media_path), config.resolution)
            if result is None:
                logger.warning("Failed to process image")
                self._save_empty_output(config)
                return 0

            frame_flat, frame_rgb = result
            frame_data.append(frame_flat)
            sample_frames.append(frame_rgb)
            processed_frames = 1
            print("Image processed")

        if not frame_data:
            logger.warning("No valid frames processed")
            self._save_empty_output(config)
            return 0

        colors, kmeans = self.extract_dominant_colors(sample_frames, config.n_colors)
        quantized_frames = [self.quantize_frame(frame, kmeans) for frame in frame_data if frame]
        compressed_frames = [self.apply_rle(frame) for frame in quantized_frames if frame]

        output_dir = os.path.dirname(config.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_data = {
            "frames": compressed_frames,
            "resolution": config.resolution,
            "num_frames": processed_frames,
            "fps": config.fps,
            "block_type": "PlasticBlock",
            "block_size": [1.0, 1.0, 1.0],
            "colors": colors,
            "compression": "rle"
        }

        with open(config.output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, separators=(',', ':'))

        output_size = os.path.getsize(config.output_path) / 1024
        logger.info(f"Output size: .build={output_size:.1f} KB")
        if output_size > 50:
            print(f"Warning: Output size ({output_size:.1f} KB) exceeds 50 KB")
        return processed_frames

    def _save_empty_output(self, config: ProcessingConfig) -> None:
        """Save an empty output file on failure."""
        output_data = {
            "frames": [],
            "resolution": config.resolution,
            "num_frames": 0,
            "fps": config.fps,
            "block_type": "PlasticBlock",
            "block_size": [1.0, 1.0, 1.0],
            "colors": {str(i): [0, 0, 0] for i in range(config.n_colors)},
            "compression": "none"
        }
        with open(config.output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, separators=(',', ':'))

    def validate_video(self, video_path: str) -> bool:
        """Validate a video file."""
        cap = cv2.VideoCapture(video_path)
        try:
            if not cap.isOpened():
                logger.error("Cannot open video file")
                return False
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            return frame_count > 0 and video_fps > 0
        finally:
            cap.release()

    def validate_image(self, image_path: str) -> bool:
        """Validate an image file."""
        try:
            img = cv2.imread(image_path)
            return img is not None
        except Exception as e:
            logger.error(f"Failed to validate image: {e}")
            return False

    def analyze_video(self, video_path: str) -> Tuple[Optional[int], Optional[float], Optional[int]]:
        """Analyze video for automatic parameter detection."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, None, None

        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            sample_indices = np.linspace(0, frame_count - 1, min(10, frame_count), dtype=int)
            sample_frames = []

            for idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    sample_frames.append(frame_rgb)

            if not sample_frames:
                return None, None, None

            pixels = np.concatenate([frame.reshape(-1, 3) for frame in sample_frames])
            sample_size = min(10000, len(pixels))
            if sample_size < len(pixels):
                indices = np.random.choice(len(pixels), sample_size, replace=False)
                pixels = pixels[indices]

            kmeans = MiniBatchKMeans(n_clusters=8, random_state=42, batch_size=1000)
            kmeans.fit(pixels)
            n_colors = min(8, max(4, len(np.unique(kmeans.labels_))))
            max_frames = min(50, frame_count)
            fps = min(24, video_fps)
            return max_frames, fps, n_colors
        except Exception as e:
            logger.error(f"Video analysis failed: {e}")
            return None, None, None
        finally:
            cap.release()

    def analyze_image(self, image_path: str) -> Optional[int]:
        """Analyze image for automatic parameter detection."""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pixels = img_rgb.reshape(-1, 3)
            sample_size = min(10000, len(pixels))
            if sample_size < len(pixels):
                indices = np.random.choice(len(pixels), sample_size, replace=False)
                pixels = pixels[indices]

            kmeans = MiniBatchKMeans(n_clusters=8, random_state=42, batch_size=1000)
            kmeans.fit(pixels)
            n_colors = min(8, max(4, len(np.unique(kmeans.labels_))))
            return n_colors
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return None

class MediaToBuildGUI:
    """GUI application for converting media to Roblox animation."""

    def __init__(self, root: ttk.Window):
        self.root = root
        self.root.title("Media Converter")
        self.style = ttk.Style(theme='darkly')
        self.config = ProcessingConfig()
        self.media_path: Optional[Path] = None
        self.processor = MediaProcessor(self.config)
        self.mode = tk.StringVar(value="manual")
        self.media_type = tk.StringVar(value="video")
        self.setup_gui()

    def setup_gui(self) -> None:
        """Set up the GUI with a table-like layout."""
        self.root.geometry("900x800")
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Header
        ttk.Label(main_frame, text="Media Converter", font=("Helvetica", 18, "bold")).grid(row=0, column=0, columnspan=3, pady=15)

        # Section 1: Media Selection
        media_frame = ttk.LabelFrame(main_frame, text="1. Select Media", padding=15)
        media_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)

        ttk.Radiobutton(media_frame, text="Video", value="video", variable=self.media_type, command=self.toggle_media_type).grid(row=0, column=0, padx=10)
        ttk.Radiobutton(media_frame, text="Photo", value="photo", variable=self.media_type, command=self.toggle_media_type).grid(row=0, column=1, padx=10)
        ttk.Button(media_frame, text="Choose File", command=self.select_media, style="primary.TButton").grid(row=1, column=0, pady=10)
        self.media_label = ttk.Label(media_frame, text="No media selected", font=("Helvetica", 12))
        self.media_label.grid(row=1, column=1, columnspan=2, sticky=tk.W)

        # Section 2: Parameters
        params_frame = ttk.LabelFrame(main_frame, text="2. Configure Parameters", padding=15)
        params_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)

        ttk.Label(params_frame, text="Resolution (WxH):").grid(row=0, column=0, sticky=tk.W)
        self.width_entry = ttk.Entry(params_frame, width=10)
        self.width_entry.insert(0, str(self.config.resolution[0]))
        self.width_entry.grid(row=0, column=1, padx=5)
        self.height_entry = ttk.Entry(params_frame, width=10)
        self.height_entry.insert(0, str(self.config.resolution[1]))
        self.height_entry.grid(row=0, column=2, padx=5)

        self.mode_frame = ttk.Frame(params_frame)
        self.mode_frame.grid(row=1, column=0, columnspan=3, pady=5)
        ttk.Radiobutton(self.mode_frame, text="Manual", value="manual", variable=self.mode, command=self.toggle_mode).grid(row=0, column=0, padx=10)
        ttk.Radiobutton(self.mode_frame, text="Auto", value="auto", variable=self.mode, command=self.toggle_mode).grid(row=0, column=1, padx=10)

        self.manual_settings = ttk.Frame(params_frame)
        self.manual_settings.grid(row=2, column=0, columnspan=3)
        ttk.Label(self.manual_settings, text="Max Frames:").grid(row=0, column=0, sticky=tk.W)
        self.frames_entry = ttk.Entry(self.manual_settings, width=10)
        self.frames_entry.insert(0, str(self.config.max_frames))
        self.frames_entry.grid(row=0, column=1, padx=5)
        ttk.Label(self.manual_settings, text="FPS:").grid(row=1, column=0, sticky=tk.W)
        self.fps_entry = ttk.Entry(self.manual_settings, width=10)
        self.fps_entry.insert(0, str(self.config.fps))
        self.fps_entry.grid(row=1, column=1, padx=5)

        ttk.Label(params_frame, text="Colors:").grid(row=3, column=0, sticky=tk.W)
        self.colors_entry = ttk.Entry(params_frame, width=10)
        self.colors_entry.insert(0, str(self.config.n_colors))
        self.colors_entry.grid(row=3, column=1, padx=5)

        ttk.Label(params_frame, text="CPU Cores:").grid(row=4, column=0, sticky=tk.W)
        self.cores_entry = ttk.Entry(params_frame, width=10)
        self.cores_entry.insert(0, str(self.config.cpu_cores))
        self.cores_entry.grid(row=4, column=1, padx=5)

        # Section 3: Output File
        output_frame = ttk.LabelFrame(main_frame, text="3. Output File", padding=15)
        output_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)

        ttk.Label(output_frame, text="Save as:").grid(row=0, column=0, sticky=tk.W)
        self.output_entry = ttk.Entry(output_frame, width=40)
        self.output_entry.insert(0, self.config.output_path)
        self.output_entry.grid(row=0, column=1, padx=5)
        ttk.Button(output_frame, text="Browse", command=self.select_output, style="secondary.TButton").grid(row=0, column=2, padx=5)
        self.generate_button = ttk.Button(output_frame, text="Generate", command=self.run_conversion, style="success.TButton")
        self.generate_button.grid(row=1, column=1, pady=10)

        # Progress and Status
        self.progress = ttk.Progressbar(main_frame, length=500, mode="determinate", style="success.Horizontal.TProgressbar")
        self.progress.grid(row=4, column=0, columnspan=3, pady=15)
        self.status_label = ttk.Label(main_frame, text="Ready", font=("Helvetica", 12))
        self.status_label.grid(row=5, column=0, columnspan=3)

        self.toggle_media_type()

    def toggle_media_type(self) -> None:
        """Adjust settings based on media type."""
        is_video = self.media_type.get() == "video"
        state = "normal" if is_video else "disabled"
        for widget in self.mode_frame.winfo_children():
            widget.configure(state=state)
        self.mode.set("manual" if is_video else "auto")
        self.toggle_mode()

    def toggle_mode(self) -> None:
        """Enable/disable manual settings based on mode."""
        is_manual = self.mode.get() == "manual" and self.media_type.get() == "video"
        state = "normal" if is_manual else "disabled"
        for widget in self.manual_settings.winfo_children():
            if isinstance(widget, ttk.Entry):
                widget.configure(state=state)

    def select_media(self) -> None:
        """Select media file."""
        if self.media_type.get() == "video":
            filetypes = [("Video files", "*.mp4 *.avi *.mov *.mkv *.gif")]
            media_path = filedialog.askopenfilename(filetypes=filetypes)
            if media_path and self.processor.validate_video(media_path):
                self.media_path = Path(media_path)
                self.media_label.config(text=f"Selected: {self.media_path.name}")
            else:
                self.media_path = None
                self.media_label.config(text="Invalid video file")
        else:
            filetypes = [("Image files", "*.png *.jpg *.jpeg *.bmp")]
            media_path = filedialog.askopenfilename(filetypes=filetypes)
            if media_path and self.processor.validate_image(media_path):
                self.media_path = Path(media_path)
                self.media_label.config(text=f"Selected: {self.media_path.name}")
            else:
                self.media_path = None
                self.media_label.config(text="Invalid image file")

    def select_output(self) -> None:
        """Select output file path."""
        output = filedialog.asksaveasfilename(
            defaultextension=".build",
            filetypes=[("Build files", "*.build"), ("All files", "*.*")],
            initialfile=self.config.output_path
        )
        if output:
            self.config.output_path = output
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, self.config.output_path)

    def validate_inputs(self) -> Optional[ProcessingConfig]:
        """Validate GUI inputs."""
        try:
            width = int(self.width_entry.get())
            height = int(self.height_entry.get())
            cpu_cores = int(self.cores_entry.get())
            n_colors = int(self.colors_entry.get())

            if width <= 0 or height <= 0:
                raise ValueError("Resolution must be positive")
            if cpu_cores < 1 or cpu_cores > os.cpu_count():
                raise ValueError(f"CPU cores must be between 1 and {os.cpu_count()}")
            if not self.media_path:
                raise ValueError("No media file selected")
            if not self.config.output_path:
                raise ValueError("No output file selected")
            if n_colors < 1:
                raise ValueError("Number of colors must be at least 1")

            config = ProcessingConfig(
                resolution=(width, height),
                cpu_cores=cpu_cores,
                output_path=self.config.output_path,
                n_colors=n_colors,
                media_type=self.media_type.get()
            )

            if config.media_type == "video":
                if self.mode.get() == "manual":
                    config.max_frames = int(self.frames_entry.get())
                    config.fps = int(self.fps_entry.get())
                    if config.max_frames <= 0:
                        raise ValueError("Max frames must be positive")
                    if config.fps <= 0:
                        raise ValueError("FPS must be positive")
                else:
                    max_frames, fps, n_colors = self.processor.analyze_video(str(self.media_path))
                    if max_frames is None:
                        raise ValueError("Failed to analyze video")
                    config.max_frames = max_frames
                    config.fps = fps
                    config.n_colors = n_colors
            else:
                config.max_frames = 1
                config.fps = 1
                if self.mode.get() == "auto":
                    n_colors = self.processor.analyze_image(str(self.media_path))
                    if n_colors is None:
                        raise ValueError("Failed to analyze image")
                    config.n_colors = n_colors

            estimated_size = config.resolution[0] * config.resolution[1] * config.max_frames / 1024
            if estimated_size > 50:
                messagebox.showwarning("Large Output", f"Estimated size ({estimated_size:.1f} KB) exceeds 50 KB")
            return config
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Error: {e}")
            return None

    def run_conversion(self) -> None:
        """Run the conversion process."""
        config = self.validate_inputs()
        if not config:
            return

        self.generate_button.config(state="disabled")
        self.progress["value"] = 0
        self.status_label.config(text="Processing...")

        try:
            num_frames = self.processor.process_media(config, self.media_path)
            logger.info(f"Processed {num_frames} frames, resolution {config.resolution}")
            self.progress["value"] = 100
            self.status_label.config(text=f"Completed: {num_frames} frames processed")
            messagebox.showinfo("Success", f"File saved at {config.output_path}")
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            self.status_label.config(text="Error occurred")
            messagebox.showerror("Error", f"Error: {e}")
        finally:
            self.generate_button.config(state="normal")

class TerminalInterface:
    """Enhanced terminal interface for media conversion."""

    def __init__(self):
        self.processor = MediaProcessor(ProcessingConfig())
        self.config = ProcessingConfig()

    def select_file(self, media_type: str) -> Optional[Path]:
        """Prompt for and validate a media file."""
        extensions = {
            "video": [".mp4", ".avi", ".mov", ".mkv", ".gif"],
            "photo": [".png", ".jpg", ".jpeg", ".bmp"]
        }
        prompt = f"Enter path to {'video' if media_type == 'video' else 'image'} (supported: {', '.join(extensions[media_type])}):"
        print(prompt)

        while True:
            file_path = input("> ").strip()
            if not file_path:
                print("Path cannot be empty")
                continue
            path = Path(file_path)
            if not path.exists():
                print("File does not exist")
                continue
            if path.suffix.lower() not in extensions[media_type]:
                print(f"Invalid format. Supported: {', '.join(extensions[media_type])}")
                continue
            validator = self.processor.validate_video if media_type == "video" else self.processor.validate_image
            if not validator(str(path)):
                print(f"Invalid {'video' if media_type == 'video' else 'image'} file")
                continue
            return path

    def select_output_file(self) -> str:
        """Prompt for and validate an output file path."""
        print("Enter output file path [default: render_output.build]:")
        while True:
            output_path = input("> ").strip() or "render_output.build"
            if not output_path.endswith(".build"):
                output_path += ".build"
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                print(f"Directory {output_dir} does not exist. Create? (y/n)")
                if input("> ").strip().lower() != 'y':
                    print("Choose an existing path or agree to create")
                    continue
                os.makedirs(output_dir)
            return output_path

    def get_int_input(self, prompt: str, default: int, min_val: int, max_val: Optional[int] = None) -> int:
        """Get validated integer input."""
        print(f"{prompt} [default: {default}]:")
        while True:
            try:
                value = input("> ").strip()
                if not value:
                    return default
                value = int(value)
                if value < min_val:
                    print(f"Value must be at least {min_val}")
                    continue
                if max_val is not None and value > max_val:
                    print(f"Value cannot exceed {max_val}")
                    continue
                return value
            except ValueError:
                print("Enter a valid integer")

    def get_resolution(self) -> Tuple[int, int]:
        """Get resolution input."""
        print("Enter resolution (e.g., 16x16) [default: 16x16]:")
        while True:
            try:
                res = input("> ").strip() or "16x16"
                width, height = map(int, res.split('x'))
                if width <= 0 or height <= 0:
                    print("Width and height must be positive")
                    continue
                return width, height
            except ValueError:
                print("Enter in format WxH (e.g., 16x16)")

    def validate_inputs(self, media_path: Path, mode: str) -> Optional[ProcessingConfig]:
        """Collect and validate terminal inputs."""
        try:
            width, height = self.get_resolution()
            cpu_cores = self.get_int_input("CPU cores", self.config.cpu_cores, 1, os.cpu_count())
            n_colors = self.get_int_input("Number of colors", self.config.n_colors, 1)
            output_path = self.select_output_file()

            config = ProcessingConfig(
                resolution=(width, height),
                cpu_cores=cpu_cores,
                output_path=output_path,
                n_colors=n_colors,
                media_type=self.config.media_type
            )

            if config.media_type == "video":
                if mode == "manual":
                    config.max_frames = self.get_int_input("Max frames", self.config.max_frames, 1)
                    config.fps = self.get_int_input("FPS", self.config.fps, 1)
                else:
                    print("Analyzing video for auto settings...")
                    max_frames, fps, n_colors = self.processor.analyze_video(str(media_path))
                    if max_frames is None:
                        raise ValueError("Failed to analyze video")
                    config.max_frames = max_frames
                    config.fps = fps
                    config.n_colors = n_colors
                    print(f"Auto settings: frames={max_frames}, FPS={fps}, colors={n_colors}")
            else:
                config.max_frames = 1
                config.fps = 1
                if mode == "auto":
                    print("Analyzing image for auto settings...")
                    n_colors = self.processor.analyze_image(str(media_path))
                    if n_colors is None:
                        raise ValueError("Failed to analyze image")
                    config.n_colors = n_colors
                    print(f"Auto setting: colors={n_colors}")

            estimated_size = config.resolution[0] * config.resolution[1] * config.max_frames / 1024
            if estimated_size > 50:
                print(f"Warning: Estimated size ({estimated_size:.1f} KB) exceeds 50 KB")
            return config
        except ValueError as e:
            logger.error(f"Input error: {e}")
            print(f"Error: {e}")
            return None

    def run(self) -> None:
        """Run the terminal interface."""
        print("\n=== Media to Roblox Animation Converter (Terminal Mode) ===\n")

        # Media type selection
        print("Select media type:")
        print("1. Video")
        print("2. Photo")
        while True:
            choice = input("> ").strip()
            if choice == '1':
                self.config.media_type = "video"
                break
            elif choice == '2':
                self.config.media_type = "photo"
                break
            print("Enter '1' for video or '2' for photo")

        # File selection
        media_path = self.select_file(self.config.media_type)
        if not media_path:
            print("Error: No media file selected. Exiting.")
            return

        # Mode selection
        mode = "auto"
        if self.config.media_type == "video":
            print("\nSelect processing mode:")
            print("1. Manual (set parameters manually)")
            print("2. Auto (parameters detected automatically)")
            while True:
                choice = input("> ").strip()
                if choice == '1':
                    mode = "manual"
                    break
                elif choice == '2':
                    mode = "auto"
                    break
                print("Enter '1' for manual or '2' for auto")
        else:
            print("\nPhoto mode set to auto")

        # Configuration and processing
        config = self.validate_inputs(media_path, mode)
        if not config:
            print("Error: Configuration failed. Exiting.")
            return

        print("\nStarting media processing...")
        try:
            num_frames = self.processor.process_media(config, media_path)
            logger.info(f"Processed {num_frames} frames, resolution {config.resolution}")
            print(f"\nSuccess: File saved at {config.output_path}")
            print(f"Processed frames: {num_frames}")
        except Exception as e:
            logger.error(f"Processing error: {e}")
            print(f"\nError: {e}")

def main(telephone: bool = False):
    """Entry point to switch between GUI and terminal."""
    if telephone or not GUI_AVAILABLE:
        TerminalInterface().run()
    else:
        root = ttk.Window()
        MediaToBuildGUI(root)
        root.mainloop()

if __name__ == "__main__":
    main(telephone=True)  # Set to False for GUI on PC
