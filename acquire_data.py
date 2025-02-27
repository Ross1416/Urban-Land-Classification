import sys
import numpy as np
import openeo
import xarray as xr
import matplotlib.pyplot as plt
import cv2
import math
import requests

from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, QPushButton,
                             QHBoxLayout)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ---------------- Global Variables ----------------
BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
MAX_CLOUD_COVER = 20
MODE = 1  # 0 = Acquire Data, 1 = Display Data

# ---------------- Helper Functions ----------------

def new_coordinates(lat, lon, distance_km, bearing_deg):
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    bearing_rad = math.radians(bearing_deg)

    R = 6371.0

    new_lat_rad = math.asin(math.sin(lat_rad) * math.cos(distance_km / R) +
                            math.cos(lat_rad) * math.sin(distance_km / R) * math.cos(bearing_rad))

    new_lon_rad = lon_rad + math.atan2(math.sin(bearing_rad) * math.sin(distance_km / R) * math.cos(lat_rad),
                                       math.cos(distance_km / R) - math.sin(lat_rad) * math.sin(new_lat_rad))

    new_lat = math.degrees(new_lat_rad)
    new_lon = math.degrees(new_lon_rad)

    return new_lat, new_lon

def postcode_to_area(postcode, height, width):
    url = f'https://api.opencagedata.com/geocode/v1/json?q={postcode}&key=ddc3764c2c2942618d17e11d9f0ed724'
    response = requests.get(url)
    data = response.json()

    if data['status']['code'] == 200:
        location = data['results'][0]['geometry']
    else:
        return None

    north, _ = new_coordinates(location['lat'], location['lng'], height / 2, 0)
    _, east = new_coordinates(location['lat'], location['lng'], width / 2, 90)
    south, _ = new_coordinates(location['lat'], location['lng'], height / 2, 180)
    _, west = new_coordinates(location['lat'], location['lng'], width / 2, 270)

    return north, south, east, west

def normalise_band(band):
    band = band - np.nanmin(band)
    band = band / np.nanmax(band)
    band = np.nan_to_num(band, nan=0)
    return (band * 255).astype(np.uint8)

def download_dataset(north, south, east, west, bands, cloud_cover, save_path, year):
    con = openeo.connect("openeo.dataspace.copernicus.eu")
    con.authenticate_oidc()

    # Generate time ranges for March of each year from 2014 to 2024
    temporal_extent = ["{}-04-01".format(year), "{}-06-28".format(year)]
    print(temporal_extent)

    datacube = con.load_collection(
        "SENTINEL2_L2A",
        spatial_extent={"west": west, "south": south, "east": east, "north": north},
        temporal_extent=temporal_extent,
        bands=bands,
        max_cloud_cover=cloud_cover,
    )

    if save_path is not None:
        datacube.download(save_path)

    return datacube

# ---------------- PyQt GUI ----------------

class SentinelViewer(QWidget):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.num_frames = self.dataset.dims["t"]
        self.dates = np.array(self.dataset["t"].values, dtype="datetime64[D]")  # Convert to readable dates
        self.current_frame = 0

        self.init_ui()
        self.update_display()

    def init_ui(self):
        self.setWindowTitle("Sentinel-2 Image Viewer")
        self.setGeometry(100, 100, 1000, 600)

        # Frame info label (shows frame number + date)
        self.frame_label = QLabel(self)
        self.frame_label.setAlignment(Qt.AlignCenter)

        # QLabel for image display
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(500, 500)

        # Matplotlib canvas for histograms
        self.figure = Figure(figsize=(5, 5))
        self.canvas = FigureCanvas(self.figure)
        self.hist_ax = self.figure.add_subplot(111)

        # Navigation buttons
        self.prev_button = QPushButton("Previous", self)
        self.prev_button.clicked.connect(self.prev_frame)
        self.next_button = QPushButton("Next", self)
        self.next_button.clicked.connect(self.next_frame)

        # Layouts
        top_layout = QVBoxLayout()
        top_layout.addWidget(self.frame_label)

        display_layout = QHBoxLayout()
        display_layout.addWidget(self.image_label)
        display_layout.addWidget(self.canvas)

        nav_layout = QHBoxLayout()
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.next_button)

        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addLayout(display_layout)
        main_layout.addLayout(nav_layout)

        self.setLayout(main_layout)

    def update_display(self):
        if self.current_frame < 0:
            self.current_frame = self.num_frames - 1
        elif self.current_frame >= self.num_frames:
            self.current_frame = 0

        # Get current frame date
        date_str = str(self.dates[self.current_frame])

        # Update frame info label
        self.frame_label.setText(f"Sentinel-2 RGB Image - Frame {self.current_frame + 1}/{self.num_frames} - Date: {date_str}")

        # Extract bands for current frame
        red = self.dataset["B04"].isel(t=self.current_frame).values
        green = self.dataset["B03"].isel(t=self.current_frame).values
        blue = self.dataset["B02"].isel(t=self.current_frame).values

        # Normalise each band to 0-255
        red = normalise_band(red)
        green = normalise_band(green)
        blue = normalise_band(blue)

        # Create RGB image
        rgb_image = np.stack([red, green, blue], axis=-1)

        # Convert numpy image (RGB) into QImage for display
        height, width, _ = rgb_image.shape
        q_image = QImage(rgb_image.data, width, height, 3 * width, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)

        # Compute histograms
        image_gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        hist_gray = cv2.calcHist([image_gray], [0], None, [256], [0, 256])

        # Clear and update histogram plot
        self.hist_ax.clear()
        self.hist_ax.plot(hist_gray, color='black', label="Gray")
        self.hist_ax.set_title(f"Histogram - Frame {self.current_frame + 1}/{self.num_frames} - {date_str}")
        self.hist_ax.set_xlim([-10, 266])
        self.hist_ax.grid(True)
        self.hist_ax.legend()
        self.canvas.draw()

    def next_frame(self):
        self.current_frame += 1
        self.update_display()

    def prev_frame(self):
        self.current_frame -= 1
        self.update_display()

# ---------------- Main Execution ----------------

if __name__ == "__main__":
    location = 'Glasgow'
    width = 3
    height = 3
    years = range(2016, 2018)

    if MODE == 0:
        north, south, east, west = postcode_to_area(location, height, width)
        for year in years:
            saveLocation = f"./data/{location}_{height}x{width}_Year{year}.nc"
            print(saveLocation)
            download_dataset(north, south, east, west, BANDS, MAX_CLOUD_COVER, saveLocation, year)
    else:
        datasets = []
        for year in years:
            file_path =  f"./data/{location}_{height}x{width}_Year{year}.nc"
            try:
                ds = xr.load_dataset(file_path)
                datasets.append(ds)
            except FileNotFoundError:
                print(f"Warning: Data for {year} not found, skipping...")

        if datasets:
            combined_dataset = xr.concat(datasets, dim="t")  # Merge along time axis
            app = QApplication(sys.argv)
            viewer = SentinelViewer(combined_dataset)
            viewer.show()
            sys.exit(app.exec_())
        else:
            print("No valid datasets found. Exiting.")
