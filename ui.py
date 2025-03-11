from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QDate, QObject, QThread, pyqtSignal

from PyQt5.QtGui import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import sys
import os
import numpy as np
from updated_acquire_data import *
from load_data import *
import openeo
import asyncio
import xarray as xr



class DownloadWorker(QObject):
    finished = pyqtSignal()

    def __init__(self, north, south, east, west, bands, cloud_cover, save_path, year):
        super().__init__()
        self.north = north
        self.south = south
        self.east = east
        self.west = west
        self.bands = bands
        self.cloud_cover = cloud_cover
        self.save_path = save_path
        self.year = year

    def run(self):
        con = openeo.connect("openeo.dataspace.copernicus.eu")
        con.authenticate_oidc()

        # Generate time ranges for March of each year from 2014 to 2024
        temporal_extent = [f"{self.year}-04-01", f"{self.year}-06-28"]
        print(temporal_extent)

        datacube = con.load_collection(
            "SENTINEL2_L2A",
            spatial_extent={"west": self.west, "south": self.south, "east": self.east, "north": self.north},
            temporal_extent=temporal_extent,
            bands=self.bands,
            max_cloud_cover=self.cloud_cover,
        )

        if self.save_path is not None:
            datacube.download(self.save_path)

        self.finished.emit()

class CombineDatasetWorker(QObject):
    finished = pyqtSignal()

    def __init__(self, location, height, width, start_year, end_year):
        super().__init__()
        self.location = location
        self.height = height
        self.width = width
        self.start_year = start_year
        self.end_year = end_year

    def run(self):
        # Combine Dataset
        datasets = []
        for year in range(self.start_year, self.end_year + 1):
            file_path = f"./data/{self.location}_{self.height}x{self.width}_{year}.nc"
            print(file_path)
            try:
                ds = xr.load_dataset(file_path)
                # Convert xarray DataSet to a (bands, t, x, y) DataArray
                data = ds[["B04", "B03", "B02"]].to_array(dim="bands")

                for i in range(0, data.sizes["t"]):
                    # Convert to Grayscale
                    weights = xr.DataArray([0.2989, 0.5870, 0.1140], dims=["bands"])
                    grayscale = (data[{"t": i}] * weights).sum(dim="bands")

                    # Determine histogram
                    grayscale_np = grayscale.values.flatten()
                    grayscale_np = grayscale_np / 10000
                    grayscale_np = grayscale_np * 255
                    hist_values, bin_edges = np.histogram(grayscale_np, bins=255, range=(0, 255))

                    # If histogram has too much white (cloud) don't include
                    if np.sum(hist_values[70:254]) > data.shape[-2]*data.shape[-1]*0.05:
                        print("Too much cloud")
                        continue

                    # If any NaN don't include:
                    if  hist_values[0] > 30:
                        print("Contained NaN")
                        continue

                    # If made it through checks then make representative of year and move on
                    datasets.append(ds[{"t": i}])
                    print("Picture Accepted")
                    break

                # Check if year was added, if not print error

                os.remove(file_path)

            except FileNotFoundError:
                print(f"Missing {year} ")

        combined_data = xr.concat(datasets, dim="t")
        file_path = f"./data/{self.location}_{self.height}x{self.width}_{self.start_year}to{self.end_year}.nc"
        if file_path is not None:
            combined_data.to_netcdf(file_path)

        self.finished.emit()

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        # Set main parameters
        self.setWindowTitle("Urban Land Classification")
        self.setGeometry(100, 100, 1200, 700)

        # Test overlay colours
        # self.class_colours = {
        #     "Trees": QColor(0, 255, 0, 100),  # Green (0)
        #     "Farm Land": QColor(255, 255, 0, 100),  # Yellow (1)
        #     "Residential": QColor(255, 0, 0, 100),  # Red (2)
        #     "Industrial": QColor(128, 128, 128, 100),  # Gray (3)
        #     "Water": QColor(0, 0, 255, 100)  # Blue (4)
        # }
        self.dataset = None
        # Test overlay colours
        self.class_colours = {
            "Trees": "Green",  # Green (0)
            "Farm Land": "Yellow",  # Yellow (1)
            "Residential": "Red",  # Red (2)
            "Industrial": "Gray",  # Gray (3)
            "Water": "Blue"  # Blue (4)
        }

        self.class_map = [
            [0, 1, 2, 3],
            [2, 4, 2, 3],
            [0, 2, 4, 3],
        ]

        # Set main layout
        main_layout = QHBoxLayout()
        left_panel = self.create_left_layout()
        centre_panel = self.create_centre_layout()
        right_panel = self.create_right_panel()

        main_layout.addLayout(left_panel, 2)
        main_layout.addLayout(centre_panel, 5)
        main_layout.addLayout(right_panel, 3)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Update
        self.create_data_folder()
        self.update_data_selection()
        self.create_overlay()
        self.update_distribution_graph()

    def create_centre_layout(self):
        image_container = self.create_image_panel()
        slider_container = self.create_slider_panel()

        layout = QVBoxLayout()
        layout.addWidget(image_container)
        layout.addWidget(slider_container)
        return layout

    def create_left_layout(self):
        downloads_container = self.create_downloads_panel()
        options_container = self.create_selections_panel()

        layout = QVBoxLayout()
        layout.addWidget(downloads_container)
        layout.addWidget(options_container)
        return layout

    def create_right_panel(self):
        layout = QVBoxLayout()
        legend = self.create_legend_panel()
        dist = self.create_distribution_panel()

        layout.addWidget(legend)
        layout.addWidget(dist)
        return layout

    def create_selections_panel(self):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)

        self.selection_dropdown = QComboBox()
        self.selection_dropdown.addItems(["Option 1", "Option 2", "Option 3"])
        layout.addWidget(self.selection_dropdown)

        self.load_button = QPushButton("Load")
        self.load_button.clicked.connect(self.load_btn_clicked)
        layout.addWidget(self.load_button)

        container = QGroupBox("Selection")
        container.setLayout(layout)
        return container

    def create_downloads_panel(self):
        # Create the layout
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)

        # Location input
        location_label = QLabel("Location:")
        self.location_input = QLineEdit()

        # Date range input (start and end dates)
        start_date_label = QLabel("Start Date:")
        self.start_date_input = QDateEdit()
        self.start_date_input.setDisplayFormat("yyyy")
        # self.start_date_input.setCalendarPopup(True)
        self.start_date_input.setDate(QDate.currentDate())  # Default to today's date
        #TODO: set minimum date to first year with data

        end_date_label = QLabel("End Date:")
        self.end_date_input = QDateEdit()
        self.end_date_input.setDisplayFormat("yyyy")
        # self.end_date_input.setCalendarPopup(True)
        self.end_date_input.setDate(QDate.currentDate())  # Default to today's date
        # TODO: set minimum date to first year with data

        # Size
        width_label = QLabel("Width (km):")
        self.width_spinbox = QSpinBox()
        self.width_spinbox.setMinimum(1)  # Set minimum value
        self.width_spinbox.setMaximum(10)  # Set maximum value
        self.width_spinbox.setValue(1)  # Set default value

        height_label = QLabel("Height (km):")
        self.height_spinbox = QSpinBox()
        self.height_spinbox.setMinimum(1)  # Set minimum value
        self.height_spinbox.setMaximum(10)  # Set maximum value
        self.height_spinbox.setValue(1)  # Set default value

        # Download button
        self.download_button = QPushButton("Download")
        self.download_button.clicked.connect(self.download_button_clicked)

        # Add widgets to the main layout
        layout.addWidget(location_label)
        layout.addWidget(self.location_input)
        layout.addWidget(start_date_label)
        layout.addWidget(self.start_date_input)
        layout.addWidget(end_date_label)
        layout.addWidget(self.end_date_input)
        layout.addWidget(width_label)
        layout.addWidget(self.width_spinbox)
        layout.addWidget(height_label)
        layout.addWidget(self.height_spinbox)
        layout.addWidget(self.download_button)

        # Set the layout for the container widget
        container = QGroupBox("Downloads")
        container.setLayout(layout)
        return container

    def create_image_panel(self):
        # Create layout and add image
        layout = QVBoxLayout()
        self.image_label = QLabel(self)
        self.pixmap = QPixmap("data/test.jpg")
        self.image_label.setPixmap(self.pixmap)
        layout.addWidget(self.image_label)
        container = QGroupBox("Image")
        container.setLayout(layout)
        return container

    def create_slider_panel(self):
        # Add lower layout
        layout = QVBoxLayout()
        options_layout = QHBoxLayout()
        slider_layout = QHBoxLayout()

        options_layout.setAlignment(Qt.AlignCenter)
        slider_layout.setAlignment(Qt.AlignCenter)

        self.toggle_overlay_checkbox = QCheckBox("Toggle Overlay")
        self.toggle_overlay_checkbox.stateChanged.connect(self.toggle_overlay)
        self.toggle_change_checkbox = QCheckBox("Toggle Change")
        self.toggle_change_checkbox.stateChanged.connect(self.toggle_change)

        options_layout.addWidget(self.toggle_overlay_checkbox)
        options_layout.addWidget(self.toggle_change_checkbox)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setMinimum(0)
        self.slider.setMaximum(10)
        self.slider.valueChanged.connect(self.slider_value_changed)

        self.start_date_label = QLabel("2016")
        self.end_date_label = QLabel("2021")

        slider_layout.addWidget(self.start_date_label)
        slider_layout.addWidget(self.slider)
        slider_layout.addWidget(self.end_date_label)

        layout.addLayout(options_layout)
        layout.addLayout(slider_layout)
        container = QGroupBox("Slider")
        container.setLayout(layout)
        return container

    def create_legend_panel(self):
        layout = QGridLayout()

        for i in range(len(self.class_colours)):
            layout.addWidget(QLabel(list(self.class_colours.keys())[i]),i,1)
            pixmap = QPixmap(64,64)
            colour = QColor(list(self.class_colours.values())[i])
            # colour.setAlpha(100)
            pixmap.fill(colour)
            label = QLabel()
            label.setPixmap(pixmap)
            layout.addWidget(label,i,0)

        container = QGroupBox("Legend")
        container.setLayout(layout)
        return container

    def create_distribution_panel(self):
        layout = QVBoxLayout()
        self.create_bar_chart(layout)
        container = QGroupBox("Distribution ")
        container.setLayout(layout)
        return container

    def update_distribution_graph(self):
        self.calculate_distribution()
        self.ax.clear()
        self.ax.barh(list(self.class_colours.keys()), self.percentages,
                     color=list(self.class_colours.values()))
        self.ax.set_xlabel("Percentage (%)")
        self.ax.set_ylabel("Classes")
        self.ax.set_title("Distribution of Classes")
        self.canvas.draw()

    def calculate_distribution(self):
        self.percentages = np.zeros(len(self.class_colours))
        size = np.array(self.class_map).shape[0]*np.array(self.class_map).shape[1]
        for i in range(len(self.class_map)):
            for j in range(len(self.class_map[i])):
                self.percentages[self.class_map[i][j]] += 1/size

    def get_class_names(self):
        return list(self.class_colours.keys())

    def get_class_values(self):
        return list(self.class_colours.values())

    def create_bar_chart(self, layout):
        figure = Figure()
        self.canvas = FigureCanvas(figure)
        self.ax = figure.add_subplot(111)
        self.ax.barh(list(self.class_colours.keys()), np.zeros(len(self.class_colours)),color=list(self.class_colours.values()))
        self.ax.set_xlabel("Percentage (%)")
        self.ax.set_ylabel("Classes")
        self.ax.set_title("Distribution of Classes")
        layout.addWidget(self.canvas)

    def load_btn_clicked(self):
        # Update image
        print("Load clicked")

        file_path = DATA_PATH+self.selection_dropdown.currentText()
        idx = file_path.find("_")
        idx = file_path.find("_", idx + 1)
        self.start_year = int(file_path[idx+1:idx+5])
        self.end_year = int(file_path[idx+7:idx+11])

        self.create_overlay()
        self.update_scroll_bar()
        # self.update_distribution_graph([30,20,10,80,20])
        self.dataset = xr.load_dataset(file_path)

    def slider_value_changed(self):
        print("Slider value changed")

    def toggle_overlay(self):
        if self.toggle_overlay_checkbox.isChecked():
            self.image_label.setPixmap(self.overlay_pixmap)
        else:
            self.image_label.setPixmap(self.pixmap)

    def toggle_change(self):
        if self.toggle_change_checkbox.isChecked():
            print("Checked toggle change")
        else:
            print("Unchecked toggle change")

    def update_data_selection(self):
        files = [f for f in os.listdir(DATA_PATH) if f.endswith(DATA_EXTENSION)]
        self.selection_dropdown.clear()
        self.selection_dropdown.addItems(files)

    def update_scroll_bar(self):
        self.slider.setMinimum(0)
        self.slider.setMaximum(abs(self.end_year-self.start_year))
        self.start_date_label.setText(str(self.start_year))
        self.end_date_label.setText(str(self.end_year))

    def update_image(self,image):
        self.pixmap = QPixmap(image)
        self.image_label.setPixmap(self.pixmap)

    def create_data_folder(self):
        if not os.path.exists(DATA_PATH):
            os.makedirs(DATA_PATH)
            print(f"Folder '{DATA_PATH}' created.")

    def create_overlay(self):
        """Draws the overlay on the image."""
        self.overlay_pixmap = self.pixmap.copy()
        painter = QPainter(self.overlay_pixmap)

        for i in range(len(self.class_map)):
            for j in range(len(self.class_map[i])):
                colour = QColor(list(self.class_colours.values())[self.class_map[i][j]])
                colour.setAlpha(100)
                painter.fillRect(j * BLOCK_SIZE, i * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, colour)

        painter.end()

    def download_button_clicked(self):
        print("Downloading...")
        self.download_button.setEnabled(False)
        self.download_button.setText("Downloading...")
        print("button changed")
        self.location = self.location_input.text()
        self.width = self.width_spinbox.value()
        self.height = self.height_spinbox.value()
        north, south, east, west = postcode_to_area(self.location, self.height, self.width)
        self.start_year = self.start_date_input.date().year()
        self.end_year = self.end_date_input.date().year()
        self.total_downloads = (self.end_year - self.start_year) + 1
        self.download_count = 0
        for year in range(self.start_year,self.end_year+1):
            saveLocation = f"./data/{self.location}_{self.height}x{self.width}_{year}.nc"
            print(saveLocation)
            self.worker = DownloadWorker(north, south, east, west, BANDS, MAX_CLOUD_COVER, saveLocation, year)
            self.thread = QThread()
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.run)
            self.worker.finished.connect(self.download_finished)
            self.thread.start()

    def download_finished(self):
        self.thread.quit()
        self.thread.wait()
        print("Download complete")
        self.download_count +=1

        if self.download_count >= self.total_downloads:
            print("All downloads finished")
            self.download_button.setText("Combining datasets...")
            self.worker = CombineDatasetWorker(self.location, self.height, self.width, self.start_year, self.end_year)
            self.thread = QThread()
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.run)
            self.worker.finished.connect(self.combined_dataset_finished)
            self.thread.start()

    def combined_dataset_finished(self):
        self.download_button.setEnabled(True)
        self.download_button.setText("Download")
        self.update_data_selection()
        print("Combined dataset finished")

# ---------------- Global Variables ----------------
BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
MAX_CLOUD_COVER = 30
DATA_PATH = "./data/"
DATA_EXTENSION = ".nc"
BLOCK_SIZE = 64
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec())