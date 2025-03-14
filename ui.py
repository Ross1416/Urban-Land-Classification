from tensorflow import keras
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
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
from UI_workers import *
from time import sleep


class App(QMainWindow):
    def __init__(self, model_path, class_labels, class_colours):
        super().__init__()
        # Set main parameters
        self.setWindowTitle("Urban Land Classification")
        self.setGeometry(100, 100, 1200, 700)

        self.dataset = None
        self.class_map = None

        self.model = keras.models.load_model(model_path)
        self.class_labels = class_labels
        self.class_colours = class_colours

        # Set main layout
        main_layout = QHBoxLayout()
        left_panel = self.create_left_layout()
        centre_panel = self.create_centre_layout()
        right_panel = self.create_right_panel()

        main_layout.addLayout(left_panel, 1)
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

        self.threadpool = QThreadPool.globalInstance()
        print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())

        # Initialize QErrorMessage
        self.error_dialog = QErrorMessage(self)

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

        for i in range(len(self.class_labels)):
            layout.addWidget(QLabel(self.class_labels[i]),i,1)
            pixmap = QPixmap(20,20)
            colour = QColor(self.class_colours[i])
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
        if self.class_map != None:
            self.calculate_distribution()
            self.ax.clear()
            self.ax.barh(self.class_labels, self.percentages,
                         color=self.class_colours)
            self.ax.set_xlabel("Percentage (%)")
            self.ax.set_ylabel("Classes")
            self.ax.set_title("Distribution of Classes")
            self.canvas.draw()

    def calculate_distribution(self):
        self.percentages = np.zeros(len(self.class_labels))

        size = np.array(self.class_map[self.slider.value()]).shape[0]*np.array(self.class_map[self.slider.value()]).shape[1]
        for i in range(len(self.class_map[self.slider.value()])):
            for j in range(len(self.class_map[self.slider.value()][i])):
                self.percentages[int(self.class_map[self.slider.value()][i][j])] += 1/size

    def create_bar_chart(self, layout):
        figure = Figure()
        self.canvas = FigureCanvas(figure)
        self.ax = figure.add_subplot(111)
        self.ax.barh(self.class_labels, np.zeros(len(self.class_labels)),color=self.class_colours)
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

        self.dataset = xr.load_dataset(file_path)
        self.classify()
        self.update_scroll_bar()
        self.update_image()
        self.create_overlay()


    def slider_value_changed(self):
        # print("Slider value changed")
        self.update_image()
        self.create_overlay()
        self.toggle_overlay()
        # self.update_distribution_graph()

    def toggle_overlay(self):
        if self.toggle_overlay_checkbox.isChecked():
            scaled_pixmap = self.overlay_pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
        else:
            scaled_pixmap = self.pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio,Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)

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

    def update_image(self):
        data = self.dataset[["B04", "B03", "B02"]].to_array(dim="bands")
        redArr = data[{"t": self.slider.value()}].values[0,:,:]
        greenArr = data[{"t": self.slider.value()}].values[1, :,:]
        blueArr = data[{"t": self.slider.value()}].values[2, :,:]
        rgb_image = np.dstack([normalise_band(redArr), normalise_band(greenArr), normalise_band(blueArr)])
        height, width, channel = rgb_image.shape
        bytes_per_line = 3 * width
        img = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.pixmap = QPixmap.fromImage(img)
        scaled_pixmap = self.pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

    def create_data_folder(self):
        if not os.path.exists(DATA_PATH):
            os.makedirs(DATA_PATH)
            print(f"Folder '{DATA_PATH}' created.")

    def create_overlay(self):
        """Draws the overlay on the image."""
        self.overlay_pixmap = self.pixmap.copy()

        if self.class_map != None:
            painter = QPainter(self.overlay_pixmap)
            for i in range(len(self.class_map[self.slider.value()])-1):
                for j in range(len(self.class_map[self.slider.value()][i])-1):
                    colour = QColor(self.class_colours[int(self.class_map[self.slider.value()][i][j])])
                    colour.setAlpha(180)
                    painter.fillRect(j * BLOCK_SIZE, i * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, colour)
            painter.end()

    def download_button_clicked(self):
        print("Downloading...")
        self.download_button.setEnabled(False)
        self.download_button.setText("Downloading...")
        print("button changed")
        self.location = self.location_input.text()

        if not self.location:
            print("Please enter a location.")
            return

        self.width = self.width_spinbox.value()
        self.height = self.height_spinbox.value()
        north, south, east, west = postcode_to_area(self.location, self.height, self.width)
        self.start_year = self.start_date_input.date().year()
        self.end_year = self.end_date_input.date().year()
        self.total_downloads = (self.end_year - self.start_year) + 1
        self.download_count = 0

        worker = Worker(download_dataset, self.location, self.width, self.height, north, south, east, west, BANDS, MAX_CLOUD_COVER, self.start_year, self.end_year)  # Any other args, kwargs are passed to the run function
        worker.signals.finished.connect(self.download_finished)
        worker.signals.error.connect(self.download_error)

        # Execute
        self.threadpool.start(worker)

    def download_error(self, err):
        # print("Download error")
        # print(type(err[0]))
        # print(type(err[1]))
        # print(type(err[2]))
        # print(err[0])
        # print(err[1])
        # print(err[2])
        # if err[1] == openeo.rest.OpenEoApiPlainError('Too Many Requests'):
        self.error_dialog.showMessage('Too many requests were made. Wait before reattempting download.')
        self.download_button.setEnabled(True)
        self.download_button.setText("Download")

    def download_finished(self):
        print("Downloads complete")
        # self.download_count +=1

        # if self.download_count >= self.total_downloads:
        # print("All downloads finished")
        self.download_button.setText("Combining datasets...")
        worker = Worker(combine_dataset,self.location, self.height, self.width, self.start_year, self.end_year)
        # worker = Worker(combined_dataset)
        worker.signals.finished.connect(self.combined_dataset_finished)
        worker.signals.error.connect(self.combined_dataset_error)
        self.threadpool.start(worker)

    def combined_dataset_error(self,err):
        print("Combine error")
        print(err)

    def combined_dataset_finished(self):
        self.download_button.setEnabled(True)
        self.download_button.setText("Download")
        self.update_data_selection()
        print("Combined dataset finished")

    def classify(self):
        self.load_button.setText("Classifying..")
        self.load_button.setEnabled(False)

        data = self.dataset[["B04", "B03", "B02"]].to_array(dim="bands")
        # classify(self.model, data, self.class_labels)
        worker = Worker(classify, self.model, data, self.class_labels)
        worker.signals.finished.connect(self.finished_classifing)
        worker.signals.error.connect(self.error_classifing)
        worker.signals.result.connect(self.handle_classification_result)

        self.threadpool.start(worker)

    def finished_classifing(self):
        print("Classification finished")
        self.load_button.setText("Load")
        self.load_button.setEnabled(True)
        self.create_overlay()
        self.toggle_overlay()
        # self.update_distribution_graph()

    def handle_classification_result(self,class_map):
        self.class_map = class_map

    def error_classifing(self):
        print("Classification error")

# ---------------- Global Variables ----------------
BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
MAX_CLOUD_COVER = 30
DATA_PATH = "./data/"
DATA_EXTENSION = ".nc"
BLOCK_SIZE = 1
class_labels = [
        "Annual Crop", "Forest", "Herbaceous Vegetation", "Highway",
        "Industrial", "Pasture", "Permanent Crop", "Residential",
        "River", "Sea/Lake", "Cloud", "Undefined"
    ]

class_colours = ["#E57373", "#64B5F6", "#81C784", "#FFD54F",
    "#BA68C8", "#F06292", "#4DB6AC", "#FF8A65",
    "#DCE775", "#A1887F", "#7986CB", "#FFB74D"]

model_path = "classification/eurosat_model_augmented.keras"

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App(model_path, class_labels, class_colours)
    window.show()
    sys.exit(app.exec())