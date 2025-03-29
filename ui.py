from tensorflow import keras

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas




# from fcns_classify import *
# from fcns_preprocess import *
# from fcns_download import *

import xarray as xr
from ui_workers import *
from data import *

import numpy as np
import os

class App(QMainWindow):
    def __init__(self, class_labels, class_colours):
        super().__init__()
        # Set main parameters
        self.setWindowTitle("Urban Land Classification")
        self.setGeometry(100, 100, 1200, 700)
        self.dataset = None
        self.class_map = None

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

        # Update all relevant aspects
        self.create_data_folder()
        self.update_data_selection()
        self.update_model_selection()
        self.create_overlay()
        self.update_distribution_graph()

        self.threadpool = QThreadPool.globalInstance()
        print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())

        # Initialize QErrorMessage
        self.error_dialog = QErrorMessage(self)

    def create_centre_layout(self):
        # Create centre layout with image and slider
        image_container = self.create_image_panel()
        slider_container = self.create_slider_panel()

        layout = QVBoxLayout()
        layout.addWidget(image_container)
        layout.addWidget(slider_container)
        return layout

    def create_left_layout(self):
        # Create left panel with download and data loading information
        downloads_container = self.create_downloads_panel()
        options_container = self.create_selections_panel()

        layout = QVBoxLayout()
        layout.addWidget(downloads_container)
        layout.addWidget(options_container)
        return layout

    def create_right_panel(self):
        # Create the right panel with data legend and class distribution
        layout = QVBoxLayout()
        legend = self.create_legend_panel()
        dist = self.create_distribution_panel()

        layout.addWidget(legend)
        layout.addWidget(dist)
        return layout

    def create_selections_panel(self):
        # Create data loading selections panel
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)

        # Data to load drop down
        selection_label = QLabel("Data:")
        self.selection_dropdown = QComboBox()
        layout.addWidget(selection_label)
        layout.addWidget(self.selection_dropdown)

        # Model to load drop down
        model_label = QLabel("Model:")
        self.model_selection_dropdown = QComboBox()
        layout.addWidget(model_label)
        layout.addWidget(self.model_selection_dropdown)

        # Stride to use, default=64 (no stride)
        stride_label = QLabel("Stride:")
        self.stride_combo = QComboBox()
        self.stride_combo.addItems(["8","16","32","64"])
        self.stride_combo.setCurrentIndex(3)
        layout.addWidget(stride_label)
        layout.addWidget(self.stride_combo)

        # Load button to load data and begin classifying
        self.load_button = QPushButton("Load")
        self.load_button.clicked.connect(self.load_btn_clicked)
        layout.addWidget(self.load_button)

        container = QGroupBox("Selection")
        container.setLayout(layout)
        return container

    def create_downloads_panel(self):
        # Create the downloads panel
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)

        # Location input
        location_label = QLabel("Location:")
        self.location_input = QLineEdit()

        # Date range input (start and end dates)
        start_date_label = QLabel("Start Date:")
        self.start_date_input = QDateEdit()
        self.start_date_input.setDisplayFormat("yyyy")
        self.start_date_input.setDate(QDate.currentDate())  # Default to today's date

        end_date_label = QLabel("End Date:")
        self.end_date_input = QDateEdit()
        self.end_date_input.setDisplayFormat("yyyy")
        self.end_date_input.setDate(QDate.currentDate())  # Default to today's date

        # Size of data to download
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
        layout.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label)
        container = QGroupBox("Image")
        container.setLayout(layout)
        return container

    def create_slider_panel(self):
        # Add lower layout of centre panel
        layout = QVBoxLayout()
        options_layout = QHBoxLayout()
        slider_layout = QHBoxLayout()

        options_layout.setAlignment(Qt.AlignCenter)
        slider_layout.setAlignment(Qt.AlignCenter)

        # Add toggle overlay checkbox
        self.toggle_overlay_checkbox = QCheckBox("Toggle Overlay")
        self.toggle_overlay_checkbox.stateChanged.connect(self.toggle_overlay)
        options_layout.addWidget(self.toggle_overlay_checkbox)

        # Add year select slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setMinimum(0)
        self.slider.setMaximum(10)
        self.slider.valueChanged.connect(self.slider_value_changed)

        # Add date range labels
        self.start_date_label = QLabel("2016")
        self.end_date_label = QLabel("2021")

        # Add all widgets to layout
        slider_layout.addWidget(self.start_date_label)
        slider_layout.addWidget(self.slider)
        slider_layout.addWidget(self.end_date_label)

        layout.addLayout(options_layout)
        layout.addLayout(slider_layout)
        container = QGroupBox("Slider")
        container.setLayout(layout)
        return container

    def create_legend_panel(self):
        # Add legend panel
        layout = QGridLayout()
        layout.setAlignment(Qt.AlignHCenter)

        # For all classes, assign and display a colour box with label
        for i in range(len(self.class_labels)):
            layout.addWidget(QLabel(self.class_labels[i]),i,1)
            pixmap = QPixmap(20,20)
            colour = QColor(self.class_colours[i])
            colour.setAlpha(ALPHA)
            pixmap.fill(colour)
            label = QLabel()
            label.setPixmap(pixmap)
            layout.addWidget(label,i,0)

        container = QGroupBox("Legend")
        container.setLayout(layout)
        return container

    def create_distribution_panel(self):
        # Create panel for class distribution graph to go
        layout = QVBoxLayout()
        self.create_bar_chart(layout)
        container = QGroupBox("Distribution ")
        container.setLayout(layout)
        return container

    def update_distribution_graph(self):
        # Update the class distribution graph
        if self.class_map != None:
            self.calculate_distribution()
            self.ax.clear()
            self.ax.barh(self.class_labels, self.percentages,
                         color=self.class_colours)
            self.ax.set_xlabel("Percentage (%)")
            # self.ax.set_ylabel("Classes")
            self.ax.set_title("Distribution of Classes")
            self.canvas.draw()

    def calculate_distribution(self):
        # Calculate the class distribution in the loaded data
        self.percentages = np.zeros(len(self.class_labels))
        size = (np.array(self.class_map[self.slider.value()]).shape[0]-1)*(np.array(self.class_map[self.slider.value()]).shape[1]-1)
        for i in range(len(self.class_map[self.slider.value()])-1):
            for j in range(len(self.class_map[self.slider.value()][i])-1):
                self.percentages[int(self.class_map[self.slider.value()][i][j])] += 1/size

    def create_bar_chart(self, layout):
        # Create the bar graph for the class distribution
        figure = Figure()
        self.canvas = FigureCanvas(figure)
        self.ax = figure.add_subplot(111)
        self.ax.barh(self.class_labels, np.zeros(len(self.class_labels)),color=self.class_colours)
        self.ax.set_xlabel("Percentage (%)")
        self.ax.set_title("Distribution of Classes")
        figure.tight_layout()
        layout.addWidget(self.canvas)

    def load_btn_clicked(self):
        # Clear overlay and distribution graph
        self.class_map = None
        self.ax.clear()

        # Extract start date from file name
        file_path = DATA_PATH+self.selection_dropdown.currentText()
        idx = file_path.find("_")
        idx = file_path.find("_", idx + 1)
        self.start_year = int(file_path[idx+1:idx+5])

        # Check if data is single or multi year and extract end date from file name
        try:
            self.end_year = int(file_path[idx+7:idx+11])
        except Exception as e:
            print(e)
            self.end_year = self.start_year

        # Load model as selected from the drop down
        self.model = keras.models.load_model(f"{MODEL_PATH}/{self.model_selection_dropdown.currentText()}")
        # Load the data as selected from the dropdown
        self.dataset = xr.load_dataset(file_path)
        # Begin classifying the data
        self.classify()
        # Update scroll bar with start and end year and number of years in dataset
        self.update_scroll_bar()
        # Update the image displayed
        self.update_image()
        # Create overlay
        self.create_overlay()

    def slider_value_changed(self):
        # When year selection slider moved update: displayed image, overlay (check if toggled) and class distribution graph
        self.update_image()
        self.create_overlay()
        self.toggle_overlay()
        self.update_distribution_graph()

    def toggle_overlay(self):
        # If overlay is toggled, display overlay
        if self.toggle_overlay_checkbox.isChecked():
            scaled_pixmap = self.overlay_pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
        else:
            scaled_pixmap = self.pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio,Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)

    def update_data_selection(self):
        # Update data selection with the most recent data files added to the folder
        files = [f for f in os.listdir(DATA_PATH) if f.endswith(DATA_EXTENSION)]
        self.selection_dropdown.clear()
        self.selection_dropdown.addItems(files)

    def update_model_selection(self):
        # Update selection selection with the most recent models added to the folder
        files = [f for f in os.listdir(MODEL_PATH) if f.endswith(MODEL_EXTENSION)]
        self.model_selection_dropdown.clear()
        self.model_selection_dropdown.addItems(files)

    def update_scroll_bar(self):
        # Update year slider with start and end year of loaded data
        self.slider.setMinimum(0)
        self.slider.setMaximum(abs(self.end_year-self.start_year))
        self.start_date_label.setText(str(self.start_year))
        self.end_date_label.setText(str(self.end_year))

    def update_image(self):
        # Extract the RGB bands from the image and normalise
        data = self.dataset[ALL_BANDS].to_array(dim="bands")
        data = data[{"t": self.slider.value()}].values
        rgb_image = []
        for x, band in enumerate(data):
            mean, std = list(BAND_NORMALISATION_VALUES.values())[x]
            if list(BAND_NORMALISATION_VALUES.keys())[x] in RGB_BANDS:
                rgb_image.append(normalise_band(band, mean, std))

        # Convert to RGB format and scale from [0,1] to [0,255]
        rgb_image.reverse()
        rgb_image = np.dstack(rgb_image)
        rgb_image = np.multiply(rgb_image,255).astype("uint8")
        height, width, channel = rgb_image.shape
        bytes_per_line = 3 * width
        # Display image on GUI
        img = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.pixmap = QPixmap.fromImage(img)
        scaled_pixmap = self.pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

    def create_data_folder(self):
        # Create the data folder if it doesn't already exist
        if not os.path.exists(DATA_PATH):
            os.makedirs(DATA_PATH)
            print(f"Folder '{DATA_PATH}' created.")

    def create_overlay(self):
        # Draws the overlay on the image according to the classification
        self.overlay_pixmap = self.pixmap.copy()

        if self.class_map != None:
            painter = QPainter(self.overlay_pixmap)
            for i in range(len(self.class_map[self.slider.value()])-1):
                for j in range(len(self.class_map[self.slider.value()][i])-1):
                    colour = QColor(self.class_colours[int(self.class_map[self.slider.value()][i][j])])
                    colour.setAlpha(ALPHA)
                    painter.fillRect(j * BLOCK_SIZE, i * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, colour)
            painter.end()

    def download_button_clicked(self):
        # Download the dataset
        print("Downloading...")
        # Disable loading button and show user that data is being downloaded
        self.download_button.setEnabled(False)
        self.download_button.setText("Downloading...")
        self.location = self.location_input.text()

        if not self.location:
            print("Please enter a location.")
            return

        # Extract parameters from ui
        self.width = self.width_spinbox.value()
        self.height = self.height_spinbox.value()
        north, south, east, west = postcode_to_area(self.location, self.height, self.width)
        self.start_year = self.start_date_input.date().year()
        self.end_year = self.end_date_input.date().year()
        self.total_downloads = (self.end_year - self.start_year) + 1
        self.download_count = 0

        # Begin asynchronous download of dataset
        worker = Worker(download_dataset, self.location, self.width, self.height, north, south, east, west, ALL_BANDS, MAX_CLOUD_COVER, self.start_year, self.end_year)  # Any other args, kwargs are passed to the run function
        worker.signals.finished.connect(self.download_finished)
        worker.signals.error.connect(self.download_error)
        self.threadpool.start(worker)

    def download_error(self, err):
        # Show when download error
        self.error_dialog.showMessage('Too many requests were made. Wait before reattempting download.')
        self.download_button.setEnabled(True)
        self.download_button.setText("Download")

    def download_finished(self):
        # When individual years are downloaded, begin combining into single dataset
        print("Downloads complete")
        self.download_button.setText("Combining datasets...")
        worker = Worker(combine_dataset,self.location, self.height, self.width, self.start_year, self.end_year)
        worker.signals.finished.connect(self.combined_dataset_finished)
        worker.signals.error.connect(self.combined_dataset_error)
        self.threadpool.start(worker)

    def combined_dataset_error(self,err):
        # If error, show to user
        print("Combine error")
        print(err)
        self.error_dialog.showMessage(err[1])
        self.download_button.setEnabled(True)
        self.download_button.setText("Download")


    def combined_dataset_finished(self):
        # Reset buttons and update dataselection drop down when completed combining
        self.download_button.setEnabled(True)
        self.download_button.setText("Download")
        self.update_data_selection()
        print("Combined dataset finished")

    def classify(self):
        # Begin classifing the selected data
        # Disable button and show to user that classifing
        self.load_button.setText("Classifying..")
        self.load_button.setEnabled(False)

        stride = int(self.stride_combo.currentText())
        data = self.dataset[ALL_BANDS].to_array(dim="bands")

        # Determine if Multispectral classification or RGB
        # and being asynchronous classification
        if "_ms_" in self.model_selection_dropdown.currentText():
            print("MS Classify")
            try:
                worker = Worker(classify, self.model, data, self.class_labels, BAND_NORMALISATION_VALUES, False, stride)
                worker.signals.finished.connect(self.finished_classifing)
                worker.signals.error.connect(self.error_classifing)
                worker.signals.result.connect(self.handle_classification_result)

                self.threadpool.start(worker)
            except:
                self.error_dialog.showMessage("Data download does not contain all 12 spectral bands. Therefore cannot be used with multispectral classification. Use RGB classifier instead")
        else:
            print("RGB Classify")
            worker = Worker(classify, self.model, data, self.class_labels, BAND_NORMALISATION_VALUES, True, stride)
            worker.signals.finished.connect(self.finished_classifing)
            worker.signals.error.connect(self.error_classifing)
            worker.signals.result.connect(self.handle_classification_result)

            self.threadpool.start(worker)

    def finished_classifing(self):
        # When finished classifing, update image overlay and class distribution graph
        print("Classification finished")
        self.load_button.setText("Load")
        self.load_button.setEnabled(True)
        self.create_overlay()
        self.toggle_overlay()
        self.update_distribution_graph()

    def handle_classification_result(self,class_map):
        # Get class map result from classification function
        self.class_map = class_map

    def error_classifing(self, err):
        print("Classification error")
        print(err)
        self.error_dialog.showMessage(err[1])
        self.load_button.setEnabled(True)
        self.load_button.setText("Load")

# ---------------- Global Variables ---------------- #
DATA_PATH = "./data/"
DATA_EXTENSION = ".nc"
MODEL_PATH = "./Classification/"
MODEL_EXTENSION = ".keras"
BLOCK_SIZE = 1
ALPHA = 150

class_labels = [
        "Annual Crop", "Forest", "Herbaceous Vegetation", "Highway",
        "Industrial", "Pasture", "Permanent Crop", "Residential",
        "River", "Sea/Lake", "Cloud", "Undefined"
    ]
class_colours = ["#E57373", "#64B5F6", "#81C784", "#FFD54F",
    "#BA68C8", "#F06292", "#4DB6AC", "#FF8A65",
    "#DCE775", "#A1887F", "#7986CB", "#4A4947"]

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App(class_labels, class_colours)
    window.show()
    sys.exit(app.exec())