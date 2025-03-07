from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import sys

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        # Set main parameters
        self.setWindowTitle("Urban Land Classification")
        self.setGeometry(100, 100, 1200, 700)

        # Set main layout
        main_layout = QHBoxLayout()
        left_panel = self.create_left_layout()
        centre_panel = self.create_centre_layout()
        right_panel = self.create_right_panel()

        main_layout.addLayout(left_panel, 2)
        main_layout.addLayout(centre_panel, 5)
        main_layout.addLayout(right_panel, 2)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

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

    def create_selections_panel(self):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)

        dropdown = QComboBox()
        dropdown.addItems(["Option 1", "Option 2", "Option 3"])
        layout.addWidget(dropdown)

        button = QPushButton("Load")
        layout.addWidget(button)

        container = QGroupBox("Selection")
        container.setLayout(layout)
        return container

    def create_downloads_panel(self):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)

        dropdown = QComboBox()
        dropdown.addItems(["Option 1", "Option 2", "Option 3"])
        layout.addWidget(dropdown)

        button = QPushButton("Load")
        layout.addWidget(button)

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
        self.toggle_change_checkbox = QCheckBox("Toggle Overlay")

        options_layout.addWidget(self.toggle_overlay_checkbox)
        options_layout.addWidget(self.toggle_change_checkbox)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setMinimum(0)
        self.slider.setMaximum(10)

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

    def create_right_panel(self):
        layout = QVBoxLayout()
        legend = self.create_legend_panel()
        dist = self.create_distribution_panel()

        layout.addWidget(legend)
        layout.addWidget(dist)
        return layout

    def create_legend_panel(self):
        layout = QVBoxLayout()

        container = QGroupBox("Legend")
        container.setLayout(layout)
        return container

    def create_distribution_panel(self):
        layout = QVBoxLayout()
        self.create_bar_chart(layout)
        container = QGroupBox("Distribution ")
        container.setLayout(layout)
        return container

    def create_bar_chart(self, layout):
        figure = Figure()
        self.canvas = FigureCanvas(figure)
        self.ax = figure.add_subplot(111)
        self.ax.bar(["A", "B", "C"], [30, 50, 70])
        layout.addWidget(self.canvas)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec())