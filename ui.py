from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import sys


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Urban Land Classification")
        self.setGeometry(100, 100, 1200, 700)

        main_layout = QHBoxLayout()
        left_panel = self.create_selection_panel()
        right_panel = self.create_right_panel()

        image_layout = QVBoxLayout()
        self.image_label = QLabel("[Image Placeholder]")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(10)

        image_layout.addWidget(self.image_label)
        image_layout.addWidget(self.slider)

        main_layout.addWidget(left_panel, 2)
        main_layout.addLayout(image_layout, 5)
        main_layout.addWidget(right_panel, 2)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def create_selection_panel(self):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)

        dropdown = QComboBox()
        dropdown.addItems(["Option 1", "Option 2", "Option 3"])
        layout.addWidget(dropdown)

        button = QPushButton("Load")
        layout.addWidget(button)

        container = QGroupBox("Selection Panel")
        container.setLayout(layout)
        return container

    def create_right_panel(self):
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Legend"))

        checkbox1 = QCheckBox("Show Overlay A")
        checkbox2 = QCheckBox("Show Overlay B")
        layout.addWidget(checkbox1)
        layout.addWidget(checkbox2)

        self.create_bar_chart(layout)

        container = QGroupBox("Settings & Graph")
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