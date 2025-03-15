from PyQt5.QtCore import *
import traceback
import sys


class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        tuple (exctype, value, traceback.format_exc() )

    result
        object data returned from processing, anything

    '''
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)



class Worker(QRunnable):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''

        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(
                *self.args, **self.kwargs
            )
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done


## OLD

# class DownloadWorker(QRunnable):
#     finished = pyqtSignal()
#
#     def __init__(self, north, south, east, west, bands, cloud_cover, save_path, year):
#         super().__init__()
#         self.north = north
#         self.south = south
#         self.east = east
#         self.west = west
#         self.bands = bands
#         self.cloud_cover = cloud_cover
#         self.save_path = save_path
#         self.year = year
#
#     @pyqtSlot()
#     def run(self):
#         print("Worker running")
#         con = openeo.connect("openeo.dataspace.copernicus.eu")
#         con.authenticate_oidc()
#
#         # Generate time ranges for March of each year from 2014 to 2024
#         temporal_extent = [f"{self.year}-04-01", f"{self.year}-06-28"]
#         print(temporal_extent)
#
#         datacube = con.load_collection(
#             "SENTINEL2_L2A",
#             spatial_extent={"west": self.west, "south": self.south, "east": self.east, "north": self.north},
#             temporal_extent=temporal_extent,
#             bands=self.bands,
#             max_cloud_cover=self.cloud_cover,
#         )
#
#         if self.save_path is not None:
#             datacube.download(self.save_path)
#
#         self.finished.emit()
#
#
# class CombineDatasetWorker(QObject):
#     finished = pyqtSignal()
#
#     def __init__(self, location, height, width, start_year, end_year):
#         super().__init__()
#         self.location = location
#         self.height = height
#         self.width = width
#         self.start_year = start_year
#         self.end_year = end_year
#
#     def run(self):
#         # Combine Dataset
#         datasets = []
#         for year in range(self.start_year, self.end_year + 1):
#             file_path = f"./data/{self.location}_{self.height}x{self.width}_{year}.nc"
#             print(file_path)
#             try:
#                 ds = xr.load_dataset(file_path)
#                 # Convert xarray DataSet to a (bands, t, x, y) DataArray
#                 data = ds[["B04", "B03", "B02"]].to_array(dim="bands")
#
#                 for i in range(0, data.sizes["t"]):
#                     # Convert to Grayscale
#                     weights = xr.DataArray([0.2989, 0.5870, 0.1140], dims=["bands"])
#                     grayscale = (data[{"t": i}] * weights).sum(dim="bands")
#
#                     # Determine histogram
#                     grayscale_np = grayscale.values.flatten()
#                     grayscale_np = grayscale_np / 10000
#                     grayscale_np = grayscale_np * 255
#                     hist_values, bin_edges = np.histogram(grayscale_np, bins=255, range=(0, 255))
#
#                     # If histogram has too much white (cloud) don't include
#                     if np.sum(hist_values[70:254]) > data.shape[-2]*data.shape[-1]*0.05:
#                         print("Too much cloud")
#                         continue
#
#                     # If any NaN don't include:
#                     if  hist_values[0] > 30:
#                         print("Contained NaN")
#                         continue
#
#                     # If made it through checks then make representative of year and move on
#                     datasets.append(ds[{"t": i}])
#                     print("Picture Accepted")
#                     break
#
#                 # Check if year was added, if not print error
#
#                 os.remove(file_path)
#
#             except FileNotFoundError:
#                 print(f"Missing {year} ")
#
#         combined_data = xr.concat(datasets, dim="t")
#         file_path = f"./data/{self.location}_{self.height}x{self.width}_{self.start_year}to{self.end_year}.nc"
#         if file_path is not None:
#             combined_data.to_netcdf(file_path)
#
#         self.finished.emit()