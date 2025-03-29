import tensorflow as tf
import os
import numpy as np
import math
import requests
import openeo
import xarray as xr
import cv2
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("agg")


def normalise_band(band, mean, std):
    band = np.nan_to_num(band, nan=0)
    band = np.clip(band, 0, 2750)
    band /= 2750
    # band = ((band-np.mean(band))/(np.std(band)+ 1e-8))*std+mean
    # band = np.clip(band,0,1)
    return band


def check_cloud(red, green, blue, width, height):
    red = np.clip(red, 0, 2750)
    green = np.clip(green, 0, 2750)
    blue = np.clip(blue, 0, 2750)
    gray = (0.2989 * red + 0.5870 * green + 0.1140 * blue) / 2750 * 255

    hist_values, bin_edges = np.histogram(gray, bins=255, range=(0, 255))

    if np.sum(hist_values[180:254]) > width * height * 0.08:
        print("Cloud")
        return 1

    if np.sum(hist_values[0:12]) > width * height * 0.02:
        print("Shadow")
        return 1

    return 0


def classify(
    model, data, class_labels, normalisation_values, RGB_only=False, stride=64
):
    class_map = []

    # Classify images for each year
    for k in range(data.sizes["t"]):
        # Create matrix for each pixel that keeps track of classes its been labelled
        classScores = np.zeros(
            (
                len(data[{"t": 0}].values[0]) + 1,
                len(data[{"t": 0}].values[0, 0]) + 1,
                len(class_labels),
            )
        )

        # Create new image with what each pixel was classified as
        newIm = np.zeros(
            (
                len(data[{"t": 0}].values[0]) + 1,
                len(data[{"t": 0}].values[0][0]) + 1,
            )
        )

        # Classify 64*64 patches at every stride
        for i in range(0, len(data[{"t": 0}].values[0]), stride):
            for j in range(0, len(data[{"t": 0}].values[0, 0]), stride):
                if i + 64 > len(data[{"t": 0}].values[0]) or j + 64 > len(
                    data[{"t": 0}].values[0, 0]
                ):
                    continue

                bands = data[{"t": k}].values[:, i : i + 64, j : j + 64]

                # Classify patch
                bands_arr = []
                rgb_bands = []
                for x, band in enumerate(bands):
                    mean, std = list(normalisation_values.values())[x]

                    if list(normalisation_values.keys())[x] in RGB_BANDS:
                        rgb_bands.append(band)
                        if RGB_only:
                            bands_arr.append(normalise_band(band, mean, std))

                    # If RGB_only=true, look at only the RGB bands

                    if not(RGB_only):
                        band = normalise_band(band, mean, std)
                        # Interpolate
                        # print("interpolating")
                        size = int(list(BAND_RESOLUTION.values())[x])
                        # if size != 64:
                        step = int(64.0 / size)
                        # print(size, step)
                        img_downsampled = band[::step, ::step]
                        # print(np.array(img_downsampled).shape)

                        x_original = np.linspace(
                            0, 1, size
                        )  # 32 points in original
                        y_original = np.linspace(0, 1, size)
                        x_new = np.linspace(0, 1, 64)  # 64 target points
                        y_new = np.linspace(0, 1, 64)

                        spline = RectBivariateSpline(
                            y_original, x_original, img_downsampled
                        )
                        band = spline(y_new, x_new)
                        bands_arr.append(band)

<<<<<<< HEAD
                rgb_bands.reverse()
                R = rgb_bands[2]  # Red
                G = rgb_bands[1]  # Green
                B = rgb_bands[0]  # Blue
                if (check_cloud(R, G, B, 64, 64)):
                    predicted_class = len(class_labels) - 2
                else:
                    if RGB_only:
                        bands_arr.reverse()
                        cnn_image = np.dstack(bands_arr)
=======
                # if RGB_only:
                #   bands_arr.reverse()
>>>>>>> 5de677f (MS RGB model)

                        cnn_image = np.expand_dims(cnn_image, axis=0)
                        # Predict
                        predictions = model.predict(cnn_image)

                        # Get the predicted class index
                        predicted_class = int(np.argmax(predictions, axis=1)[0])
                    else:
                        cnn_image = np.dstack(bands_arr)

                        cnn_image = np.expand_dims(cnn_image, axis=0)
                        # Predict
                        predictions = model.predict(cnn_image)

                        # Get the predicted class index
                        predicted_class = int(np.argmax(predictions, axis=1)[0])

                # Add to each pixel of patch that it was found to be class x
                for rows in range(i, i + 64):
                    for cols in range(j, j + 64):
                        classScores[rows, cols, predicted_class] += 1

        # Iterate through each pixel and check which class it was assigned to most
        for i in range(0, len(newIm) - 1):
            for j in range(0, len(newIm[0]) - 1):
                foundClass = 0
                currScore = classScores[i, j, 0]
                for l in range(1, len(class_labels) - 1):
                    if currScore < classScores[i, j, l]:
                        foundClass = l
                        currScore = classScores[i, j, l]

                if foundClass == 0 and currScore == 0:
                    foundClass = len(class_labels) - 1

                newIm[i, j] = foundClass

        # print(f"{100 * k / data.sizes["t"]}%")
        class_map.append(newIm)

    return class_map


def new_coordinates(lat, lon, distance_km, bearing_deg):
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    bearing_rad = math.radians(bearing_deg)

    R = 6371.0

    new_lat_rad = math.asin(
        math.sin(lat_rad) * math.cos(distance_km / R)
        + math.cos(lat_rad) * math.sin(distance_km / R) * math.cos(bearing_rad)
    )

    new_lon_rad = lon_rad + math.atan2(
        math.sin(bearing_rad) * math.sin(distance_km / R) * math.cos(lat_rad),
        math.cos(distance_km / R) - math.sin(lat_rad) * math.sin(new_lat_rad),
    )

    new_lat = math.degrees(new_lat_rad)
    new_lon = math.degrees(new_lon_rad)

    return new_lat, new_lon


def postcode_to_area(postcode, height, width):
    url = f"https://api.opencagedata.com/geocode/v1/json?q={postcode}&key=ddc3764c2c2942618d17e11d9f0ed724"
    response = requests.get(url)
    data = response.json()

    if data["status"]["code"] == 200:
        location = data["results"][0]["geometry"]
    else:
        return None

    north, _ = new_coordinates(location["lat"], location["lng"], height / 2, 0)
    _, east = new_coordinates(location["lat"], location["lng"], width / 2, 90)
    south, _ = new_coordinates(
        location["lat"], location["lng"], height / 2, 180
    )
    _, west = new_coordinates(location["lat"], location["lng"], width / 2, 270)

    return north, south, east, west


def download_dataset(
    location,
    width,
    height,
    north,
    south,
    east,
    west,
    bands,
    cloud_cover,
    start_year,
    end_year,
):
    con = openeo.connect("openeo.dataspace.copernicus.eu")
    con.authenticate_oidc()
    datacube = None
    for year in range(start_year, end_year + 1):
        saveLocation = f"./data/{location}_{height}x{width}_{year}.nc"

        if not os.path.exists(saveLocation) and saveLocation is not None:
            # Generate time ranges for March of each year from 2014 to 2024
            temporal_extent = [f"{year}-04-01", f"{year}-06-28"]
            # temporal_extent = [f"{year}-01-01", f"{year}-12-28"]
            print(north,south,east,west)
            datacube = con.load_collection(
                "SENTINEL2_L2A",
                spatial_extent={
                    "west": west,
                    "south": south,
                    "east": east,
                    "north": north,
                },
                temporal_extent=temporal_extent,
                bands=bands,
                max_cloud_cover=cloud_cover,
            )
            datacube.download(saveLocation)
            print(f"Downloaded {saveLocation}")
        else:
            print(f"Year already downloaded: {saveLocation}")
    return datacube


def combine_dataset(location, height, width, start_year, end_year):
    # Combine Dataset
    datasets = []
    for year in range(start_year, end_year + 1):
        print(year)
        file_path = f"./data/{location}_{height}x{width}_{year}.nc"
        try:
            ds = xr.load_dataset(file_path)
            # Convert xarray DataSet to a (bands, t, x, y) DataArray
            data = ds[["B04", "B03", "B02"]].to_array(dim="bands")
            for i in range(0, data.sizes["t"]):
                # Convert to Grayscale
                maxVal = 2750

                # Convert to Grayscale
                weights = xr.DataArray(
                    [0.2989, 0.5870, 0.1140], dims=["bands"]
                )
                grayscale = (data[{"t": i}] * weights).sum(dim="bands")

                # (SAME CODE AS CLOUD FUNCTION JUST INPUT IN DIFFERENT FORMAT AND PLOTTING IMAGES HERE)
                grayscale_np = grayscale.values

                grayscale_np = np.minimum(grayscale_np, maxVal)

                grayscale_np = grayscale_np / maxVal * 255

                hist_values, bin_edges = np.histogram(
                    grayscale_np, bins=256, range=(0, 256)
                )

                # If histogram has too much white (cloud) don't include
                if hist_values[255] > data.shape[-2] * data.shape[-1] * 0.03:
                    print("Too much cloud")
                    continue

                # # If any NaN don't include:
                # if hist_values[0] > 30:
                #     print("Contained NaN")
                #     continue

                # If any in shadow don't include:
                if (
                    np.sum(hist_values[0:5])
                    > data.shape[-2] * data.shape[-1] * 0.01
                ):
                    print("Too dark")
                    continue

                # If made it through checks then make representative of year and move on
                datasets.append(ds[{"t": i}])
                print("Picture Accepted")
                break

        except FileNotFoundError:
            print(f"Missing {year}")

    combined_data = xr.concat(datasets, dim="t")
    file_path = (
        f"./data/{location}_{height}x{width}_{start_year}to{end_year}.nc"
    )
    if file_path is not None:
        combined_data.to_netcdf(file_path)
        print(f"Combined: {file_path}")


# ---------------- Global Variables ---------------- #
ALL_BANDS = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B11",
    "B12",
]
RGB_BANDS = ["B04", "B03", "B02"]
# BAND_NORMALISATION_VALUES = {
#     "B01": (0.0763954,0.013842635),
#     "B02": (0.039900124, 0.011908601),
#     "B03": (0.0372102, 0.014114987),
#     "B04": (0.033805475, 0.021231387),
#     "B05": (0.049949486,0.023617674),
#     "B06": (0.07207413,0.030981975),
#     "B07": (0.084782995,0.038819022),
#     "B08": (0.08217743,0.039935715),
#     "B8A": (0.047593687,0.026250929),
#     "B09": (0.06611737,0.025843639),
#     "B11": (0.050346848,0.034246925),
#     "B12": (0.092849344,0.04398858),
# }

# BAND_NORMALISATION_VALUES = {
#     "B01": (0.27073746,0.048857192),
#     "B02": (0.223406, 0.06586568),
#     "B03": (0.20833384, 0.0781746),
#     "B04": (0.18923364, 0.11786194),
#     "B05": (0.23979506,0.11283984),
#     "B06": (0.40054204,0.1717241),
#     "B07": (0.47429512,0.21576442),
#     "B08": (0.4598104,0.22216458),
#     "B8A": (0.14643246,0.0807013),
#     "B09": (0.066117596,0.025843601),
#     "B11": (0.22353914,0.15123488),
#     "B12": (0.51832534,0.24237852),
# }

BAND_RESOLUTION = {
    "B01": 16,  # 60, ## SHOULD BE 10
    "B02": 64,  # 10,
    "B03": 64,  # 10,
    "B04": 64,  # 10,
    "B05": 32,  # 20,
    "B06": 32,  # 20,
    "B07": 32,  # 20,
    "B08": 64,  # 10,
    "B8A": 32,  # 20,
    "B09": 16,  # 60, ## SHOULD BE 10
    "B11": 32,  # 20,
    "B12": 32,  # 20,
}

# RGB_BAND_NORMALISATION_VALUES = {
#     "B02": (0.4025, 0.1161),
#     "B03": (0.3804, 0.1375),
#     "B04": (0.3398, 0.2037),
# }
BAND_NORMALISATION_VALUES = {
    "B01": (0.0763954, 0.013842635),
    "B02": (0.4025, 0.1161),
    "B03": (0.3804, 0.1375),
    "B04": (0.3398, 0.2037),
    "B05": (0.049949486, 0.023617674),
    "B06": (0.07207413, 0.030981975),
    "B07": (0.084782995, 0.038819022),
    "B08": (0.08217743, 0.039935715),
    "B8A": (0.047593687, 0.026250929),
    "B09": (0.06611737, 0.025843639),
    "B11": (0.050346848, 0.034246925),
    "B12": (0.092849344, 0.04398858),
}


MAX_CLOUD_COVER = 5
if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt
    from tensorflow import keras

    matplotlib.use("qtagg")

    MODEL_PATH = "./Classification/eurosat_model.keras"
    FILE_PATH = "./data/Arles_3x3_2016to2023.nc"

    model = keras.models.load_model(MODEL_PATH)
    dataset = xr.load_dataset(FILE_PATH)

    arr = []

    for i, band in enumerate(ALL_BANDS):
        data = dataset[band]
        data = data[{"t": 0}].values[0:64, 0:64]
        arr.append(data)

    fig, axes = plt.subplots(1, 12, figsize=(12, 9))
    for i, ax in enumerate(axes.flat):
        ax.imshow(arr[i], cmap="gray")  # Use cmap='gray' for grayscale images
        ax.axis("off")  # Remove axes for a cleaner look

    plt.tight_layout()
    plt.show()
