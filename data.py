import tensorflow as tf
import os
import numpy as np
import math
import requests
import openeo
import xarray as xr
import cv2

def normalise_band(band, mean, std):
    band = np.nan_to_num(band, nan=0)
    band = ((band-np.mean(band))/(np.std(band)+ 1e-8))*std+mean
    band = np.clip(band,0,1)
    return band

def classify(model, data, class_labels, normalisation_values, RGB_only=False, stride=64):
    class_map = []

    # Classify images for each year
    for k in range(data.sizes["t"]):
        # Create matrix for each pixel that keeps track of classes its been labelled
        classScores = np.zeros((
            len(data[{"t": 0}].values[0]) + 1,
            len(data[{"t": 0}].values[0, 0]) + 1,
            len(class_labels)
        ))

        # Create new image with what each pixel was classified as
        newIm = np.zeros((
            len(data[{"t": 0}].values[0]) + 1,
            len(data[{"t": 0}].values[0][0]) + 1
        ))

        # Classify 64*64 patches at every stride
        for i in range(0, len(data[{"t": 0}].values[0]), stride):
            for j in range(0, len(data[{"t": 0}].values[0, 0]), stride):
                if i + 64 > len(data[{"t": 0}].values[0]) or j + 64 > len(data[{"t": 0}].values[0, 0]):
                    continue

                bands = data[{"t": k}].values[:, i:i + 64, j:j + 64]

                # # Double check not too much cloud cover before NN
                # if check_cloud(redArr, greenArr, blueArr, 64, 64):
                #     # Too much cloud
                #     predicted_class = len(class_labels) - 2
                # else:

                # Classify patch
                bands_arr = []
                for x, band in enumerate(bands):
                    mean, std = list(normalisation_values.values())[x]

                    # If RGB_only=true, look at only the RGB bands

                    if RGB_only:
                        if list(normalisation_values.keys())[x] in RGB_BANDS:
                            bands_arr.append(normalise_band(band, mean, std))
                            bands_arr.reverse()
                    else:
                        bands_arr.append(normalise_band(band, mean, std))

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

def download_dataset(location, width, height, north, south, east, west, bands, cloud_cover, start_year, end_year):
    con = openeo.connect("openeo.dataspace.copernicus.eu")
    con.authenticate_oidc()

    for year in range(start_year, end_year + 1):
        saveLocation = f"./data/{location}_{height}x{width}_{year}.nc"

        if not os.path.exists(saveLocation) and saveLocation is not None:
            # Generate time ranges for March of each year from 2014 to 2024
            temporal_extent = [f"{year}-04-01", f"{year}-06-28"]

            datacube = con.load_collection(
                "SENTINEL2_L2A",
                spatial_extent={"west": west, "south": south, "east": east, "north": north},
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
        file_path = f"./data/{location}_{height}x{width}_{year}.nc"
        try:
            ds = xr.load_dataset(file_path)
            # Convert xarray DataSet to a (bands, t, x, y) DataArray
            data = ds[["B04", "B03", "B02"]].to_array(dim="bands")
            for i in range(0, data.sizes["t"]):
                # Convert to Grayscale
                # TODO: check this works for all bands correctly
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
                if hist_values[0] > 30:
                    print("Contained NaN")
                    continue

                # If made it through checks then make representative of year and move on
                datasets.append(ds[{"t": i}])
                print("Picture Accepted")
                break

            # Check if year was added, if not print error

        except FileNotFoundError:
            print(f"Missing {year}")

    combined_data = xr.concat(datasets, dim="t")
    file_path = f"./data/{location}_{height}x{width}_{start_year}to{end_year}.nc"
    if file_path is not None:
        combined_data.to_netcdf(file_path)
        print(f"Combined: {file_path}")

# ---------------- Global Variables ---------------- #
ALL_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09","B11", "B12"]
RGB_BANDS = ["B04", "B03", "B02"]
BAND_NORMALISATION_VALUES = {
    "B01": (0,0),
    "B02": (0.4025, 0.1161),
    "B03": (0.3804, 0.1375),
    "B04": (0.3398, 0.2037),
    "B05": (0,0),
    "B06": (0,0),
    "B07": (0,0),
    "B08": (0,0),
    "B8A": (0,0),
    "B09": (0,0),
    "B11": (0,0),
    "B12": (0,0),
}



MAX_CLOUD_COVER = 30
if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt
    from tensorflow import keras

    matplotlib.use('qtagg')


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
        ax.imshow(arr[i], cmap='gray')  # Use cmap='gray' for grayscale images
        ax.axis('off')  # Remove axes for a cleaner look

    plt.tight_layout()
    plt.show()
