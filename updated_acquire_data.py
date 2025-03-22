import sys
from cmath import isnan

import numpy as np
import openeo
import xarray as xr
import matplotlib.pyplot as plt
import math
import requests
import os

# ---------------- Global Variables ----------------
BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09","B11", "B12"]
MAX_CLOUD_COVER = 30

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

def download_dataset(location, width, height, north, south, east, west, bands, cloud_cover, start_year, end_year):
    con = openeo.connect("openeo.dataspace.copernicus.eu")
    con.authenticate_oidc()

    for year in range(start_year, end_year + 1):
        saveLocation = f"./data/{location}_{height}x{width}_{year}.nc"
        print(saveLocation)

        # Generate time ranges for March of each year from 2014 to 2024
        temporal_extent = [f"{year}-04-01", f"{year}-06-28"]
        print(temporal_extent)

        datacube = con.load_collection(
            "SENTINEL2_L2A",
            spatial_extent={"west": west, "south": south, "east": east, "north": north},
            temporal_extent=temporal_extent,
            bands=bands,
            max_cloud_cover=cloud_cover,
        )

        if saveLocation is not None:
            datacube.download(saveLocation)

    return datacube

def combine_dataset(location, height, width, start_year, end_year):
    # Combine Dataset
    datasets = []
    for year in range(start_year, end_year + 1):
        file_path = f"./data/{location}_{height}x{width}_{year}.nc"
        print(file_path)
        try:
            ds = xr.load_dataset(file_path)
            # Convert xarray DataSet to a (bands, t, x, y) DataArray
            # data = ds[["B04", "B03", "B02"]].to_array(dim="bands")
            data = ds[BANDS].to_array(dim="bands")

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
                if hist_values[0] > 30:
                    print("Contained NaN")
                    continue

                # If made it through checks then make representative of year and move on
                datasets.append(ds[{"t": i}])
                print("Picture Accepted")
                break

            # Check if year was added, if not print error

            os.remove(file_path)

        except FileNotFoundError:
            print(f"Missing {year}")

    combined_data = xr.concat(datasets, dim="t")
    file_path = f"./data/{location}_{height}x{width}_{start_year}to{end_year}.nc"
    if file_path is not None:
        combined_data.to_netcdf(file_path)

# ---------------- Main Execution ----------------

if __name__ == "__main__":
    location = 'Arles'
    width = 3
    height = 3
    years = range(2016, 2023+1)

    mode = 1    # 0 = Download Datasets, 1 = Filter to get one image from each dataset

    if mode == 0:
        north, south, east, west = postcode_to_area(location, height, width)
        for year in years:
            saveLocation = f"./data/{location}_{height}x{width}_{year}.nc"
            print(saveLocation)
            download_dataset(north, south, east, west, BANDS, MAX_CLOUD_COVER, saveLocation, year)
    else:
        # Combine Dataset
        datasets = []
        con = openeo.connect("openeo.dataspace.copernicus.eu")
        con.authenticate_oidc()
        for year in years:
            file_path = f"./data/{location}_{height}x{width}_{year}.nc"
            try:
                ds = xr.load_dataset(file_path)
                # Convert xarray DataSet to a (bands, t, x, y) DataArray
                data = ds[["B04", "B03", "B02"]].to_array(dim="bands")


                for i in range(0, data.sizes["t"]):
                    # Create a new figure for each set of images
                    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=90, sharey=True)

                    # Plot RGB image
                    data[{"t": i}].plot.imshow(vmin=0, vmax=2000, ax=axes[0])

                    # Convert to Grayscale
                    weights = xr.DataArray([0.2989, 0.5870, 0.1140], dims=["bands"])
                    grayscale = (data[{"t": i}] * weights).sum(dim="bands")
                    grayscale.plot.imshow(vmin=0, vmax=2000, ax=axes[1])

                    # Determine histogram
                    grayscale_np = grayscale.values.flatten()
                    nanFlag = 0
                    maxVal = 10000
                    grayscale_np = grayscale_np / 10000
                    grayscale_np = grayscale_np * 255
                    hist_values, bin_edges = np.histogram(grayscale_np, bins=255, range=(0, 255))
                    plt.figure(figsize=(6, 4), dpi=90)
                    plt.bar(bin_edges[:-1], hist_values, width=np.diff(bin_edges), color="black", edgecolor="white")
                    plt.title("Histogram")
                    plt.xlabel("Pixel Value")
                    plt.ylabel("Frequency")
                    plt.show()  # Show histogram separately

                    print(data.shape[-2]*data.shape[-1]*0.1)
                    # If histogram has too much white (cloud) don't include
                    if np.sum(hist_values[100:254]) > data.shape[-2]*data.shape[-1]*0.05:
                        print("Too much cloud")
                        continue

                    # If any NaN don't include:
                    if  hist_values[0] > 30:
                        print("Contained NaN")
                        continue
                    print(grayscale_np)

                    # If made it through checks then make representative of year and move on
                    datasets.append(ds[{"t": i}])
                    print("Picture Accepted")
                    break

                # Check if year was added, if not print error

            except FileNotFoundError:
                print(f"Missing {year} ")

        # Check all images appropriate
        fig, axes = plt.subplots(nrows=math.floor(len(datasets) / 2), ncols=math.ceil(len(datasets) / 2), figsize=(8, 3), dpi=90, sharey=True)
        axes = axes.flatten()
        data = []
        for i in range(len(datasets)):
            data = datasets[i][["B04", "B03", "B02"]].to_array(dim="bands")

            data.plot.imshow(vmin=0, vmax=2000, ax=axes[i])
        plt.show()

        combined_data = xr.concat(datasets, dim="t")
        file_path = f"./data/{location}_{height}x{width}_{years[0]}to{years[len(datasets)-1]}.nc"
        if file_path is not None:
            combined_data.to_netcdf(file_path)






