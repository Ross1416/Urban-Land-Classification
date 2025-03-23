import numpy as np
import openeo
import xarray as xr
import matplotlib.pyplot as plt
import math
import requests
import os

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