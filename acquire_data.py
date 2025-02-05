import numpy as np
import openeo
import xarray
import matplotlib.pyplot as plt
import cv2
import math
import requests

def prompt_user_for_next_frame():
    user_input = input("Press Enter to view the next frame, or type 'exit' to stop: ")
    if user_input.lower() == 'exit':
        return False
    return True

def new_coordinates(lat, lon, distance_km, bearing_deg):
    # Convert latitude, longitude, and bearing to radians
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    bearing_rad = math.radians(bearing_deg)

    # Earth's radius in kilometers
    R = 6371.0

    # Compute new latitude
    new_lat_rad = math.asin(math.sin(lat_rad) * math.cos(distance_km / R) +
                            math.cos(lat_rad) * math.sin(distance_km / R) * math.cos(bearing_rad))

    # Compute new longitude
    new_lon_rad = lon_rad + math.atan2(math.sin(bearing_rad) * math.sin(distance_km / R) * math.cos(lat_rad),
                                       math.cos(distance_km / R) - math.sin(lat_rad) * math.sin(new_lat_rad))

    # Convert back to degrees
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
        return

    print("Centre: ", location)

    # Perform calculations based on the coordinates (same as before)
    north, _ = new_coordinates(location['lat'], location['lng'], height / 2, 0)
    _, east = new_coordinates(location['lat'], location['lng'], width / 2, 90)
    south, _ = new_coordinates(location['lat'], location['lng'], height / 2, 180)
    _, west = new_coordinates(location['lat'], location['lng'], width / 2, 270)

    print("north: ", north)
    print("east: ", east)
    print("south: ", south)
    print("west: ", west)

    return north, south, east, west



def normalise_band(band):
    band = band - band.min()
    band = band / band.max()
    return (band * 255).astype(np.uint8)

def interpolate_image(img, spatial_resolution):
    """Decimate Nearest Neighbour input images and then interpolate with nearest neighbour interpolation to 10m^2 spatial resolution."""
    # TODO could optimise to just filer image instead of decimating and re-interpolating
    scale_factor = int(spatial_resolution / 10)
    # Decimate to remove nearest neighbour
    img = img[::scale_factor, ::scale_factor]
    # Interpolate using Bicubic interpolation
    new_size = (int(img.shape[0] * scale_factor), int(img.shape[1] * scale_factor))
    resized_img = cv2.resize(img, new_size, interpolation=cv2.INTER_CUBIC)
    return resized_img

def crop_to_patches(img,patch_size=64):
    """input large greyscale image array, output array of {patch_size}x{patch_size} image patches."""
    x, y = img.shape[0], img.shape[1]
    x//=patch_size
    y//=patch_size
    patches = np.zeros([patch_size,patch_size,x*y])
    for i in range(x):
        for j in range(y):
            patches[:,:,i*j] = img[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
    return patches

def download_dataset(north, south, east, west, bands,cloud_cover,save_path):
    # Connect to dataset
    con = openeo.connect("openeo.dataspace.copernicus.eu")
    con.authenticate_oidc()

    # Load datacube
    datacube = con.load_collection(
        "SENTINEL2_L2A",
        spatial_extent={"west": west, "south": south, "east": east, "north": north},
        temporal_extent = ["2021-02-01", "2021-04-30"],
        bands=bands,
        max_cloud_cover=cloud_cover,
    )

    # Download data cube
    if save_path != None:
        datacube.download(save_path)

    return datacube

BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
MAX_CLOUD_COVER=10
MODE = 1    #0=Acquire Data / 1=Display data

if __name__ == "__main__":
    # Get Dataset
    if MODE == 0:
        north, south, east, west = postcode_to_area('FK17 8FD', 3, 3)
        ds = download_dataset(north, south, east, west, BANDS,MAX_CLOUD_COVER,"./data/data.nc")
    else:
        # Load data
        ds = xarray.load_dataset("./data/data.nc")

        num_frames = ds.dims["t"]  # Number of time steps

        plt.figure(figsize=(8, 8))

        delay = 2

        for t in range(num_frames):
            print(f"Displaying frame {t + 1}/{num_frames}...")

            # Extract RGB bands
            red = ds["B04"].isel(t=t).values
            green = ds["B03"].isel(t=t).values
            blue = ds["B02"].isel(t=t).values

            # Normalize bands
            red = normalise_band(red)
            green = normalise_band(green)
            blue = normalise_band(blue)

            # Stack RGB image
            rgb_image = np.stack([red, green, blue], axis=-1)

            # Display the image
            plt.imshow(rgb_image)
            plt.axis("off")
            plt.title(f"Sentinel-2 RGB Image - Frame {t + 1}/{num_frames}")
            plt.draw()
            plt.pause(0.1)

            # Pause for user input before showing the next frame
            if not prompt_user_for_next_frame():
                break  # Exit the loop if the user types 'exit'

            plt.clf()  # Clear the figure for the next frame

        plt.close()