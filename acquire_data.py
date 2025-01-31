import numpy as np
import openeo
import xarray
import matplotlib.pyplot as plt
import cv2

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

def download_dataset(BANDS,cloud_cover=10,save_path=None):
    # Connect to dataset
    con = openeo.connect("openeo.dataspace.copernicus.eu")
    con.authenticate_oidc()

    # Load datacube
    datacube = con.load_collection(
        "SENTINEL2_L2A",
        spatial_extent={"west": -4.27, "south": 55.83, "east": -4.20, "north": 55.87},
        temporal_extent = ["2021-02-01", "2021-04-30"],
        bands=BANDS,
        max_cloud_cover=cloud_cover,
    )

    # Download data cube
    if save_path != None:
        datacube.download(save_path)

    return datacube

BANDS=["BO1","BO2","BO3","B04","B05","B06","B07","B08","B08A","B09","B010","B11","B12"]
MAX_CLOUD_COVER=10

if __name__ == "__main__":
    # Get Dataset
    ds = download_dataset(BANDS,MAX_CLOUD_COVER,"./data/data.nc")

    # Load data
    ds = xarray.load_dataset("./data/data.nc")

    # Separate out bands - not necessary
    r=ds[["B04"]].to_array(dim="bands")[{"t": -1}]
    a=ds[["B01"]].to_array(dim="bands")[{"t": -1}]

    # Interpolate all bands to 10m^2 spatial Resolution
    
    # Crop all images to 64x64 patches

    # Download single tiff image
    # datacube.band("B04").max_time().download("./data/image.tiff")