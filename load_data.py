
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

import math

BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]

def normalise_band(band):
    band = band - np.nanmin(band)
    # band = np.clip(band, a_min=0, a_max=5000)
    band = band / np.nanmax(band)
    # band = band/5000
    band = np.nan_to_num(band, nan=0)
    return (band * 255).astype(np.uint8)

if __name__ == "__main__":
    file_path = 'data/Glasgow_3x3_2016to2020.nc'

    # Combine Dataset
    dataset = xr.load_dataset(file_path)

    # Check all images appropriate
    data = dataset[["B04", "B03", "B02"]].to_array(dim="bands")
    fig, axes = plt.subplots(nrows=math.floor(data.sizes["t"] / 2), ncols=math.ceil(data.sizes["t"] / 2), figsize=(8, 3), dpi=90, sharey=True)
    axes = axes.flatten()
    for i in range(data.sizes["t"]):
        data[{"t": i}].plot.imshow(vmin=0, vmax=2000, ax=axes[i])

    plt.show()

    # Split an image into 64*64 segments (All set for image 0 can be put in a loop and change 0s to loop var

    print(f"Red Values: {data[{"t": 0}].values[0]}")
    print(f"Rows: {len(data[{"t": 0}].values[0])}")
    print(f"Columns: {len(data[{"t": 0}].values[0,0])}")

    fig, axes = plt.subplots(nrows=math.floor(len(data[{"t": 0}].values[0]) / 64), ncols=math.floor(len(data[{"t": 0}].values[0,0]) / 64),
                             figsize=(8, 3), dpi=90, sharex=True, sharey=True)
    axes = axes.flatten()
    idImage = 0
    for i in range(0,len(data[{"t": 0}].values[0]), 64):
        for j in range(0, len(data[{"t": 0}].values[0,0]), 64):
            if i+64 > len(data[{"t": 0}].values[0]) or j+64 > len(data[{"t": 0}].values[0,0]):
                print("\nData out of range")
                continue

            redArr = data[{"t": 0}].values[0, i:i+64, j:j+64]
            greenArr = data[{"t": 0}].values[0, i:i + 64, j:j + 64]
            blueArr = data[{"t": 0}].values[0, i:i + 64, j:j + 64]

            # For testing display cropped image
            rgb_image = np.dstack([normalise_band(redArr), normalise_band(greenArr), normalise_band(blueArr)])
            axes[idImage].imshow(rgb_image)
            axes[idImage].set_xticks([])
            axes[idImage].set_yticks([])
            axes[idImage].set_frame_on(False)
            idImage = idImage + 1
            print(f"\nStarting position ({j},{i})")
            print(f"Rows taken: {len(redArr)}")
            print(f"Columns taken: {len(redArr[0])}")

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()






