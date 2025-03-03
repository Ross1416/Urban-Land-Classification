
import xarray as xr
import matplotlib.pyplot as plt

import math

BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]

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






