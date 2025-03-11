
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

import math

from tensorflow import keras

BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]

def normalise_band_for_CNN(band):
    band = band / 10000
    return band

def normalise_band(band):
    band = band / 2000
    band = np.nan_to_num(band, nan=0)
    band = (band * 255)
    band[band > 255] = 255
    return band.astype(np.uint8)

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

    model = keras.models.load_model('classification/eurosat_model.keras')
    class_labels = [
        "Annual Crop", "Forest", "Herbaceous Vegetation", "Highway",
        "Industrial", "Pasture", "Permanent Crop", "Residential",
        "River", "Sea/Lake"
    ]

    for i in range(0,len(data[{"t": 0}].values[0]), 64):
        for j in range(0, len(data[{"t": 0}].values[0,0]), 64):
            if i+64 > len(data[{"t": 0}].values[0]) or j+64 > len(data[{"t": 0}].values[0,0]):
                print("\nData out of range")
                continue

            redArr = data[{"t": 1}].values[0, i:i+64, j:j+64]
            greenArr = data[{"t": 1}].values[1, i:i + 64, j:j + 64]
            blueArr = data[{"t": 1}].values[2, i:i + 64, j:j + 64]

            cnn_image = np.dstack([normalise_band_for_CNN(redArr),
                                   normalise_band_for_CNN(greenArr),
                                   normalise_band_for_CNN(blueArr)])
            cnn_image = np.expand_dims(cnn_image, axis=0)
            print(f"{len(cnn_image)},{len(cnn_image[0])},{len(cnn_image[0,0])}")

            # Predict
            predictions = model.predict(cnn_image)

            # Get the predicted class index
            predicted_class = np.argmax(predictions, axis=1)[0]
            predicted_label = class_labels[predicted_class]

            # For testing display cropped image
            rgb_image = np.dstack([normalise_band(redArr), normalise_band(greenArr), normalise_band(blueArr)])
            axes[idImage].imshow(rgb_image)
            axes[idImage].set_xticks([])
            axes[idImage].set_yticks([])
            axes[idImage].set_frame_on(False)
            axes[idImage].set_title(predicted_label, fontsize=8)
            idImage = idImage + 1
            print(f"\nStarting position ({j},{i})")
            print(f"Rows taken: {len(redArr)}")
            print(f"Columns taken: {len(redArr[0])}")

    #plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()






