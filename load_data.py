import cv2
from tensorflow import keras
import matplotlib
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
matplotlib.use('qtagg')
import math



import matplotlib.colors as mcolors

BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09","B11", "B12"]

def normalise_band_for_CNN(band, mean, std):
    band = np.nan_to_num(band, nan=0)
    band = ((band-np.mean(band))/np.std(band)+ 1e-8)*std+mean
    band = np.clip(band,0,1)
    return band


def normalise_band(band):
    band = band / 2000
    band = np.nan_to_num(band, nan=0)
    band = (band * 255)
    band[band > 255] = 255
    return band.astype(np.uint8)

def check_cloud(red, green, blue, width, height):
    r = normalise_band(red)
    g = normalise_band(green)
    b = normalise_band(blue)

    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    hist_values, bin_edges = np.histogram(gray, bins=255, range=(0, 255))

    if np.sum(hist_values[230:254]) > width * height * 0.05:
        return 1

    return 0

def classify(model, data, class_labels, stride):
    class_map = []

    # Classify images for each year
    for k in range(data.sizes["t"]):
        strideRows = stride
        strideCols = stride

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
        for i in range(0, len(data[{"t": 0}].values[0]), strideRows):
            for j in range(0, len(data[{"t": 0}].values[0, 0]), strideCols):
                if i + 64 > len(data[{"t": 0}].values[0]) or j + 64 > len(data[{"t": 0}].values[0, 0]):
                    continue

                # Get RGB
                redArr = data[{"t": k}].values[0, i:i + 64, j:j + 64]
                greenArr = data[{"t": k}].values[1, i:i + 64, j:j + 64]
                blueArr = data[{"t": k}].values[2, i:i + 64, j:j + 64]

                # # Double check not too much cloud cover before NN
                # if check_cloud(redArr, greenArr, blueArr, 64, 64):
                #     # Too much cloud
                #     predicted_class = len(class_labels) - 2
                # else:
                # Classify patch
                cnn_image = np.dstack([normalise_band_for_CNN(redArr, 0.3398, 0.2037),
                                       normalise_band_for_CNN(greenArr, 0.3804, 0.1375),
                                       normalise_band_for_CNN(blueArr,0.4025, 0.1161)])
                print(cnn_image.shape)
                cnn_image = np.expand_dims(cnn_image, axis=0)
                print(cnn_image.shape)
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
                    foundClass = len(class_labels)-1

                newIm[i, j] = foundClass


        print(f"{100*k/data.sizes["t"]}%")
        class_map.append(newIm)
    return class_map

def classify_ms(model, data, class_labels, stride):
    class_map = []

    # Classify images for each year
    for k in range(data.sizes["t"]):
        strideRows = stride
        strideCols = stride

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
        for i in range(0, len(data[{"t": 0}].values[0]), strideRows):
            for j in range(0, len(data[{"t": 0}].values[0, 0]), strideCols):
                if i + 64 > len(data[{"t": 0}].values[0]) or j + 64 > len(data[{"t": 0}].values[0, 0]):
                    continue

                # Get RGB

                bands = data[{"t": k}].values[:, i:i + 64, j:j + 64]
                # print("classify ms")
                # print(bands.shape)
                # redArr = data[{"t": k}].values[0, i:i + 64, j:j + 64]
                # print(redArr.shape)
                # greenArr = data[{"t": k}].values[1, i:i + 64, j:j + 64]
                # blueArr = data[{"t": k}].values[2, i:i + 64, j:j + 64]

                # # Doubl   e check not too much cloud cover before NN
                # if check_cloud(redArr, greenArr, blueArr, 64, 64):
                #     # Too much cloud
                #     predicted_class = len(class_labels) - 2
                # else:
                # Classify patch
                bands_arr = []
                [bands_arr.append(normalise_band_for_CNN(x)) for x in bands]
                cnn_image = np.dstack(bands_arr)
                print(cnn_image.shape)
                cnn_image = np.expand_dims(cnn_image, axis=0)
                print(cnn_image)
                print(cnn_image.shape)

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
                    foundClass = len(class_labels)-1

                newIm[i, j] = foundClass


        print(f"{100*k/data.sizes["t"]}%")
        class_map.append(newIm)
    return class_map

def classify_basic(model, data, class_labels):
    class_map = []
    num_rows = math.floor(data[{"t": 0}].shape[1]/64)
    num_cols = math.floor(data[{"t": 0}].shape[2]/64)

    print(num_rows, num_cols)

    for k in range(data.sizes["t"]):
        # idImage = 0
        class_map_year = np.zeros((num_rows, num_cols))
        for i in range(0,len(data[{"t": 0}].values[0]), 64):
            for j in range(0, len(data[{"t": 0}].values[0,0]), 64):
                if i+64 > len(data[{"t": 0}].values[0]) or j+64 > len(data[{"t": 0}].values[0,0]):
                    print("\nData out of range")
                    continue

                redArr = data[{"t": k}].values[0, i:i+64, j:j+64]
                greenArr = data[{"t": k}].values[1, i:i + 64, j:j + 64]
                blueArr = data[{"t": k}].values[2, i:i + 64, j:j + 64]

                cnn_image = np.dstack([normalise_band_for_CNN(redArr),
                                       normalise_band_for_CNN(greenArr),
                                       normalise_band_for_CNN(blueArr)])
                cnn_image = np.expand_dims(cnn_image, axis=0)

                # Predict
                predictions = model.predict(cnn_image)

                # Get the predicted class index
                predicted_class = np.argmax(predictions, axis=1)[0]
                class_map_year[int(i/64)][int(j/64)] = int(predicted_class)
                # predicted_label = class_labels[predicted_class]

                # For testing display cropped image
                # rgb_image = np.dstack([normalise_band(redArr), normalise_band(greenArr), normalise_band(blueArr)])
                # axes[idImage].imshow(rgb_image)
                # axes[idImage].set_xticks([])
                # axes[idImage].set_yticks([])
                # axes[idImage].set_frame_on(False)
                # axes[idImage].set_title(predicted_label, fontsize=8)
                # idImage = idImage + 1

        # plt.subplots_adjust(wspace=0, hspace=0)
        # plt.show()

        class_map.append(class_map_year)
    return class_map


if __name__ == "__main__":
    # file_path = 'data/Stepps_3x3_2016to2021.nc'
    file_path = 'data/Arles_3x3_2016to2023.nc'

    # Combine Dataset
    dataset = xr.load_dataset(file_path)

    # Check all images appropriate
    data = dataset[["B04", "B03", "B02"]].to_array(dim="bands")
    fig, axes = plt.subplots(nrows=math.floor(data.sizes["t"] / 2), ncols=math.ceil(data.sizes["t"] / 2), figsize=(8, 3), dpi=90, sharey=True)
    axes = axes.flatten()
    for i in range(data.sizes["t"]):
        data[{"t": i}].plot.imshow(vmin=0, vmax=2000, ax=axes[i])

    plt.show()

    model = keras.models.load_model('classification/eurosat_model_augmented.keras')
    class_labels = [
        "Annual Crop", "Forest", "Herbaceous Vegetation", "Highway",
        "Industrial", "Pasture", "Permanent Crop", "Residential",
        "River", "Sea/Lake", "Cloud", "Undefined"
    ]

    for k in range(data.sizes["t"]):
        fig, axes = plt.subplots(nrows=1,
                                 ncols=2)
        axes = axes.flatten()
        idImage = 0
        strideRows = 16
        strideCols = 16
        # strideRows = round(len(data[{"t": 0}].values[0])/64)
        # strideCols = round(len(data[{"t": 0}].values[0]) / 64)
        classScores = np.zeros((
            len(data[{"t": 0}].values[0])+1,
            len(data[{"t": 0}].values[0,0])+1,
            len(class_labels)
        ))

        newIm = np.zeros((
            len(data[{"t": 0}].values[0])+1,
            len(data[{"t": 0}].values[0][0])+1
        ))

        for i in range(0,len(data[{"t": 0}].values[0]), strideRows):
            for j in range(0, len(data[{"t": 0}].values[0,0]), strideCols):
                if i+64 > len(data[{"t": 0}].values[0]) or j+64 > len(data[{"t": 0}].values[0,0]):
                    print("\nData out of range")
                    continue

                redArr = data[{"t": k}].values[0, i:i+64, j:j+64]
                greenArr = data[{"t": k}].values[1, i:i + 64, j:j + 64]
                blueArr = data[{"t": k}].values[2, i:i + 64, j:j + 64]

                if check_cloud(redArr,greenArr,blueArr,64,64):
                    # Too much cloud
                    predicted_class = len(class_labels)-2
                else:
                    # cnn_image = np.dstack([normalise_band_for_CNN(redArr),
                    #                        normalise_band_for_CNN(greenArr),
                    #                        normalise_band_for_CNN(blueArr)])
                    cnn_image = np.dstack([normalise_band_for_CNN(redArr, 0.3398, 0.2037),
                     normalise_band_for_CNN(greenArr, 0.3804, 0.1375),
                     normalise_band_for_CNN(blueArr, 0.4025, 0.1161)])

                    cnn_image = np.expand_dims(cnn_image, axis=0)

                    print(cnn_image.shape)

                    # Predict
                    predictions = model.predict(cnn_image)

                    # Get the predicted class index
                    predicted_class = np.argmax(predictions, axis=1)[0]

                for rows in range(i,i+64):
                    for cols in range(j,j+64):
                        classScores[rows,cols,predicted_class] += 1

        print(len(newIm[0])-1)
        for i in range(0, len(newIm)-1):
            for j in range(0, len(newIm[0])-1):
                foundClass = 0
                currScore = classScores[i,j,0]
                for l in range(1, len(class_labels)-1):
                    if currScore < classScores[i,j,l]:
                        foundClass = l
                        currScore = classScores[i, j, l]

                if foundClass == 0 and currScore == 0:
                    foundClass = len(class_labels)

                newIm[i,j] = foundClass

        data[{"t": k}].plot.imshow(vmin=0, vmax=2000, ax=axes[0])
        axes[1].imshow(newIm, cmap="viridis")  # or cmap="viridis"

        # Define a colormap with distinct colors for each class
        num_classes = len(class_labels)
        colors = plt.cm.get_cmap("tab20", num_classes)  # Use 'tab11' for 11 distinct colors

        # Create a mappable object for colorbar
        cmap = mcolors.ListedColormap([colors(i) for i in range(num_classes)])
        bounds = np.arange(num_classes + 1) - 0.5
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        # Display classification map
        im = axes[1].imshow(newIm, cmap=cmap, norm=norm)

        # Add colorbar with class labels
        cbar = fig.colorbar(im, ax=axes[1], ticks=np.arange(num_classes))
        cbar.ax.set_yticklabels(class_labels)  # Label the colorbar with class names
        cbar.set_label("Land Cover Class")  # Colorbar title

        #plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()




