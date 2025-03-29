from tensorflow import keras
import matplotlib
import matplotlib.pyplot as plt
import xarray as xr
matplotlib.use('qtagg')

import matplotlib.colors as mcolors

from data import *


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
        strideRows = 64
        strideCols = 64
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
                    cnn_image = np.dstack([normalise_band(redArr, 0.3398, 0.2037),
                     normalise_band(greenArr, 0.3804, 0.1375),
                     normalise_band(blueArr, 0.4025, 0.1161)])

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




