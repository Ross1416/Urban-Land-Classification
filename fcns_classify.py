import math
import numpy as np

from fcns_preprocess import *

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
                cnn_image = np.dstack([normalise_band_for_CNN(redArr),# 0.3398, 0.2037),
                                       normalise_band_for_CNN(greenArr),# 0.3804, 0.1375),
                                       normalise_band_for_CNN(blueArr)]),# 0.4025, 0.1161)])
                #print(cnn_image.shape)
                cnn_image = np.expand_dims(cnn_image, axis=0)
                #print(cnn_image.shape)
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