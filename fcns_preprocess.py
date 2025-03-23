import numpy as np

def normalise_band_for_CNN(band):#, mean, std):
    # method 1
    # band = band / 10000
    # method 2
    # band = np.clip(band, 0, 2000)
    # band /= 2000
    #method 3
    band = (band - np.min(band)) / (np.max(band) - np.min(band))
    #method 4
    # band = band / 2000
    # band = np.nan_to_num(band, nan=0)
    # band = (band * 255)
    # band[band > 255] = 255
    # band /= 255
    # method 5
    # band = band / 2000
    # band = (band - np.min(band)) / (np.max(band) - np.min(band))
    # method 6
    # band = ((band-np.mean(band))/np.std(band))*std+mean
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