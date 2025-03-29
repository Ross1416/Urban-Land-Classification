import numpy as np

def normalise_band_for_CNN(band, mean, std):

    band = np.nan_to_num(band, nan=0)
    band = ((band-np.mean(band))/np.std(band)+ 1e-8)*std+mean
    band = np.clip(band,0,1)

    return band

def check_cloud(red, green, blue, width, height):
    gray = 0.2989 * red + 0.5870 * green + 0.1140 * blue

    hist_values, bin_edges = np.histogram(gray, bins=255, range=(0, 255))

    if hist_values[255] > width * height * 0.01:
        return 1
    
    return 0