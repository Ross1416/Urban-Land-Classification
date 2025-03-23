import numpy as np

def normalise_band_for_CNN(band, mean, std):

    band = np.nan_to_num(band, nan=0)
    band = ((band-np.mean(band))/np.std(band)+ 1e-8)*std+mean
    band = np.clip(band,0,1)

    return band