#%%
import openeo
import xarray
import matplotlib.pyplot as plt
import cv2

def interpolate_image(img, spatial_resolution):
    """Interpolate input images to correct spatial resolution."""
    scale_factor = int(spatial_resolution / 10)
    new_size = (int(img.shape[0] * scale_factor), int(img.shape[1] * scale_factor))
    resized_img = cv2.resize(img, new_size, interpolation=cv2.INTER_CUBIC)
    return resized_img

# First, we connect to the back-end and authenticate. 
# con = openeo.connect("openeo.dataspace.copernicus.eu")
# con.authenticate_oidc()

##### Uncomment to download data
# datacube = con.load_collection(
#     "SENTINEL2_L2A",
#     spatial_extent={"west": -4.27, "south": 55.83, "east": -4.20, "north": 55.87},
#     temporal_extent = ["2021-02-01", "2021-04-30"],
#     bands=["B02", "B03", "B04","B01"],
#     max_cloud_cover=10,
# )
#
# # Download data cube
# datacube.download("./data/data.nc")



# Load data
ds = xarray.load_dataset("./data/data.nc")

r=ds[["B04"]].to_array(dim="bands")[{"t": -1}]
a=ds[["B01"]].to_array(dim="bands")[{"t": -1}]

print(r.shape)
print(a.shape)

r=r[0][:][:]

# Input is nearest neighbour interpolated to be equal dimensions
a=a[0][:][:]

# Decimate
a_decimated = a[::6,::6]
a_bicubic = interpolate_image(a_decimated.to_numpy(), spatial_resolution=60)

error = abs(a[:300,:300]-a_bicubic[:300,:300])


plt.figure()
a.plot.imshow(vmin=0, vmax=2000,cmap="grey")
plt.figure()
plt.imshow(a_bicubic, vmin=0, vmax=2000,cmap="gray")
# a_bicubic.plot.imshow(vmin=0, vmax=2000,cmap="grey")
# plt.figure()
# error.plot.imshow(vmin=0, vmax=2000,cmap="grey")
plt.figure()
plt.imshow(error,cmap="gray")
plt.show()

exit()
data = ds[["B04", "B03", "B02"]].to_array(dim="bands")
data[{"t": -1}].plot.imshow(vmin=0, vmax=2000)
# plt.show()

#download
# datacube.band("B04").max_time().download("./data/image.tiff")