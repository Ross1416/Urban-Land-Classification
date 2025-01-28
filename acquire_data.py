import openeo
import xarray
import matplotlib.pyplot as plt

# First, we connect to the back-end and authenticate. 
con = openeo.connect("openeo.dataspace.copernicus.eu")
con.authenticate_oidc()

##### Uncomment to download data
# datacube = con.load_collection(
#     "SENTINEL2_L2A",
#     spatial_extent={"west": -4.27, "south": 55.83, "east": -4.20, "north": 55.87},
#     temporal_extent = ["2021-02-01", "2021-04-30"],
#     bands=["B02", "B03", "B04"],
#     max_cloud_cover=10,
# )

# Download data cube
# datacube.download("./data/data.nc")


# Load data
ds = xarray.load_dataset("./data/data.nc")
data = ds[["B04", "B03", "B02"]].to_array(dim="bands")
data[{"t": -1}].plot.imshow(vmin=0, vmax=2000)
plt.show()

#download
# datacube.band("B04").max_time().download("./data/image.tiff")