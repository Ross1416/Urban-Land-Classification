from data import *
import matplotlib
matplotlib.use("QtAgg")  # or "TkAgg"
import matplotlib.pyplot as plt
# ---------------- Global Variables ----------------
BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09","B11", "B12"]
MAX_CLOUD_COVER = 30

# ---------------- Main Execution ----------------

if __name__ == "__main__":
    location = 'Arles'
    width = 3
    height = 3
    years = range(2016, 2023+1)

    mode = 1    # 0 = Download Datasets, 1 = Filter to get one image from each dataset

    if mode == 0:
        north, south, east, west = postcode_to_area(location, height, width)
        for year in years:
            saveLocation = f"./data/{location}_{height}x{width}_{year}.nc"
            print(saveLocation)
            download_dataset(north, south, east, west, BANDS, MAX_CLOUD_COVER, saveLocation, year)
    else:
        # Combine Dataset
        datasets = []
        con = openeo.connect("openeo.dataspace.copernicus.eu")
        con.authenticate_oidc()
        for year in years:
            file_path = f"./data/{location}_{height}x{width}_{year}.nc"
            try:
                ds = xr.load_dataset(file_path)
                # Convert xarray DataSet to a (bands, t, x, y) DataArray
                data = ds[["B04", "B03", "B02"]].to_array(dim="bands")


                for i in range(0, data.sizes["t"]):
                    # Create a new figure for each set of images
                    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=90, sharey=True)

                    maxVal = 2750

                    # Plot RGB image
                    data[{"t": i}].plot.imshow(vmin=0, vmax=maxVal, ax=axes[0])

                    # Convert to Grayscale
                    weights = xr.DataArray([0.2989, 0.5870, 0.1140], dims=["bands"])
                    grayscale = (data[{"t": i}] * weights).sum(dim="bands")
                    grayscale.plot.imshow(vmin=0, vmax=maxVal, ax=axes[1])

                    # (SAME CODE AS CLOUD FUNCTION JUST INPUT IN DIFFERENT FORMAT AND PLOTTING IMAGES HERE)
                    grayscale_np = grayscale.values

                    grayscale_np = np.minimum(grayscale_np, maxVal)

                    grayscale_np = grayscale_np / maxVal * 255

                    hist_values, bin_edges = np.histogram(grayscale_np, bins=256, range=(0, 256))

                    plt.figure(figsize=(6, 4), dpi=90)
                    plt.bar(bin_edges[:-1], hist_values, width=np.diff(bin_edges), color="black", edgecolor="white")
                    plt.title("Histogram (Clipping=2750)")
                    plt.xlabel("Pixel Value")
                    plt.ylabel("Frequency")
                    plt.show()

                    # If histogram has too much white (cloud) don't include
                    if hist_values[255] > data.shape[-2]*data.shape[-1]*0.03:
                        print("Too much cloud")
                        continue

                    # If any NaN don't include:
                    if  hist_values[0] > 30:
                        print("Contained NaN")
                        continue

                    # If any in shadow don't include:
                    if np.sum(hist_values[0:5]) > data.shape[-2]*data.shape[-1]*0.05:
                        print("Too dark")
                        continue

                    # If made it through checks then make representative of year and move on
                    datasets.append(ds[{"t": i}])
                    print("Picture Accepted")
                    break

                # Check if year was added, if not print error

            except FileNotFoundError:
                print(f"Missing {year} ")

        # Check all images appropriate
        fig, axes = plt.subplots(nrows=math.floor(len(datasets) / 2), ncols=math.ceil(len(datasets) / 2), figsize=(8, 3), dpi=90, sharey=True)
        axes = axes.flatten()
        data = []
        for i in range(len(datasets)):
            data = datasets[i][["B04", "B03", "B02"]].to_array(dim="bands")

            data.plot.imshow(vmin=0, vmax=2000, ax=axes[i])
        plt.show()

        combined_data = xr.concat(datasets, dim="t")
        file_path = f"./data/{location}_{height}x{width}_{years[0]}to{years[len(datasets)-1]}.nc"
        if file_path is not None:
            combined_data.to_netcdf(file_path)






