## Urban-Land-Classification

Contains code to aquire data from ESA's Sentinel 2 Satellite

The code used to train RGB and MS CNN models is provided in CNN.py and CNN_MS.py

Also contains code to run a GUI to classify Sentinel 2 data and display it with an overlay. 

The required python libraries can be installed by running:
```bash
pip install -r requirements.txt
```

The ui tool can be loaded:
```bash
python ui.py

Note that the OpenCage Geocoder API key has since been disabled and therefore no new data can be downloaded without a new key. 
