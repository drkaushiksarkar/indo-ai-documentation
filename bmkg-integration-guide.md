# BMKG API Integration Guide for Climate Smart Indonesia

## Overview

This document provides a comprehensive guide for integrating data from BMKG (Badan Meteorologi, Klimatologi, dan Geofisika - Indonesian Agency for Meteorology, Climatology and Geophysics) into the Climate Smart Indonesia project. It covers API endpoints, data retrieval methods, and conversion processes for transforming JSON responses into NetCDF format for climate analysis and dengue prediction models.

## Table of Contents

- [BMKG Data Sources](#bmkg-data-sources)
- [API Access Methods](#api-access-methods)
- [API Endpoints](#api-endpoints)
- [Authentication](#authentication)
- [Data Retrieval Examples](#data-retrieval-examples)
- [JSON to NetCDF Conversion](#json-to-netcdf-conversion)
- [Integration with Climate Models](#integration-with-climate-models)
- [Code Examples](#code-examples)
- [Troubleshooting](#troubleshooting)
- [Additional Resources](#additional-resources)

## BMKG Data Sources

BMKG provides various meteorological, climatological, and geophysical data relevant to the Climate Smart Indonesia project:

1. **Weather Forecast Data**: Short-term weather predictions
2. **Climate Data**: Historical and current climate observations
3. **Rainfall Data**: Precipitation measurements across Indonesia
4. **WMS Map Services**: Web Map Services for visualizing data

For dengue prediction models, the most relevant data sources are weather forecasts, climate data, and rainfall data.

## API Access Methods

BMKG data can be accessed through several methods:

1. **REST API**: Direct HTTP requests to BMKG endpoints
2. **WMS Services**: Web Map Services for geospatial data
3. **JavaScript Library**: Using the `bmkg-wrapper` library
4. **Python Libraries**: Custom wrappers available via PyPI

## API Endpoints

### Weather Forecast API

```
https://api.bmkg.go.id/publik/prakiraan-cuaca?adm4={kode_wilayah_tingkat_iv}
```

Parameters:
- `adm4`: Administrative area code (level 4 - village/urban community)

### Rainfall Map WMS Service

```
https://gis.bmkg.go.id/arcgis/services/Peta_Curah_Hujan_dan_Hari_Hujan_/MapServer/WMSServer
```

For specific WMS capabilities and parameters, you can use:

```
https://gis.bmkg.go.id/arcgis/services/Peta_Curah_Hujan_dan_Hari_Hujan_/MapServer/WMSServer?request=GetCapabilities&service=WMS
```

### Climate Data API

```
https://api.bmkg.go.id/publik/iklim/suhu?loc={location_id}&start={start_date}&end={end_date}
```

Parameters:
- `loc`: Location identifier
- `start`: Start date in YYYY-MM-DD format
- `end`: End date in YYYY-MM-DD format

## Authentication

Most BMKG public APIs don't require authentication, but for advanced access or higher rate limits, you may need to request an API key from BMKG directly. Contact them at:

```
Email: info@bmkg.go.id
Web: https://www.bmkg.go.id/
```

## Data Retrieval Examples

### Using cURL

```bash
# Get weather forecast for Lebak Bulus area
curl "https://api.bmkg.go.id/publik/prakiraan-cuaca?adm4=31.74.06.1002"

# Get rainfall data for a specific area and date range
curl "https://api.bmkg.go.id/publik/iklim/curah-hujan?loc=501&start=2023-01-01&end=2023-01-31"
```

### Using JavaScript with bmkg-wrapper

```javascript
import BMKG from 'bmkg-wrapper';
const bmkg = new BMKG();

// Get weather forecast for Lebak Bulus
async function prakiraanCuaca(kelurahan) {
  const res = await bmkg.prakiraanCuaca(kelurahan);
  console.log(res);
}
prakiraanCuaca('Lebak Bulus');
```

### Using Python with Requests

```python
import requests
import json

def get_weather_forecast(adm4_code):
    url = f"https://api.bmkg.go.id/publik/prakiraan-cuaca?adm4={adm4_code}"
    response = requests.get(url)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        return None

# Example for Lebak Bulus
data = get_weather_forecast("31.74.06.1002")
print(json.dumps(data, indent=2))
```

## JSON to NetCDF Conversion

Converting JSON data from BMKG to NetCDF format is essential for compatibility with climate models and data analysis tools. Here's how to do it:

### Using Python with xarray and netCDF4

```python
import json
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime

def convert_bmkg_json_to_netcdf(json_data, output_file):
    """
    Convert BMKG JSON data to NetCDF format
    
    Parameters:
    -----------
    json_data : dict
        JSON data from BMKG API
    output_file : str
        Path to output NetCDF file
    """
    # Extract location information
    location = json_data['lokasi']
    lat = location['lat']
    lon = location['lon']
    
    # Extract weather forecast data
    forecasts = []
    for forecast_group in json_data['cuaca']:
        forecasts.extend(forecast_group)
    
    # Create a pandas DataFrame from the data
    data = []
    for forecast in forecasts:
        data.append({
            'time': pd.to_datetime(forecast['utc_datetime']),
            'temperature': forecast['t'],
            'humidity': forecast['hu'],
            'wind_speed': forecast['ws'],
            'wind_direction': forecast['wd_deg'],
            'cloud_cover': forecast['tcc'],
            'weather_code': forecast['weather']
        })
    
    df = pd.DataFrame(data)
    df = df.set_index('time')
    
    # Convert to xarray Dataset
    ds = xr.Dataset.from_dataframe(df)
    
    # Add coordinates
    ds = ds.assign_coords(
        lat=np.array([lat]),
        lon=np.array([lon])
    )
    
    # Add metadata
    ds.attrs['title'] = f"BMKG Weather Forecast for {location['desa']}, {location['kecamatan']}, {location['kotkab']}, {location['provinsi']}"
    ds.attrs['source'] = "BMKG API"
    ds.attrs['creation_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Add variable attributes
    ds['temperature'].attrs['units'] = 'degC'
    ds['temperature'].attrs['long_name'] = 'Air Temperature'
    
    ds['humidity'].attrs['units'] = '%'
    ds['humidity'].attrs['long_name'] = 'Relative Humidity'
    
    ds['wind_speed'].attrs['units'] = 'km/h'
    ds['wind_speed'].attrs['long_name'] = 'Wind Speed'
    
    ds['wind_direction'].attrs['units'] = 'degrees'
    ds['wind_direction'].attrs['long_name'] = 'Wind Direction'
    
    ds['cloud_cover'].attrs['units'] = '%'
    ds['cloud_cover'].attrs['long_name'] = 'Total Cloud Cover'
    
    ds['weather_code'].attrs['long_name'] = 'Weather Condition Code'
    
    # Save to NetCDF file
    ds.to_netcdf(output_file)
    print(f"NetCDF file created: {output_file}")
    
    return ds

# Example usage
with open('bmkg_data.json', 'r') as f:
    json_data = json.load(f)

convert_bmkg_json_to_netcdf(json_data, 'bmkg_forecast.nc')
```

### Converting WMS Data to NetCDF

For WMS services like rainfall maps, you can use GDAL to convert to NetCDF:

```python
import os
from osgeo import gdal

def wms_to_netcdf(wms_url, layer, bbox, width, height, output_file):
    """
    Download WMS data and convert to NetCDF
    
    Parameters:
    -----------
    wms_url : str
        URL of the WMS service
    layer : str
        WMS layer name
    bbox : list
        Bounding box [minx, miny, maxx, maxy]
    width, height : int
        Image dimensions
    output_file : str
        Path to output NetCDF file
    """
    # Create a temporary GeoTIFF file
    temp_file = 'temp_wms.tif'
    
    # Download WMS data as GeoTIFF
    wms_request = f"{wms_url}?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&LAYERS={layer}&STYLES=&CRS=EPSG:4326&BBOX={bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]}&WIDTH={width}&HEIGHT={height}&FORMAT=image/tiff"
    
    gdal.UseExceptions()
    
    # Download and save the GeoTIFF
    ds = gdal.Open(wms_request)
    driver = gdal.GetDriverByName('GTiff')
    driver.CreateCopy(temp_file, ds)
    ds = None
    
    # Convert GeoTIFF to NetCDF
    options = gdal.TranslateOptions(format='NetCDF', 
                                   creationOptions=['FORMAT=NC4', 'COMPRESS=DEFLATE'])
    gdal.Translate(output_file, temp_file, options=options)
    
    # Clean up temporary file
    os.remove(temp_file)
    print(f"NetCDF file created: {output_file}")

# Example usage
wms_url = "https://gis.bmkg.go.id/arcgis/services/Peta_Curah_Hujan_dan_Hari_Hujan_/MapServer/WMSServer"
layer = "2"  # Peta Curah Hujan layer
bbox = [94.971952, -11.007615, 141.020042, 6.076768]  # Indonesia bounding box
wms_to_netcdf(wms_url, layer, bbox, 1000, 500, 'bmkg_rainfall.nc')
```

## Integration with Climate Models

After converting BMKG data to NetCDF format, you can integrate it with the Variational Autoencoder (VAE) models used in the Climate Smart Indonesia project:

```python
import xarray as xr
import tensorflow as tf
import numpy as np

def preprocess_netcdf_for_vae(netcdf_file, scaler=None):
    """
    Preprocess NetCDF data for input into VAE models
    
    Parameters:
    -----------
    netcdf_file : str
        Path to NetCDF file
    scaler : sklearn.preprocessing.StandardScaler, optional
        Scaler for standardizing data
        
    Returns:
    --------
    processed_data : numpy.ndarray
        Processed data ready for VAE input
    scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler (if not provided as input)
    """
    # Load NetCDF data
    ds = xr.open_dataset(netcdf_file)
    
    # Select relevant variables
    variables = ['temperature', 'humidity', 'wind_speed', 'cloud_cover']
    data = np.stack([ds[var].values.flatten() for var in variables], axis=1)
    
    # Handle missing values
    data = np.nan_to_num(data)
    
    # Scale data
    if scaler is None:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        processed_data = scaler.fit_transform(data)
    else:
        processed_data = scaler.transform(data)
        
    return processed_data, scaler

# Example integration with VAE model
def integrate_with_vae_model(netcdf_file, vae_model_path):
    """
    Process NetCDF data and encode using trained VAE model
    
    Parameters:
    -----------
    netcdf_file : str
        Path to NetCDF file
    vae_model_path : str
        Path to saved VAE model
    
    Returns:
    --------
    latent_vectors : numpy.ndarray
        Encoded latent vectors
    """
    # Load and preprocess data
    data, _ = preprocess_netcdf_for_vae(netcdf_file)
    
    # Load VAE model
    vae_model = tf.keras.models.load_model(vae_model_path)
    
    # Get encoder part of VAE
    encoder = vae_model.get_layer('encoder')
    
    # Encode data to latent space
    latent_vectors = encoder.predict(data)
    
    return latent_vectors
```

## Code Examples

### Complete Example: Fetching Data, Converting to NetCDF, and Using with VAE

```python
import requests
import json
import xarray as xr
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import os

# 1. Fetch data from BMKG API
def fetch_bmkg_data(adm4_code):
    url = f"https://api.bmkg.go.id/publik/prakiraan-cuaca?adm4={adm4_code}"
    response = requests.get(url)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API Error: {response.status_code}")

# 2. Convert JSON to NetCDF
def json_to_netcdf(json_data, output_file):
    # Extract location information
    location = json_data['lokasi']
    lat = location['lat']
    lon = location['lon']
    
    # Extract weather forecast data
    forecasts = []
    for forecast_group in json_data['cuaca']:
        forecasts.extend(forecast_group)
    
    # Process the data
    times = []
    temp = []
    humidity = []
    wind_speed = []
    cloud_cover = []
    
    for forecast in forecasts:
        times.append(np.datetime64(forecast['utc_datetime']))
        temp.append(forecast['t'])
        humidity.append(forecast['hu'])
        wind_speed.append(forecast['ws'])
        cloud_cover.append(forecast['tcc'])
    
    # Create xarray dataset
    ds = xr.Dataset(
        data_vars=dict(
            temperature=(['time'], np.array(temp)),
            humidity=(['time'], np.array(humidity)),
            wind_speed=(['time'], np.array(wind_speed)),
            cloud_cover=(['time'], np.array(cloud_cover)),
        ),
        coords=dict(
            lon=float(lon),
            lat=float(lat),
            time=times,
        ),
        attrs=dict(
            description=f"Weather forecast for {location['desa']}, {location['kecamatan']}, {location['provinsi']}",
            source="BMKG API",
        ),
    )
    
    # Add variable metadata
    ds.temperature.attrs['units'] = 'degC'
    ds.humidity.attrs['units'] = '%'
    ds.wind_speed.attrs['units'] = 'km/h'
    ds.cloud_cover.attrs['units'] = '%'
    
    # Save to NetCDF file
    ds.to_netcdf(output_file)
    print(f"NetCDF file created: {output_file}")
    
    return ds

# 3. Preprocess NetCDF for VAE
def preprocess_for_vae(netcdf_file, scaler_file=None):
    # Load data
    ds = xr.open_dataset(netcdf_file)
    
    # Extract variables and stack them
    variables = ['temperature', 'humidity', 'wind_speed', 'cloud_cover']
    data_arrays = [ds[var].values for var in variables]
    
    # Reshape for VAE input
    X = np.column_stack([arr.flatten() for arr in data_arrays])
    
    # Scale data
    if scaler_file and os.path.exists(scaler_file):
        import joblib
        scaler = joblib.load(scaler_file)
    else:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        if scaler_file:
            import joblib
            joblib.dump(scaler, scaler_file)
    
    return X, scaler

# 4. Encode with VAE
def encode_with_vae(preprocessed_data, model_path):
    # Load trained VAE model
    model = tf.keras.models.load_model(model_path)
    
    # Get encoder part
    encoder = model.get_layer('encoder')
    
    # Encode data
    latent_vectors = encoder.predict(preprocessed_data)
    
    return latent_vectors

# Main pipeline
def main():
    # Configuration
    adm4_code = "31.74.06.1002"  # Lebak Bulus
    netcdf_file = "bmkg_forecast.nc"
    scaler_file = "climate_scaler.joblib"
    vae_model_path = "models/saved_models/climate_vae.h5"
    
    # Step 1: Fetch data
    print("Fetching data from BMKG API...")
    json_data = fetch_bmkg_data(adm4_code)
    
    # Step 2: Convert to NetCDF
    print("Converting JSON to NetCDF...")
    ds = json_to_netcdf(json_data, netcdf_file)
    
    # Step 3: Preprocess for VAE
    print("Preprocessing data for VAE...")
    X, scaler = preprocess_for_vae(netcdf_file, scaler_file)
    
    # Step 4: Encode with VAE
    print("Encoding data with VAE model...")
    latent_vectors = encode_with_vae(X, vae_model_path)
    
    print("Pipeline completed successfully!")
    print(f"Latent vectors shape: {latent_vectors.shape}")
    
    # Can now use latent_vectors for dengue prediction models
    
if __name__ == "__main__":
    main()
```

### Fetching and Processing WMS Rainfall Data

```python
import requests
from owslib.wms import WebMapService
from osgeo import gdal
import xarray as xr
import numpy as np
import os

def get_wms_rainfall_data(output_geotiff, output_netcdf):
    """
    Fetch rainfall map from BMKG WMS and save as GeoTIFF and NetCDF
    """
    # Connect to WMS service
    wms_url = "https://gis.bmkg.go.id/arcgis/services/Peta_Curah_Hujan_dan_Hari_Hujan_/MapServer/WMSServer"
    wms = WebMapService(wms_url, version='1.3.0')
    
    # Get the image
    img = wms.getmap(
        layers=['2'],  # Layer ID for Peta Curah Hujan
        srs='EPSG:4326',
        bbox=(-11.007615, 94.971952, 6.076768, 141.020042),  # (miny, minx, maxy, maxx)
        size=(1000, 500),
        format='image/tiff',
        transparent=True
    )
    
    # Save the image
    with open(output_geotiff, 'wb') as out:
        out.write(img.read())
    
    # Convert to NetCDF using GDAL
    options = gdal.TranslateOptions(format='NetCDF')
    gdal.Translate(output_netcdf, output_geotiff, options=options)
    
    # Load the NetCDF file and add metadata
    ds = xr.open_dataset(output_netcdf)
    
    # Add metadata
    ds.attrs['title'] = 'BMKG Rainfall Map'
    ds.attrs['source'] = 'BMKG WMS Service'
    
    # Save the enhanced NetCDF file
    ds.to_netcdf(output_netcdf)
    
    # Clean up GeoTIFF
    os.remove(output_geotiff)
    
    return ds

# Example usage
rainfall_data = get_wms_rainfall_data('rainfall.tiff', 'rainfall.nc')
print("Rainfall data saved as NetCDF")
```

## Troubleshooting

### Common Issues and Solutions

1. **API Connection Errors**
   - **Issue**: Unable to connect to BMKG API
   - **Solution**: Check your network connection and verify that the API endpoint is correct. BMKG may occasionally change their API structure.

2. **Data Format Changes**
   - **Issue**: JSON structure from API is different than expected
   - **Solution**: Update your parsing code to match the new structure. Print the raw JSON response to understand the current format.

3. **WMS Service Unavailable**
   - **Issue**: Cannot connect to WMS services
   - **Solution**: Verify the WMS URL is correct and that the service is currently running. BMKG occasionally performs maintenance on their servers.

4. **NetCDF Conversion Errors**
   - **Issue**: Errors when converting to NetCDF format
   - **Solution**: Ensure you have all required dependencies installed (xarray, netCDF4). Check that your data is properly structured before conversion.

### Debugging Tips

- Use proper error handling and logging in your code
- Print intermediate results to understand data structure
- Start with small data samples before processing large datasets
- Check API documentation for any updates or changes

## Additional Resources

### Libraries and Tools

- **xarray**: Python library for working with labeled multi-dimensional arrays
- **netCDF4**: Python interface to the netCDF C library
- **owslib**: Python library for accessing OGC web services
- **GDAL**: Geospatial Data Abstraction Library
- **bmkg-wrapper**: JavaScript library for easy BMKG API access

### Documentation

- [BMKG Official Website](https://www.bmkg.go.id/)
- [BMKG Data Online](https://dataonline.bmkg.go.id/)
- [xarray Documentation](https://xarray.pydata.org/en/stable/)
- [netCDF Documentation](https://unidata.github.io/netcdf4-python/)
- [Climate Smart Indonesia Project Documentation](https://github.com/NormanMul/Climate-Smart-Indonesia---Documentation)

---

## Contact Information

For questions or assistance with BMKG data integration, please contact:

- **Climate Smart Indonesia Development Team**
  - GitHub: https://github.com/NormanMul/Climate-Smart-Indonesia---Documentation

- **BMKG Data Services**
  - Email: info@bmkg.go.id
  - Website: https://www.bmkg.go.id/


