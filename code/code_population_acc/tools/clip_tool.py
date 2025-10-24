# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 13:32:46 2025

@author: Ruichen MA
"""

import os
import glob
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.errors import RasterioIOError
import matplotlib.pyplot as plt

def setup_directories(input_dir: str, output_dir: str):
    """
    setup paths for input_dir and output_dir
    """
    os.makedirs(output_dir, exist_ok=True)
    return input_dir, output_dir

def load_and_prepare_geojson(geojson_path: str):
    """
    load GeoJSON file。
    """
    counties_gdf = gpd.read_file(geojson_path, engine='pyogrio')
    
    # repair invalid geometries
    if not counties_gdf.geometry.is_valid.all():
        print("warning: GeoDataFrame include invalid geometries，and trying to repair...")
        counties_gdf["geometry"] = counties_gdf.geometry.buffer(0)
    
    # remove empty geometries
    counties_gdf = counties_gdf[~counties_gdf.geometry.is_empty]
    if counties_gdf.empty:
        print("Error: no valid geometries after repairing GeoJSON！")
        exit()
    
    # check the coordinate system is EPSG:4326
    if counties_gdf.crs.to_epsg() != 4326:
        print("Warning: GeoJSON is not EPSG:4326 and changing...")
        counties_gdf = counties_gdf.to_crs(epsg=4326)
    return counties_gdf

def get_tif_files(input_dir: str, pattern: str):
    """
    obtain a list of TIF files matching a specific pattern in the input directory.
    """
    read_path = os.path.join(input_dir, pattern)
    return glob.glob(read_path)

def clip_raster_files(tif_files: str, output_dir: str, counties_gdf: gpd.GeoDataFrame):
    """
    clip the list of TIF files according to the county boundaries of the United States and save them to the specified location.
    """
    for tif_path in tif_files:
        clip_raster(tif_path, output_dir, counties_gdf)


def plot_tiff(tiff_file: str):
    """
    visualize the tiff map
    """
    with rasterio.open(tiff_file) as src:
        # read 1st band
        band1 = src.read(1) 
        # visualization
        plt.imshow(band1, cmap='gray') 
        plt.colorbar() 
        plt.title('TIFF Raster Data Visualization')
        plt.show()

def clip_raster(tif_path: str, output_dir: str, counties_gdf: gpd.GeoDataFrame):
    """
    clip a single TIF file according to the county boundaries of the United States and save it to the specified location.
    
    Parameters
    ----------
    tif_path : str
        Path to the input raster (.tif) file.
    output_dir : str
        Directory where the clipped raster will be saved.
    counties_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing the county boundary geometries for clipping.
    
    """
    output_filename = os.path.basename(tif_path).replace('.tif', '_clipped_usa.tif')
    output_path = os.path.join(output_dir, output_filename)
    
    try:
        print(f"start processing: {tif_path}")
        with rasterio.open(tif_path) as src:
            if src.crs.to_epsg() != 4326:
                raise ValueError(f" {tif_path} is not EPSG:4326。")
            
            # check intersection
            tif_bounds = src.bounds
            geo_bounds = counties_gdf.total_bounds
            if not (tif_bounds[0] <= geo_bounds[2] and tif_bounds[2] >= geo_bounds[0] and
                    tif_bounds[1] <= geo_bounds[3] and tif_bounds[3] >= geo_bounds[1]):
                print(f"Warning: {tif_path} has no intersection with GeoJSON")
                return
            
            # Reproject county geometries to an equal-area projection (EPSG:5070) 
            projected_gdf = counties_gdf.to_crs(epsg=5070)  

            # Apply a small buffer (0.1) to avoid boundary data loss during clipping 
            buffered_geometries = projected_gdf.geometry.buffer(0.1)

            # Convert the buffered geometries back to geographic coordinates (EPSG:4326)
            buffered_geometries = buffered_geometries.to_crs(epsg=4326)
            out_image, out_transform = mask(src, buffered_geometries, crop=False)

            # check null
            if out_image.size == 0:
                print(f"Warning: {tif_path} is null and will not save。")
                return

            out_meta = src.meta.copy()

            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "dtype": 'int32'  # update data type into int32
            })

            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(out_image)
            
            print(f"clip and save to: {output_path}")
            
            plot_tiff(output_path)
            
    except RasterioIOError as rio_err:
        print(f"Can not open {tif_path}: {rio_err}")
    except Exception as e:
        print(f"Errors {tif_path}: {e}")
        
def clip_tiff(input_tif_directory: str,output_tif_directory: str,geojson_path: str,tiff_file: str):
    
    print("Step1:Start Clipping ...")
    # create paths
    input_dir, output_dir = setup_directories(input_tif_directory, output_tif_directory)
    # load GeoJSON
    counties_gdf = load_and_prepare_geojson(geojson_path)
    # laod population TIF
    population_tifs = get_tif_files(input_dir, tiff_file)
    # clip TIF and save
    clip_raster_files(population_tifs, output_dir, counties_gdf)
    print("--" * 50)
