# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 10:45:13 2025

@author:  Xingjun Huang & Ruichen MA
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import geopandas as gpd
import rasterio
from shapely.geometry import box
from tools import global_config, accessibility_calculator_poi
from multiprocessing import Pool, cpu_count
import gc
from scipy.spatial.distance import cdist
from pyproj import Transformer


# Define a global (top-level) function for parallel processing
def process_evse_counts(args):
    geom, stations_gdf, evse_type = args
    # Count the number of EVSE of a specific type within the given geometry
    return count_stations_in_grid(geom, stations_gdf, evse_type)


def count_stations_in_grid(grid_geom, stations_gdf, evse_type):
    # Find possible matching indices using the spatial index (bounding box check)
    possible_matches_index = list(stations_gdf.sindex.intersection(grid_geom.bounds))
    possible_matches = stations_gdf.iloc[possible_matches_index]
    # Refine to exact matches using precise intersection
    precise_matches = possible_matches[possible_matches.intersects(grid_geom)]
    # Sum the number of EVSE of the given type within the geometry
    return precise_matches[evse_type].sum()


def add_evse_count_to_population_gdf(population_gdf, stations_gdf):
    # Create tasks for each EVSE type for parallel processing
    tasks_dc = [(geom, stations_gdf, 'ev_dc_fast_num') for geom in population_gdf['geometry']]
    tasks_l1 = [(geom, stations_gdf, 'ev_level1_evse_num') for geom in population_gdf['geometry']]
    tasks_l2 = [(geom, stations_gdf, 'ev_level2_evse_num') for geom in population_gdf['geometry']]

    # Parallel processing using all available CPU cores
    with Pool(cpu_count()) as pool:
        population_gdf['dc_evse_count'] = pool.map(process_evse_counts, tasks_dc)
        population_gdf['l1_evse_count'] = pool.map(process_evse_counts, tasks_l1)
        population_gdf['l2_evse_count'] = pool.map(process_evse_counts, tasks_l2)

    # Compute total EVSE count per grid cell
    population_gdf['evse_count'] = (
        population_gdf['dc_evse_count'] +
        population_gdf['l1_evse_count'] +
        population_gdf['l2_evse_count']
    )

    return population_gdf


def read_population_data(population_tif):
    with rasterio.open(population_tif) as src:
        population_data = src.read(1)
        population_transform = src.transform
        population_crs = src.crs
    return population_data, population_transform, population_crs

def create_population_gdf(population_data, population_transform, population_crs):
    print('creating population geodataframe ...')
    rows, cols = np.where(population_data >= 0)
    population_geometries = []
    population_values = []

    for row, col in tqdm(zip(rows, cols), total=len(rows), desc='Processing pixels'):
        x, y = population_transform * (col, row)
        geom = box(x, y, x + population_transform[0], y + population_transform[4])
        population_geometries.append(geom)
        population_values.append(population_data[row, col])

    population_gdf = gpd.GeoDataFrame({'geometry': population_geometries, 'population': population_values}, crs=population_crs)
    return population_gdf

def read_stations_data(stations_file, target_crs, columns,bbox=None):      
    stations_gdf = gpd.read_file(stations_file, engine='pyogrio', columns= columns,bbox=bbox)   
    stations_gdf = stations_gdf.to_crs(target_crs)
    stations_gdf = stations_gdf[stations_gdf.geometry.notnull()].copy()
    columns_to_fill = ['ev_dc_fast_num','ev_level1_evse_num', 'ev_level2_evse_num']
    stations_gdf[columns_to_fill] = stations_gdf[columns_to_fill].fillna(0)
    stations_gdf['evse_num'] = stations_gdf[columns[0]] + stations_gdf[columns[1]] + stations_gdf[columns[2]]
    return stations_gdf

def count_pois_in_grid(city_pois,grid_geom, category_labels,POITYPE):
    """
    Count the number of Points of Interest (POIs) within a given grid cell.
    It can count overall POIs or POIs by specific categories depending on POITYPE.
    
    Parameters:
    - city_pois: GeoDataFrame of POIs in the city
    - grid_geom: Shapely geometry of the grid cell
    - category_labels: Dictionary mapping full category names to short labels
    - POITYPE: Type of POI counting ("POI", "POI Class_manual", "Purpose_manual")
    
    """
    # Create a spatial index for efficient spatial queries
    sindex = city_pois.sindex

    # Find candidate POIs whose bounding boxes intersect the grid
    possible_matches_index = list(sindex.intersection(grid_geom.bounds))
    possible_matches = city_pois.iloc[possible_matches_index]

    # Refine to only POIs that truly intersect with the grid geometry
    pois_in_grid = possible_matches[possible_matches.intersects(grid_geom)]

    if POITYPE == "POI":
        # If there are no POIs in the grid, return a list with a single 0
        if pois_in_grid.empty:
            # print("No points of interest found in the grid.")
            return [0]  
        
        # Otherwise, return the total number of POIs in the grid
        total_count = len(pois_in_grid)
        return [total_count]
    
    elif POITYPE == "POI Class_manual":
        # If empty, return a list of zeros: first element for total, rest for each category
        if pois_in_grid.empty:
            # print("No points of interest found in the grid.")
            return [0] * (len(category_labels) + 1) 
        
        # Count POIs in each category
        category_counts = {
            short_label: len(pois_in_grid[pois_in_grid["POI Class_manual"].astype(str) == str(full_label)])
            for full_label, short_label in category_labels.items()
        }
        
        # Return total POIs followed by counts per category
        total_count = len(pois_in_grid)
        return [total_count] + list(category_counts.values())
    
    elif POITYPE == "Purpose_manual":
        # Similar to POI Class_manual, but based on the "Purpose_manual" column
        if pois_in_grid.empty:
            # print("No points of interest found in the grid.")
            return [0] * (len(category_labels) + 1)
        category_counts = {
            short_label: len(pois_in_grid[pois_in_grid["Purpose_manual"].astype(str) == str(full_label)])
            for full_label, short_label in category_labels.items()
            }


        total_count = len(pois_in_grid)
        return [total_count] + list(category_counts.values())


def add_poi_count_to_city_gdf(city_pois,poi_count_type, category_labels, POITYPE):
    """
    Add POI counts to each grid cell in the city GeoDataFrame.
    This includes total POI counts and counts per category (if applicable).
    """
    # Compute POI counts for each grid cell (returns a 2D numpy array)
    poi_2D = np.array([count_pois_in_grid(city_pois,geom, category_labels,POITYPE) for geom in global_config.grid_gdf['geometry']])
    # Dynamically assign each column in grid_gdf with corresponding counts
    for i, suffix in enumerate(poi_count_type):
        global_config.grid_gdf[suffix] = poi_2D[:, i]


def compute_w_batch(args):
    """
    Compute a Gaussian-decayed weight matrix W for a batch of grid cells.
    """
    grid_batch, station_proj, d0, gaussian_decay = args
    # Compute pairwise Euclidean distances in kilometers
    batch_distances = cdist(grid_batch, station_proj, metric='euclidean').astype(np.float32) / 1000
    # Apply Gaussian decay function to distances
    W_batch = gaussian_decay(batch_distances, d0) # W_ij
    # Set weights to 0 for distances exceeding the threshold
    W_batch[batch_distances > d0] = 0
    return W_batch


def project_coordinates(lats, lons, target_crs='EPSG:3857'):
    """
    Project latitude and longitude coordinates to a target CRS.
    """
    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    xs, ys = transformer.transform(lons, lats)
    return np.vstack((xs, ys)).T.astype(np.float32)


def gaussian_decay(distance, d0):
    """
    Gaussian decay function for accessibility weighting.
    """
    norm_factor = (1 - np.exp(-0.5)).astype(np.float32)
    decay_values = np.where(distance <= d0,
                            ((np.exp(-0.5 * (distance / d0) ** 2) - np.exp(-0.5)) / norm_factor).astype(np.float32),
                            0).astype(np.float32)
    return decay_values.astype(np.float32)


def overlay_tiff_poi(d0=None,poi_file: str = None,usa_grid_1km_path: str = None,stations_file: str = None, poi_count_type=None,category_labels=None,cs_type=None,POITYPE=None):
        
    """
    Overlay POI data and EVSE stations on a grid and compute accessibility and equity metrics.
    
    """
    all_city_accessibility_summaries = [pd.DataFrame() for _ in range(4)]
    all_city_gini_summaries = [pd.DataFrame() for _ in range(4)]

    # Initialize DataFrames to store results for all cities (4 types assumed)
    if POITYPE == "POI":
        poi_gdf = gpd.read_parquet(poi_file, columns=['geometry'])
    elif POITYPE == "POI Class_manual":
        poi_gdf = gpd.read_parquet(poi_file, columns=['POI Class_manual', 'geometry'])
    else:
        poi_gdf = gpd.read_parquet(poi_file, columns=['Purpose_manual', 'geometry'])

    # Read POI data once and keep in memory, selecting only necessary columns
    global_config.usa_map_gdf['simplified_geometry'] = global_config.usa_map_gdf.geometry.simplify(tolerance=0.0001, preserve_topology=True)

    # Iterate over each city (identified by unique GID_2) 
    for idx in global_config.usa_map_gdf['GID_2'].unique(): 

        # Extract the city's polygon
        city = global_config.usa_map_gdf[global_config.usa_map_gdf['GID_2'] == idx].copy()
    
        # Use simplified geometry for bounding-box filtering
        bbox = city['simplified_geometry'].bounds.values.flatten()
        # Spatial filter using bbox
        city_pois = poi_gdf.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]  

        # Extract grid cells and stations within the city boundary
        global_config.grid_gdf = gpd.read_file(usa_grid_1km_path, layer="grid", bbox=city.geometry).sjoin(city, predicate="intersects", how="left")
        global_config.station_gdf = read_stations_data(stations_file, 4326, columns=['ev_dc_fast_num','ev_level1_evse_num', 'ev_level2_evse_num','geometry'], bbox=city.geometry)
        
        # If either grid or station data is empty, create placeholder rows with zeros
        if global_config.grid_gdf.empty or global_config.station_gdf.empty:
            for i in range(len(all_city_accessibility_summaries)):
    
                all_city_accessibility_summaries[i] = pd.concat(
                    [all_city_accessibility_summaries[i], 
                     pd.DataFrame({f'Average_Accessibility_{"" if poi_count_col == "poi_count" else "_" + poi_count_col}': [0]
                                    for poi_count_col in poi_count_type})],
                    ignore_index=True
                )

                all_city_gini_summaries[i] = pd.concat(
                    [all_city_gini_summaries[i], 
                     pd.DataFrame({f'Gini_Coefficient_{poi_count_col if poi_count_col != "poi_count" else ""}': [0]
                                   for poi_count_col in poi_count_type})], 
                    ignore_index=True
                )
            print(f"{idx} grid or station data is empty")
            continue

        # Count POIs per grid cell
        add_poi_count_to_city_gdf(city_pois,poi_count_type, category_labels,POITYPE)
        global_config.grid_gdf.reset_index(drop=True)
        
        # Extract coordinates
        station_coords = np.array([(geom.y, geom.x) for geom in global_config.station_gdf.geometry], dtype=np.float32)
        grid_centroids = [(geom.centroid.y, geom.centroid.x) for geom in global_config.grid_gdf.geometry]
        station_proj = project_coordinates(*zip(*station_coords))
        grid_proj = project_coordinates(*zip(*grid_centroids))
        
        # Compute pairwise distance matrix in km
        distances = cdist(grid_proj, station_proj, metric='euclidean').astype(np.float32) / 1000
        global_config.W = gaussian_decay(distances, d0) 
        global_config.W[distances > d0] = 0 
        
        num_grids = grid_proj.shape[0]

         # For each charging station type, compute accessibility and Gini coefficient
        for i, cstype in enumerate(cs_type):
            # Create AccessibilityCalculator object
            calculator = accessibility_calculator_poi.AccessibilityCalculator_poi(
                d0=d0, 
                supply_type=cstype,
                poi_count_type=poi_count_type
            )

            # Compute accessibility and Gini metrics
            calculator.calculate_accessibility_m2sfca(num_grids)
            city_accessibility_result = calculator.summarize_accessibility()
            gini_coefficient_result = calculator.summarize_gini_coefficient()

            # Append results to overall DataFrames
            all_city_accessibility_summaries[i] = pd.concat(
                [all_city_accessibility_summaries[i], city_accessibility_result], ignore_index=True
            )
            all_city_gini_summaries[i] = pd.concat(
                [all_city_gini_summaries[i], gini_coefficient_result], ignore_index=True
            )

        global_config.grid_gdf = None  # Clear global variables to free memory
        del city_accessibility_result,gini_coefficient_result
        global_config.station_gdf = None
        gc.collect()  
        print(f"{idx} is finished")

    # Return results as a dictionary
    return {
        "accessibility_summaries": {
            cs: all_city_accessibility_summaries[i] for i, cs in enumerate(cs_type)
        },
        "gini_summaries": {
            cs: all_city_gini_summaries[i] for i, cs in enumerate(cs_type)
        },
    }

