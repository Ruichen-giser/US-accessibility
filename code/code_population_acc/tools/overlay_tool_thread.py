# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 10:45:13 2025

@author:  Ruichen MA
"""
import geopandas as gpd
import rasterio
from shapely.geometry import box
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from tools import accessibility_calculator
import matplotlib.pyplot as plt


def read_population_data(population_tif):
    """
    Read a population raster (.tif) file and extract the data array,
    spatial transformation, and coordinate reference system (CRS).
    """
    with rasterio.open(population_tif) as src:
        population_data = src.read(1)
        population_transform = src.transform
        population_crs = src.crs
    return population_data, population_transform, population_crs


def create_population_gdf(population_data, population_transform, population_crs):
    """
    Convert population raster data into a GeoDataFrame where each
    pixel (with population ≥ 1) becomes a polygon feature with its value.
    """
    print('creating population geodataframe ...')
    rows, cols = np.where(population_data >= 1)
    population_geometries = []
    population_values = []

    for row, col in tqdm(zip(rows, cols), total=len(rows), desc='Processing pixels'):
        x, y = population_transform * (col, row)
        geom = box(x, y, x + population_transform[0], y + population_transform[4])
        population_geometries.append(geom)
        population_values.append(population_data[row, col])

    population_gdf = gpd.GeoDataFrame({'geometry': population_geometries, 'population': population_values}, crs=population_crs)
    return population_gdf


def read_stations_data(stations_file, target_crs):
    """
    Read and preprocess an EV charging station dataset into a standardized GeoDataFrame.
    """
    print('reading stations data ...')
    stations_gdf = gpd.read_file(stations_file)
    stations_gdf = stations_gdf.to_crs(target_crs)
    stations_gdf = stations_gdf[['id','country','state','city','zip','ev_dc_fast_num','ev_level1_evse_num', 'ev_level2_evse_num','geometry']]
    columns_to_fill = ['ev_dc_fast_num', 'ev_level1_evse_num', 'ev_level2_evse_num']
    stations_gdf[columns_to_fill] = stations_gdf[columns_to_fill].fillna(0)
    stations_gdf['evse_num'] = stations_gdf['ev_dc_fast_num'] + stations_gdf['ev_level1_evse_num'] + stations_gdf['ev_level2_evse_num']
    return stations_gdf


def plot_accessibility_with_stations(grids_gdf, station_gdf, 
                                     column='Accessibility', 
                                     cmap='GnBu', 
                                     title='Charging Accessibility'):
    """
    Visualize spatial accessibility results and overlay EV charging station locations.
    
    Parameters
    ----------
    :param grids_gdf: accessibility metrics for spatial grid cells or regions.
    :param station_gdf: charging station locations.
    :param column: Name of the column in `grids_gdf` representing accessibility values.
    :param cmap: Matplotlib colormap used for the accessibility heatmap.
    :param title: Title of the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the accessibility results as a color-coded map
    grids_gdf.plot(column=column, ax=ax, legend=True, cmap=cmap, edgecolor='none')
    
    # Plot charging station locations (shown as red dots)
    station_gdf.plot(ax=ax, color='red', markersize=10, label='Charging Station')

    # Add title, legend, and formatting
    ax.set_title(title, fontsize=14)
    ax.set_axis_off()
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    
def plot_population_with_stations(grids_gdf, station_gdf, 
                                   column='population', 
                                   cmap='Oranges', 
                                   title='Population Distribution and Charging Stations'):
    """
    Visualize population distribution and EV charging station locations.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the population distribution as a color-coded map
    grids_gdf.plot(column=column, ax=ax, legend=True, cmap=cmap, edgecolor='none')
    
    # Plot charging station locations
    station_gdf.plot(ax=ax, color='red', markersize=10, label='Charging Station')

    ax.set_title(title, fontsize=14)
    ax.set_axis_off()
    ax.legend()
    plt.tight_layout()
    plt.show()


def process_evse_type(city, joined_pop_gdf, joined_sta_gdf, supply_col, d0, plot=False,  method='m2sfca'):
    """
    Process and evaluate EVSE (Electric Vehicle Supply Equipment) accessibility for a given city.
    """
    print(f"\nProcessing {supply_col} in City {city}...")
    
    # Initialize the accessibility calculator with population and station data
    calculator = accessibility_calculator.AccessibilityCalculator(
        joined_pop_gdf.reset_index(drop=True),
        joined_sta_gdf,
        d0=d0,
        supply_col=supply_col
    )
    # Choose the accessibility method based on user input
    if method.lower() == '2sfca':
        calculator.calculate_accessibility_2sfca()
    elif method.lower() == 'm2sfca':
        calculator.calculate_accessibility_m2sfca()
    else:
        raise ValueError("Invalid method. Please choose '2sfca' or 'm2sfca'.")
    
    # Optionally visualize results
    if plot:
        plot_accessibility_with_stations(calculator.grids_gdf, calculator.station_gdf)
        plot_population_with_stations(calculator.grids_gdf, calculator.station_gdf)
    
    # Summarize outputs
    summary_accessibility = calculator.summarize_accessibility()
    gini = calculator.summarize_gini_coefficient()

    print(f"{supply_col} City_level Acc: {summary_accessibility}")
    print(f"{supply_col} Gini Coef.: {gini}")

    return summary_accessibility, gini


def overlay_tiff(output_tif_directory: str, tiff_file_after_clipped: str, stations_file: str, country_file: str, d0: float, year: int, method='m2sfca'): 
    """
    Overlay population raster data with EV charging station locations and compute accessibility
    metrics for each city using the specified accessibility model (2SFCA or M2SFCA)
    
    Parameters
    ----------
    :param output_tif_directory : Directory of clipped population TIFF file.
    :param tiff_file_after_clipped : Name of the clipped population raster file.
    :param stations_file : File path to the EV charging station dataset (GeoJSON, Shapefile, etc.).
    :param country_file : File path to the administrative boundary file containing cities or regions.
    :param d0 : Search radius (distance threshold) used in the accessibility model.
    :param year : Year of analysis.
    :param method : Accessibility model to use ('2sfca' or 'm2sfca'). Default is 'm2sfca'.
    
    """
    print("Step2: Start Overlaying ...")

    # Create an output directory named by the given year
    output_dir = os.path.join(os.getcwd(), "output", str(year))
    os.makedirs(output_dir, exist_ok=True)

    # Read population raster (TIFF) data
    population_data, population_transform, population_crs = read_population_data(
        os.path.join(output_tif_directory, tiff_file_after_clipped))
    print('Reading population TIFF dataset finished.')

    # Convert raster population data to a GeoDataFrame
    population_gdf = create_population_gdf(population_data, population_transform, population_crs)
    print('Creating population GeoDataFrame finished.')
    
    # Read EV charging station data and convert to target CRS
    stations_gdf = read_stations_data(stations_file, population_crs)
    print('Reading EVCS data finished.')
    
    # Read country/city boundary data (used for spatial join)
    country_gdf = gpd.read_file(country_file, engine='pyogrio')[['NAME_1', 'NAME_2', 'GID_1', 'GID_2', 'geometry']]
    print('Reading city boundary data finished.')
    
    # Define EVSE (charging equipment) types to process
    evse_types = [
        ('evse_num', 'evse_count'),
        ('ev_dc_fast_num', 'dc_evse_count'),
        ('ev_level1_evse_num', 'l1_evse_count'),
        ('ev_level2_evse_num', 'l2_evse_count'),
    ]
    # Create empty dictionaries to store accessibility and Gini results
    all_accessibility = {col: pd.DataFrame() for _, col in evse_types}
    all_gini = {col: pd.DataFrame() for _, col in evse_types}
    
    # Loop through each city and perform accessibility calculation
    for city in tqdm(list(country_gdf['GID_1'].unique()), desc="Processing cities"):
        city_boundary_gdf = country_gdf[country_gdf['GID_1'] == city]
        joined_pop_gdf = gpd.sjoin(population_gdf, city_boundary_gdf, how='inner')
        joined_sta_gdf = gpd.sjoin(stations_gdf, city_boundary_gdf, how='inner')

        print(f"City: {city}, EVSE Count: {joined_sta_gdf['evse_num'].sum()}, Population: {joined_pop_gdf['population'].sum()}")
        
        # Process each EVSE type (e.g., all, DC fast, AC Level 1, AC Level 2)
        for supply_col, suffix in evse_types:
            summary, gini_summary = process_evse_type(
                city, joined_pop_gdf, joined_sta_gdf, supply_col, d0=d0, plot=False, method='m2sfca')
            all_accessibility[suffix] = pd.concat([all_accessibility[suffix], summary], ignore_index=True)
            all_gini[suffix] = pd.concat([all_gini[suffix], gini_summary], ignore_index=True)

    # Save all results as CSV files for the current year
    for _, suffix in evse_types:
        all_accessibility[suffix].to_csv(os.path.join(output_dir, f'{year}_city_accessibility_summaries_{suffix}.csv'), index=False)
        all_gini[suffix].to_csv(os.path.join(output_dir, f'{year}_city_gini_summaries_{suffix}.csv'), index=False)

    print("-" * 50)
