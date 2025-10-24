# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 09:29:11 2025

@author: Xingjun Huang & Ruichen MA
"""

from tools import global_config, overlay_tool_thread_poi
import os
import time
import geopandas as gpd
import pandas as pd
pd.set_option('display.max_columns', None) 
from pathlib import Path

if __name__ == "__main__":
    
    # main.py directory
    current_dir = Path(__file__).resolve().parent
    
    # Start a timer to measure execution time
    start_time = time.perf_counter()
 
    # Define suffixes for POI columns, used to create DataFrame column names
    poi_count_type = ["poi_count"]

    # Define the types of charging stations and corresponding cumulative DataFrame
    cs_type = ['evse_num', 'ev_dc_fast_num','ev_level1_evse_num', 'ev_level2_evse_num']
    
    # usa boundary 
    usa_map_path = current_dir.parent.parent / 'data' /'US-map'/ 'usa_map.geojson'
    usa_grid_1km_path = current_dir.parent.parent / 'data' /'US-map'/ 'merged_usa_grid_1km.gpkg' 
    
    # Read USA boundary GeoDataFrame with specific columns
    global_config.usa_map_gdf = gpd.read_file(usa_map_path, engine='pyogrio', columns=['NAME_1', 'NAME_2', 'GID_2', 'geometry'])
    print('reading USA boundary data finished!')

    # Loop through years 2014 to 2024
    for year in range(2014,2025):
        print (f"{year}: main poi")
        
        # Define file paths for POI data and EV station data for the current year
        poi_file = current_dir.parent.parent / 'data' /'US-poi-2014-2024' / f'data_{year}_geoparquet.parquet'
        stations_file = current_dir.parent.parent / 'data' /'US-EV-Station-2014-2024'/f'alt_fuel_stations_historical_day (Dec 31 {year}).geojson'

        # Compute accessibility and equity
        d0=1 # catchment size (km)
        POITYPE = "POI"
        result = overlay_tool_thread_poi.overlay_tiff_poi(d0, poi_file, usa_grid_1km_path, stations_file, poi_count_type, None, cs_type,POITYPE)
    
        # Define the output directory based on distance threshold
        if d0 == 0.25:
            output_dir = current_dir.parent.parent / 'code' /'code_facility_acc'/ 'output' / f"POI_resutls_{year}_poi_250m"
        else:
            output_dir = current_dir.parent.parent / 'code' /'code_facility_acc'/ 'output' /f"POI_resutls_{year}_poi_{d0}km"

        os.makedirs(output_dir, exist_ok=True)
        
        # Save accessibility summaries to CSV files
        for supply_type, summary in result["accessibility_summaries"].items():
            filepath = os.path.join(output_dir, f"{year}_accessibility_summary_{supply_type}.csv")
            summary.to_csv(filepath, index=False)
            print(f"Accessibility summary for {supply_type} saved to: {filepath}")
        
        # Save Gini coefficient summaries to CSV files
        for supply_type, gini_summary in result["gini_summaries"].items():
            filepath = os.path.join(output_dir, f"{year}_gini_summary_{supply_type}.csv")
            gini_summary.to_csv(filepath, index=False)
            print(f"Gini summary for {year}_{supply_type} saved to: {filepath}")
        
        # Print elapsed time in minutes for the current iteration
        end_time = time.perf_counter()
        elapsed_time_minutes = (end_time - start_time) / 60
    
        print(f"Elapsed time in minutes: {elapsed_time_minutes:.2f} minutes")
