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
    
    start_time = time.perf_counter()
    
    """
    Facility-based Acc. by trip purpose.
    
    poi_count_P20: School/Daycare/Religious
    poi_count_P30: Medical/Dental Services
    poi_count_P40: Shopping/Errands
    poi_count_P50: Social/Recreational 
    poi_count_P80: Meal
    
    """
    poi_count_type = [
        "poi_count", "poi_count_P20", "poi_count_P30",
        "poi_count_P40", "poi_count_P50", "poi_count_P80"]
        
    # charger types
    cs_type = ['evse_num', 'ev_dc_fast_num','ev_level1_evse_num', 'ev_level2_evse_num']

    # labels for facility types
    category_labels = {
        "20": "P20",
        "30": "P30",
        "40": "P40",
        "50": "P50",
        "80": "P80"
    }
    
    for year in range(2014,2025):
    #for year in [2017]: # code if you want to run a single year
        d0 = 1 # catchment size (km) 
        print(f"{year}: {d0} for poi purpose")
        
        # poi dataset
        poi_file = current_dir.parent.parent / 'data' /'US-poi-2014-2024' / f'data_{year}_purpose_geoparquet.parquet'
        
        # EVCS dataset
        stations_file = current_dir.parent.parent / 'data' /'US-EV-Station-2014-2024'/f'alt_fuel_stations_historical_day (Dec 31 {year}).geojson'
        
        # usa boundary 
        usa_map_path = current_dir.parent.parent / 'data' /'US-map'/ 'usa_map.geojson'
        usa_grid_1km_path = current_dir.parent.parent / 'data' /'US-map'/ 'merged_usa_grid_1km.gpkg' 
        global_config.usa_map_gdf = gpd.read_file(usa_map_path, engine='pyogrio', columns=['NAME_1', 'NAME_2', 'GID_2', 'geometry'])
        print('reading USA boundary data finished!')
        
        # Acc and equity calculation
        POITYPE = "Purpose_manual" # "POI","POI Class_manual","Purpose_manual"
        result = overlay_tool_thread_poi.overlay_tiff_poi(d0, poi_file, usa_grid_1km_path, stations_file, poi_count_type, category_labels, cs_type,POITYPE)
    
        # Define the output directory based on distance threshold
        if d0 == 0.25:
            output_dir = f"./results/POI_resutls/{year}_poi_purpose_250m"
        else:
            output_dir = f"./results/POI_resutls/{year}_poi_purpose_{d0}km"
            
        os.makedirs(output_dir, exist_ok=True)
        
        # save accessibility_summaries
        for supply_type, summary in result["accessibility_summaries"].items():
            filepath = os.path.join(output_dir, f"{year}_accessibility_summary_{supply_type}.csv")
            summary.to_csv(filepath, index=False)
            print(f"{supply_type} saved to: {filepath}")
        
        # save gini_summaries
        for supply_type, gini_summary in result["gini_summaries"].items():
            filepath = os.path.join(output_dir, f"{year}_gini_summary_{supply_type}.csv")
            gini_summary.to_csv(filepath, index=False)
            print(f"{year}_{supply_type} saved to: {filepath}")

        end_time = time.perf_counter()
        elapsed_time_minutes = (end_time - start_time) / 60
    
        print(f"Elapsed time in minutes: {elapsed_time_minutes:.2f} minutes")

