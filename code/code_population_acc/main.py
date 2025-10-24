# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 13:56:07 2025

@author:  Ruichen MA
"""

from tools import clip_tool, overlay_tool_thread
from pathlib import Path
import os

if __name__ == "__main__":
    
    # main.py directory
    current_dir = Path(__file__).resolve().parent
    
    # set the study period of population-based acc. and equity
    for year in range(2014, 2024):
        
        """
        load data
        """
        
        # usa pop dataset
        input_tif_directory = current_dir.parent.parent / 'data' /'US-WorldPOP-2014-2020' /'raw'
        
        # clipped usa pop dataset
        output_tif_directory = current_dir.parent.parent / 'data' /'US-WorldPOP-2014-2020'/'clipped'
        
        # usa boundary
        geojson_path = current_dir.parent.parent / 'data' /'US-map'/ 'usa_map.geojson'
        
        # fix tiff year if beyond 2020 as Worldpop supports data before 2021.
        tiff_year = min(year, 2020)
    
        # specific pop tiff 2014-2020 (use 2020 if year > 2020)
        tiff_file = f'usa_ppp_{tiff_year}_1km_Aggregated_UNadj.tif'
        
        # clipped pop tiff
        tiff_file_after_clipped = f'usa_ppp_{tiff_year}_1km_Aggregated_UNadj_clipped_usa.tif'
    
        # EVCS dataset (always use current year)
        stations_file =current_dir.parent.parent / 'data' /'US-EV-Station-2014-2024'/f'alt_fuel_stations_historical_day (Dec 31 {year}).geojson'
    
        """
        population-based accessibility and equity calculation procedure
        """
        # Step1: clip tiff file
        tiff_clipped_path = os.path.join(output_tif_directory,
                                         tiff_file_after_clipped)
        
        # Check whether the US boundary has been clipped; if so, skip the clipping step to excute step2.
        if not os.path.exists(tiff_clipped_path):
            print(f"Clipped file not found for year {year}, performing clip...")
            clip_tool.clip_tiff(input_tif_directory, output_tif_directory, geojson_path, tiff_file)
        else:
            print(f"Clipped file already exists for year {year}, skipping clip.")
    
        # Step2: overlay pop and EVCS 
        results = overlay_tool_thread.overlay_tiff( output_tif_directory,
                                                    tiff_file_after_clipped,
                                                    stations_file,
                                                    geojson_path,
                                                    d0=1,
                                                    year=year,
                                                    method='m2sfca'
                                                    )


