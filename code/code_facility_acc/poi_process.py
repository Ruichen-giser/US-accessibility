# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 09:18:45 2025

@author: Xingjun Huang
"""

import pandas as pd
from shapely.geometry import Point 
import os
import geopandas as gpd
from tqdm import tqdm
import transbigdata as tbd
import time
import glob
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
pd.set_option('display.max_columns', None) 

# Process complex Parquet data and duplicate rows according to their Purpose
def process_batch(batch: pd.DataFrame, reference_dict: dict) -> pd.DataFrame:
    results = []

    for _, row in batch.iterrows():
        unique_purposes = {}
        for label in row["fsq_category_ids"]:
            # Look up the Purpose from the reference dictionary
            purpose = reference_dict.get(label)
            # If the purpose exists and is not duplicated, add it to unique_purposes
            if purpose and purpose not in unique_purposes.values():
                unique_purposes[label] = purpose

        # Generate results based on the number of unique purposes
        if len(unique_purposes) == 1:
            # Only one unique purpose — create a single row
            fsq_category_id = list(unique_purposes.keys())[0]
            results.append({**row, "fsq_category_ids_merged": fsq_category_id})
        else:
            # Multiple unique purposes — create one row per purpose
            for fsq_category_id in unique_purposes.keys():
                results.append({**row, "fsq_category_ids_merged": fsq_category_id})

    # Convert the results list to a DataFrame
    return pd.DataFrame(results)


# Define a function to extract the part before '>'
def extract_category(label_list):
    return label_list[0].split('>')[0]


# Define a function to process a single POI Parquet file
def process_file(file):
    """
    Read and process a single Parquet file, returning a filtered DataFrame.
    """
    try:
        # Read the Parquet file
        df = pd.read_parquet(file)

        # Keep only the specified columns and filter by country (U.S.)
        df = df.loc[df["country"] == "US", columns_to_keep]

        # Drop rows with missing latitude, longitude, or category information
        df = df.dropna(subset=["latitude", "longitude", "fsq_category_labels", "fsq_category_ids"])

        # # Optionally, extract the main category label and store it in a new column
        # df["category_label"] = df["fsq_category_labels"].apply(extract_category)

        print(f"Processed: {file}")
        return df
    except Exception as e:
        print(f"Error processing {file}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame to avoid merge errors


# Use a thread pool to process files in parallel
def process_files_in_parallel(files):
    """
    Process multiple Parquet files in parallel and return a combined DataFrame.
    """
    filtered_dataframes = []

    with ThreadPoolExecutor() as executor:
        # Submit each file to the thread pool for processing
        future_to_file = {executor.submit(process_file, file): file for file in files}

        # Collect results as they complete
        for future in as_completed(future_to_file):
            filtered_dataframes.append(future.result())

    # Combine all processed DataFrames into one
    combined_df = pd.concat(filtered_dataframes, ignore_index=True)
    return combined_df


def merge_geojsons(input_folder, output_file):
    """
    Merge all GeoJSON files within a specified folder.
    
    :param input_folder: Path to the input folder containing GeoJSON files.
    :param output_file: Path to the output merged GeoJSON file.
    """
    # Get all GeoJSON files in the folder
    all_files = [f for f in os.listdir(input_folder)
                 if f.endswith('.geojson') or f.endswith('.json')]

    # Store all GeoDataFrames
    gdf_list = []

    # Read each file with a progress bar
    for filename in tqdm(all_files, desc="Merging files"):
        file_path = os.path.join(input_folder, filename)
        try:
            # Read geographic data
            gdf = gpd.read_file(file_path)

            # Optionally, add a column recording the source file name
            gdf['source_file'] = filename

            gdf_list.append(gdf)
        except Exception as e:
            print(f"\nError: Unable to read file {filename} - {str(e)}")
            continue

    if not gdf_list:
        raise ValueError("No GeoJSON files found for merging.")

    # Merge all GeoDataFrames
    merged_gdf = gpd.GeoDataFrame(
        pd.concat(gdf_list, ignore_index=True),
        crs=gdf_list[0].crs  # Inherit CRS from the first file
    )

    # Save the merged result
    merged_gdf.to_file(output_file, driver='GeoJSON')
    print(f"\nMerging completed! Output file saved to: {output_file}")


# Extract the primary category ID
def extract_category_id(label_list):
    if len(label_list) == 1:
        return label_list[0]  # Return the single ID in the list
    return None  # Return None if the list is empty or invalid



def convert_to_geoparquet(input_parquet, output_geoparquet, 
                          lat_col='latitude', lon_col='longitude', 
                          crs="EPSG:4326"):
    """
    Convert a standard Parquet file into a GeoParquet file with point geometries.
    """
    # Read the input Parquet file
    df = pd.read_parquet(input_parquet)

    # Check whether latitude and longitude columns exist
    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError(f"Columns '{lat_col}' and '{lon_col}' must be present in the input file.")
    
    # Create point geometries (ensure correct coordinate order: longitude, latitude)
    df['geometry'] = df.apply(
        lambda row: Point(row[lon_col], row[lat_col]), 
        axis=1
    )
    
    # Convert to a GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=crs)
    
    # Important: specify encoding method when saving to ensure bbox query compatibility
    gdf.to_parquet(output_geoparquet, engine='pyarrow')
    
    return gdf



if __name__ == "__main__":
    
    """
    This section contains four code blocks:
        1.Rasterize the U.S. boundary into a 1×1 km grid.
        2.Extract each year’s POI data from Foursquare and save as GeoParquet.
        3.For each year’s POI data, perform data cleaning by class: split one case into one or more cases based on class.
        4.For each year’s POI data, perform data cleaning by purpose: split one case into one or more cases based on purpose.
    """
    
    
    # -------------------------------------------------------------------------------------------------------------
    # Rasterize the U.S. boundary into a 1×1 km grid.
    # -------------------------------------------------------------------------------------------------------------
    # main.py directory
    current_dir = Path(__file__).resolve().parent
    usa_map = gpd.read_file(current_dir.parent.parent / 'data' /'US-map'/ 'usa_map.geojson')
    county_gdf = usa_map[['NAME_1', 'NAME_2', 'GID_2', 'geometry']]
    # grid_rec, params_rec = tbd.area_to_grid(county_gdf, accuracy=1000)
    del usa_map

    # Process the data by county to avoid memory overflow when handling the entire U.S. at once
    for county in county_gdf['NAME_2'].unique():
        if os.path.exists(current_dir.parent.parent / 'data' /'US-map'/ 'grid_us_1km_county' / f'{county}_grid_1km.geojson'):
            continue  # Skip if the county grid file already exists

        # Generate 1 km × 1 km grid cells for the current county
        county_grid, params_rec = tbd.area_to_grid(
            county_gdf[county_gdf['NAME_2'] == county],
            accuracy=1000
        )

        # Save the generated grid as a GeoJSON file
        county_grid.to_file(
            current_dir.parent.parent / 'data' /'US-map'/ 'grid_us_1km_county' / f'{county}_grid_1km.geojson',
            driver="GeoJSON"
        )
        del county_grid, params_rec

    # Define input and output paths
    input_folder = current_dir.parent.parent / 'data' /'US-map'/ 'grid_us_1km_county'         # Folder containing per-county grid files
    output_file = current_dir.parent.parent / 'data' /'US-map'/ 'merged_usa_grid_1km1.geojson' # Output path for the merged national grid

    # Merge all county-level GeoJSON grids into one national grid
    merge_geojsons(input_folder, output_file)



    # ---------------------------------------------------------------------------------------------------------------
    # Extract each year’s POI data from Foursquare and save as GeoParquet.
    # -------------------------------------------------------------------------------------------------------------
    # Specify the directory containing Parquet files
    directory_path = current_dir.parent.parent / 'data'   # Replace with your actual path
    
    # Find all Parquet files in the directory
    parquet_files = glob.glob(f"{directory_path}*.parquet")
    
    # Optionally, select only a subset of files (currently using all)
    selected_files = parquet_files
    
    # Specify the columns to keep
    columns_to_keep = [
        "latitude", "longitude", "fsq_category_ids", "fsq_category_labels", "country", "region",
        "date_created", "date_closed"
    ]
    
    # Process all Parquet files in parallel
    combined_df = process_files_in_parallel(selected_files)
    
    # Ensure 'date_created' and 'date_closed' are in string (date) format
    combined_df["date_created"] = combined_df["date_created"].astype(str)
    combined_df["date_closed"] = combined_df["date_closed"].fillna("9999-12-31").astype(str)
    
    # Extract all unique years from 'date_created'
    unique_years = sorted(combined_df["date_created"].str[:4].unique())
    
    # Filter and save data for each year
    for year in range(2014, 2025):
        # Select POIs that were active during the given year
        filtered_gdf = combined_df[
            (combined_df["date_created"] < f"{year}-12-31") &  # Created before the end of the year
            (combined_df["date_closed"] > f"{year}-01-01")     # Closed after the beginning of the year
        ]
    
        # Save the filtered data as a Parquet file
        output_path = current_dir.parent.parent / 'data' /'US-poi-2003-2025'/ f'data_{year}.parquet'
        filtered_gdf.to_parquet(output_path, index=False)
        print(f"Saved data for year {year} to {output_path}")
    
    # -------------------------------------------------------------------------------------------------------------
    # Convert data_year.parquet files into GeoParquet format
    # -------------------------------------------------------------------------------------------------------------
    input_dir = current_dir.parent.parent / 'data' /'US-poi-2003-2025'
    parquet_files = [os.path.join(input_dir, f"data_{year}.parquet") for year in range(2015, 2025)]
    geojson_files = [os.path.join(input_dir, f"data_{year}_geoparquet.parquet") for year in range(2015, 2025)]
    os.makedirs(input_dir, exist_ok=True)
    
    # Pair each Parquet file with its corresponding GeoParquet output
    zipfiles = list(zip(parquet_files, geojson_files))
    
    # Convert all Parquet files to GeoParquet with latitude and longitude geometry
    for input_parquet, output_geoparquet in tqdm(zipfiles, desc="Processing parquet files"):
        convert_to_geoparquet(
            input_parquet, output_geoparquet,
            lat_col='latitude', lon_col='longitude',
            crs="EPSG:4326"
        )




    # -------------------------------------------------------------------------------------------------------------
    # Data cleaning for POI "class" classification
    # -------------------------------------------------------------------------------------------------------------
    
    # Define input and reference file paths
    input_dir = current_dir.parent.parent / 'data' / 'US-poi-2003-2025'
    reference_path = current_dir / 'poi_result.csv'
    
    # Load reference dataset (category definitions and manual filters)
    reference_type = pd.read_csv(reference_path, encoding="ISO-8859-1")
    reference_type.rename(columns={"Category ID": "fsq_category_ids_merged"}, inplace=True)
    
    # Get Parquet file paths for the years of interest (here only 2014)
    parquet_files = [os.path.join(input_dir, f) for f in [f"data_{year}.parquet" for year in range(2014, 2015)]]
    os.makedirs(input_dir, exist_ok=True)
    
    # -------------------------------------------------------------------------------------------------------------
    # Loop through each Parquet file for cleaning and classification
    # -------------------------------------------------------------------------------------------------------------
    for file_path in tqdm(parquet_files, desc="Processing parquet files"):
        start_time = time.perf_counter()
        parquet_data = pd.read_parquet(file_path)
    
        # Extract the main (merged) category ID
        parquet_data["fsq_category_ids_merged"] = parquet_data["fsq_category_ids"].apply(extract_category_id)
        
        # -------------------------------
        # Step 1: Process simple cases
        # -------------------------------
        simple_parquet = parquet_data[parquet_data["fsq_category_ids_merged"].notna()]
        simple_parquet = pd.merge(simple_parquet, reference_type, on="fsq_category_ids_merged", how="inner")
        simple_parquet = simple_parquet[simple_parquet['Filter_manual'] == 1]
        print("Simple processing finished")
        
        # -------------------------------
        # Step 2: Process complex cases
        # -------------------------------
        complex_parquet = parquet_data[parquet_data["fsq_category_ids_merged"].isna()]
        reference_dict = reference_type.set_index("fsq_category_ids_merged")["POI Class_manual"].to_dict()
        
        # Split data into smaller batches for parallel processing
        batch_size = 1000
        batches = [
            complex_parquet.iloc[i:i + batch_size]
            for i in range(0, len(complex_parquet), batch_size)
        ]
        
        # Process batches in parallel using ThreadPoolExecutor
        results = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_batch, batch, reference_dict) for batch in batches]
            results = [future.result() for future in futures]
        
        # Combine all processed batches
        complex_parquet = pd.concat(results, ignore_index=True)
        complex_parquet = pd.merge(complex_parquet, reference_type, on="fsq_category_ids_merged", how="inner")
        complex_parquet = complex_parquet[complex_parquet['Filter_manual'] == 1]
        
        # -------------------------------
        # Step 3: Merge simple and complex results
        # -------------------------------
        merged_data = pd.concat([simple_parquet, complex_parquet], ignore_index=True)
        print("Data merging finished")
        
        # -------------------------------
        # Step 4: Extract category labels and save as GeoParquet
        # -------------------------------
        merged_data["category_label"] = merged_data["fsq_category_labels"].apply(extract_category)
        merged_data.drop(
            columns=["fsq_category_ids", "fsq_category_labels", "region", "Category Label", "Filter_manual"],
            inplace=True
        )
        merged_data.rename(columns={"fsq_category_ids_merged": "fsq_category_ids"}, inplace=True)
        
        # Define output file path
        output_path = os.path.join(
            input_dir,
            os.path.basename(file_path).replace(".parquet", "_poi_class_geoparquet.parquet")
        )
        
        # Create point geometry and save as GeoParquet
        geometry = gpd.points_from_xy(merged_data["longitude"], merged_data["latitude"], crs="EPSG:4326")
        merged_data = gpd.GeoDataFrame(merged_data, geometry=geometry)
        merged_data[['latitude', 'longitude', 'POI Class_manual', 'geometry']].to_parquet(output_path, engine='pyarrow')
        print(f"Processed {file_path} -> {output_path}")
        
        # Log processing time
        end_time = time.perf_counter()
        elapsed_time_minutes = (end_time - start_time) / 60
        print(f"Elapsed time in minutes: {elapsed_time_minutes:.2f} minutes")





    # -------------------------------------------------------------------------------------------------------------
    # Data cleaning for POI "purpose" classification
    # -------------------------------------------------------------------------------------------------------------
    
    # Define input and reference file paths
    input_dir = current_dir.parent.parent / 'data' / 'US-poi-2003-2025'
    reference_path = current_dir / 'poi_result.csv'
    
    # Load reference dataset (category definitions and manual filters)
    reference_type = pd.read_csv(reference_path, encoding="ISO-8859-1")
    reference_type.rename(columns={"Category ID": "fsq_category_ids_merged"}, inplace=True)
    
    # Get all yearly Parquet file paths
    parquet_files = [os.path.join(input_dir, f) for f in [f"data_{year}.parquet" for year in range(2014, 2025)]]
    os.makedirs(input_dir, exist_ok=True)
    
    # -------------------------------------------------------------------------------------------------------------
    # Loop through each year's dataset for cleaning and classification
    # -------------------------------------------------------------------------------------------------------------
    for file_path in tqdm(parquet_files, desc="Processing parquet files"):
        start_time = time.perf_counter()
        parquet_data = pd.read_parquet(file_path)
        
        # Extract the main (merged) category ID
        parquet_data["fsq_category_ids_merged"] = parquet_data["fsq_category_ids"].apply(extract_category_id)
        
        # -------------------------------
        # Step 1. Process simple cases
        # -------------------------------
        # Select rows with valid category IDs and merge with reference data
        simple_parquet = parquet_data[parquet_data["fsq_category_ids_merged"].notna()]
        simple_parquet = pd.merge(simple_parquet, reference_type, on="fsq_category_ids_merged", how="inner")
        simple_parquet = simple_parquet[simple_parquet['Filter_manual'] == 1]
        print("Simple processing finished")
        
        # -------------------------------
        # Step 2. Process complex cases
        # -------------------------------
        # Handle records with missing or multiple category IDs
        complex_parquet = parquet_data[parquet_data["fsq_category_ids_merged"].isna()]
        reference_dict = reference_type.set_index("fsq_category_ids_merged")["Purpose_manual"].to_dict()
        
        # Split into smaller batches for parallel processing
        batch_size = 1000
        batches = [
            complex_parquet.iloc[i:i + batch_size]
            for i in range(0, len(complex_parquet), batch_size)
        ]
        
        # Process batches in parallel using ThreadPoolExecutor
        results = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_batch, batch, reference_dict) for batch in batches]
            results = [future.result() for future in futures]
        
        # Combine all processed batches
        complex_parquet = pd.concat(results, ignore_index=True)
        complex_parquet = pd.merge(complex_parquet, reference_type, on="fsq_category_ids_merged", how="inner")
        complex_parquet = complex_parquet[complex_parquet['Filter_manual'] == 1]
        
        # Merge simple and complex results
        merged_data = pd.concat([simple_parquet, complex_parquet], ignore_index=True)
        print("Data merging finished")
        
        # -------------------------------
        # Step 3. Extract and save results
        # -------------------------------
        # Extract category labels and remove unnecessary columns
        merged_data["category_label"] = merged_data["fsq_category_labels"].apply(extract_category)
        merged_data.drop(
            columns=["fsq_category_ids", "fsq_category_labels", "region", "Category Label", "Filter_manual"],
            inplace=True
        )
        merged_data.rename(columns={"fsq_category_ids_merged": "fsq_category_ids"}, inplace=True)
        
        # Define output file path
        output_path = os.path.join(
            input_dir,
            os.path.basename(file_path).replace(".parquet", "_purpose_geoparquet.parquet")
        )
        
        # Create point geometry and save as GeoParquet
        geometry = gpd.points_from_xy(merged_data["longitude"], merged_data["latitude"], crs="EPSG:4326")
        merged_data = gpd.GeoDataFrame(merged_data, geometry=geometry)
        merged_data[['latitude', 'longitude', 'Purpose_manual', 'geometry']].to_parquet(output_path, engine='pyarrow')
        print(f"Processed {file_path} -> {output_path}")
        
        # Log processing time
        end_time = time.perf_counter()
        elapsed_time_minutes = (end_time - start_time) / 60
        print(f"Elapsed time in minutes: {elapsed_time_minutes:.2f} minutes")
    


