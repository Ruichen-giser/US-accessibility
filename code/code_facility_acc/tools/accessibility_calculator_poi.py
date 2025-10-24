# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 21:44:09 2025

@author: Xingjun Huang & Ruichen MA
"""


import numpy as np
import pandas as pd
import gc
from tools import global_config


def calculate_gini(population, accessibility):
    """
    Calculate the Gini coefficient for accessibility distribution.
    
    Parameters:
    - population: numpy array of population values for each grid
    - accessibility: numpy array of accessibility values for each grid
    
    """
    
    # If input arrays are empty, return 0
    if len(accessibility) == 0 or len(population) == 0:
        return 0
    
    # Filter out grid cells with zero population
    valid_mask = population > 0
    population = population[valid_mask]
    accessibility = accessibility[valid_mask]

    if len(accessibility) == 0 or len(population) == 0:
        return 0

    # If only one grid has non-zero accessibility, maximum inequality
    if np.count_nonzero(accessibility) == 1:
        return 1.0

    sorted_indices = np.argsort(accessibility)
    sorted_population = population[sorted_indices]
    sorted_accessibility = accessibility[sorted_indices]

    cum_population = np.cumsum(sorted_population)
    cum_accessibility = np.cumsum(sorted_accessibility)

    if cum_accessibility[-1] == 0:
        return 0

    cum_population = cum_population / cum_population[-1]
    cum_accessibility = cum_accessibility / cum_accessibility[-1]

    gini = 1 - np.sum((cum_accessibility[1:] + cum_accessibility[:-1]) * \
                  (cum_population[1:] - cum_population[:-1]))

    return max(0, min(gini, 1))


class AccessibilityCalculator_poi:
    def __init__(self,poi_count_type, d0, supply_type):
        self.d0 = d0
        self.supply_type = supply_type
        self.poi_count_type = poi_count_type 

    @staticmethod
    def calc_accessibility_single(args):
        """
        Calculate accessibility for a single POI type for a batch of grid cells.
        
        """
        supplies, batch_size, num_grids, poi_type,grid_gdf,W = args
        
        # # Extract population/demand for the specified POI type
        population = grid_gdf[poi_type].values.astype(np.float32)
        total_demands = W.T @ population 
        total_demands = np.where(total_demands > 0, total_demands, 1).astype(np.float32) 

        # D_ij = (supply_j * W_ij) / total_demand_j
        D = (W * supplies.reshape(1, -1)) / total_demands.reshape(1, -1)
        del total_demands,population
        gc.collect()

        # A_i = sum_j D_ij * W_ij
        accessibility_scores = np.sum(D * W, axis=1).astype(np.float32)
        del D
        gc.collect()

        # Multiply accessibility scores by population for population-weighted accessibility
        pop_accessibility = accessibility_scores * grid_gdf[poi_type].values
        return (poi_type, accessibility_scores, pop_accessibility)


    def calculate_accessibility_m2sfca(self,num_grids):
        """
        Calculate accessibility for each grid cell using the modified 2-step floating catchment area (m2SFCA) method.
        This method directly updates the global grid GeoDataFrame with accessibility and population-weighted accessibility.
        
        Parameters:
        - num_grids: total number of grid cells to process
        
        """

        batch_size = 1000  # Process 1000 grids per batch
        
        # Extract supply values for the specified station type
        supplies = global_config.station_gdf[self.supply_type].values.astype(np.float32)  # shape: (M,)
        args_list = [
            (supplies, batch_size, num_grids, poi_type, global_config.grid_gdf, global_config.W)
            for poi_type in self.poi_count_type
        ]
        
        # Parallel computation using multiprocessing
        results = []
        for args in args_list:
            result = self.calc_accessibility_single(args)
            results.append(result)

        for poi_type, accessibility_scores, pop_accessibility in results:
            accessibility_col = "accessibility_" + poi_type
            pop_accessibility_col = "pop_" + accessibility_col
            global_config.grid_gdf[accessibility_col] = pd.Series(accessibility_scores).fillna(0).astype(np.float32)
            global_config.grid_gdf[pop_accessibility_col] = pd.Series(pop_accessibility).fillna(0).astype(np.float32) 
    
        return True
    
    
    def summarize_gini_coefficient(self):
        """
        Compute the Gini coefficient for each city, including both overall and per-category POIs.
        """
    
        result = global_config.grid_gdf.groupby(['GID_2']).apply(
            lambda group: pd.Series({
                f'Gini_Coefficient_{poi_count_col if poi_count_col != "poi_count" else ""}': 
                calculate_gini( group[poi_count_col].values, group["pop_" + "accessibility_" + poi_count_col].values)
                for poi_count_col in self.poi_count_type
            })
        ).reset_index()
        result = result.rename(columns={'Gini_Coefficient_': 'Gini_Coefficient'})
        return result    
    
    
    def summarize_accessibility(self):
        """
        Compute the average per-capita accessibility for each city, including overall and per-category POIs.
        """
    
        def calculate_average_accessibility(group, poi_count_col):
            total_poi_count = group[poi_count_col].sum()
            if total_poi_count == 0: 
                return 0
            return (group["accessibility_" + poi_count_col] * group[poi_count_col]).sum() * 100000 / total_poi_count
    
        result = global_config.grid_gdf.groupby(['GID_2']).apply(
            lambda group: pd.Series({
                f'Average_Accessibility {"" if poi_count_col == "poi_count" else "_" + poi_count_col}': 
                calculate_average_accessibility(group, poi_count_col)
                for poi_count_col in self.poi_count_type
            })
        ).reset_index()
        return result


    def summarize_spatial__accessibility(self):
        """
        Compute the overall spatial accessibility level for each city.
        """
        #  Check if the grid GeoDataFrame is empty
        if self.grids_gdf.empty:
            # If empty, return an empty DataFrame with expected columns
            return pd.DataFrame(columns=['GID_2', 'Average_Accessibility'])
        
        # Group grid cells by city (GID_2) and compute average spatial accessibility
        city_spatial_accessibility_summary = self.grids_gdf.groupby(['GID_2']).apply(
            lambda group: (group[self.pop_accessibility_col].sum() / group[self.pop_accessibility_col].count())
        ).reset_index(name='Average_Accessibility')
        
        return city_spatial_accessibility_summary
    