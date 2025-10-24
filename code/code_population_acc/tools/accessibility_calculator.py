# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 21:44:09 2025

@author: Ruichen MA
"""

import numpy as np
from scipy.spatial.distance import cdist
from pyproj import Transformer
import gc

def calculate_gini(population, accessibility):
    """
    Calculate the Gini coefficient to measure inequality in accessibility distribution weighted by population.
    """
    
    # Return 0 if either population or accessibility arrays are empty
    if len(accessibility) == 0 or len(population) == 0:
        return 0
    
    # Filter out areas with zero population
    valid_mask = population > 0
    population = population[valid_mask]
    accessibility = accessibility[valid_mask]

    if len(accessibility) == 0 or len(population) == 0:
        return 0
    
    # Special case: if only one grid cell has nonzero accessibility
    # Return 0 if all population entries were filtered out
    if np.count_nonzero(accessibility) == 1:
        return 1.0

    # Sort values by accessibility in ascending order
    sorted_indices = np.argsort(accessibility)
    sorted_population = population[sorted_indices]
    sorted_accessibility = accessibility[sorted_indices]

    # Compute cumulative population and cumulative accessibility
    cum_population = np.cumsum(sorted_population)
    cum_accessibility = np.cumsum(sorted_accessibility)

    # Normalize cumulative population and accessibility (range 0–1)
    cum_population = cum_population / cum_population[-1]
    cum_accessibility = cum_accessibility / cum_accessibility[-1]

    # Compute the Gini coefficient using the trapezoidal rule (Lorenz curve area)
    gini = 1 - np.sum((cum_accessibility[1:] + cum_accessibility[:-1]) * \
                  (cum_population[1:] - cum_population[:-1]))

    # Ensure the result is within [0, 1] bounds
    return max(0, min(gini, 1))

class AccessibilityCalculator:
    def __init__(self, grids_gdf, station_gdf, d0, supply_col='evse_count'):
        
        """
        Initialize the accessibility calculator.
        
        Parameters
        ----------
        :param grids_gdf: Demand-side grid data.
        :param station_gdf: EV station data.
        :param d0:  Distance threshold (km).
        :param supply_col: Name of the supply field (e.g., 'evse_count').
        """
        
        self.grids_gdf = grids_gdf
        self.station_gdf = station_gdf
        self.d0 = d0
        self.supply_col = supply_col
        self.accessibility_col = 'Accessibility'
        self.pop_accessibility_col = 'Populated Accessibility'
        self.grids_gdf[self.accessibility_col] = 0.0
        self.grids_gdf[self.pop_accessibility_col] = 0.0

    @staticmethod
    def gaussian_decay(distance, d0):
        """Gaussian distance decay function."""
        norm_factor = (1 - np.exp(-0.5)).astype(np.float32)
        decay_values = np.where(distance <= d0,
                                ((np.exp(-0.5 * (distance / d0) ** 2) - np.exp(-0.5)) / norm_factor).astype(np.float32),
                                0).astype(np.float32)
        return decay_values.astype(np.float32)

    @staticmethod
    def project_coordinates(lats, lons, target_crs='EPSG:3857'):
        """Project latitude/longitude to target coordinate system."""
        transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
        xs, ys = transformer.transform(lons, lats)
        return np.vstack((xs, ys)).T.astype(np.float32)

    def calculate_accessibility_m2sfca(self):
        """
        Compute accessibility using the Modified 2-Step Floating Catchment Area (M2SFCA) method.
        """
        if self.grids_gdf.empty or self.station_gdf.empty:
            print("Warning: Empty grid or station data.")
            return None
    
        # Extract coordinates and variables
        station_coords = np.array([(geom.y, geom.x) for geom in self.station_gdf.geometry], dtype=np.float32)
        supplies = self.station_gdf[self.supply_col].values.astype(np.float32)  # shape: (M,)
        grid_centroids = [(geom.centroid.y, geom.centroid.x) for geom in self.grids_gdf.geometry]
        population = self.grids_gdf['population'].values.astype(np.float32)    # shape: (N,)
    
        # Project to planar coordinates (meters)
        station_proj = self.project_coordinates(*zip(*station_coords))  # shape: (M, 2)
        grid_proj = self.project_coordinates(*zip(*grid_centroids))     # shape: (N, 2)
    
        # Distance matrix (km)
        distances = cdist(grid_proj, station_proj, metric='euclidean').astype(np.float32) / 1000  # shape: (N, M)
    
        # Gaussian decay weights
        W = self.gaussian_decay(distances, d0=self.d0)
        W[distances > self.d0] = 0  # 超过阈值设为0
    
        # Step 1: Weighted demand per station
        weighted_demand = W.T @ population  # shape: (M,)
        weighted_demand = np.where(weighted_demand > 0, weighted_demand, 1.0).astype(np.float32)
    
        # Step 2: Supply-demand ratio
        D = (W * supplies.reshape(1, -1)) / weighted_demand.reshape(1, -1)  # shape: (N, M)
    
        # Step 3: Accessibility score per grid
        accessibility_scores = np.sum(D * W, axis=1).astype(np.float32)  # shape: (N,)
    
        # Store results
        self.grids_gdf[self.accessibility_col] = np.nan_to_num(accessibility_scores)
        self.grids_gdf[self.pop_accessibility_col] = self.grids_gdf[self.accessibility_col] * population
    
        del distances, W, D, weighted_demand, station_coords, station_proj, grid_proj, supplies
        gc.collect()
        return True
    
    def calculate_accessibility_2sfca(self):
        """
        Compute accessibility using the traditional 2-Step Floating Catchment Area (2SFCA) method.
        """
        if self.grids_gdf.empty or self.station_gdf.empty:
            print("Warning: Empty grid or station data.")
            return None

        station_coords = np.array([(geom.y, geom.x) for geom in self.station_gdf.geometry], dtype=np.float32)
        supplies = self.station_gdf[self.supply_col].values.astype(np.float32)

        station_proj = self.project_coordinates(*zip(*station_coords))

        grid_centroids = [(geom.centroid.y, geom.centroid.x) for geom in self.grids_gdf.geometry]
        grid_proj = self.project_coordinates(*zip(*grid_centroids))

        distances = cdist(grid_proj, station_proj, metric='euclidean').astype(np.float32) / 1000

        decay_matrix = self.gaussian_decay(distances, d0=self.d0)
        decay_matrix[distances > self.d0] = 0  # 超过阈值设为 0

        population = self.grids_gdf['population'].values.astype(np.float32)
        total_demands = decay_matrix.T @ population
        total_demands = np.where(total_demands > 0, total_demands, 1).astype(np.float32)

        R_j = (supplies / total_demands).astype(np.float32)
        accessibility_scores = (decay_matrix @ R_j).astype(np.float32)

        self.grids_gdf[self.accessibility_col] = np.nan_to_num(accessibility_scores).astype(np.float32)
        self.grids_gdf[self.pop_accessibility_col] = self.grids_gdf[self.accessibility_col] * population

        del distances, decay_matrix, total_demands, R_j, grid_proj, station_proj, station_coords, supplies
        gc.collect()
        return True
    
    # ---------- Summary Metrics ----------
    
    def summarize_gini_coefficient(self):
        """
        Compute Gini coefficient of accessibility for each city.
        """
        # 按城市分组
        city_gini_summary = self.grids_gdf.groupby(['NAME_1', 'NAME_2', 'GID_1', 'GID_2']).apply(
            lambda group: calculate_gini(group['population'].values, group[self.pop_accessibility_col].values)
        ).reset_index(name='Gini_Coefficient')
        
        return city_gini_summary

    def summarize_accessibility(self):
        """
        Compute population-weighted average accessibility for each city.
        """
        city_accessibility_summary = self.grids_gdf.groupby(['NAME_1','NAME_2','GID_1','GID_2']).apply(
            lambda group: (group[self.accessibility_col] * group['population']).sum() * 100000 / group['population'].sum()
        ).reset_index(name='Average_Accessibility')
        return city_accessibility_summary

    