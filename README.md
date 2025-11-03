# US-accessibility

# Overview
This repository contains all codes and (sample) dataset of the paper - 
***Improved Equity of U.S. Public Electric Vehicle Charger System with Geospatial Disparities***.  

**Authors:** Ruichen Ma# , Xingjun Huang# , Xiong Yang, Mingjia He, Justin Hayse Chiwing G. Tang, Binru Wei, Chengxiang Zhuge*  

\# Co-first authors, * Corresponding author  

**Note:** Some data files are too large to upload to GitHub. Only a sample dataset is provided here. The **full dataset** can be requested separately through our [Global EV Data Initiative](https://globalevdata.github.io/data.html).

# Requirements and Installation
The whole calculation- and analysis-related codes should run with a **Python** environment, regardless of operating systems theoretically. 
More detailed info is as below:

## Prerequisites 
It is highly recommended to install and use the following versions of python/packages to run the codes:
- ``python``: 3.12.7
- ``numpy``: 1.26.4
- ``pandas``: 2.2.2
- ``matplotlib``: 3.9.2
- ``scipy``: 1.13.1
- ``geopandas``: 1.0.1
- ``pyproj``: 3.7.1
- ``time``: 3.12.7

## Project Structure
├── code  
│ ├── code_facility_acc/ # Code for facility-based accessibility and equity calculation  
│ ├── code_population_acc/ # Code for population-based accessibility and equity calculation  
│ └── code_visualization/ # Code for visualization  
│  
├── data # Full dataset for facility-based analysis (too large for GitHub)  
│ ├── US-accessibility  
│ │ ├── facility-based/ # Dataset for acc analysis by facility  
│ │ └── population-based/ # Dataset for acc analysis by population  
│ │  
│ ├── US-equity   
│ │ ├── facility-based/ # Dataset for equity analysis by facility  
│ │ └── population-based/ # Dataset for equity analysis by population  
│ │  
│ ├── US-EV-Station-2014-2024/ # Historical EV charging station data  
│ ├── US-map/ # US map GeoJSONs  
│ ├── US-poi-2014-2024/ # Points of Interest data  
│ └── US-WorldPOP-2014-2020/ # Population raster data  
│  
└── sample data/ # Small sample datasets for testing and demonstration (i.e. L.A., U.S.)  

  
# Contact
- Leave questions in [Issues on GitHub](https://github.com/Ruichen-giser/US-accessibility/issues)
- Get in touch with the Corresponding Author: [Dr. Chengxiang Zhuge](mailto:chengxiang.zhuge@polyu.edu.hk)
or visit our research group website: [The TIP](https://thetipteam.editorx.io/website) for more information

# License
This repository is licensed under the MIT License - see the [LICENSE](https://github.com/Ruichen-giser/US-accessibility/blob/main/LICENSE) file for details.

