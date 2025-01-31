# Python LiDAR tools

Python LiDAR tools

# Installation

## Requirements

- [Python](https://www.python.org/)

## Setup (Windows)

1. Open a Windows PowerShell
2. Navigate to the directory where you want to store the project:

   ```
   cd PATH
   ```

3. Clone the github repository with:

   ```
   git clone https://github.com/sitn/py-lidar-tools.git
   ```

4. Navigate into the project directory:

   ```
   cd .\py-lidar-tools\
   ```

5. Create a new python virtual environment:

   ```
   python -m venv venv
   ```

This creates a `venv` folder inside the project directory.

6. Activate the virtual environnement:

   ```
   .\venv\Scripts\activate
   ```

7. Install the depenencies:

   ```
   pip install -r requirements.txt
   ```

## Usage

1. Within the python venv, navigate to the `scripts` directory:

   ```
   cd .\scripts\
   ```

2. Open the `config.yaml` file and adjust the parameters as needed.

3. Run the scripts according to the desired workflow

## Worflows

### Tree top detection

1. Run `raster_tiling.py` to create a set of raster tiles with overlap (to avoid edge effects)
2. Run `detect_tree_tops.py` to detect the tree tops from the canopy height model tiles
3. Run `merge_` to merge the results into a single file


## Overview

| Script                | Description                                                                    |
| --------------------- | ------------------------------------------------------------------------------ |
| `raster_tiling.py`    | Creates a raster tile set from a vector tile index and raster source directory |
| `detect_tree_tops.py` | Detects tree top locations in a raster canopy height model                     |
