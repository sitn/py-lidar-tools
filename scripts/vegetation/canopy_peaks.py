# -*- coding: utf-8 -*-
import numpy as np
from scipy.ndimage import gaussian_filter, grey_dilation, distance_transform_edt
from skimage.morphology import reconstruction
import matplotlib.pyplot as plt
import rasterio

def canopy_peaks(chm, transform, gaussian_filter_sigma=None, method='default', 
                 min_tree_height=2, search_radius=lambda h: 1, 
                 min_height_difference=0.1, fig=False, verbose=False):

    '''
    CANOPY_PEAKS - find local maxima coordinates in a raster Canopy Height Model (CHM).
    
    [CRH, XYH] = CANOPYPEAKS(CHM, REFMAT, METHOD, ...) computes candidate
    tree top coordinates in the image (CRH) and map (XYH) reference systems
    using the specified METHOD. Two methods are implemented:
    1. Search within a fixed or variable size circular window (based on allometry) convolution, see
    Popescu et al. (2004) [1] and Chen et al. (2006) [2].
    2. H-maxima transform, see Kwak et al. (2007) [3]
    
    [1] S. C. Popescu and R. H. Wynne, "Seeing the trees in the forest: Using lidar and 
    multispectral data fusion with local filtering and variable window size for estimating tree height,"
    Photogrammetric Engineering and Remote Sensing, vol. 70, no. 5, pp. 589–604, 2004.
    
    [2] Q. Chen, D. Baldocchi, P. Gong, and M. Kelly, "Isolating Individual Trees in a 
    Savanna Woodland Using Small Footprint Lidar Data," Photogrammetric Engineering & 
    Remote Sensing, vol. 72, no. 8, pp. 923–932, Aug. 2006.
    
    [3] D.-A. Kwak, W.-K. Lee, J.-H. Lee, G. S. Biging, and P. Gong, 
    "Detection of individual trees and estimation of tree height using LiDAR data,"
    J For Res, vol. 12, no. 6, pp. 425–434, Oct. 2007.
    
    Syntax:  [crh, xyh] = canopyPeaks(chm, refmat, method, ...)
    
    Inputs:
    - chm: 2D numpy array, raster Canopy Height Model (CHM)
    
    - transform - affine transform object (from rasterio), mapping between image and map coordinates
    
    - gaussian_filter_sigma - scalar (optional, default: None), standard deviation for the Gaussian kernel used to smooth the CHM
    
    - method: 'default' or 'hMaxima' (default: 'default'), peak detection method
    
    - min_tree_height: scalar (optional, default: 2.0), minimum tree top height
    
    - search_radius: anonymous function (optional, default: lambda h: 1),  function
      specifying the radius (as a function of height) for the circular convolution window used to detect local maxima. 
        Example models:
         @(h) (3.09632 + 0.00895 * h^2)/2  deciduous forest (Popescu et al, 2004)
         @(h) (3.75105 - 0.17919 * h + 0.01241 * h^2)/2 coniferous forests (Popescu et al, 2004)
         @(h) (2.51503 + 0.00901 * h^2)/2; mixed forests (Popescu et al, 2004)
         @(h) (1.7425 * h^0.5566)/2; mixed forests (Chen et al., 2006)
         @(h) (1.2 + 0.16 * h)/2; mixed forests (Pitkänen et al., 2004)
         Use '@(h) c' where c is an arbitrary constant radius, if you do not want to vary the search radius as a function of height.
    
    - minHeightDifference - scalar (optional, default: 0.1), threshold
      height difference below which the H-maxima transform suppresses all local maxima (only when method is set to 'hMaxima')
    
    - fig, boolean (optional, default: False), switch to plot figures
    
    - verbose (optional, default: False) - boolean value, verbosity switch
    
    Outputs:
    - crh - Mx3 numeric matrix, images coordinates (col, row) and height values of tree tops
    - xyh - Mx3 numeric matrix, map coordinates (x, y) and height values of tree tops
    
    Example:
        
    crh, xyh = canopy_peaks(
        subset,
        transform,
        smoothing_filter=1,
        method='default',      
        min_tree_height=2,   
        search_radius=lambda h: 0.28 * h**0.59, 
        fig=True,           
        verbose=True   
    )

    
    Author: Matthew Parkan, SITN
    Last revision: Janjuary 31, 2025
    Licence: BSD 3-Clause License
    '''
    
    # Smoothing the CHM (if applicable)
    if gaussian_filter_sigma is not None:
        if verbose:
            print("Smoothing CHM...")
        chm = gaussian_filter(chm, gaussian_filter_sigma)


    # Pre-process CHM
    chm[chm < 0] = 0
    chm[np.isnan(chm)] = 0
    chm = chm.astype(float)

    # Find local maxima (tree tops)
    if verbose:
        print("Detecting peaks...")

    nrows, ncols = chm.shape
    transformer = rasterio.transform.AffineTransformer(transform)
    
    if method == "default":
        # Default method: search for peaks using a variable-sized window based on tree height
        crown_radius = np.vectorize(search_radius)(chm)
        grid_resolution = abs(transform[0])  # Pixel size in x direction
        window_radius = np.maximum(np.round(crown_radius / grid_resolution).astype(int), 1)
        unique_window_radius = np.unique(window_radius)
        
        idxl_lm = np.zeros_like(chm, dtype=bool)
        mask = np.ones_like(chm, dtype=bool)

        for r in np.sort(unique_window_radius)[::-1]:
            # Create a circular structuring element
            y, x = np.ogrid[-r:r+1, -r:r+1]
            se = x**2 + y**2 <= r**2
            local_max = (chm >= grey_dilation(chm, footprint=se)) & (window_radius == r) & mask
            idxl_lm = idxl_lm | local_max
            mask = mask & (distance_transform_edt(~local_max) > r)

        row_lm, col_lm = np.where(idxl_lm)
        h_lm = chm[row_lm, col_lm]
    
    elif method == "hMaxima":
        # H-Maxima transform method
        h_min = chm - min_height_difference
        h_min[h_min < 0] = 0
        h_maxima = reconstruction(h_min, chm, method="dilation")
        local_max = (chm - h_maxima) >= min_height_difference
        row_lm, col_lm = np.where(local_max)
        h_lm = chm[row_lm, col_lm]
    
    # Convert image coordinates (row, col) to map coordinates (x, y)
    if verbose:
        print("Transforming to map coordinates...")
    
    xy = np.column_stack(transformer.xy(row_lm, col_lm))
    
    # Combine image coordinates with heights
    crh = np.column_stack((col_lm, row_lm, h_lm))
    xyh = np.column_stack((xy[:, 0], xy[:, 1], h_lm))
    
    # Filter tree tops below the height threshold
    if verbose:
        print("Filtering low peaks...")
    
    mask = h_lm >= min_tree_height
    crh = crh[mask]
    xyh = xyh[mask]
    
    # Sort peaks by decreasing height
    sort_idx = np.argsort(-xyh[:, 2])
    crh = crh[sort_idx]
    xyh = xyh[sort_idx]
    
    # Optionally, plot the results
    if fig:
        plt.imshow(chm, cmap="gray")
        plt.plot(crh[:, 0], crh[:, 1], "rx", markersize=3)
        plt.colorbar()
        plt.title("Detected Tree Tops")
        plt.xlabel("Column")
        plt.ylabel("Row")
        plt.show()
    
    
    return crh, xyh