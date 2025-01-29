# -*- coding: utf-8 -*-
"""
Author: Matthew Parkan, SITN
Description: Utility functions
Last revision: January 23, 2025
Licence: BSD 3-Clause License 
"""

import re

#%% Functions

def get_filepath(files_in, pattern):
    for element in files_in:
        if re.search(pattern, element):
            return element
        
