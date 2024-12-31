# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 00:21:33 2022

@author: adh
"""

# Standard library imports
import copy
import os
import sys
import copy

wd = os.path.dirname(os.path.abspath(__file__))
os.chdir(wd)
# os.chdir("C:\\Users\\adh\\OneDrive - Cambridge Econometrics\\ADH CE\\Phd\\KDP_2023\\CLEARS_CEE")

# Third party imports
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import itertools

# Local library imports
import source_code.paths_append
from model_class import ModelRun

print('Start')

# Instantiate the run
model = ModelRun()

print('Initiated')

# Fetch ModelRun attributes, for examination
# Titles of the model
titles = model.titles
# Dimensions of model variables
dims = model.dims
# Converters
converter = model.converter
# Data
data = model.data
# Timeline
timeline = model.timeline
# ID
runid = model.name
# Set random seed
np.random.seed(123)
# Run model
model.run()
print('Model run finished')
print('Export results')
# Export results
results = copy.deepcopy(model.data)

main_outputs = {'battery_cap': 'Battery storage capacity (GWh)',
                'battery_cum': 'Cumulative number of batteries',
                'battery_npv': 'NPV of battery (EUR)',
                'battery_share': 'Share of batteries owners (%)',
                'battery_size': 'Size of battery system (kWh)',
                'charge_level': 'Charge level of batteries (kWh)',
                'charge_total_2050': 'Avg. charging 2050 (kW)',
                'discharge_total_2050': 'Avg. discharging 2050 (kW)',
                'pv_cap_est': 'PV capacity (GW)',
                'pv_cum': 'Cumulative number of PVs',
                'pv_gen_adj': 'PV generation profiles (kW)',
                'pv_gen_total_2050': 'Avg. PV generation, 2050 (kW)',
                'pv_npv': 'NPV of rooftop PVs (EUR)',
                'pv_share': 'Share of rooftop PVs (%)',
                'pv_size': 'Size of solar PV system (kW)',
                'profile_shares': 'Shares of profile types (%)',
                'profiles_adj': 'Adjusted load profiles (kW)',
                'consumption_adj': 'Electricity consumption (kWh)',
                'p': 'Innovation parameter',
                'q': 'Imitation parameter'}

# Export to Excel
out_fn = 'output_' + runid + '.xlsx'
with pd.ExcelWriter(os.path.join('output', out_fn)) as writer:

    for var, var_nm in main_outputs.items():
        print(var)
    
        # Extract data
        data = results[var]
        # Get dimension names and titles
        dim_nm = list(dims[var])
        dim_dict = {d: dims[var][d] for d in range(7)}
        titles_ = [list(titles[dims[var][d]]) for d in range(7)]
        titles_nona = [list(titles[dims[var][d]]) for d in range(7) if dims[var][d] != 'NA']
    
    
        # Get length of dimensions
        len_d = [len(list(titles_)[d]) for d in range(7)]
        len_d = list(np.array(len_d))
    
        # Sort values to match titles
        sort_dims = []
        dim_order = []
        for dim in dim_nm:
            if dim != 'NA' and dim != 'iterations':
                dim_order.append(dim)
                sort_dims.append(dim)
    
            if dim == 'iterations':
                dim_order.append('feed-in-tariff')
                sort_dims.append('feed-in-tariff')
        
        if var in ['charge_total_2050', 'discharge_total_2050', 'pv_gen_total_2050']:
            titles_nona = titles_nona[0:3] + titles_nona[5:]
            sort_dims.pop(3)
            dim_order.pop(3)
            sort_dims.pop(3)
            dim_order.pop(3)
            data = data[:, :, :, :, np.newaxis, :, np.newaxis, :, :].sum(axis = 3).mean(axis = 4) 
        
        else:
            if 'azimuth' in dim_order:
                titles_nona.pop(3)
                sort_dims.pop(3)
                dim_order.pop(3)

                if var in ['charge_level', 'pv_gen_adj', 'pv_npv', 'battery_npv', 'profiles_adj', 'consumption_adj', 'pv_size', 'pv_share', 'battery_share', 'battery_size', 'pv_size']:
                    data = data[:, :, :, 1, np.newaxis, :, :, :]
                else:
                    data = data[:, :, :, :, np.newaxis, :, :, :].sum(axis = 3)
                
            if 'timeline' in dim_order:
                titles_nona[-1] = titles_nona[-1][16:]
    
                data = data[:, :, :, :, :, :, 16:]
                
        multi_idx = list(itertools.product(*titles_nona))
        total_len = np.prod(np.array(data.shape)) 

        idx_name = ', '.join(sort_dims)
        idx_name = idx_name.replace('iterations', 'feed-in-tariff')
    
            
        df_out = pd.DataFrame(data.reshape([total_len]), index = multi_idx, columns = ['Value'])
        df_out.index.name = idx_name
        df_out = df_out.reset_index()
    
        df_out[sort_dims] = pd.DataFrame(df_out[idx_name].tolist(), index = df_out.index)
        dim_order = dim_order + ['Value']
        df_out = df_out[dim_order]
        
        # Export
        df_out.to_excel(writer, sheet_name=var_nm, index = False)
        print('finished')
    writer.save()
    writer.close()
    
print('Ready')
