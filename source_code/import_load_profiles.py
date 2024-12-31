# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 14:45:19 2024

@author: adh
"""

import numpy as np
import pandas as pd
import h5py
import os

########################################
######## Import load components ########
########################################

os.chdir("C:/Users/adh/OneDrive - Cambridge Econometrics/ADH CE/Phd/KDP_2023/CLEARS_CEE")
filename = "data/loadprofiles.hdf5"

titles_fn = 'utilities/classification_titles.xlsx'
titles = pd.read_excel(titles_fn, sheet_name = None)

conv_fn = 'utilities/converters.xlsx'
conv = pd.read_excel(titles_fn, sheet_name = None)
conv['country']

f1 = h5py.File(filename,'r+')

data = dict()

with h5py.File(filename, "r") as f:
    # Print all root level object names (aka keys)
    # these can be group or dataset names
    print("Keys: %s" % f.keys())
    for k1 in f.keys():
        data[k1] = dict()

        for k2 in f[k1].keys():

            if k2 in conv['country']['Short name'].values:

                for k3 in f[k1][k2].keys():
                    arr = f[k1][k2][k3]['table'][()]  # returns as a numpy array
                    arr2 = np.asarray([np.asarray([sublist[0], sublist[1]]) for sublist in arr])
                    if k2 not in data[k1].keys():
                        data[k1][k2] = pd.DataFrame(arr2, columns = ['index', k3]).set_index('index')
                    else:
                        data[k1][k2][k3] = pd.DataFrame(arr2, columns = ['index', k3]).set_index('index')

# Mapping appliances to profiles
profiles = dict()
profiles['normal'] = ['hot_water', 'ict', 'lighting', 'mechanical_energy',
                      'process_heat']
profiles['heating'] = ['hot_water', 'ict', 'lighting', 'mechanical_energy',
       'process_heat', 'space_heating']
profiles['cooling'] = ['cooling', 'hot_water', 'ict', 'lighting', 'mechanical_energy',
       'process_heat']

# Create indices
dates = np.repeat(titles['date']['Full name'].values, len(titles['hour']['Full name'].values))
hours = []
for _ in range(len(titles['date']['Full name'].values)):
    hours.extend(titles['hour']['Full name'].values)

idx = pd.MultiIndex.from_arrays([dates, hours], names=('date', 'hour'))
cols = ['country', 'profile_type', 'date', 'hour', 'Value']
load_cols = ['country']
load_cols.extend(data['2014']['HU'].columns)

# Create empty dataframes
country_profiles = pd.DataFrame(columns = cols)
norm_country_profiles = pd.DataFrame(columns = cols)

load_df_long = pd.DataFrame(columns = load_cols)

# Convert data into long datasets
for i, cty in enumerate(conv['country']['Short name'].values):
    cty_long = conv['country']['Full name'][i]
    load_df = data['2014'][cty].copy()
    cty_df = pd.DataFrame(0, columns = titles['profile_type']['Full name'].values,
                                     index = idx)

    for p in titles['profile_type']['Full name'].values:
        appliances = profiles[p]
        cty_df[p] = load_df[profiles[p]].sum(axis = 1).values

    # Save load by appliance
    load_df.insert(0, 'country', cty_long)
    load_df_long = pd.concat([load_df_long, load_df]).reset_index(drop = True)

    # Save results in long format
    cty_df_long = pd.melt(cty_df.reset_index(), id_vars = ['date', 'hour'],
                                         var_name = 'profile_type', value_name = 'Value')
    cty_df_long = cty_df_long[['profile_type', 'date', 'hour', 'Value']]
    cty_df_long.insert(0, 'country', cty_long)

    country_profiles = pd.concat([country_profiles, cty_df_long]).reset_index(drop = True)

    # Save results in long format
    cty_df_long = pd.melt(cty_df.reset_index(), id_vars = ['date', 'hour'],
                                         var_name = 'profile_type', value_name = 'Value')
    cty_df_long = cty_df_long[['profile_type', 'date', 'hour', 'Value']]
    cty_df_long.insert(0, 'country', cty_long)

    country_profiles = pd.concat([country_profiles, cty_df_long]).reset_index(drop = True)


    # Normalise consumption
    norm_cty_df = cty_df / cty_df.normal.sum()

    norm_cty_df_long = pd.melt(norm_cty_df.reset_index(), id_vars = ['date', 'hour'],
                                         var_name = 'profile_type', value_name = 'Value')
    norm_cty_df_long = norm_cty_df_long[['profile_type', 'date', 'hour', 'Value']]
    norm_cty_df_long.insert(0, 'country', cty_long)

    norm_country_profiles = pd.concat([norm_country_profiles, norm_cty_df_long]).reset_index(drop = True)

# Export results
out_fn = 'input/Baseline/profiles.csv'

# Remove file if exists
try:
    os.remove(out_fn)
except FileNotFoundError:
    pass

# Create comments to the csv file
first_row = 'Load profiles by country'
second_row = 'Normalised to normal profile'
third_row = 'Schlemminger et al. (2021)'


with open(out_fn, 'a', newline='') as f:
    f.write(first_row + ' \n')
    f.write(second_row + ' \n')
    f.write(third_row + ' \n')
    norm_country_profiles.to_csv(f, header = True, index = False)



#########################################
##### Calculate share of profiles ######
#########################################

hh_nr_fn = 'input/Baseline/hh_nr.csv'
hh_nr = pd.read_csv(hh_nr_fn, skiprows = 3, index_col = 0)
end_use_fn = 'input/Baseline/end_use_consumption.csv'
end_use = pd.read_csv(end_use_fn, skiprows = 3, index_col = 0)


profile_sums = norm_country_profiles.groupby(by = ['country', 'profile_type']).Value.sum()
appliance_sums = load_df_long.groupby(by = ['country']).sum()
# appliance_sums = pd.melt(appliance_sums.reset_index(), id_vars = 'country', var_name = 'appliance', value_name = 'Value')

# Get share of households using cooling
appliances = ['space_heating', 'cooling']
cooling_sums = appliance_sums.cooling
cooling_share = end_use.loc[end_use.end_use == 'cooling', 'Value'] * 1000000 / cooling_sums / hh_nr.Value
# Use the average of Slovakia and Czechia for Poland
cooling_share['Poland'] = (cooling_share['Czechia'] + cooling_share['Slovakia']) / 2


end_use.loc[end_use.end_use == 'space_heating', 'Value'] / end_use.loc[end_use.end_use == 'total', 'Value']

# Get avg. consumption by profile type
prof_share_fn = 'input/Baseline/profile_shares.csv'
prof_share = pd.read_csv(prof_share_fn, skiprows = 3, index_col = [0, 1])

# Avg. total consumption
avg_total_cons = end_use.loc[end_use.end_use == 'total', 'Value'] / hh_nr.Value * 1000000


# Calculate normal profile consumption based on profile shares and avg. consumption
prof_share_cons = profile_sums * prof_share.Value
prof_share_cons = prof_share_cons.reset_index().set_index('country')
prof_share = prof_share.reset_index().set_index('country')

heating_share_cons = prof_share_cons.loc[prof_share_cons.profile_type == 'heating', 'Value']
cooling_share_cons = prof_share_cons.loc[prof_share_cons.profile_type == 'cooling', 'Value']
normal_share = prof_share.loc[prof_share.profile_type == 'normal', 'Value']


normal_cons = avg_total_cons * (1 - heating_share_cons - cooling_share_cons) / normal_share

# Export results
out_fn = 'input/Baseline/consumption.csv'

# Remove file if exists
try:
    os.remove(out_fn)
except FileNotFoundError:
    pass

# Create comments to the csv file
first_row = 'Average annual electricity consumption of normal profiles by country'
second_row = 'kWh'
third_row = 'Calculated from Eurostat'


with open(out_fn, 'a', newline='') as f:
    f.write(first_row + ' \n')
    f.write(second_row + ' \n')
    f.write(third_row + ' \n')
    normal_cons.to_csv(f, header = True)




########################################
##### Import heating technologies ######
########################################

# filename = "data/FTT_heat_data.xlsx"

# heat_df = pd.read_excel(filename, skiprows = 2)

# df_list = np.split(heat_df, heat_df[heat_df.isnull().all(1)].index)

# heat_dict = {}

# for df in df_list[1:]:
#     cty = df.iloc[1, 0].strip()
#     cty_df = pd.DataFrame(df.iloc[3:, 1:].values, columns = df.iloc[2, 1:].astype(int), index = df.iloc[3:, 0].values)
#     heat_dict[cty] = cty_df