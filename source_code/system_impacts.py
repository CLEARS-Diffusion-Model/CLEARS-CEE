# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 21:58:37 2024

@author: adh
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def peak_battery_use(data, titles):
    # Set main parameters
    yearly_cons = data['consumption'][:, :, :, :, :, :, :]
    cons_size = np.array([1.5, 1, 0.75])[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    eff_idx = titles['battery_data'].index('efficiency')
    battery_eff = data['battery_specs'][eff_idx, 0, 0, 0, 0, 0, 0]
    # Get number of potential adopters by profile
    size_w = np.expand_dims([0.25, 0.5, 0.25], axis = (0, 1, 3))
    azi_w = np.expand_dims([0.25, 0.5, 0.25], axis = (0, 1, 2))
    nr_houses_4d = np.repeat(data['hh_total'][:, :, :, :, 0, 0, 0], len(titles['profile_type']), axis = 1)
    nr_houses_4d = np.repeat(nr_houses_4d, len(titles['cons_size']), axis = 2)
    nr_houses_4d = np.repeat(nr_houses_4d, len(titles['azimuth']), axis = 3)

    nr_houses_profile = data['profile_shares'][:, :, :, :, 0, 0, 0] * nr_houses_4d
    nr_houses_profile = size_w * nr_houses_profile
    nr_houses_profile = azi_w * nr_houses_profile
    owner_sh = data['owner_share'][:, :, :, :, 0, 0, 0] / 100
    hh_sh = data['hh_share'][:, :, :, :, 0, 0, 0] / 100
    # Restrict to 1-3 apartment houses
    data['hh_nr'][:, :, :, :, 0, 0, 0] = nr_houses_profile * hh_sh * owner_sh

    # Get DoD
    dod_idx = titles['battery_data'].index('depth_of_discharge')
    dod = data['battery_specs'][dod_idx, 0, 0, 0, 0, 0, 0]

    # Adjust profiles with consumption
    cons_prof = np.repeat(yearly_cons, len(titles['profile_type']), axis = 1)
    cons_prof_size = np.repeat(cons_prof, len(titles['cons_size']), axis = 2)
    cons_prof_size_azi = np.repeat(cons_prof_size, len(titles['azimuth']), axis = 3)

    profile_sum = np.expand_dims(data['profiles'][:, :, :, :, :, :, :].sum(axis = 6).sum(axis = 5), axis = (5, 6))

    data['consumption_adj'] = cons_prof_size_azi * profile_sum * cons_size

    # Adjust profile with consumption profiles
    adj_profile = data['profiles'][:, :, :, :, 0, :, :] / profile_sum[:, :, :, :, 0, :, :]
    data['profiles_adj'] = adj_profile[:, :, np.newaxis, :, :, :, :] * data['consumption_adj'][:, :, :, :, :, :, :]

    # Adjust solar profile to meet annual consumption
    pv_sum = data['pv_gen'][:, 0, 0, :, 0, :, :].sum(axis = 3).sum(axis = 2)
    # pv_size = data['consumption_adj'][:, :, :, :, 0, 0, 0] / pv_sum[:, np.newaxis, np.newaxis, :]
    pv_size = data['pv_size'][:, :, :, :, 0, 0, 0]
    # data['pv_size'][:, :, :, :, 0, 0, 0] = pv_size
    data['pv_gen_adj'] = data['pv_gen'][:, :, :, :, :, :, :] * pv_size[:, :, :, :, np.newaxis, np.newaxis, np.newaxis]
    # Set battery size
    battery_size = data['pv_size'][:, :, :, :, 0, 0, 0] * 2
    data['battery_size'][:, :, :, :, 0, 0, 0] = battery_size

    for i, country in enumerate(titles['country']):
        # Get county profiles and solar generation
        adj_profile = data['profiles_adj'][i, :, :, :, 0, :, :]
        adj_pv_gen = data['pv_gen_adj'][i, :, :, :, 0, :, :]
        # Calculate PV overproduction
        overprod = adj_pv_gen - adj_profile
        overprod[overprod < 0] = 0

        # Calculate residual load
        residual_demand = adj_profile - adj_pv_gen
        residual_demand[residual_demand < 0] = 0
        reg_pv_size = pv_size[i, :, :, :]

        charge = np.zeros_like(residual_demand)
        charge_level = np.zeros_like(residual_demand)
        discharge = np.zeros_like(residual_demand)


        for d, day in enumerate(titles['date']):

            for h, hour in enumerate(titles['hour']):
                # Add charge to battery and adjust for efficiency
                if h > 0:
                    prev_charge_level = charge_level[:, :, :, d, h - 1]
                    charge_level[:, :, :, d, h]  = prev_charge_level + np.minimum(overprod[:, :, :, d, h] * battery_eff, reg_pv_size)
                    # Get hourly charge
                    charge[:, :, :, d, h] = charge_level[:, :, :, d, h] - prev_charge_level

                elif h == 0 and d > 0:
                    last_h = max(list(titles['hour_short']))
                    prev_charge_level = charge_level[:, :, :, d - 1, last_h]
                    charge_level[:, :, :, d, h]  = prev_charge_level + np.minimum(overprod[:, :, :, d, h] * battery_eff, reg_pv_size)
                    # Get hourly charge
                    charge[:, :, :, d, h] = charge_level[:, :, :, d, h] - prev_charge_level
                else:
                    charge_level[:, :, :, d, h]  = np.minimum(overprod[:, :, :, d, h] * battery_eff, reg_pv_size)
                    # Get hourly charge
                    charge[:, :, :, d, h] = charge_level[ :, :, :, d, h]



                # Create holder for discharge potential
                discharge_potential = np.zeros_like(discharge[:, :, :, d, h])
                # Cap charge with battery size
                for s, size in enumerate(titles['cons_size']):
                    charge_level[:, s, :, d, h] = np.minimum(charge_level[:, s, :, d, h], battery_size[i, :, s, :])
                    # Do not allow battery to go below 20%
                    # Calculate discharge potential
                    discharge_potential[:, s, :] = np.maximum(0, charge_level[:, s, :, d, h] - (1 - dod) * battery_size[i, :, s, :])

                # Calculate hourly charge
                if h > 0:
                    prev_charge_level = charge_level[:, :, :, d, h - 1]
                    charge[:, :, :, d, h] = (charge_level[:, :, :, d, h] - prev_charge_level) / battery_eff

                elif h == 0 and d > 0:
                    last_h = max(list(titles['hour_short']))
                    prev_charge_level = charge_level[:, :, :, d - 1, last_h]
                    charge[:, :, :, d, h] = (charge_level[:, :, :, d, h] - prev_charge_level) / battery_eff
                else:
                    charge[:, :, :, d, h] = (charge_level[:, :, :, d, h]) / battery_eff


                # Calculate discharge
                discharge[:, :, :, d, h] = np.minimum(discharge_potential[:, :, :], residual_demand[:, :, :, d, h])

                # If non-peak hour
                peak_h = [16, 17, 18, 19]
                sun_h = list(range(11, 16))
                if h in peak_h:
                    # Find where charge level is not enough to supply the whole demand
                    # Move charge from earlier hours
                    # missing_charge = np.maximum(residual_demand[:, :, :, d, h] - discharge[:, :, :, d, h], 0)
                    # if np.any(missing_charge > 0):
                    #     shift_charge = missing_charge.copy()
                    #     for s_h in sun_h:
                    #         h_shift = np.minimum(shift_charge[:, :, :], discharge[:, :, :, d, s_h])
                    #         # Remove discharge from the afternoon hours
                    #         discharge[:, :, :, d, s_h] -= h_shift
                    #         shift_charge -= h_shift
                    #     # Add discharge to peak hour
                    #     discharge[:, :, :, d, h] += missing_charge - shift_charge

                    # Get the 30-day average residual demand (demand potentially supplied by battery)
                    prev_month = d - 30
                    # Consider the following 16 hours to supply with the battery
                    follow_h = np.array(list(range(h + 1, 24)) + list(range(0, 24 - h + 1)))

                    remaining_potential = discharge_potential[:, :, :] - discharge[:, :, :, d, h]


                    if d >= 30:
                        date_idx = np.array(list(range(prev_month, d)))
                    else:
                        # If date in January then use December data
                        date_idx = np.array(list(range(365 + d - 30, 365)) + list(range(0, d + 1)))
                    avg_daily_demand = residual_demand[:, :, :, date_idx[:,None], follow_h[None, :]].mean(axis = 3).sum(axis = 3)

                    # If stored energy is more than the 30-day average sell excess energy to the system
                    if np.any(remaining_potential > avg_daily_demand):
                        # Sell excess energy but limited to battery power capacity
                        peak_sell = np.minimum(reg_pv_size, (remaining_potential - avg_daily_demand) / (20 - h))
                        peak_sell[peak_sell < 0] = 0
                        # Add discharge to peak hour
                        discharge[:, :, :, d, h] += peak_sell


                # Remove discharge
                charge_level[:, :, :, d, h] = charge_level[:, :, :, d, h] - discharge[:, :, :, d, h]

        data['charge'][i, :, :, :, :, :, 0] = charge[:, :, :, :, :]
        data['charge_level'][i, :, :, :, :, :, 0] = charge_level[:, :, :, :, :]
        data['discharge'][i, :, :, :, :, :, 0] = discharge[:, :, :, :, :]



    return data





def battery_use(data, titles):
    # Set main parameters
    yearly_cons = data['consumption'][:, :, :, :, :, :, :]
    cons_size = np.array([1.5, 1, 0.75])[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    eff_idx = titles['battery_data'].index('efficiency')
    battery_eff = data['battery_specs'][eff_idx, 0, 0, 0, 0, 0, 0]
    # Get number of potential adopters by profile
    size_w = np.expand_dims([0.25, 0.5, 0.25], axis = (0, 1, 3))
    azi_w = np.expand_dims([0.25, 0.5, 0.25], axis = (0, 1, 2))
    nr_houses_4d = np.repeat(data['hh_total'][:, :, :, :, 0, 0, 0], len(titles['profile_type']), axis = 1)
    nr_houses_4d = np.repeat(nr_houses_4d, len(titles['cons_size']), axis = 2)
    nr_houses_4d = np.repeat(nr_houses_4d, len(titles['azimuth']), axis = 3)

    nr_houses_profile = data['profile_shares'][:, :, :, :, 0, 0, 0] * nr_houses_4d
    nr_houses_profile = size_w * nr_houses_profile
    nr_houses_profile = azi_w * nr_houses_profile
    owner_sh = data['owner_share'][:, :, :, :, 0, 0, 0] / 100
    hh_sh = data['hh_share'][:, :, :, :, 0, 0, 0] / 100
    # Restrict to 1-3 apartment houses
    data['hh_nr'][:, :, :, :, 0, 0, 0] = nr_houses_profile * hh_sh * owner_sh

    # Get DoD
    dod_idx = titles['battery_data'].index('depth_of_discharge')
    dod = data['battery_specs'][dod_idx, 0, 0, 0, 0, 0, 0]

    # Adjust profiles with consumption
    cons_prof = np.repeat(yearly_cons, len(titles['profile_type']), axis = 1)
    cons_prof_size = np.repeat(cons_prof, len(titles['cons_size']), axis = 2)
    cons_prof_size_azi = np.repeat(cons_prof_size, len(titles['azimuth']), axis = 3)

    profile_sum = np.expand_dims(data['profiles'][:, :, :, :, :, :, :].sum(axis = 6).sum(axis = 5), axis = (5, 6))

    data['consumption_adj'] = cons_prof_size_azi * profile_sum * cons_size

    # Adjust profile with consumption profiles
    adj_profile = data['profiles'][:, :, :, :, 0, :, :] / profile_sum[:, :, :, :, 0, :, :]
    data['profiles_adj'] = adj_profile[:, :, np.newaxis, :, :, :, :] * data['consumption_adj'][:, :, :, :, :, :, :]

    # Adjust solar profile to meet annual consumption
    pv_sum = data['pv_gen'][:, 0, 0, :, 0, :, :].sum(axis = 3).sum(axis = 2)
    # pv_size = data['consumption_adj'][:, :, :, :, 0, 0, 0] / pv_sum[:, np.newaxis, np.newaxis, :]
    pv_size = data['pv_size'][:, :, :, :, 0, 0, 0]
    # data['pv_size'][:, :, :, :, 0, 0, 0] = pv_size
    data['pv_gen_adj'] = data['pv_gen'][:, :, :, :, :, :, :] * pv_size[:, :, :, :, np.newaxis, np.newaxis, np.newaxis]
    # Set battery size
    battery_size = data['pv_size'][:, :, :, :, 0, 0, 0] * 2
    data['battery_size'][:, :, :, :, 0, 0, 0] = battery_size

    for i, country in enumerate(titles['country']):
        # Get county profiles and solar generation
        adj_profile = data['profiles_adj'][i, :, :, :, 0, :, :]
        adj_pv_gen = data['pv_gen_adj'][i, :, :, :, 0, :, :]
        # Calculate PV overproduction
        overprod = adj_pv_gen - adj_profile
        overprod[overprod < 0] = 0

        # Calculate residual load
        residual_demand = adj_profile - adj_pv_gen
        residual_demand[residual_demand < 0] = 0
        reg_pv_size = pv_size[i, :, :, :]

        charge = np.zeros_like(residual_demand)
        charge_level = np.zeros_like(residual_demand)
        discharge = np.zeros_like(residual_demand)


        for d, day in enumerate(titles['date']):

            for h, hour in enumerate(titles['hour']):
                # Add charge to battery and adjust for efficiency
                if h > 0:
                    prev_charge_level = charge_level[:, :, :, d, h - 1]
                    charge_level[:, :, :, d, h]  = prev_charge_level + np.minimum(overprod[:, :, :, d, h] * battery_eff, reg_pv_size)
                    # Get hourly charge
                    charge[:, :, :, d, h] = charge_level[:, :, :, d, h] - prev_charge_level

                elif h == 0 and d > 0:
                    last_h = max(list(titles['hour_short']))
                    prev_charge_level = charge_level[:, :, :, d - 1, last_h]
                    charge_level[:, :, :, d, h]  = prev_charge_level + np.minimum(overprod[:, :, :, d, h] * battery_eff, reg_pv_size)
                    # Get hourly charge
                    charge[:, :, :, d, h] = charge_level[:, :, :, d, h] - prev_charge_level
                else:
                    charge_level[:, :, :, d, h]  = np.minimum(overprod[:, :, :, d, h] * battery_eff, reg_pv_size)
                    # Get hourly charge
                    charge[:, :, :, d, h] = charge_level[ :, :, :, d, h]



                # Create holder for discharge potential
                discharge_potential = np.zeros_like(discharge[:, :, :, d, h])
                # Cap charge with battery size
                for s, size in enumerate(titles['cons_size']):
                    charge_level[:, s, :, d, h] = np.minimum(charge_level[:, s, :, d, h], battery_size[i, :, s, :])
                    # Do not allow battery to go below 20%
                    # Calculate discharge potential
                    discharge_potential[:, s, :] = np.maximum(0, charge_level[:, s, :, d, h] - (1 - dod) * battery_size[i, :, s, :])

                # Calculate hourly charge
                if h > 0:
                    prev_charge_level = charge_level[:, :, :, d, h - 1]
                    charge[:, :, :, d, h] = (charge_level[:, :, :, d, h] - prev_charge_level) / battery_eff

                elif h == 0 and d > 0:
                    last_h = max(list(titles['hour_short']))
                    prev_charge_level = charge_level[:, :, :, d - 1, last_h]
                    charge[:, :, :, d, h] = (charge_level[:, :, :, d, h] - prev_charge_level) / battery_eff
                else:
                    charge[:, :, :, d, h] = (charge_level[:, :, :, d, h]) / battery_eff


                # Calculate discharge
                discharge[:, :, :, d, h] = np.minimum(discharge_potential[:, :, :], residual_demand[:, :, :, d, h])

                # Remove discharge
                charge_level[:, :, :, d, h] = charge_level[:, :, :, d, h] - discharge[:, :, :, d, h]

        data['charge'][i, :, :, :, :, :, 0] = charge[:, :, :, :, :]
        data['charge_level'][i, :, :, :, :, :, 0] = charge_level[:, :, :, :, :]
        data['discharge'][i, :, :, :, :, :, 0] = discharge[:, :, :, :, :]



    return data


def total_battery_use(data, titles, timeline, period, f):

    if f == 10:
        data['charge_total'][:, :, :, :, :, :, period] = data['charge'][:, :, :, :, :, :, 0] * data['battery_cum'][:, :, :, :, f, np.newaxis, :, period]
        data['discharge_total'][:, :, :, :, :, :, period] = data['discharge'][:, :, :, :, :, :, 0] * data['battery_cum'][:, :, :, :, f, np.newaxis, :, period]
    data['charge_total_2050'][:, :, :, :, :, :, f] = data['charge'][:, :, :, :, :, :, 0] * data['battery_cum'][:, :, :, :, f, np.newaxis, :, period]
    data['discharge_total_2050'][:, :, :, :, :, :, f] = data['discharge'][:, :, :, :, :, :, 0] * data['battery_cum'][:, :, :, :, f, np.newaxis, :, period]

    data['battery_cap'][:, 0, 0, 0, f, 0, period] = (data['battery_cum'][:, :, :, :, f, 0, period] * data['battery_size'][:, :, :, :, 0, 0, 0]).sum(axis = 1).sum(axis = 1).sum(axis = 1) / 1000000

    return data

def total_pv_generation(data, titles, timeline, period, f):

    if f == 10:
        data['pv_gen_total'][:, :, :, :, :, :, period] = data['pv_gen_adj'][:, :, :, :, 0, :, :] * data['pv_cum'][:, :, :, :, f, np.newaxis, :, period]
    data['pv_gen_total_2050'][:, :, :, :, :, :, f] = data['pv_gen_adj'][:, :, :, :, 0, :, :] * data['pv_cum'][:, :, :, :, f, np.newaxis, :, period]

    if period > 15:
        data['pv_cap_est'][:, 0, 0, 0, f, 0, period] = (data['pv_cum'][:, :, :, :, f, 0, period] * data['pv_size'][:, :, :, :, 0, 0, 0]).sum(axis = 1).sum(axis = 1).sum(axis = 1) / 1000000
    else:
        data['pv_cap_est'][:, 0, 0, 0, f, 0, period] = data['pv_cap'][:, 0, 0, 0, 0, 0, period]

    return data

def self_consumption(data, titles, timeline, period, f):
    for c, cty in enumerate(titles['country']):
        pv_gen = data['pv_gen_total_2050'][c, :, :, :, :, :, f]
        total_load = data['pv_cum'][c, :, :, :, f, np.newaxis, :, period] * data['profiles_adj'][c, :, :, :, 0, :, :]
        battery_charge = data['charge_total_2050'][c, :, :, :, :, :, f]
        grid_cons = total_load - pv_gen
        grid_cons[grid_cons < 0] = 0
        self_cons_no_battery = total_load - grid_cons
        self_cons_no_battery = self_cons_no_battery.sum() / pv_gen.sum()

        self_cons_battery = total_load + battery_charge - grid_cons
        data['self_consumption'][c, 0, 0, 0, f, 0, period]  = self_cons_battery.sum() / pv_gen.sum()

    return data
