# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 13:54:29 2024

@author: adh
"""

# Standard library imports
import copy
import os
import sys
import copy

os.chdir("D:\\KDP_2023\\GitHub\\CLEARS-CEE")


# Third party imports
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
# Local library imports
import source_code.paths_append
from model_class import ModelRun
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.axes as ax
import seaborn as sns

# run_id = 'Dev1'
# with open('output\{}.pickle'.format(run_id), 'wb') as f:
#     model = pickle.load(f)


results = copy.deepcopy(model.data)

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


# colourmap = 'gnuplot'
colourmap = 'turbo'
cmap = matplotlib.cm.get_cmap('turbo')

plt.style.use('seaborn-darkgrid')
csfont = {'fontname':'Times New Roman'}
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 500
plt.rcParams.update({'font.size': 14})

summer_idx = list(range(151, 243))
summer = titles['date'][151:243]
winter_idx = list(range(0, 59)) + list(range(334, 365))
winter = titles['date'][0:59] + titles['date'][334:365]
off_season_idx = list(range(59, 151)) + list(range(243, 334))
off_season = titles['date'][59:151] + titles['date'][243:334]

# Hungary data

cz_con = data['consumption'][titles['country'].index('Czechia'), 0, 0, 0] * 12
cz_idx = titles['country'].index('Czechia')
pv_sum = data['pv_gen'][cz_idx, 0, :, :].sum()
pv_size = cz_con / pv_sum
adj_pv_gen = data['pv_gen'][cz_idx, 0, :, :] * pv_size



########################################################################################
######################################### NPV #########################################
########################################################################################

################# Battery

fn = "Battery_NPV.jpg"
fp = os.path.join('figures', fn)

# Figure size
figsize = (12.0, 8)
# Create subplot
fig, axes = plt.subplots(nrows=2, ncols=4,
                         figsize=figsize,
                         sharex=True, sharey='row')


tl_out = np.arange(2024, 2050+1)

# colors = ["green", "black", "firebrick", "gray", "blue", "aqua", "red", "orange", "magenta", "navy", "tan", "maroon", "peru", "olive", "khaki"]
line_info  = {}

colors = ['#C5446E', '#49C9C5', '#AAB71D']#, '#009FE3', '#909090']
colors = [cmap(0), cmap(80), cmap(220)]

for r, reg in enumerate(list(titles['country'])):
    for c, cons in enumerate(list(titles['cons_size'])):
        for p, prof in enumerate(list(titles['profile_type'])):
            i=0
            # Set color
            colour = colors[c]
            # Set line style
            if prof == 'normal':
                linestyle = '--'
            elif prof == 'heating':
                linestyle = '-'
            elif prof == 'cooling':
                linestyle = ':'

            reg_npv = results['battery_npv'][r, p, c, 1, 0, 0, 16:]
            lbl = cons + ' ' + prof

            if r > 3:
                col = r - 4
                row = 1
            else:
                col = r
                row = 0
            axes[row, col].plot(np.asarray(tl_out),
                      reg_npv,
                      label=lbl,
                      color=colour,
                      linewidth=1.5,
                      linestyle=linestyle)
            axes[row, col].set_title(reg, fontstyle='italic', fontsize=14)

            axes[row, col].set_xlim([tl_out[0], tl_out[-1]]);
            min_npv = results['battery_npv'][:, :, :, 1, 0, 0, 16:].min() - 200
            max_npv = results['battery_npv'][:, :, :, 1, 0, 0, 16:].max() + 200
            axes[row, col].set_ylim([min_npv, max_npv]);
            axes[row, col].grid(color = 'grey', alpha=0.4, linestyle = '--', linewidth = 0.5)
            axes[row, col].tick_params('x', labelrotation=60)
            # axes[row, col].label_outer()
            axes[row, col].set_xticks([2030, 2040, 2050])

    axes[0, 0].set_ylabel("Net Present Value (EUR)")
    axes[1, 0].set_ylabel("Net Present Value (EUR)")

    h1, l1 = axes[0, 0].get_legend_handles_labels()
    # l1[4] = l1[4].split(' ')[0] + '\n' + l1[4].split(' ')[1]
    handles = h1[1::2] + [h1[0]]
    labels = l1[1::2] + [l1[0]]
    #labels[12] = labels[12].split(';')[0] + '\n' + labels[12].split(';')[1]

    # l1 = [lab.split(';')[0]+'\n' +lab.split(';')[1] for lab in l1]

    fig.legend(handles=h1,
               labels=l1,
               loc="upper right",
               bbox_to_anchor=(0.9, 0.85),
               frameon=False,
               borderaxespad=0.,
               ncol=1,
               title="Profiles",
               fontsize=12)


fig.subplots_adjust(hspace=0.15, wspace=0.3, right=0.75, bottom=0.2, left=0.17)

fig.savefig(fp)
plt.show()

################# PV


fn = "PV_NPV.jpg"
fp = os.path.join('figures', fn)

# Figure size
figsize = (12.0, 8)
# Create subplot
fig, axes = plt.subplots(nrows=2, ncols=4,
                         figsize=figsize,
                         sharex=True, sharey='row')


tl_out = np.arange(2024, 2050+1)

# colors = ["green", "black", "firebrick", "gray", "blue", "aqua", "red", "orange", "magenta", "navy", "tan", "maroon", "peru", "olive", "khaki"]
line_info  = {}

colors = ['#C5446E', '#49C9C5', '#AAB71D']#, '#009FE3', '#909090']
colors = [cmap(0), cmap(80), cmap(220)]

for r, reg in enumerate(list(titles['country'])):
    for c, cons in enumerate(list(titles['cons_size'])):
        for p, prof in enumerate(list(titles['profile_type'])):
            i=0
            # Set color
            colour = colors[c]
            # Set line style
            if prof == 'normal':
                linestyle = '--'
            elif prof == 'heating':
                linestyle = '-'
            elif prof == 'cooling':
                linestyle = ':'

            reg_npv = results['pv_npv'][r, p, c, 1, 0, 0, 16:]
            lbl = cons + ' ' + prof

            if r > 3:
                col = r - 4
                row = 1
            else:
                col = r
                row = 0
            axes[row, col].plot(np.asarray(tl_out),
                      reg_npv,
                      label=lbl,
                      color=colour,
                      linewidth=1.5,
                      linestyle=linestyle)
            axes[row, col].set_title(reg, fontstyle='italic', fontsize=14)

            axes[row, col].set_xlim([tl_out[0], tl_out[-1]]);
            min_npv = results['pv_npv'][:, :, :, 1, 0, 0, 16:].min() - 200
            max_npv = results['pv_npv'][:, :, :, 1, 0, 0, 16:].max() + 200
            axes[row, col].set_ylim([min_npv, max_npv]);
            axes[row, col].grid(color = 'grey', alpha=0.4, linestyle = '--', linewidth = 0.5)
            axes[row, col].tick_params('x', labelrotation=60)
            # axes[row, col].label_outer()
            axes[row, col].set_xticks([2030, 2040, 2050])

    axes[0, 0].set_ylabel("Net Present Value (EUR)")
    axes[1, 0].set_ylabel("Net Present Value (EUR)")

    h1, l1 = axes[0, 0].get_legend_handles_labels()
    # l1[4] = l1[4].split(' ')[0] + '\n' + l1[4].split(' ')[1]
    handles = h1[1::2] + [h1[0]]
    labels = l1[1::2] + [l1[0]]
    #labels[12] = labels[12].split(';')[0] + '\n' + labels[12].split(';')[1]

    # l1 = [lab.split(';')[0]+'\n' +lab.split(';')[1] for lab in l1]

    fig.legend(handles=h1,
               labels=l1,
               loc="upper right",
               bbox_to_anchor=(0.9, 0.85),
               frameon=False,
               borderaxespad=0.,
               ncol=1,
               title="Profiles",
               fontsize=12)


fig.subplots_adjust(hspace=0.15, wspace=0.3, right=0.75, bottom=0.2, left=0.17)

fig.savefig(fp)
plt.show()


# ########################################################################################
# ####################################### Profiles #######################################
# ########################################################################################

# ################# Summer

# # Profiles
# fn = "Profiles_summer.jpg"
# fp = os.path.join('figures', fn)

# # Figure size
# figsize = (12.0, 8)
# # Create subplot
# fig, axes = plt.subplots(nrows=2, ncols=4,
#                          figsize=figsize,
#                          sharex=True, sharey='row')


# tl_out = np.arange(2024, 2050+1)

# # colors = ["green", "black", "firebrick", "gray", "blue", "aqua", "red", "orange", "magenta", "navy", "tan", "maroon", "peru", "olive", "khaki"]
# line_info  = {}

# colors = ['#C5446E', '#49C9C5', '#AAB71D'] #, '#009FE3', '#909090']
# colors = [cmap(0), cmap(80), cmap(220)]

# for r, reg in enumerate(list(titles['country'])):
#     for c, cons in enumerate(list(titles['cons_size'])):
#         for p, prof in enumerate(list(titles['profile_type'])):
#             i=0
#             # Set color
#             colour = colors[c]
#             # Set line style
#             if prof == 'normal':
#                 linestyle = '--'
#             elif prof == 'heating':
#                 linestyle = '-'
#             elif prof == 'cooling':
#                 linestyle = ':'

#             reg_profile = results['profiles_adj'][r, p, c, 1, 0, summer_idx, :].mean(axis = 0)
#             lbl = cons + ' ' + prof

#             if r > 3:
#                 col = r - 4
#                 row = 1
#             else:
#                 col = r
#                 row = 0
#             axes[row, col].plot(np.asarray(range(0, 24)),
#                       reg_profile,
#                       label=lbl,
#                       color=colour,
#                       linewidth=1.5,
#                       linestyle=linestyle)
#             axes[row, col].set_title(reg, fontstyle='italic', fontsize=14)

#             axes[row, col].set_xlim([0, 23]);
#             min_load = 0
#             max_load = results['profiles_adj'][:, :, :, 1, 0, summer_idx, :].mean(axis = 3).max() + 0.1
#             axes[row, col].set_ylim([min_load, max_load]);
#             axes[row, col].grid(color = 'grey', alpha=0.4, linestyle = '--', linewidth = 0.5)
#             axes[row, col].tick_params('x', labelrotation=60)
#             # axes[row, col].label_outer()
#             axes[row, col].set_xticks([0, 6, 12, 18])

#     axes[0, 0].set_ylabel("kW")
#     axes[1, 0].set_ylabel("kW")

#     h1, l1 = axes[0, 0].get_legend_handles_labels()
#     # l1[4] = l1[4].split(' ')[0] + '\n' + l1[4].split(' ')[1]
#     handles = h1[1::2] + [h1[0]]
#     labels = l1[1::2] + [l1[0]]
#     #labels[12] = labels[12].split(';')[0] + '\n' + labels[12].split(';')[1]

#     # l1 = [lab.split(';')[0]+'\n' +lab.split(';')[1] for lab in l1]

#     fig.legend(handles=h1,
#                labels=l1,
#                loc="upper right",
#                bbox_to_anchor=(0.9, 0.85),
#                frameon=False,
#                borderaxespad=0.,
#                ncol=1,
#                title="Profiles",
#                fontsize=12)


# fig.subplots_adjust(hspace=0.15, wspace=0.3, right=0.75, bottom=0.2, left=0.17)

# fig.savefig(fp)
# plt.show()


# ################# Winter


# fn = "Profiles_winter.jpg"
# fp = os.path.join('figures', fn)

# # Figure size
# figsize = (12.0, 8)
# # Create subplot
# fig, axes = plt.subplots(nrows=2, ncols=4,
#                          figsize=figsize,
#                          sharex=True, sharey='row')


# tl_out = np.arange(2024, 2050+1)

# # colors = ["green", "black", "firebrick", "gray", "blue", "aqua", "red", "orange", "magenta", "navy", "tan", "maroon", "peru", "olive", "khaki"]
# line_info  = {}

# colors = ['#C5446E', '#49C9C5', '#AAB71D'] #, '#009FE3', '#909090']
# colors = [cmap(0), cmap(80), cmap(220)]

# for r, reg in enumerate(list(titles['country'])):
#     for c, cons in enumerate(list(titles['cons_size'])):
#         for p, prof in enumerate(list(titles['profile_type'])):
#             i=0
#             # Set color
#             colour = colors[c]
#             # Set line style
#             if prof == 'normal':
#                 linestyle = '--'
#             elif prof == 'heating':
#                 linestyle = '-'
#             elif prof == 'cooling':
#                 linestyle = ':'

#             reg_profile = results['profiles_adj'][r, p, c, 1, 0, winter_idx, :].mean(axis = 0)
#             lbl = cons + ' ' + prof

#             if r > 3:
#                 col = r - 4
#                 row = 1
#             else:
#                 col = r
#                 row = 0
#             axes[row, col].plot(np.asarray(range(0, 24)),
#                       reg_profile,
#                       label=lbl,
#                       color=colour,
#                       linewidth=1.5,
#                       linestyle=linestyle)
#             axes[row, col].set_title(reg, fontstyle='italic', fontsize=14)

#             axes[row, col].set_xlim([0, 23]);
#             min_load = 0
#             max_load = results['profiles_adj'][:, :, :, 1, 0, winter_idx, :].mean(axis = 3).max() + 0.1
#             axes[row, col].set_ylim([min_load, max_load]);
#             axes[row, col].grid(color = 'grey', alpha=0.4, linestyle = '--', linewidth = 0.5)
#             axes[row, col].tick_params('x', labelrotation=60)
#             # axes[row, col].label_outer()
#             axes[row, col].set_xticks([0, 6, 12, 18])

#     axes[0, 0].set_ylabel("kW")
#     axes[1, 0].set_ylabel("kW")

#     h1, l1 = axes[0, 0].get_legend_handles_labels()
#     # l1[4] = l1[4].split(' ')[0] + '\n' + l1[4].split(' ')[1]
#     handles = h1[1::2] + [h1[0]]
#     labels = l1[1::2] + [l1[0]]
#     #labels[12] = labels[12].split(';')[0] + '\n' + labels[12].split(';')[1]

#     # l1 = [lab.split(';')[0]+'\n' +lab.split(';')[1] for lab in l1]

#     fig.legend(handles=h1,
#                labels=l1,
#                loc="upper right",
#                bbox_to_anchor=(0.9, 0.85),
#                frameon=False,
#                borderaxespad=0.,
#                ncol=1,
#                title="Profiles",
#                fontsize=12)


# fig.subplots_adjust(hspace=0.15, wspace=0.3, right=0.75, bottom=0.2, left=0.17)

# fig.savefig(fp)
# plt.show()



# ########################################################################################
# ####################################### Cum. nr. #######################################
# ########################################################################################

# ################# Battery

# fn = "Battery_nr_sub.jpg"
# fp = os.path.join('figures', fn)

# # Figure size
# figsize = (12.0, 8)
# # Create subplot
# fig, axes = plt.subplots(nrows=2, ncols=4,
#                          figsize=figsize,
#                          sharex=True, sharey=False)


# tl_out = np.arange(2024, 2050+1)

# # colors = ["green", "black", "firebrick", "gray", "blue", "aqua", "red", "orange", "magenta", "navy", "tan", "maroon", "peru", "olive", "khaki"]
# line_info  = {}

# colors = ['#C5446E', '#49C9C5', '#AAB71D']#, '#009FE3', '#909090']
# colors = [cmap(0), cmap(80), cmap(220)]

# for r, reg in enumerate(list(titles['country'])):
#     for c, cons in enumerate(list(titles['cons_size'])):
#         i=0
#         # Set color
#         colour = colors[c]
#         # Set line style
#         linestyle = '-'


#         reg_battery = results['battery_cum'][r, :, c, :, 0, 0, 16:].sum(axis = 0).sum(axis = 0) / 1000
#         lbl = cons

#         if r > 3:
#             col = r - 4
#             row = 1
#         else:
#             col = r
#             row = 0
#         axes[row, col].plot(np.asarray(tl_out),
#                   reg_battery,
#                   label=lbl,
#                   color=colour,
#                   linewidth=1.5,
#                   linestyle=linestyle)
#         axes[row, col].set_title(reg, fontstyle='italic', fontsize=14)

#         axes[row, col].set_xlim([tl_out[0], tl_out[-1]]);
#         # min_npv = results['battery_cum'][:, :, :, 1, 0, 0, 16:].min() - 200
#         # max_npv = results['battery_cum'][:, :, :, 1, 0, 0, 16:].max() + 200
#         # axes[row, col].set_ylim([min_npv, max_npv]);
#         axes[row, col].grid(color = 'grey', alpha=0.4, linestyle = '--', linewidth = 0.5)
#         axes[row, col].tick_params('x', labelrotation=60)
#         # axes[row, col].label_outer()
#         axes[row, col].set_xticks([2030, 2040, 2050])

#     axes[0, 0].set_ylabel("Thousand battery owners")
#     axes[1, 0].set_ylabel("Thousand battery owners")

#     h1, l1 = axes[0, 0].get_legend_handles_labels()
#     # l1[4] = l1[4].split(' ')[0] + '\n' + l1[4].split(' ')[1]
#     handles = h1[1::2] + [h1[0]]
#     labels = l1[1::2] + [l1[0]]
#     #labels[12] = labels[12].split(';')[0] + '\n' + labels[12].split(';')[1]

#     # l1 = [lab.split(';')[0]+'\n' +lab.split(';')[1] for lab in l1]

#     fig.legend(handles=h1,
#                labels=l1,
#                loc="lower center",
#                bbox_to_anchor=(0.45, 0.05),
#                frameon=False,
#                borderaxespad=0.,
#                ncol=3,
#                title="Profiles",
#                fontsize=12)


# fig.subplots_adjust(hspace=0.15, wspace=0.3, right=0.75, bottom=0.2, left=0.17)

# fig.savefig(fp)
# plt.show()



# ################# PV

# fn = "PV_nr_sub.jpg"
# fp = os.path.join('figures', fn)

# # Figure size
# figsize = (12.0, 8)
# # Create subplot
# fig, axes = plt.subplots(nrows=2, ncols=4,
#                          figsize=figsize,
#                          sharex=True, sharey=False)


# tl_out = np.arange(2024, 2050+1)

# # colors = ["green", "black", "firebrick", "gray", "blue", "aqua", "red", "orange", "magenta", "navy", "tan", "maroon", "peru", "olive", "khaki"]
# line_info  = {}

# colors = ['#C5446E', '#49C9C5', '#AAB71D']#, '#009FE3', '#909090']
# colors = [cmap(0), cmap(80), cmap(220)]

# for r, reg in enumerate(list(titles['country'])):
#     for c, cons in enumerate(list(titles['cons_size'])):
#         i=0
#         # Set color
#         colour = colors[c]
#         # Set line style
#         linestyle = '-'


#         reg_battery = results['pv_cum'][r, :, c, :, 0, 0, 16:].sum(axis = 0).sum(axis = 0) / 1000
#         lbl = cons

#         if r > 3:
#             col = r - 4
#             row = 1
#         else:
#             col = r
#             row = 0
#         axes[row, col].plot(np.asarray(tl_out),
#                   reg_battery,
#                   label=lbl,
#                   color=colour,
#                   linewidth=1.5,
#                   linestyle=linestyle)
#         axes[row, col].set_title(reg, fontstyle='italic', fontsize=14)

#         axes[row, col].set_xlim([tl_out[0], tl_out[-1]]);
#         # min_npv = results['battery_cum'][:, :, :, 1, 0, 0, 16:].min() - 200
#         # max_npv = results['battery_cum'][:, :, :, 1, 0, 0, 16:].max() + 200
#         # axes[row, col].set_ylim([min_npv, max_npv]);
#         axes[row, col].grid(color = 'grey', alpha=0.4, linestyle = '--', linewidth = 0.5)
#         axes[row, col].tick_params('x', labelrotation=60)
#         # axes[row, col].label_outer()
#         axes[row, col].set_xticks([2030, 2040, 2050])

#     axes[0, 0].set_ylabel("Thousand PV owners")
#     axes[1, 0].set_ylabel("Thousand PV owners")

#     h1, l1 = axes[0, 0].get_legend_handles_labels()
#     # l1[4] = l1[4].split(' ')[0] + '\n' + l1[4].split(' ')[1]
#     handles = h1[1::2] + [h1[0]]
#     labels = l1[1::2] + [l1[0]]
#     #labels[12] = labels[12].split(';')[0] + '\n' + labels[12].split(';')[1]

#     # l1 = [lab.split(';')[0]+'\n' +lab.split(';')[1] for lab in l1]

#     fig.legend(handles=h1,
#                labels=l1,
#                loc="lower center",
#                bbox_to_anchor=(0.45, 0.05),
#                frameon=False,
#                borderaxespad=0.,
#                ncol=3,
#                title="Profiles",
#                fontsize=12)


# fig.subplots_adjust(hspace=0.15, wspace=0.3, right=0.75, bottom=0.2, left=0.17)

# fig.savefig(fp)
# plt.show()


# ########################################################################################
# #################################### Adoption share ####################################
# ########################################################################################

# ################# Battery

# # Battery share

# fn = "Battery_sh_sub.jpg"
# fp = os.path.join('figures', fn)

# # Figure size
# figsize = (12.0, 8)
# # Create subplot
# fig, axes = plt.subplots(nrows=2, ncols=4,
#                          figsize=figsize,
#                          sharex=True, sharey='row')


# tl_out = np.arange(2024, 2050+1)

# # colors = ["green", "black", "firebrick", "gray", "blue", "aqua", "red", "orange", "magenta", "navy", "tan", "maroon", "peru", "olive", "khaki"]
# line_info  = {}

# colors = ['#C5446E', '#49C9C5', '#AAB71D']#, '#009FE3', '#909090']
# colors = [cmap(0), cmap(80), cmap(220)]

# for r, reg in enumerate(list(titles['country'])):
#     for c, cons in enumerate(list(titles['cons_size'])):
#         i=0
#         # Set color
#         colour = colors[c]
#         # Set line style
#         linestyle = '-'


#         reg_battery = results['battery_cum'][r, :, c, :, 0, 0, 16:].sum(axis = 0).sum(axis = 0)
#         houses = results['hh_nr'][r, :, c, :, 0, 0, 0].sum(axis = 0).sum(axis = 0)
#         reg_battery_sh = reg_battery / houses * 100

#         lbl = cons

#         if r > 3:
#             col = r - 4
#             row = 1
#         else:
#             col = r
#             row = 0
#         axes[row, col].plot(np.asarray(tl_out),
#                   reg_battery_sh,
#                   label=lbl,
#                   color=colour,
#                   linewidth=1.5,
#                   linestyle=linestyle)
#         axes[row, col].set_title(reg, fontstyle='italic', fontsize=14)

#         axes[row, col].set_xlim([tl_out[0], tl_out[-1]]);
#         # min_npv = results['battery_cum'][:, :, :, 1, 0, 0, 16:].min() - 200
#         # max_npv = results['battery_cum'][:, :, :, 1, 0, 0, 16:].max() + 200
#         # axes[row, col].set_ylim([min_npv, max_npv]);
#         axes[row, col].grid(color = 'grey', alpha=0.4, linestyle = '--', linewidth = 0.5)
#         axes[row, col].tick_params('x', labelrotation=60)
#         # axes[row, col].label_outer()
#         axes[row, col].set_xticks([2030, 2040, 2050])

#     axes[0, 0].set_ylabel("Share of battery owners (%)")
#     axes[1, 0].set_ylabel("Share of battery owners (%)")

#     h1, l1 = axes[0, 0].get_legend_handles_labels()
#     # l1[4] = l1[4].split(' ')[0] + '\n' + l1[4].split(' ')[1]
#     handles = h1[1::2] + [h1[0]]
#     labels = l1[1::2] + [l1[0]]
#     #labels[12] = labels[12].split(';')[0] + '\n' + labels[12].split(';')[1]

#     # l1 = [lab.split(';')[0]+'\n' +lab.split(';')[1] for lab in l1]

#     fig.legend(handles=h1,
#                labels=l1,
#                loc="lower center",
#                bbox_to_anchor=(0.45, 0.05),
#                frameon=False,
#                borderaxespad=0.,
#                ncol=3,
#                title="Profiles",
#                fontsize=12)


# fig.subplots_adjust(hspace=0.15, wspace=0.3, right=0.75, bottom=0.2, left=0.17)

# fig.savefig(fp)
# plt.show()



# ################# PV

# # PV share

# fn = "PV_sh_sub.jpg"
# fp = os.path.join('figures', fn)

# # Figure size
# figsize = (12.0, 8)
# # Create subplot
# fig, axes = plt.subplots(nrows=2, ncols=4,
#                          figsize=figsize,
#                          sharex=True, sharey='row')


# tl_out = np.arange(2024, 2050+1)

# # colors = ["green", "black", "firebrick", "gray", "blue", "aqua", "red", "orange", "magenta", "navy", "tan", "maroon", "peru", "olive", "khaki"]
# line_info  = {}

# colors = ['#C5446E', '#49C9C5', '#AAB71D']#, '#009FE3', '#909090']
# colors = [cmap(0), cmap(80), cmap(220)]

# for r, reg in enumerate(list(titles['country'])):
#     for c, cons in enumerate(list(titles['cons_size'])):
#         i=0
#         # Set color
#         colour = colors[c]
#         # Set line style
#         linestyle = '-'


#         reg_battery = results['pv_cum'][r, :, c, :, 0, 0, 16:].sum(axis = 0).sum(axis = 0)
#         houses = results['hh_nr'][r, :, c, :, 0, 0, 0].sum(axis = 0).sum(axis = 0)
#         reg_battery_sh = reg_battery / houses * 100

#         lbl = cons

#         if r > 3:
#             col = r - 4
#             row = 1
#         else:
#             col = r
#             row = 0
#         axes[row, col].plot(np.asarray(tl_out),
#                   reg_battery_sh,
#                   label=lbl,
#                   color=colour,
#                   linewidth=1.5,
#                   linestyle=linestyle)
#         axes[row, col].set_title(reg, fontstyle='italic', fontsize=14)

#         axes[row, col].set_xlim([tl_out[0], tl_out[-1]]);
#         # min_npv = results['battery_cum'][:, :, :, 1, 0, 0, 16:].min() - 200
#         # max_npv = results['battery_cum'][:, :, :, 1, 0, 0, 16:].max() + 200
#         # axes[row, col].set_ylim([min_npv, max_npv]);
#         axes[row, col].grid(color = 'grey', alpha=0.4, linestyle = '--', linewidth = 0.5)
#         axes[row, col].tick_params('x', labelrotation=60)
#         # axes[row, col].label_outer()
#         axes[row, col].set_xticks([2030, 2040, 2050])

#     axes[0, 0].set_ylabel("Share of PV owners (%)")
#     axes[1, 0].set_ylabel("Share of PV owners (%)")

#     h1, l1 = axes[0, 0].get_legend_handles_labels()
#     # l1[4] = l1[4].split(' ')[0] + '\n' + l1[4].split(' ')[1]
#     handles = h1[1::2] + [h1[0]]
#     labels = l1[1::2] + [l1[0]]
#     #labels[12] = labels[12].split(';')[0] + '\n' + labels[12].split(';')[1]

#     # l1 = [lab.split(';')[0]+'\n' +lab.split(';')[1] for lab in l1]

#     fig.legend(handles=h1,
#                labels=l1,
#                loc="lower center",
#                bbox_to_anchor=(0.45, 0.05),
#                frameon=False,
#                borderaxespad=0.,
#                ncol=3,
#                title="Profiles",
#                fontsize=12)


# fig.subplots_adjust(hspace=0.15, wspace=0.3, right=0.75, bottom=0.2, left=0.17)

# fig.savefig(fp)
# plt.show()



# ########################################################################################
# ############################### Adoption share by profile ##############################
# ########################################################################################

# # Battery share by profile type

# fn = "Battery_sh_profile.jpg"
# fp = os.path.join('figures', fn)

# # Figure size
# figsize = (12.0, 8)
# # Create subplot
# fig, axes = plt.subplots(nrows=2, ncols=4,
#                          figsize=figsize,
#                          sharex=True, sharey='row')


# tl_out = np.arange(2024, 2050+1)

# # colors = ["green", "black", "firebrick", "gray", "blue", "aqua", "red", "orange", "magenta", "navy", "tan", "maroon", "peru", "olive", "khaki"]
# line_info  = {}

# colors = ['#C5446E', '#49C9C5', '#AAB71D']#, '#009FE3', '#909090']
# colors = [cmap(0), cmap(80), cmap(220)]

# for r, reg in enumerate(list(titles['country'])):
#     for c, cons in enumerate(list(titles['cons_size'])):
#         for p, prof in enumerate(list(titles['profile_type'])):
#             i=0
#             # Set color
#             colour = colors[c]
#             # Set line style
#             if prof == 'normal':
#                 linestyle = '--'
#             elif prof == 'heating':
#                 linestyle = '-'
#             elif prof == 'cooling':
#                 linestyle = ':'
#             i=0

#             reg_battery = results['battery_cum'][r, p, c, :, 0, 0, 16:].sum(axis = 0)
#             houses = results['hh_nr'][r, p, c, :, 0, 0, 0].sum(axis = 0)
#             reg_battery_sh = reg_battery / houses * 100

#             lbl = cons + ' ' + prof

#             if r > 3:
#                 col = r - 4
#                 row = 1
#             else:
#                 col = r
#                 row = 0
#             axes[row, col].plot(np.asarray(tl_out),
#                       reg_battery_sh,
#                       label=lbl,
#                       color=colour,
#                       linewidth=1.5,
#                       linestyle=linestyle)
#             axes[row, col].set_title(reg, fontstyle='italic', fontsize=14)

#             axes[row, col].set_xlim([tl_out[0], tl_out[-1]]);
#             # min_npv = results['battery_cum'][:, :, :, 1, 0, 0, 16:].min() - 200
#             # max_npv = results['battery_cum'][:, :, :, 1, 0, 0, 16:].max() + 200
#             # axes[row, col].set_ylim([min_npv, max_npv]);
#             axes[row, col].grid(color = 'grey', alpha=0.4, linestyle = '--', linewidth = 0.5)
#             axes[row, col].tick_params('x', labelrotation=60)
#             # axes[row, col].label_outer()
#             axes[row, col].set_xticks([2030, 2040, 2050])

#     axes[0, 0].set_ylabel("Share of battery owners (%)")
#     axes[1, 0].set_ylabel("Share of battery owners (%)")

#     h1, l1 = axes[0, 0].get_legend_handles_labels()
#     # l1[4] = l1[4].split(' ')[0] + '\n' + l1[4].split(' ')[1]
#     handles = h1[1::2] + [h1[0]]
#     labels = l1[1::2] + [l1[0]]
#     #labels[12] = labels[12].split(';')[0] + '\n' + labels[12].split(';')[1]

#     # l1 = [lab.split(';')[0]+'\n' +lab.split(';')[1] for lab in l1]

#     fig.legend(handles=h1,
#                labels=l1,
#                loc="upper right",
#                bbox_to_anchor=(0.9, 0.85),
#                frameon=False,
#                borderaxespad=0.,
#                ncol=1,
#                title="Profiles",
#                fontsize=12)


# fig.subplots_adjust(hspace=0.15, wspace=0.3, right=0.75, bottom=0.2, left=0.17)

# fig.savefig(fp)
# plt.show()

# ########################################################################################
# ############################## Feed-in-tariff adapt. share #############################
# ########################################################################################

# ################# Battery share by feed-in-tariff

# fn = "Battery_sh_tariff.jpg"
# fp = os.path.join('figures', fn)

# # Figure size
# figsize = (12.0, 8)
# # Create subplot
# fig, axes = plt.subplots(nrows=2, ncols=4,
#                          figsize=figsize,
#                          sharex=True, sharey='row')


# tl_out = np.arange(2024, 2050+1)

# # colors = ["green", "black", "firebrick", "gray", "blue", "aqua", "red", "orange", "magenta", "navy", "tan", "maroon", "peru", "olive", "khaki"]
# line_info  = {}

# colors = ['#C5446E', '#49C9C5', '#AAB71D']#, '#009FE3', '#909090']
# colors = [cmap(0), cmap(80), cmap(220)]

# for r, reg in enumerate(list(titles['country'])):
#     for f in titles['iterations']:
#         i=0
#         f = int(f)
#         # Set color
#         colour = cmap(50 + f * 10)
#         # Set line style
#         linestyle = '-'
#         reg_battery_sh = results['battery_cum'][r, :, :, :, f, 0, 16:].sum(axis = 0).sum(axis = 0).sum(axis = 0)
#         houses = results['hh_nr'][r, :, :, :, 0, 0, 0].sum(axis = 0).sum(axis = 0).sum(axis = 0)
#         reg_battery_sh = reg_battery_sh / houses * 100
#         cents = f
#         lbl = str(cents) + ' cents'

#         if r > 3:
#             col = r - 4
#             row = 1
#         else:
#             col = r
#             row = 0
#         axes[row, col].plot(np.asarray(tl_out),
#                   reg_battery_sh,
#                   label=lbl,
#                   color=colour,
#                   linewidth=1.5,
#                   linestyle=linestyle)
#         axes[row, col].set_title(reg, fontstyle='italic', fontsize=14)

#         axes[row, col].set_xlim([tl_out[0], tl_out[-1]]);
#         # min_npv = results['battery_cum'][:, :, :, 1, 0, 0, 16:].min() - 200
#         # max_npv = results['battery_cum'][:, :, :, 1, 0, 0, 16:].max() + 200
#         # axes[row, col].set_ylim([min_npv, max_npv]);
#         axes[row, col].grid(color = 'grey', alpha=0.4, linestyle = '--', linewidth = 0.5)
#         axes[row, col].tick_params('x', labelrotation=60)
#         # axes[row, col].label_outer()
#         axes[row, col].set_xticks([2030, 2040, 2050])

# axes[0, 0].set_ylabel("Share of battery owners (%)")
# axes[1, 0].set_ylabel("Share of battery owners (%)")

# h1, l1 = axes[0, 0].get_legend_handles_labels()
# # l1[4] = l1[4].split(' ')[0] + '\n' + l1[4].split(' ')[1]
# handles = h1[1::2] + [h1[0]]
# labels = l1[1::2] + [l1[0]]
# #labels[12] = labels[12].split(';')[0] + '\n' + labels[12].split(';')[1]

# # l1 = [lab.split(';')[0]+'\n' +lab.split(';')[1] for lab in l1]

# fig.legend(handles=h1,
#            labels=l1,
#            loc="upper right",
#            bbox_to_anchor=(0.87, 0.85),
#            frameon=False,
#            borderaxespad=0.,
#            ncol=1,
#            title="Feed-in-tariff",
#            fontsize=12)


# fig.subplots_adjust(hspace=0.15, wspace=0.3, right=0.75, bottom=0.2, left=0.17)

# fig.savefig(fp)
# plt.show()



# ################# PV share by feed-in-tariff

# fn = "PV_sh_tariff.jpg"
# fp = os.path.join('figures', fn)

# # Figure size
# figsize = (12.0, 8)
# # Create subplot
# fig, axes = plt.subplots(nrows=2, ncols=4,
#                          figsize=figsize,
#                          sharex=True, sharey='row')


# tl_out = np.arange(2024, 2050+1)

# # colors = ["green", "black", "firebrick", "gray", "blue", "aqua", "red", "orange", "magenta", "navy", "tan", "maroon", "peru", "olive", "khaki"]
# line_info  = {}

# colors = ['#C5446E', '#49C9C5', '#AAB71D']#, '#009FE3', '#909090']
# colors = [cmap(0), cmap(80), cmap(220)]

# for r, reg in enumerate(list(titles['country'])):
#     for f in titles['iterations']:
#         i=0
#         f = int(f)
#         # Set color
#         colour = cmap(50 + f * 10)
#         # Set line style
#         linestyle = '-'
#         reg_pv_sh = results['pv_cum'][r, :, :, :, f, 0, 16:].sum(axis = 0).sum(axis = 0).sum(axis = 0)
#         houses = results['hh_nr'][r, :, :, :, 0, 0, 0].sum(axis = 0).sum(axis = 0).sum(axis = 0)
#         reg_pv_sh = reg_pv_sh / houses * 100
#         cents = f
#         lbl = str(cents) + ' cents'

#         if r > 3:
#             col = r - 4
#             row = 1
#         else:
#             col = r
#             row = 0
#         axes[row, col].plot(np.asarray(tl_out),
#                   reg_pv_sh,
#                   label=lbl,
#                   color=colour,
#                   linewidth=1.5,
#                   linestyle=linestyle)
#         axes[row, col].set_title(reg, fontstyle='italic', fontsize=14)

#         axes[row, col].set_xlim([tl_out[0], tl_out[-1]]);
#         # min_npv = results['battery_cum'][:, :, :, 1, 0, 0, 16:].min() - 200
#         # max_npv = results['battery_cum'][:, :, :, 1, 0, 0, 16:].max() + 200
#         # axes[row, col].set_ylim([min_npv, max_npv]);
#         axes[row, col].grid(color = 'grey', alpha=0.4, linestyle = '--', linewidth = 0.5)
#         axes[row, col].tick_params('x', labelrotation=60)
#         # axes[row, col].label_outer()
#         axes[row, col].set_xticks([2030, 2040, 2050])

# axes[0, 0].set_ylabel("Share of PV owners (%)")
# axes[1, 0].set_ylabel("Share of PV owners (%)")

# h1, l1 = axes[0, 0].get_legend_handles_labels()
# # l1[4] = l1[4].split(' ')[0] + '\n' + l1[4].split(' ')[1]
# handles = h1[1::2] + [h1[0]]
# labels = l1[1::2] + [l1[0]]
# #labels[12] = labels[12].split(';')[0] + '\n' + labels[12].split(';')[1]

# # l1 = [lab.split(';')[0]+'\n' +lab.split(';')[1] for lab in l1]

# fig.legend(handles=h1,
#            labels=l1,
#            loc="upper right",
#            bbox_to_anchor=(0.87, 0.85),
#            frameon=False,
#            borderaxespad=0.,
#            ncol=1,
#            title="Feed-in-tariff",
#            fontsize=12)


# fig.subplots_adjust(hspace=0.15, wspace=0.3, right=0.75, bottom=0.2, left=0.17)

# fig.savefig(fp)
# plt.show()



# ########################################################################################
# ################################ Feed-in-tariff capacity ###############################
# ########################################################################################

# ################# Battery share by feed-in-tariff

# fn = "Battery_capacity.jpg"
# fp = os.path.join('figures', fn)

# # Figure size
# figsize = (12.0, 8)
# # Create subplot
# fig, axes = plt.subplots(nrows=2, ncols=4,
#                          figsize=figsize,
#                          sharex=True, sharey=False)


# tl_out = np.arange(2024, 2050+1)

# # colors = ["green", "black", "firebrick", "gray", "blue", "aqua", "red", "orange", "magenta", "navy", "tan", "maroon", "peru", "olive", "khaki"]
# line_info  = {}

# colors = ['#C5446E', '#49C9C5', '#AAB71D']#, '#009FE3', '#909090']
# colors = [cmap(0), cmap(80), cmap(220)]

# for r, reg in enumerate(list(titles['country'])):
#     for f in titles['iterations']:
#         i=0
#         f = int(f)
#         # Set color
#         colour = cmap(50 + f * 10)
#         # Set line style
#         linestyle = '-'
#         reg_battery_cap = results['battery_cap'][r, 0, 0, 0, f, 0, 16:]

#         cents = f
#         lbl = str(cents) + ' cents'

#         if r > 3:
#             col = r - 4
#             row = 1
#         else:
#             col = r
#             row = 0
#         axes[row, col].plot(np.asarray(tl_out),
#                   reg_battery_cap,
#                   label=lbl,
#                   color=colour,
#                   linewidth=1.5,
#                   linestyle=linestyle)
#         axes[row, col].set_title(reg, fontstyle='italic', fontsize=14)

#         axes[row, col].set_xlim([tl_out[0], tl_out[-1]]);
#         # min_npv = results['battery_cum'][:, :, :, 1, 0, 0, 16:].min() - 200
#         # max_npv = results['battery_cum'][:, :, :, 1, 0, 0, 16:].max() + 200
#         # axes[row, col].set_ylim([min_npv, max_npv]);
#         axes[row, col].grid(color = 'grey', alpha=0.4, linestyle = '--', linewidth = 0.5)
#         axes[row, col].tick_params('x', labelrotation=60)
#         # axes[row, col].label_outer()
#         axes[row, col].set_xticks([2030, 2040, 2050])

# axes[0, 0].set_ylabel("GWh")
# axes[1, 0].set_ylabel("GWh")

# h1, l1 = axes[0, 0].get_legend_handles_labels()
# # l1[4] = l1[4].split(' ')[0] + '\n' + l1[4].split(' ')[1]
# handles = h1[1::2] + [h1[0]]
# labels = l1[1::2] + [l1[0]]
# #labels[12] = labels[12].split(';')[0] + '\n' + labels[12].split(';')[1]

# # l1 = [lab.split(';')[0]+'\n' +lab.split(';')[1] for lab in l1]

# fig.legend(handles=h1,
#            labels=l1,
#            loc="upper right",
#            bbox_to_anchor=(0.87, 0.85),
#            frameon=False,
#            borderaxespad=0.,
#            ncol=1,
#            title="Feed-in-tariff",
#            fontsize=12)


# fig.subplots_adjust(hspace=0.15, wspace=0.3, right=0.75, bottom=0.2, left=0.17)

# fig.savefig(fp)
# plt.show()



# ################# PV capacity by feed-in-tariff

# fn = "PV_capacity.jpg"
# fp = os.path.join('figures', fn)

# # Figure size
# figsize = (12.0, 8)
# # Create subplot
# fig, axes = plt.subplots(nrows=2, ncols=4,
#                          figsize=figsize,
#                          sharex=True, sharey=False)


# tl_out = np.arange(2024, 2050+1)

# # colors = ["green", "black", "firebrick", "gray", "blue", "aqua", "red", "orange", "magenta", "navy", "tan", "maroon", "peru", "olive", "khaki"]
# line_info  = {}

# colors = ['#C5446E', '#49C9C5', '#AAB71D']#, '#009FE3', '#909090']
# colors = [cmap(0), cmap(80), cmap(220)]

# for r, reg in enumerate(list(titles['country'])):
#     for f in titles['iterations']:
#         i=0
#         f = int(f)
#         # Set color
#         colour = cmap(50 + f * 10)
#         # Set line style
#         linestyle = '-'
#         reg_pv_cap = results['pv_cap_est'][r, 0, 0, 0, f, 0, 16:]
#         cents = f
#         lbl = str(cents) + ' cents'

#         if r > 3:
#             col = r - 4
#             row = 1
#         else:
#             col = r
#             row = 0
#         axes[row, col].plot(np.asarray(tl_out),
#                   reg_pv_cap,
#                   label=lbl,
#                   color=colour,
#                   linewidth=1.5,
#                   linestyle=linestyle)
#         axes[row, col].set_title(reg, fontstyle='italic', fontsize=14)

#         axes[row, col].set_xlim([tl_out[0], tl_out[-1]]);
#         # min_npv = results['battery_cum'][:, :, :, 1, 0, 0, 16:].min() - 200
#         # max_npv = results['battery_cum'][:, :, :, 1, 0, 0, 16:].max() + 200
#         # axes[row, col].set_ylim([min_npv, max_npv]);
#         axes[row, col].grid(color = 'grey', alpha=0.4, linestyle = '--', linewidth = 0.5)
#         axes[row, col].tick_params('x', labelrotation=60)
#         # axes[row, col].label_outer()
#         axes[row, col].set_xticks([2030, 2040, 2050])

# axes[0, 0].set_ylabel("GW")
# axes[1, 0].set_ylabel("GW")

# h1, l1 = axes[0, 0].get_legend_handles_labels()
# # l1[4] = l1[4].split(' ')[0] + '\n' + l1[4].split(' ')[1]
# handles = h1[1::2] + [h1[0]]
# labels = l1[1::2] + [l1[0]]
# #labels[12] = labels[12].split(';')[0] + '\n' + labels[12].split(';')[1]

# # l1 = [lab.split(';')[0]+'\n' +lab.split(';')[1] for lab in l1]

# fig.legend(handles=h1,
#            labels=l1,
#            loc="upper right",
#            bbox_to_anchor=(0.87, 0.85),
#            frameon=False,
#            borderaxespad=0.,
#            ncol=1,
#            title="Feed-in-tariff",
#            fontsize=12)


# fig.subplots_adjust(hspace=0.15, wspace=0.3, right=0.75, bottom=0.2, left=0.17)

# fig.savefig(fp)
# plt.show()



########################################################################################
################################ System impacts in 2050 ################################
########################################################################################

################# Summer system impacts in 2050

fn = "Total_system_impact_summer_2050.jpg"
fp = os.path.join('figures', fn)

# Figure size
figsize = (12.0, 8)
# Create subplot
fig, axes = plt.subplots(nrows=2, ncols=4,
                         figsize=figsize,
                         sharex=True, sharey=False)


tl_out = np.arange(0, 24)

# colors = ["green", "black", "firebrick", "gray", "blue", "aqua", "red", "orange", "magenta", "navy", "tan", "maroon", "peru", "olive", "khaki"]
line_info  = {}

colors = ['#C5446E', '#49C9C5', '#AAB71D']#, '#009FE3', '#909090']
colors = [cmap(0), cmap(80), cmap(220)]

for r, reg in enumerate(list(titles['country'])):
    for f in titles['iterations']:
        i=0
        f = int(f)
        # Set color
        colour = cmap(50 + f * 10)
        # Set line style
        linestyle = '-'

        reg_discharge = results['discharge_total_2050'][r, :, :, :, summer_idx, :, f].sum(axis = 1).sum(axis = 1).sum(axis = 1) / 1000
        reg_charge = results['charge_total_2050'][r, :, :, :, summer_idx, :, f].sum(axis = 1).sum(axis = 1).sum(axis = 1) / 1000
        reg_pv_gen = results['pv_gen_total_2050'][r, :, :, :, summer_idx, :, f].sum(axis = 1).sum(axis = 1).sum(axis = 1) / 1000
        reg_output = (reg_pv_gen - reg_charge + reg_discharge).mean(axis = 0) / 1000

        tot_discharge = results['discharge_total_2050'][r, :, :, :, summer_idx, :, f].sum(axis = 1).sum(axis = 1).sum(axis = 1) / 1000
        tot_charge = results['charge_total_2050'][r, :, :, :, summer_idx, :, f].sum(axis = 1).sum(axis = 1).sum(axis = 1) / 1000
        tot_pv_gen = results['pv_gen_total_2050'][r, :, :, :, summer_idx, :, f].sum(axis = 1).sum(axis = 1).sum(axis = 1) / 1000
        tot_output = (tot_pv_gen - tot_charge + tot_discharge).mean(axis = 0) / 1000

        cents = f
        lbl = str(cents) + ' cents'

        if r > 3:
            col = r - 4
            row = 1
        else:
            col = r
            row = 0
        axes[row, col].plot(np.asarray(tl_out),
                  reg_output,
                  label=lbl,
                  color=colour,
                  linewidth=1.5,
                  linestyle=linestyle)
        axes[row, col].set_title(reg, fontstyle='italic', fontsize=14)

        axes[row, col].set_xlim([tl_out[0], tl_out[-1]]);
        # min_npv = results['battery_cum'][:, :, :, 1, 0, 0, 16:].min() - 200
        # max_npv = results['battery_cum'][:, :, :, 1, 0, 0, 16:].max() + 200
        axes[row, col].set_ylim([0, max(tot_output) * 1.1]);
        axes[row, col].grid(color = 'grey', alpha=0.4, linestyle = '--', linewidth = 0.5)
        axes[row, col].tick_params('x', labelrotation=60)
        # axes[row, col].label_outer()
        axes[row, col].set_xticks([0, 6, 12, 18])

axes[0, 0].set_ylabel("GW")
axes[1, 0].set_ylabel("GW")

h1, l1 = axes[0, 0].get_legend_handles_labels()
# l1[4] = l1[4].split(' ')[0] + '\n' + l1[4].split(' ')[1]
handles = h1[1::2] + [h1[0]]
labels = l1[1::2] + [l1[0]]
#labels[12] = labels[12].split(';')[0] + '\n' + labels[12].split(';')[1]

# l1 = [lab.split(';')[0]+'\n' +lab.split(';')[1] for lab in l1]

fig.legend(handles=h1,
           labels=l1,
           loc="upper right",
           bbox_to_anchor=(0.87, 0.85),
           frameon=False,
           borderaxespad=0.,
           ncol=1,
           title="Feed-in-tariff",
           fontsize=12)


fig.subplots_adjust(hspace=0.15, wspace=0.3, right=0.75, bottom=0.2, left=0.17)

fig.savefig(fp)
plt.show()


################# Winter system impacts in 2050

fn = "Total_system_impact_winter_2050.jpg"
fp = os.path.join('figures', fn)

# Figure size
figsize = (12.0, 8)
# Create subplot
fig, axes = plt.subplots(nrows=2, ncols=4,
                         figsize=figsize,
                         sharex=True, sharey=False)


tl_out = np.arange(0, 24)

# colors = ["green", "black", "firebrick", "gray", "blue", "aqua", "red", "orange", "magenta", "navy", "tan", "maroon", "peru", "olive", "khaki"]
line_info  = {}

colors = ['#C5446E', '#49C9C5', '#AAB71D']#, '#009FE3', '#909090']
colors = [cmap(0), cmap(80), cmap(220)]

for r, reg in enumerate(list(titles['country'])):
    for f in titles['iterations']:
        i=0
        f = int(f)
        # Set color
        colour = cmap(50 + f * 10)
        # Set line style
        linestyle = '-'

        reg_discharge = results['discharge_total_2050'][r, :, :, :, winter_idx, :, f].sum(axis = 1).sum(axis = 1).sum(axis = 1) / 1000
        reg_charge = results['charge_total_2050'][r, :, :, :, winter_idx, :, f].sum(axis = 1).sum(axis = 1).sum(axis = 1) / 1000
        reg_pv_gen = results['pv_gen_total_2050'][r, :, :, :, winter_idx, :, f].sum(axis = 1).sum(axis = 1).sum(axis = 1) / 1000
        reg_output = (reg_pv_gen - reg_charge + reg_discharge).mean(axis = 0) / 1000

        tot_discharge = results['discharge_total_2050'][r, :, :, :, summer_idx, :, f].sum(axis = 1).sum(axis = 1).sum(axis = 1) / 1000
        tot_charge = results['charge_total_2050'][r, :, :, :, summer_idx, :, f].sum(axis = 1).sum(axis = 1).sum(axis = 1) / 1000
        tot_pv_gen = results['pv_gen_total_2050'][r, :, :, :, summer_idx, :, f].sum(axis = 1).sum(axis = 1).sum(axis = 1) / 1000
        tot_output = (tot_pv_gen - tot_charge + tot_discharge).mean(axis = 0) / 1000

        cents = f
        lbl = str(cents) + ' cents'

        if r > 3:
            col = r - 4
            row = 1
        else:
            col = r
            row = 0
        axes[row, col].plot(np.asarray(tl_out),
                  reg_output,
                  label=lbl,
                  color=colour,
                  linewidth=1.5,
                  linestyle=linestyle)
        axes[row, col].set_title(reg, fontstyle='italic', fontsize=14)

        axes[row, col].set_xlim([tl_out[0], tl_out[-1]]);
        # min_npv = results['battery_cum'][:, :, :, 1, 0, 0, 16:].min() - 200
        # max_npv = results['battery_cum'][:, :, :, 1, 0, 0, 16:].max() + 200
        # axes[row, col].set_ylim([min_npv, max_npv]);
        axes[row, col].set_ylim([0, max(tot_output) * 1.1]);
        axes[row, col].grid(color = 'grey', alpha=0.4, linestyle = '--', linewidth = 0.5)
        axes[row, col].tick_params('x', labelrotation=60)
        # axes[row, col].label_outer()
        axes[row, col].set_xticks([0, 6, 12, 18])

axes[0, 0].set_ylabel("GW")
axes[1, 0].set_ylabel("GW")

h1, l1 = axes[0, 0].get_legend_handles_labels()
# l1[4] = l1[4].split(' ')[0] + '\n' + l1[4].split(' ')[1]
handles = h1[1::2] + [h1[0]]
labels = l1[1::2] + [l1[0]]
#labels[12] = labels[12].split(';')[0] + '\n' + labels[12].split(';')[1]

# l1 = [lab.split(';')[0]+'\n' +lab.split(';')[1] for lab in l1]

fig.legend(handles=h1,
           labels=l1,
           loc="upper right",
           bbox_to_anchor=(0.87, 0.85),
           frameon=False,
           borderaxespad=0.,
           ncol=1,
           title="Feed-in-tariff",
           fontsize=12)


fig.subplots_adjust(hspace=0.15, wspace=0.3, right=0.75, bottom=0.2, left=0.17)

fig.savefig(fp)
plt.show()





# ########################################################################################
# ############################### System impacts over time ###############################
# ########################################################################################


# ################# Summer System impacts

# fn = "Battery_charge_summer.jpg"
# fp = os.path.join('figures', fn)

# # Figure size
# figsize = (12.0, 8)
# # Create subplot
# fig, axes = plt.subplots(nrows=2, ncols=4,
#                          figsize=figsize,
#                          sharex=True, sharey=False)


# tl_out = np.arange(0, 24)

# # colors = ["green", "black", "firebrick", "gray", "blue", "aqua", "red", "orange", "magenta", "navy", "tan", "maroon", "peru", "olive", "khaki"]
# line_info  = {}

# colors = ['#C5446E', '#49C9C5', '#AAB71D']#, '#009FE3', '#909090']
# colors = [cmap(10), cmap(40), cmap(80), cmap(220), cmap(320)]

# for r, reg in enumerate(list(titles['country'])):
#     for y, year in enumerate(list([2030, 2035, 2040, 2045, 2050])):
#         i=0
#         # Set color
#         colour = colors[y]
#         # Set line style
#         linestyle = '-'
#         period = year - 2008

#         reg_discharge = results['discharge_total'][r, :, :, :, summer_idx, :, period].sum(axis = 1).sum(axis = 1).sum(axis = 1).mean(axis = 0) / 1000
#         reg_charge = results['charge_total'][r, :, :, :, summer_idx, :, period].sum(axis = 1).sum(axis = 1).sum(axis = 1).mean(axis = 0) / 1000
#         reg_output = reg_charge - reg_discharge

#         lbl = str(year)

#         if r > 3:
#             col = r - 4
#             row = 1
#         else:
#             col = r
#             row = 0
#         axes[row, col].plot(np.asarray(tl_out),
#                   reg_output,
#                   label=lbl,
#                   color=colour,
#                   linewidth=1.5,
#                   linestyle=linestyle)
#         axes[row, col].set_title(reg, fontstyle='italic', fontsize=14)

#         axes[row, col].set_xlim([tl_out[0], tl_out[-1]]);
#         # min_npv = results['battery_cum'][:, :, :, 1, 0, 0, 16:].min() - 200
#         # max_npv = results['battery_cum'][:, :, :, 1, 0, 0, 16:].max() + 200
#         # axes[row, col].set_ylim([min_npv, max_npv]);
#         axes[row, col].grid(color = 'grey', alpha=0.4, linestyle = '--', linewidth = 0.5)
#         axes[row, col].tick_params('x', labelrotation=60)
#         # axes[row, col].label_outer()
#         axes[row, col].set_xticks([0, 6, 12, 18])

#     axes[0, 0].set_ylabel("MW")
#     axes[1, 0].set_ylabel("MW")

#     h1, l1 = axes[0, 0].get_legend_handles_labels()
#     # l1[4] = l1[4].split(' ')[0] + '\n' + l1[4].split(' ')[1]
#     handles = h1[0:5]
#     labels = l1[0:5]
#     #labels[12] = labels[12].split(';')[0] + '\n' + labels[12].split(';')[1]

#     # l1 = [lab.split(';')[0]+'\n' +lab.split(';')[1] for lab in l1]

#     fig.legend(handles=handles,
#                labels=labels,
#                loc="lower center",
#                bbox_to_anchor=(0.45, 0.1),
#                frameon=False,
#                borderaxespad=0.,
#                ncol=5,
#                title="Year",
#                fontsize=12)

# fig.subplots_adjust(hspace=0.15, wspace=0.3, right=0.75, bottom=0.2, left=0.17)

# fig.savefig(fp)
# plt.show()



# ################# Winter System impacts

# fn = "Battery_charge_winter.jpg"
# fp = os.path.join('figures', fn)

# # Figure size
# figsize = (12.0, 8)
# # Create subplot
# fig, axes = plt.subplots(nrows=2, ncols=4,
#                          figsize=figsize,
#                          sharex=True, sharey=False)


# tl_out = np.arange(0, 24)

# # colors = ["green", "black", "firebrick", "gray", "blue", "aqua", "red", "orange", "magenta", "navy", "tan", "maroon", "peru", "olive", "khaki"]
# line_info  = {}

# colors = ['#C5446E', '#49C9C5', '#AAB71D']#, '#009FE3', '#909090']
# colors = [cmap(10), cmap(40), cmap(80), cmap(220), cmap(320)]

# for r, reg in enumerate(list(titles['country'])):
#     for y, year in enumerate(list([2030, 2035, 2040, 2045, 2050])):
#         i=0
#         # Set color
#         colour = colors[y]
#         # Set line style
#         linestyle = '-'
#         period = year - 2008

#         reg_discharge = results['discharge_total'][r, :, :, :, winter_idx, :, period].sum(axis = 1).sum(axis = 1).sum(axis = 1).mean(axis = 0) / 1000
#         reg_charge = results['charge_total'][r, :, :, :, winter_idx, :, period].sum(axis = 1).sum(axis = 1).sum(axis = 1).mean(axis = 0) / 1000
#         reg_output = reg_charge - reg_discharge

#         lbl = str(year)

#         if r > 3:
#             col = r - 4
#             row = 1
#         else:
#             col = r
#             row = 0
#         axes[row, col].plot(np.asarray(tl_out),
#                   reg_output,
#                   label=lbl,
#                   color=colour,
#                   linewidth=1.5,
#                   linestyle=linestyle)
#         axes[row, col].set_title(reg, fontstyle='italic', fontsize=14)

#         axes[row, col].set_xlim([tl_out[0], tl_out[-1]]);
#         # min_npv = results['battery_cum'][:, :, :, 1, 0, 0, 16:].min() - 200
#         # max_npv = results['battery_cum'][:, :, :, 1, 0, 0, 16:].max() + 200
#         # axes[row, col].set_ylim([min_npv, max_npv]);
#         axes[row, col].grid(color = 'grey', alpha=0.4, linestyle = '--', linewidth = 0.5)
#         axes[row, col].tick_params('x', labelrotation=60)
#         # axes[row, col].label_outer()
#         axes[row, col].set_xticks([0, 6, 12, 18])

#     axes[0, 0].set_ylabel("MW")
#     axes[1, 0].set_ylabel("MW")

#     h1, l1 = axes[0, 0].get_legend_handles_labels()
#     # l1[4] = l1[4].split(' ')[0] + '\n' + l1[4].split(' ')[1]
#     handles = h1[0:5]
#     labels = l1[0:5]
#     #labels[12] = labels[12].split(';')[0] + '\n' + labels[12].split(';')[1]

#     # l1 = [lab.split(';')[0]+'\n' +lab.split(';')[1] for lab in l1]

#     fig.legend(handles=handles,
#                labels=labels,
#                loc="lower center",
#                bbox_to_anchor=(0.45, 0.05),
#                frameon=False,
#                borderaxespad=0.,
#                ncol=5,
#                title="Year",
#                fontsize=12)


# fig.subplots_adjust(hspace=0.15, wspace=0.3, right=0.75, bottom=0.2, left=0.17)

# fig.savefig(fp)
# plt.show()



# ##################################################

# # System impacts

# fn = "Total_system_impact_summer.jpg"
# fp = os.path.join('figures', fn)

# # Figure size
# figsize = (12.0, 8)
# # Create subplot
# fig, axes = plt.subplots(nrows=2, ncols=4,
#                          figsize=figsize,
#                          sharex=True, sharey=False)


# tl_out = np.arange(0, 24)

# # colors = ["green", "black", "firebrick", "gray", "blue", "aqua", "red", "orange", "magenta", "navy", "tan", "maroon", "peru", "olive", "khaki"]
# line_info  = {}

# colors = ['#C5446E', '#49C9C5', '#AAB71D']#, '#009FE3', '#909090']
# colors = [cmap(10), cmap(40), cmap(80), cmap(220), cmap(320)]

# for r, reg in enumerate(list(titles['country'])):
#     for y, year in enumerate(list([2030, 2035, 2040, 2045, 2050])):
#         i=0
#         # Set color
#         colour = colors[y]
#         # Set line style
#         linestyle = '-'
#         period = year - 2008

#         reg_discharge = results['discharge_total'][r, :, :, :, summer_idx, :, period].sum(axis = 1).sum(axis = 1).sum(axis = 1) / 1000
#         reg_charge = results['charge_total'][r, :, :, :, summer_idx, :, period].sum(axis = 1).sum(axis = 1).sum(axis = 1) / 1000
#         reg_pv_gen = results['pv_gen_total'][r, :, :, :, summer_idx, :, period].sum(axis = 1).sum(axis = 1).sum(axis = 1) / 1000
#         reg_output = (reg_pv_gen - reg_charge + reg_discharge).mean(axis = 0) / 1000

#         lbl = str(year)

#         if r > 3:
#             col = r - 4
#             row = 1
#         else:
#             col = r
#             row = 0
#         axes[row, col].plot(np.asarray(tl_out),
#                   reg_output,
#                   label=lbl,
#                   color=colour,
#                   linewidth=1.5,
#                   linestyle=linestyle)
#         axes[row, col].set_title(reg, fontstyle='italic', fontsize=14)

#         axes[row, col].set_xlim([tl_out[0], tl_out[-1]]);
#         # min_npv = results['battery_cum'][:, :, :, 1, 0, 0, 16:].min() - 200
#         # max_npv = results['battery_cum'][:, :, :, 1, 0, 0, 16:].max() + 200
#         # axes[row, col].set_ylim([min_npv, max_npv]);
#         axes[row, col].grid(color = 'grey', alpha=0.4, linestyle = '--', linewidth = 0.5)
#         axes[row, col].tick_params('x', labelrotation=60)
#         # axes[row, col].label_outer()
#         axes[row, col].set_xticks([0, 6, 12, 18])

#     axes[0, 0].set_ylabel("GW")
#     axes[1, 0].set_ylabel("GW")

#     h1, l1 = axes[0, 0].get_legend_handles_labels()
#     # l1[4] = l1[4].split(' ')[0] + '\n' + l1[4].split(' ')[1]
#     handles = h1[0:5]
#     labels = l1[0:5]
#     #labels[12] = labels[12].split(';')[0] + '\n' + labels[12].split(';')[1]

#     # l1 = [lab.split(';')[0]+'\n' +lab.split(';')[1] for lab in l1]

#     fig.legend(handles=handles,
#                labels=labels,
#                loc="lower center",
#                bbox_to_anchor=(0.45, 0.05),
#                frameon=False,
#                borderaxespad=0.,
#                ncol=5,
#                title="Year",
#                fontsize=12)


# fig.subplots_adjust(hspace=0.15, wspace=0.3, right=0.75, bottom=0.2, left=0.17)

# fig.savefig(fp)
# plt.show()



# ##################################################

# # System impacts

# fn = "Total_system_impact_winter.jpg"
# fp = os.path.join('figures', fn)

# # Figure size
# figsize = (12.0, 8)
# # Create subplot
# fig, axes = plt.subplots(nrows=2, ncols=4,
#                          figsize=figsize,
#                          sharex=True, sharey=False)


# tl_out = np.arange(0, 24)

# # colors = ["green", "black", "firebrick", "gray", "blue", "aqua", "red", "orange", "magenta", "navy", "tan", "maroon", "peru", "olive", "khaki"]
# line_info  = {}

# colors = ['#C5446E', '#49C9C5', '#AAB71D']#, '#009FE3', '#909090']
# colors = [cmap(10), cmap(40), cmap(80), cmap(220), cmap(320)]

# for r, reg in enumerate(list(titles['country'])):
#     for y, year in enumerate(list([2030, 2035, 2040, 2045, 2050])):
#         i=0
#         # Set color
#         colour = colors[y]
#         # Set line style
#         linestyle = '-'
#         period = year - 2008

#         reg_discharge = results['discharge_total'][r, :, :, :, winter_idx, :, period].sum(axis = 1).sum(axis = 1).sum(axis = 1) / 1000
#         reg_charge = results['charge_total'][r, :, :, :, winter_idx, :, period].sum(axis = 1).sum(axis = 1).sum(axis = 1) / 1000
#         reg_pv_gen = results['pv_gen_total'][r, :, :, :, winter_idx, :, period].sum(axis = 1).sum(axis = 1).sum(axis = 1) / 1000
#         reg_output = (reg_pv_gen - reg_charge + reg_discharge).mean(axis = 0) / 1000

#         lbl = str(year)

#         if r > 3:
#             col = r - 4
#             row = 1
#         else:
#             col = r
#             row = 0
#         axes[row, col].plot(np.asarray(tl_out),
#                   reg_output,
#                   label=lbl,
#                   color=colour,
#                   linewidth=1.5,
#                   linestyle=linestyle)
#         axes[row, col].set_title(reg, fontstyle='italic', fontsize=14)

#         axes[row, col].set_xlim([tl_out[0], tl_out[-1]]);
#         # min_npv = results['battery_cum'][:, :, :, 1, 0, 0, 16:].min() - 200
#         # max_npv = results['battery_cum'][:, :, :, 1, 0, 0, 16:].max() + 200
#         # axes[row, col].set_ylim([min_npv, max_npv]);
#         axes[row, col].grid(color = 'grey', alpha=0.4, linestyle = '--', linewidth = 0.5)
#         axes[row, col].tick_params('x', labelrotation=60)
#         # axes[row, col].label_outer()
#         axes[row, col].set_xticks([0, 6, 12, 18])

#     axes[0, 0].set_ylabel("GW")
#     axes[1, 0].set_ylabel("GW")

#     h1, l1 = axes[0, 0].get_legend_handles_labels()
#     # l1[4] = l1[4].split(' ')[0] + '\n' + l1[4].split(' ')[1]
#     handles = h1[0:5]
#     labels = l1[0:5]
#     #labels[12] = labels[12].split(';')[0] + '\n' + labels[12].split(';')[1]

#     # l1 = [lab.split(';')[0]+'\n' +lab.split(';')[1] for lab in l1]

#     fig.legend(handles=handles,
#                labels=labels,
#                loc="lower center",
#                bbox_to_anchor=(0.45, 0.05),
#                frameon=False,
#                borderaxespad=0.,
#                ncol=5,
#                title="Year",
#                fontsize=12)


# fig.subplots_adjust(hspace=0.15, wspace=0.3, right=0.75, bottom=0.2, left=0.17)

# fig.savefig(fp)
# plt.show()


########################################################################################
##################################### Charge level #####################################
########################################################################################

################# Summer

fn = "Charge_level_summer.jpg"
fp = os.path.join('figures', fn)

# Figure size
figsize = (12.0, 8)
# Create subplot
fig, axes = plt.subplots(nrows=2, ncols=4,
                         figsize=figsize,
                         sharex=True, sharey='row')


# colors = ["green", "black", "firebrick", "gray", "blue", "aqua", "red", "orange", "magenta", "navy", "tan", "maroon", "peru", "olive", "khaki"]
line_info  = {}

colors = ['#C5446E', '#49C9C5', '#AAB71D'] #, '#009FE3', '#909090']
colors = [cmap(0), cmap(80), cmap(220)]

for r, reg in enumerate(list(titles['country'])):
    for c, cons in enumerate(list(titles['cons_size'])):
        for p, prof in enumerate(list(titles['profile_type'])):
            i=0
            # Set color
            colour = colors[c]
            # Set line style
            if prof == 'normal':
                linestyle = '--'
            elif prof == 'heating':
                linestyle = '-'
            elif prof == 'cooling':
                linestyle = ':'

            reg_profile = results['charge_level'][r, p, c, 1, summer_idx, :, 0].mean(axis = 0)
            reg_profile = reg_profile / results['battery_size'][r, p, c, 1, 0, 0, 0]
            lbl = cons + ' ' + prof

            if r > 3:
                col = r - 4
                row = 1
            else:
                col = r
                row = 0
            axes[row, col].plot(np.asarray(range(0, 24)),
                      reg_profile,
                      label=lbl,
                      color=colour,
                      linewidth=1.5,
                      linestyle=linestyle)
            axes[row, col].set_title(reg, fontstyle='italic', fontsize=14)

            axes[row, col].set_xlim([0, 23]);
            min_load = 0
            max_load = 1
            axes[row, col].set_ylim([min_load, max_load]);
            axes[row, col].grid(color = 'grey', alpha=0.4, linestyle = '--', linewidth = 0.5)
            axes[row, col].tick_params('x', labelrotation=60)
            # axes[row, col].label_outer()
            axes[row, col].set_xticks([0, 6, 12, 18])

    axes[0, 0].set_ylabel("kW")
    axes[1, 0].set_ylabel("kW")

    h1, l1 = axes[0, 0].get_legend_handles_labels()
    # l1[4] = l1[4].split(' ')[0] + '\n' + l1[4].split(' ')[1]
    handles = h1[1::2] + [h1[0]]
    labels = l1[1::2] + [l1[0]]
    #labels[12] = labels[12].split(';')[0] + '\n' + labels[12].split(';')[1]

    # l1 = [lab.split(';')[0]+'\n' +lab.split(';')[1] for lab in l1]

    fig.legend(handles=h1,
               labels=l1,
               loc="upper right",
               bbox_to_anchor=(0.9, 0.85),
               frameon=False,
               borderaxespad=0.,
               ncol=1,
               title="Profiles",
               fontsize=12)


fig.subplots_adjust(hspace=0.15, wspace=0.3, right=0.75, bottom=0.2, left=0.17)

fig.savefig(fp)
plt.show()

################# Winter

fn = "Charge_level_winter.jpg"
fp = os.path.join('figures', fn)

# Figure size
figsize = (12.0, 8)
# Create subplot
fig, axes = plt.subplots(nrows=2, ncols=4,
                         figsize=figsize,
                         sharex=True, sharey='row')


# colors = ["green", "black", "firebrick", "gray", "blue", "aqua", "red", "orange", "magenta", "navy", "tan", "maroon", "peru", "olive", "khaki"]
line_info  = {}

colors = ['#C5446E', '#49C9C5', '#AAB71D'] #, '#009FE3', '#909090']
colors = [cmap(0), cmap(80), cmap(220)]

for r, reg in enumerate(list(titles['country'])):
    for c, cons in enumerate(list(titles['cons_size'])):
        for p, prof in enumerate(list(titles['profile_type'])):
            i=0
            # Set color
            colour = colors[c]
            # Set line style
            if prof == 'normal':
                linestyle = '--'
            elif prof == 'heating':
                linestyle = '-'
            elif prof == 'cooling':
                linestyle = ':'

            reg_profile = results['charge_level'][r, p, c, 1, winter_idx, :, 0].mean(axis = 0)
            reg_profile = reg_profile / results['battery_size'][r, p, c, 1, 0, 0, 0]
            lbl = cons + ' ' + prof

            if r > 3:
                col = r - 4
                row = 1
            else:
                col = r
                row = 0
            axes[row, col].plot(np.asarray(range(0, 24)),
                      reg_profile,
                      label=lbl,
                      color=colour,
                      linewidth=1.5,
                      linestyle=linestyle)
            axes[row, col].set_title(reg, fontstyle='italic', fontsize=14)

            axes[row, col].set_xlim([0, 23]);
            min_load = 0
            max_load = 1
            axes[row, col].set_ylim([min_load, max_load]);
            axes[row, col].grid(color = 'grey', alpha=0.4, linestyle = '--', linewidth = 0.5)
            axes[row, col].tick_params('x', labelrotation=60)
            # axes[row, col].label_outer()
            axes[row, col].set_xticks([0, 6, 12, 18])

    axes[0, 0].set_ylabel("kW")
    axes[1, 0].set_ylabel("kW")

    h1, l1 = axes[0, 0].get_legend_handles_labels()
    # l1[4] = l1[4].split(' ')[0] + '\n' + l1[4].split(' ')[1]
    handles = h1[1::2] + [h1[0]]
    labels = l1[1::2] + [l1[0]]
    #labels[12] = labels[12].split(';')[0] + '\n' + labels[12].split(';')[1]

    # l1 = [lab.split(';')[0]+'\n' +lab.split(';')[1] for lab in l1]

    fig.legend(handles=h1,
               labels=l1,
               loc="upper right",
               bbox_to_anchor=(0.9, 0.85),
               frameon=False,
               borderaxespad=0.,
               ncol=1,
               title="Profiles",
               fontsize=12)


fig.subplots_adjust(hspace=0.15, wspace=0.3, right=0.75, bottom=0.2, left=0.17)

fig.savefig(fp)
plt.show()

# # Summer
# summer_charge_level = pd.DataFrame(d6_results['charge_level'][pest_idx, 1, 1, summer_idx, :, 0], columns = titles['hour'],
#                                    index = summer)
# summer_charge_level.quantile(q = 0.25).plot(xlabel = 'Hour', ylabel = 'Battery charge level (kWh)',
#                 figsize = (16, 10), linewidth = 2, ylim = (0, 5), style = '--', color = 'darkred',
#                 xticks = range(0, 25, 4)).legend(loc = 'upper left', fontsize = 14)
# summer_charge_level.quantile(q = 0.5).plot(xlabel = 'Hour', ylabel = 'Battery charge level (kWh)',
#                 figsize = (16, 10), linewidth = 2, ylim = (0, 5), color = 'red',
#                 xticks = range(0, 25, 4)).legend(loc = 'upper left', fontsize = 14)
# summer_charge_level.quantile(q = 0.75).plot(xlabel = 'Hour', ylabel = 'Battery charge level (kWh)',
#                 figsize = (16, 10), linewidth = 2, ylim = (0, 5), style = '--', color = 'darkred',
#                 xticks = range(0, 25, 4)).legend(loc = 'upper left', fontsize = 14)

# d = summer_charge_level.quantile(q = 0.25).index.values
# plt.fill_between(d, summer_charge_level.quantile(q = 0.75), summer_charge_level.quantile(q = 0.25),
#                 where=summer_charge_level.quantile(q = 0.75) >= summer_charge_level.quantile(q = 0.25),
#                 facecolor='red', alpha=0.2, interpolate=True)
# plt.plot(d, summer_charge_level.quantile(q = 0.5), 'k-', color = 'red')
# plt.plot(d, summer_charge_level.quantile(q = 0.25), '--', color = 'darkred')
# plt.plot(d, summer_charge_level.quantile(q = 0.75), '--', color = 'darkred')
# plt.ylim(0, 5)
# plt.xticks(range(0, 25, 4))
# plt.show()

# Sytem impacts
# discharge = pd.DataFrame(d6_results['discharge_total'][:, :, :, :, :, :].sum(axis = 0).sum(axis = 0).sum(axis = 0).mean(axis = 0) / 1000, index = titles['hour'],
#               columns = timeline)[range(2020, 2051)]
# discharge[range(2022, 2051, 2)].plot(xlabel = 'Hour', ylabel = 'Battery output (MW)',
#                 figsize = (16, 10), linewidth = 2, ylim = (0, 110), colormap = 'RdGy',
#                 xticks = range(0, 25, 4)).legend(loc = 'upper left', fontsize = 14)

# Summer
# summer_discharge = pd.DataFrame(results['discharge_total'][:, :, :, :, summer_idx, :, :].sum(axis = 0).sum(axis = 0).sum(axis = 0).sum(axis = 0).mean(axis = 0) / 1000, index = titles['hour'],
#               columns = timeline)[range(2020, 2051)]

# summer_discharge[range(2024, 2051, 2)].plot(xlabel = 'Hour', ylabel = 'Battery output (MW)',
#                 figsize = (16, 10), linewidth = 2, ylim = (0, 10), colormap = colourmap,
#                 xticks = range(0, 25, 4)).legend(loc = 'upper left', fontsize = 14)

# Winter
# winter_discharge = pd.DataFrame(d6_results['discharge_total'][:, :, :, winter_idx, :, :].sum(axis = 0).sum(axis = 0).sum(axis = 0).mean(axis = 0) / 1000, index = titles['hour'],
#               columns = timeline)[range(2020, 2051)]
# winter_discharge[range(2024, 2051, 2)].plot(xlabel = 'Hour', ylabel = 'Battery output (MW)',
#                 figsize = (16, 10), linewidth = 2, ylim = (0, 10), colormap = colourmap,
#                 xticks = range(0, 25, 4)).legend(loc = 'upper left', fontsize = 14)

# Do not allow to go below 20%
# Reasoning for NPV distribution --> batteries can only be installed for 3 phase systems --> extra cost
# Estimate how would the RLDC would look liek if all current PV owners had batteries
# Extra balacing service from batteries

# charge = pd.DataFrame(d6_results['charge_total'][:, :, :, :, :, :].sum(axis = 0).sum(axis = 0).sum(axis = 0).mean(axis = 0) / 1000, index = titles['hour'],
#               columns = timeline)[range(2020, 2051)]

# charge[range(2024, 2051, 2)].plot(xlabel = 'Hour', ylabel = 'Battery input (MW)',
#                                 figsize = (16, 10), linewidth = 2,
#                                 xticks = range(0, 25, 4)).legend(loc = 'upper left', fontsize = 14)

# Summer
# summer_charge = pd.DataFrame(-d6_results['charge_total'][:, :, :, summer_idx, :, :].sum(axis = 0).sum(axis = 0).sum(axis = 0).mean(axis = 0) / 1000, index = titles['hour'],
#               columns = timeline)[range(2020, 2051)]
# summer_charge[range(2024, 2051, 2)].plot(xlabel = 'Hour', ylabel = 'Battery output (MW)',
#                 figsize = (16, 10), linewidth = 2, ylim = (-20, 0), colormap = colourmap,
#                 xticks = range(0, 25, 4)).legend(loc = 'upper left', fontsize = 14)

# Winter
# winter_charge = pd.DataFrame(-d6_results['charge_total'][:, :, :, winter_idx, :, :].sum(axis = 0).sum(axis = 0).sum(axis = 0).mean(axis = 0) / 1000, index = titles['hour'],
#               columns = timeline)[range(2020, 2051)]
# winter_charge[range(2024, 2051, 2)].plot(xlabel = 'Hour', ylabel = 'Battery output (MW)',
#                 figsize = (16, 10), linewidth = 2, ylim = (-20, 0), colormap = colourmap,
#                 xticks = range(0, 25, 4)).legend(loc = 'upper left', fontsize = 14)




# charge_level = pd.DataFrame(d6_results['charge_level'][:, :, :, :, :, 0].mean(axis = 0).mean(axis = 0).mean(axis = 1), columns = titles['hour'],
#               index = titles['cons_size'])

# charge_level.T.plot(xlabel = 'Hour', ylabel = 'Avg. battery charge level (MW)',
#                                 figsize = (16, 10), linewidth = 2,
#                                 xticks = range(0, 25, 4)).legend(loc = 'upper left', fontsize = 14)

# # Summer
# summer_charge_level = pd.DataFrame(d6_results['charge_level'][pest_idx, 1, 1, summer_idx, :, 0], columns = titles['hour'],
#                                    index = summer)
# summer_charge_level.quantile(q = 0.25).plot(xlabel = 'Hour', ylabel = 'Battery charge level (kWh)',
#                 figsize = (16, 10), linewidth = 2, ylim = (0, 5), style = '--', color = 'darkred',
#                 xticks = range(0, 25, 4)).legend(loc = 'upper left', fontsize = 14)
# summer_charge_level.quantile(q = 0.5).plot(xlabel = 'Hour', ylabel = 'Battery charge level (kWh)',
#                 figsize = (16, 10), linewidth = 2, ylim = (0, 5), color = 'red',
#                 xticks = range(0, 25, 4)).legend(loc = 'upper left', fontsize = 14)
# summer_charge_level.quantile(q = 0.75).plot(xlabel = 'Hour', ylabel = 'Battery charge level (kWh)',
#                 figsize = (16, 10), linewidth = 2, ylim = (0, 5), style = '--', color = 'darkred',
#                 xticks = range(0, 25, 4)).legend(loc = 'upper left', fontsize = 14)

# d = summer_charge_level.quantile(q = 0.25).index.values
# plt.fill_between(d, summer_charge_level.quantile(q = 0.75), summer_charge_level.quantile(q = 0.25),
#                 where=summer_charge_level.quantile(q = 0.75) >= summer_charge_level.quantile(q = 0.25),
#                 facecolor='red', alpha=0.2, interpolate=True)
# plt.plot(d, summer_charge_level.quantile(q = 0.5), 'k-', color = 'red')
# plt.plot(d, summer_charge_level.quantile(q = 0.25), '--', color = 'darkred')
# plt.plot(d, summer_charge_level.quantile(q = 0.75), '--', color = 'darkred')
# plt.ylim(0, 5)
# plt.xticks(range(0, 25, 4))
# plt.show()


# # Winter
# winter_charge_level = pd.DataFrame(d6_results['charge_level'][pest_idx, 1, 1, winter_idx, :, 0], columns = titles['hour'],
#                                    index = winter)
# winter_charge_level.quantile(q = 0.25).plot(xlabel = 'Hour', ylabel = 'Battery charge level (kWh)',
#                 figsize = (16, 10), linewidth = 2, ylim = (0, 5), style = '--', color = 'darkred',
#                 xticks = range(0, 25, 4)).legend(loc = 'upper left', fontsize = 14)
# winter_charge_level.quantile(q = 0.5).plot(xlabel = 'Hour', ylabel = 'Battery charge level (kWh)',
#                 figsize = (16, 10), linewidth = 2, ylim = (0, 5), color = 'red',
#                 xticks = range(0, 25, 4)).legend(loc = 'upper left', fontsize = 14)
# winter_charge_level.quantile(q = 0.75).plot(xlabel = 'Hour', ylabel = 'Battery charge level (kWh)',
#                 figsize = (16, 10), linewidth = 2, ylim = (0, 5), style = '--', color = 'darkred',
#                 xticks = range(0, 25, 4)).legend(loc = 'upper left', fontsize = 14)



# d = winter_charge_level.quantile(q = 0.25).index.values
# plt.fill_between(d, winter_charge_level.quantile(q = 0.75), winter_charge_level.quantile(q = 0.25),
#                 where=winter_charge_level.quantile(q = 0.75) >= winter_charge_level.quantile(q = 0.25),
#                 facecolor='red', alpha=0.2, interpolate=True)
# plt.plot(d, winter_charge_level.quantile(q = 0.5), 'k-', color = 'red')
# plt.plot(d, winter_charge_level.quantile(q = 0.25), '--', color = 'darkred')
# plt.plot(d, winter_charge_level.quantile(q = 0.75), '--', color = 'darkred')
# plt.ylim(0, 5)
# plt.xticks(range(0, 25, 4))
# plt.show()






# General profile with battery
# Summer

# summer_profile = pd.Series(results['profiles'][0, 1, summer_idx, :].mean(axis = 0), index = titles['hour']) * pest_con / results['profiles'][0, 0, :, :].sum()
# summer_charge = pd.Series(d6_results['charge'][pest_idx, 1, 1, summer_idx, :, 0].mean(axis = 0), index = titles['hour'])
# summer_discharge = pd.Series(d6_results['discharge'][pest_idx, 1, 1, summer_idx, :, 0].mean(axis = 0), index = titles['hour'])
# summer_pv = pd.Series(adj_pv_gen[summer_idx, :].mean(axis = 0), index = titles['hour'])
# summer_output = summer_discharge - summer_charge
# generation = (summer_discharge + summer_pv)

# summer_profile = pd.Series(np.median(results['profiles'][0, 1, summer_idx, :], axis = 0), index = titles['hour']) * pest_con / results['profiles'][0, 0, :, :].sum()
# summer_charge = pd.Series(np.median(d6_results['charge'][pest_idx, 1, 1, summer_idx, :, 0], axis = 0), index = titles['hour'])
# summer_discharge = pd.Series(np.median(d6_results['discharge'][pest_idx, 1, 1, summer_idx, :, 0], axis = 0), index = titles['hour'])
# summer_pv = pd.Series(np.median(adj_pv_gen[summer_idx, :], axis = 0), index = titles['hour'])
# summer_output = summer_discharge - summer_charge
# generation = (summer_discharge + summer_pv)


# d = summer_profile.index.values

# summer_profile.plot(xlabel = 'Hour', ylabel = 'Avg. hourly load (kW)',
#                 figsize = (16, 10), linewidth = 2, ylim = (0, 5), color = 'red',
#                 xticks = range(0, 25, 4), fontsize = 20).legend(loc = 'upper left', fontsize = 20)
# plt.fill_between(d, summer_profile - summer_output, summer_pv,
#                   where = summer_profile - summer_output >= summer_pv,
#                   facecolor='black', alpha=0.7, interpolate=True)
# plt.fill_between(d, summer_profile, summer_profile - summer_output,
#                   where = summer_profile <= summer_profile - summer_output,
#                   facecolor='darkorange', alpha=0.9, interpolate=True)
# plt.fill_between(d, summer_profile - summer_output, summer_pv,
#                   where = summer_profile - summer_output <= (summer_pv),
#                   facecolor='orangered', alpha=0.9, interpolate=True)
# plt.fill_between(d, summer_profile - summer_output, summer_profile,
#                   where = summer_profile - summer_output <= (summer_profile),
#                   facecolor='grey', alpha=1, interpolate=True)

# plt.fill_between(d, summer_pv, facecolor='coral', alpha=0.7, interpolate=True)

# plt.plot(d, summer_pv, '--', color = 'darkred', linewidth = 2)
# plt.plot(d, summer_profile - summer_output, '-k', color = 'black', linewidth = 2)
# plt.plot(d, summer_profile, 'k-', color = 'red', linewidth = 2)
# plt.ylim(-0, 1.6)
# # plt.xlabel('Hour')
# # plt.ylabel('Avg. hourly load (kW)')
# # plt.figsize((8, 5))
# plt.ylabel('Avg. hourly load (kW)',fontdict={'fontsize':20})
# plt.xlabel('Hour',fontdict={'fontsize':20})
# plt.xticks(range(0, 25, 4))
# plt.show()


# # Winter

# winter_profile = pd.Series(results['profiles'][0, 1, winter_idx, :].mean(axis = 0), index = titles['hour']) * pest_con / results['profiles'][0, 0, :, :].sum()
# winter_charge = pd.Series(d6_results['charge'][pest_idx, 1, 1, winter_idx, :, 0].mean(axis = 0), index = titles['hour'])
# winter_discharge = pd.Series(d6_results['discharge'][pest_idx, 1, 1, winter_idx, :, 0].mean(axis = 0), index = titles['hour'])
# winter_pv = pd.Series(adj_pv_gen[winter_idx, :].mean(axis = 0), index = titles['hour'])
# winter_output = winter_discharge - winter_charge
# generation = (winter_discharge + winter_pv)

# # winter_profile = pd.Series(np.median(results['profiles'][0, 1, winter_idx, :], axis = 0), index = titles['hour']) * pest_con / results['profiles'][0, 0, :, :].sum()
# # winter_charge = pd.Series(np.median(d6_results['charge'][pest_idx, 1, 1, winter_idx, :, 0], axis = 0), index = titles['hour'])
# # winter_discharge = pd.Series(np.median(d6_results['discharge'][pest_idx, 1, 1, winter_idx, :, 0], axis = 0), index = titles['hour'])
# # winter_pv = pd.Series(np.median(adj_pv_gen[winter_idx, :], axis = 0), index = titles['hour'])
# # winter_output = winter_discharge - winter_charge
# # generation = (winter_discharge + winter_pv)


# d = winter_profile.index.values
# winter_profile.plot(xlabel = 'Hour', ylabel = 'Avg. hourly load (kW)',
#                 figsize = (16, 10), linewidth = 2, ylim = (0, 5), color = 'red',
#                 xticks = range(0, 25, 4)).legend(loc = 'upper left', fontsize = 14)
# plt.fill_between(d, winter_profile, winter_profile - winter_output,
#                   where = winter_profile <= winter_profile - winter_output,
#                   facecolor='black', alpha=0.8, interpolate=True)
# plt.fill_between(d, winter_profile - winter_output, winter_pv,
#                   where = winter_profile - winter_output <= (winter_pv),
#                   facecolor='red', alpha=0.7, interpolate=True)
# plt.fill_between(d, winter_profile - winter_output, winter_profile,
#                   where = winter_profile - winter_output <= (winter_profile),
#                   facecolor='grey', alpha=1, interpolate=True)
# plt.fill_between(d, winter_profile - winter_output, winter_pv,
#                   where = winter_profile - winter_output >= winter_pv,
#                   facecolor='black', alpha=0.7, interpolate=True)
# plt.fill_between(d, winter_pv, facecolor='darkred', alpha=0.7, interpolate=True)

# plt.plot(d, winter_pv, '--', color = 'darkred', linewidth = 2)
# plt.plot(d, winter_profile - winter_output, '-k', color = 'black', linewidth = 2)
# plt.plot(d, winter_profile, 'k-', color = 'red', linewidth = 2)
# plt.ylim(-0, 1.6)
# # plt.xlabel('Hour')
# # plt.ylabel('Avg. hourly load (kW)')
# # plt.figsize((8, 5))
# plt.xticks(range(0, 25, 4))
# plt.show()


# Calculate peak shaving
# adj_load_2030 = results['load'][0, 0, :, :] + d6_results['charge_total'][:, :, :, :, :, 22].sum(axis = 0).sum(axis = 0).sum(axis = 0) / 1000 - d6_results['discharge_total'][:, :, :, :, :, 22].sum(axis = 0).sum(axis = 0).sum(axis = 0) / 1000
# adj_load_2040 = results['load'][0, 1, :, :] + d6_results['charge_total'][:, :, :, :, :, 32].sum(axis = 0).sum(axis = 0).sum(axis = 0) / 1000 - d6_results['discharge_total'][:, :, :, :, :, 32].sum(axis = 0).sum(axis = 0).sum(axis = 0) / 1000
# adj_load_2050 = results['load'][0, 2, :, :] + d6_results['charge_total'][:, :, :, :, :, 42].sum(axis = 0).sum(axis = 0).sum(axis = 0) / 1000 - d6_results['discharge_total'][:, :, :, :, :, 42].sum(axis = 0).sum(axis = 0).sum(axis = 0) / 1000
# adj_load = np.array([adj_load_2030, adj_load_2040, adj_load_2050])

# summer_load = pd.DataFrame(results['load'][0, :, summer_idx, :].mean(axis = 0), columns = titles['hour'],
#               index = titles['year'])
# summer_adj_load = pd.DataFrame(adj_load[:, summer_idx, :].mean(axis = 1), columns = titles['hour'],
#               index = titles['year'])
# summer_adj_load.index = ['2030 Adj.', '2040 Adj.', '2050 Adj.']

# summer_load.T.plot(xlabel = 'Hour', ylabel = 'Total projected load (MW)',
#                 figsize = (16, 10), linewidth = 3, ylim = (3000, 6000), colormap = colourmap,
#                 xticks = range(0, 25, 4)).legend(loc = 'upper left', fontsize = 14)
# summer_adj_load.loc['2030 Adj.'].plot(style = '--', linewidth = 2, color = 'magenta').legend(loc = 'upper left', fontsize = 14)
# summer_adj_load.loc['2040 Adj.'].plot(style = '--', linewidth = 2, color = 'darkgreen').legend(loc = 'upper left', fontsize = 14)
# summer_adj_load.loc['2050 Adj.'].plot(style = '--', linewidth = 2, color = 'orange').legend(loc = 'upper left', fontsize = 14)

# summer_adj_load.T.plot(xlabel = 'Hour', ylabel = 'Total projected load (MW)',
#                 figsize = (16, 10), linewidth = 2, ylim = (3000, 6000), colormap = colourmap,
#                 xticks = range(0, 25, 4)).legend(loc = 'upper left', fontsize = 14)


# winter_load = pd.DataFrame(results['load'][0, :, winter_idx, :].mean(axis = 0), columns = titles['hour'],
#               index = titles['year'])
# winter_adj_load = pd.DataFrame(adj_load[:, winter_idx, :].mean(axis = 1), columns = titles['hour'],
#               index = titles['year'])
# winter_adj_load.index = ['2030 Adj.', '2040 Adj.', '2050 Adj.']

# winter_load.T.plot(xlabel = 'Hour', ylabel = 'Total projected load (MW)',
#                 figsize = (16, 10), linewidth = 2, ylim = (3000, 6000), colormap = colourmap,
#                 xticks = range(0, 25, 4)).legend(loc = 'upper left', fontsize = 14)

# winter_adj_load.loc['2030 Adj.'].plot(style = '--', linewidth = 2, color = 'magenta').legend(loc = 'upper left', fontsize = 14)
# winter_adj_load.loc['2040 Adj.'].plot(style = '--', linewidth = 2, color = 'darkgreen').legend(loc = 'upper left', fontsize = 14)
# winter_adj_load.loc['2050 Adj.'].plot(style = '--', linewidth = 2, color = 'orange').legend(loc = 'upper left', fontsize = 14)




# winter_adj_load.T.plot(xlabel = 'Hour', ylabel = 'Total projected load (MW)',
#                 figsize = (16, 10), linewidth = 2, ylim = (3000, 6000), colormap = colourmap,
#                 xticks = range(0, 25, 4)).legend(loc = 'upper left', fontsize = 14)

# charge = pd.DataFrame(d6_results['charge_total'][:, :, :, :, :, 42].sum(axis = 0).sum(axis = 0).sum(axis = 0) / 1000, columns = titles['hour'],
#               index = titles['date'])
# discharge = pd.DataFrame(d6_results['discharge_total'][:, :, :, :, :, 42].sum(axis = 0).sum(axis = 0).sum(axis = 0) / 1000, columns = titles['hour'],
#               index = titles['date'])
# output = charge - discharge

# load = pd.DataFrame(results['load'][0, 0, :, :], columns = titles['hour'],
#               index = titles['date'])

# adj_load_680 = load + output

# adj_load_870 = load + output

# adj_load_40 = load + output

# adj_load_50 = load + output

# adj_load_0 = load + output

# load.index.name = 'date'
# adj_load_0.index.name = 'date'
# adj_load_50.index.name = 'date'
# adj_load_40.index.name = 'date'
# adj_load_870.index.name = 'date'
# adj_load_680.index.name = 'date'

# loads = load.reset_index().melt(id_vars = 'date', var_name = 'hour', value_name = 'value')
# loads[0] = adj_load_0.reset_index().melt(id_vars = 'date')['value']
# loads[0.5] = adj_load_50.reset_index().melt(id_vars = 'date')['value']
# loads[0.4] = adj_load_40.reset_index().melt(id_vars = 'date')['value']
# loads[635] = adj_load_680.reset_index().melt(id_vars = 'date')['value']
# loads[838] = adj_load_870.reset_index().melt(id_vars = 'date')['value']
# loads = loads.set_index(['date', 'hour'])
# loads.plot()

# loads.groupby(by = 'date').max()

# summer_loads_plot = pd.DataFrame(0, index = titles['hour'], columns = ['Load', 'Baseline', '40%', '50%', 'EUR 635', 'EUR 838'])
# summer_loads_plot[''] = 0
# summer_loads_plot['Load'] = load.iloc[summer_idx, :].mean()
# summer_loads_plot['Baseline'] = adj_load_0.iloc[summer_idx, :].mean()
# summer_loads_plot['40%'] = adj_load_40.iloc[summer_idx, :].mean()
# summer_loads_plot['50%'] = adj_load_50.iloc[summer_idx, :].mean()
# summer_loads_plot['EUR 635'] = adj_load_680.iloc[summer_idx, :].mean()
# summer_loads_plot['EUR 838'] = adj_load_870.iloc[summer_idx, :].mean()


# plot = summer_loads_plot.plot(figsize = (16, 10), linewidth = 3, colormap = colourmap, ylim = (3500, 6200),
#                 xticks = range(0, 25, 4), fontsize = 24)
# plot.legend(loc = 'upper left', fontsize = 24)
# plot.set_ylabel('Total load (MW)',fontdict={'fontsize':24})
# plot.set_xlabel('Hour',fontdict={'fontsize':24})

# winter_loads_plot = pd.DataFrame(0, index = titles['hour'], columns = ['Load','Baseline', '40%', '50%', 'EUR 635', 'EUR 838'])
# winter_loads_plot[''] = 0
# winter_loads_plot['Load'] = load.iloc[winter_idx, :].mean()
# winter_loads_plot['Baseline'] = adj_load_0.iloc[winter_idx, :].mean()
# winter_loads_plot['40%'] = adj_load_40.iloc[winter_idx, :].mean()
# winter_loads_plot['50%'] = adj_load_50.iloc[winter_idx, :].mean()
# winter_loads_plot['EUR 635'] = adj_load_680.iloc[winter_idx, :].mean()
# winter_loads_plot['EUR 838'] = adj_load_870.iloc[winter_idx, :].mean()


# plot = winter_loads_plot.plot(figsize = (16, 10), linewidth = 3, colormap = colourmap, ylim = (3500, 6200),
#                 xticks = range(0, 25, 4), fontsize = 24)
# plot.legend(loc = 'upper left', fontsize = 24)
# plot.set_ylabel('Total load (MW)',fontdict={'fontsize':24})
# plot.set_xlabel('Hour',fontdict={'fontsize':24})
