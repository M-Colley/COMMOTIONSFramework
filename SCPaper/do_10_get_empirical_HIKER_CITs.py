# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 18:28:33 2021

@author: tragma
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import sc_fitting

INCLUDE_EHMI = False


def extract_hiker_cit_data(hiker_df, include_ehmi=False):
    """Extract empirical HIKER crossing initiation time DataFrames per scenario.

    When include_ehmi is False (the default), extracts the 24 non-eHMI scenarios
    (has_ehmi == False). When include_ehmi is True, also extracts the yielding
    eHMI trials (is_braking == True and has_ehmi == True) under '_y_e' keys.
    """
    hiker_data = {}
    n_data_points = 0
    for veh_speed_mph in sc_fitting.HIKER_VEH_SPEEDS_MPH:
        for veh_time_gap in sc_fitting.HIKER_VEH_TIME_GAPS:
            for veh_yielding in (False, True):
                ehmi_options = (False, True) if (include_ehmi and veh_yielding) else (False,)
                for veh_ehmi in ehmi_options:
                    scenario_name = sc_fitting.get_hiker_scen_name(
                        veh_speed_mph, veh_time_gap, veh_yielding,
                        veh_ehmi=veh_ehmi)
                    scenario_rows = ((hiker_df['orig_speed'] == veh_speed_mph)
                                     & (hiker_df['time_gap'] == veh_time_gap)
                                     & (hiker_df['is_braking'] == veh_yielding)
                                     & (hiker_df['has_ehmi'] == veh_ehmi))
                    hiker_data[scenario_name] = \
                        hiker_df[['subject', 'crossing_time']][scenario_rows]
                    n_data_points += int(np.count_nonzero(scenario_rows))
    return hiker_data, n_data_points


if __name__ == '__main__':
    hiker_df = pd.read_csv(sc_fitting.DATA_FOLDER + '/hiker_cts.csv')
    hiker_data, n_data_points = extract_hiker_cit_data(
        hiker_df, include_ehmi=INCLUDE_EHMI)
    print(f'Total number of data points: {n_data_points}')

    # plot CDFs
    plt.close('all')
    sc_fitting.do_hiker_cit_cdf_plot(hiker_data)

    # save data
    print('Saving HIKER CIT data...')
    with open(sc_fitting.DATA_FOLDER + '/' + sc_fitting.HIKER_DATA_FILE_NAME,
              'wb') as file_obj:
        pickle.dump(hiker_data, file_obj)
