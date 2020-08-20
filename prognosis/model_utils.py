# MIT License
#
# Copyright (c) 2020-2022 Quoc Tran
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from sklearn import linear_model
import streamlit as st
import pwlf_mod as pwlf
from csv import writer

#DEATH_RATE = 0.01
#ICU_RATE = 0.05
#HOSPITAL_RATE = 0.15
#SYMPTOM_RATE = 0.2
#INFECT_2_HOSPITAL_TIME = 13
#HOSPITAL_2_ICU_TIME = 2
#ICU_2_DEATH_TIME = 5
#ICU_2_RECOVER_TIME = 11
#NOT_ICU_DISCHARGE_TIME = 7


def get_data(file_template='../csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_{type}_{scope}.csv',
            type='deaths', scope='global'):
    """
    type = enum('deaths', 'confirmed', 'recovered'),
    scope = enum('global', 'US')
    """
    csv_data = pd.read_csv(file_template.format(type=type, scope=scope))
    return csv_data.rename(index=str, columns={"Country/Region": "Country",
                                                 "Province/State": "State",
                                                 "Country_Region": "Country",
                                                 "Province_State": "State",
                                                 "Admin2": "County"})


def get_US_State_hospital_cap_data(file_template='data/Hospital_Capacity_by_State_Harvard.csv'):
    """
    Get total hospital beds and ICUs for all US states
    """
    csv_data = pd.read_csv(file_template, index_col='State')
    return csv_data


def process_local_data(local_data):
    local_data.index = pd.to_datetime(local_data.index)
    # Remove non positive value
    local_data = local_data[local_data > 0].dropna()
    # Pad first value with 0
    local_data.loc[min(local_data.index)+dt.timedelta(-1)] = 0
    return local_data.sort_index()


def get_data_by_country(country, type='deaths'):
    global_data = get_data(scope='global', type=type)
    local_data = global_data.query('Country == "{}"'.format(country)).iloc[:,4:].T.sum(axis=1).to_frame()
    return process_local_data(local_data)


def get_data_by_state(state, type='deaths'):
    US_data = get_data(scope='US', type=type)
    local_data = US_data.query('State == "{}"'.format(state)).iloc[:,12:].T.sum(axis=1).to_frame()
    return process_local_data(local_data)


def get_data_by_county_and_state(county, state, type='deaths'):
    US_data = get_data(scope='US', type=type)
    local_data = US_data.query('County == "{}" and State == "{}"'.format(county, state)).iloc[:,12:].T.sum(axis=1).to_frame()
    return process_local_data(local_data)


def get_lockdown_date_global(csv_file='data/lockdown_date_country.csv'):
    return pd.read_csv(csv_file)[['country', 'lockdown_date', 'relax_date']].set_index('country')


def get_lockdown_date_by_country(country):
    try:
        lockdown_date = pd.to_datetime(get_lockdown_date_global().loc[country][0]).date()
    except KeyError:
        lockdown_date = None
    return lockdown_date


def get_relax_date_by_country(country):
    try:
        relax_date = pd.to_datetime(get_lockdown_date_global().loc[country][1]).date()
    except KeyError:
        return None
    if relax_date > pd.to_datetime('2020/01/01'):
        return relax_date
    else:
        return None


def get_lockdown_date_US(csv_file='data/lockdown_date_state_US.csv'):
    return pd.read_csv(csv_file)[['state', 'lockdown_date', 'relax_date']].set_index('state')


def get_lockdown_date_by_state_US(state):
    try:
        lockdown_date = pd.to_datetime(get_lockdown_date_US().loc[state][0]).date()
    except KeyError:
        lockdown_date = None
    return lockdown_date


def get_relax_date_by_state_US(state):
    try:
        relax_date = pd.to_datetime(get_lockdown_date_US().loc[state][1]).date()
    except KeyError:
        return None
    if relax_date > pd.to_datetime('2020/01/01'):
        return relax_date
    else:
        return None


def get_daily_data(cum_data):
    return cum_data.diff().fillna(0)


def get_impute_from_death(death_row, periods, end_date_offset=0):
    date_ind = death_row.name
    end_date = date_ind + dt.timedelta(end_date_offset)
    date_range = pd.date_range(end=end_date, periods=periods)
    return pd.DataFrame(death_row.tolist()*periods, index=date_range)


def get_hospital_beds_from_death(death_row):
    '''Get imputation of hospital beds needed from one day record of new death'''
    dead_hospital_use_periods = HOSPITAL_2_ICU_TIME+ICU_2_DEATH_TIME
    dead_hospital_use = get_impute_from_death(death_row=death_row, 
                                              periods=dead_hospital_use_periods)
    ICU_recovered_hospital_use_periods = HOSPITAL_2_ICU_TIME+ICU_2_RECOVER_TIME+NOT_ICU_DISCHARGE_TIME
    ICU_recovered_hospital_use_end_date_offset = ICU_2_RECOVER_TIME-ICU_2_DEATH_TIME+NOT_ICU_DISCHARGE_TIME
    ICU_recovered_hospital_use = get_impute_from_death(death_row=death_row, 
                                                       periods=ICU_recovered_hospital_use_periods,
                                                       end_date_offset=ICU_recovered_hospital_use_end_date_offset)
    no_ICU_hospital_use_periods = NOT_ICU_DISCHARGE_TIME
    no_ICU_hospital_use_end_date_offset = -HOSPITAL_2_ICU_TIME-ICU_2_DEATH_TIME+NOT_ICU_DISCHARGE_TIME
    no_ICU_hospital_use = get_impute_from_death(death_row=death_row, 
                                                periods=no_ICU_hospital_use_periods,
                                                end_date_offset=no_ICU_hospital_use_end_date_offset)
    hospital_beds = dead_hospital_use.add(((ICU_RATE-DEATH_RATE)/DEATH_RATE)*ICU_recovered_hospital_use, fill_value=0)\
            .add(((HOSPITAL_RATE-ICU_RATE)/DEATH_RATE)*no_ICU_hospital_use, fill_value=0)
    hospital_beds.columns = ['hospital_beds']
    return hospital_beds


def get_ICU_from_death(death_row):
    '''Get imputation of ICU needed from one day record of new death'''
    dead_ICU_use = get_impute_from_death(death_row=death_row, periods=ICU_2_DEATH_TIME)
    recovered_ICU_use_end_date_offset = ICU_2_RECOVER_TIME-ICU_2_DEATH_TIME
    recovered_ICU_use = get_impute_from_death(death_row=death_row, 
                                              periods=ICU_2_RECOVER_TIME,
                                              end_date_offset=recovered_ICU_use_end_date_offset)
    ICU_n = dead_ICU_use.add(((ICU_RATE-DEATH_RATE)/DEATH_RATE)*recovered_ICU_use, fill_value=0)
    ICU_n.columns = ['ICU']
    return ICU_n


def get_infected_cases(local_death_data):
    '''This number only is close to number of confirmed case in country very early in the disease and 
    can still do contact tracing or very wide testing, eg. South Korea, Germany'''
    delay_time = INFECT_2_HOSPITAL_TIME + HOSPITAL_2_ICU_TIME + ICU_2_DEATH_TIME
    infected_cases = (100/DEATH_RATE)*local_death_data.tshift(-delay_time)
    infected_cases.columns = ['infected']
    return infected_cases


def get_symptomatic_cases(local_death_data):
    '''This is number of cases that show clear symptoms (severe),
    in country without investigative testing this is close to number of confirmed case, most country'''
    delay_time = HOSPITAL_2_ICU_TIME + ICU_2_DEATH_TIME
    symptomatic_cases = (SYMPTOM_RATE/DEATH_RATE)*local_death_data.tshift(-delay_time)
    symptomatic_cases.columns = ['symptomatic']
    return symptomatic_cases


def get_hospitalized_cases(local_death_data):
    '''In country with severe lack of testing, this is close to number of confirmed case, eg. Italy, Iran'''
    delay_time = HOSPITAL_2_ICU_TIME + ICU_2_DEATH_TIME
    hospitalized_cases = (HOSPITAL_RATE/DEATH_RATE)*local_death_data.tshift(-delay_time)
    hospitalized_cases.columns = ['hospitalized']
    return hospitalized_cases


def get_number_hospital_beds_need(daily_local_death_new):
    '''Calculate number of hospital bed needed from number of daily new death '''
    # Start by first date
    hospital_beds = get_hospital_beds_from_death(daily_local_death_new.iloc[0])
    # Run through all days
    for i in range(len(daily_local_death_new)-1):
        hospital_beds = hospital_beds.add(get_hospital_beds_from_death(daily_local_death_new.iloc[i+1]), 
                                          fill_value=0)
    hospital_beds = hospital_beds.iloc[:-(HOSPITAL_2_ICU_TIME+ICU_2_RECOVER_TIME+NOT_ICU_DISCHARGE_TIME)]
    return hospital_beds


def get_number_ICU_need(daily_local_death_new):
    '''Calculate number of ICU needed from number of daily new death '''
    # Start by first date
    ICU_n = get_ICU_from_death(daily_local_death_new.iloc[0])
    # Run through all days
    for i in range(len(daily_local_death_new)-1):
        ICU_n = ICU_n.add(get_ICU_from_death(daily_local_death_new.iloc[i+1]), fill_value=0)
    ICU_n = ICU_n.iloc[:-ICU_2_RECOVER_TIME]
    return ICU_n


def remove_outliers(log_daily_death, break_points):
    """ Remove outliers by running robust linear regression in each section"""
    robust_reg = linear_model.HuberRegressor(fit_intercept=True)
    outliers = np.array([], dtype=bool)
    for i in range(len(break_points)-1):
        data_train = log_daily_death.query('time_idx>={}&time_idx<{}'.format(break_points[i], break_points[i+1]))
        try:
            robust_reg.fit(data_train.time_idx.values.reshape(-1, 1), data_train.death)
            outliers_pw = robust_reg.outliers_
        except:
            outliers_pw = np.array([False] * len(data_train), dtype=bool)
        outliers = np.concatenate((outliers, outliers_pw))
    return log_daily_death[~outliers]


def get_log_daily_predicted_death(local_death_data, forecast_horizon=60, policy_change_dates=[], contain_rate=0.8):
    '''Since this is highly contagious disease. Daily new death, which is a proxy for daily new infected cases
    is model as d(t)=a*d(t-1) or equivalent to d(t) = b*a^(t). After a log transform, it becomes linear.
    log(d(t))=logb+t*loga, so we can use linear regression to provide forecast (use robust linear regressor to avoid
    data anomaly in death reporting)
    There are two seperate linear curves, one before the lockdown is effective(21 days after lockdown) and one after
    For using this prediction to infer back the other metrics (infected cases, hospital, ICU, etc..) only the before
    curve is used and valid. If we assume there is no new infection after lock down (perfect lockdown), the after
    curve only depends on the distribution of time to death since ICU.
    WARNING: if lockdown_date is not provided, we will default to no lockdown to raise awareness of worst case
    if no action. If you have info on lockdown date please use it to make sure the model provide accurate result'''
    daily_local_death_new = get_daily_data(local_death_data)
    smoothing_days = 3
    daily_local_death_new = daily_local_death_new.rolling(smoothing_days, min_periods=1).mean()
    # Because of this smoothing step, we need to time var of prediction by smoothing_days=3.
    #shift ahead 1 day to avoid overfitted due to average of exponential value
    #daily_local_death_new = daily_local_death_new.shift(1)
    daily_local_death_new.columns = ['death']
    log_daily_death = np.log(daily_local_death_new)
    # log_daily_death.dropna(inplace=True)
    data_start_date = min(local_death_data.index)
    data_end_date = max(local_death_data.index)
    forecast_end_date = data_end_date + dt.timedelta(forecast_horizon)
    forecast_date_index = pd.date_range(start=data_start_date, end=forecast_end_date)
    policy_effective_dates = pd.to_datetime(policy_change_dates) + dt.timedelta(
        INFECT_2_HOSPITAL_TIME + HOSPITAL_2_ICU_TIME + ICU_2_DEATH_TIME)
    data_start_date_idx = 0
    data_end_date_idx = (data_end_date - data_start_date).days
    forecast_end_date_idx = data_end_date_idx + forecast_horizon
    forecast_time_idx = (forecast_date_index - data_start_date).days.values
    data_time_idx = (log_daily_death.index - data_start_date).days.values
    policy_effective_dates_idx = (policy_effective_dates - data_start_date).days.values
    log_daily_death['time_idx'] = data_time_idx
    log_daily_death = log_daily_death.replace([np.inf, -np.inf], np.nan).dropna()
    break_points = np.array([data_start_date_idx,] +
                            policy_effective_dates_idx[(~np.isnan(policy_effective_dates_idx))&
                                                       (policy_effective_dates_idx < forecast_end_date_idx)].tolist() +
                            [forecast_end_date_idx,])
    log_daily_death = remove_outliers(log_daily_death, break_points)
    regr_pw = pwlf.PiecewiseLinFit(x=log_daily_death.time_idx.values, y=log_daily_death.death)
    regr_pw.fit_with_breaks(break_points)
    model_beta = regr_pw.beta
    log_predicted_death_pred_var = smoothing_days * regr_pw.prediction_variance(forecast_time_idx)

    # Use default slope when data is not enough to fit last line, less than 4 data point, with contain_rate=1 mean slope
    # is the same as previous slope (same policy) and 0 mean (relax 100%) slope will be same as before lockdown
    # import pdb;pdb.set_trace()
    if ((data_end_date_idx-break_points[-2]) < 4) | (model_beta[-1] > max(0.3, abs(model_beta[1]))):
        model_beta[-1] = (-model_beta[-2])*(1-contain_rate)
        print("Use default last slope due to not enough data")
        variance = log_predicted_death_pred_var[sum(forecast_time_idx <= break_points[-2])]
        log_predicted_death_pred_var_oos = variance * (forecast_time_idx[forecast_time_idx > break_points[-2]] -
                                                       break_points[-2])
        log_predicted_death_pred_var = np.concatenate(
                (log_predicted_death_pred_var[:sum(forecast_time_idx <= break_points[-2])],
                 log_predicted_death_pred_var_oos))

    log_predicted_death_values = regr_pw.predict(forecast_time_idx, beta=model_beta, breaks=break_points)

    log_predicted_death_lower_bound_values = log_predicted_death_values - 1.96 * np.sqrt(log_predicted_death_pred_var)
    log_predicted_death_upper_bound_values = log_predicted_death_values + 1.96 * np.sqrt(log_predicted_death_pred_var)

    log_predicted_death = pd.DataFrame(log_predicted_death_values, index=forecast_date_index)
    log_predicted_death_lower_bound = pd.DataFrame(log_predicted_death_lower_bound_values, index=forecast_date_index)
    log_predicted_death_upper_bound = pd.DataFrame(log_predicted_death_upper_bound_values, index=forecast_date_index)
    log_predicted_death.columns = ['predicted_death']
    log_predicted_death_lower_bound.columns = ['lower_bound']
    log_predicted_death_upper_bound.columns = ['upper_bound']
    return log_predicted_death, log_predicted_death_lower_bound, log_predicted_death_upper_bound, regr_pw.beta


def get_daily_predicted_death(local_death_data, forecast_horizon=60, policy_change_dates=[], contain_rate=0.8):
    log_daily_predicted_death, lb, ub, model_beta = get_log_daily_predicted_death(local_death_data,
                                                                                  forecast_horizon,
                                                                                  policy_change_dates,
                                                                                  contain_rate)
    return np.exp(log_daily_predicted_death), np.exp(lb), np.exp(ub), model_beta


def get_cumulative_predicted_death(local_death_data, forecast_horizon=60, policy_change_dates=[], contain_rate=0.8):
    daily, lb, ub, model_beta = get_daily_predicted_death(local_death_data, forecast_horizon, policy_change_dates,
                                                          contain_rate)
    return daily.cumsum(), lb.cumsum(), ub.cumsum(), model_beta


def get_daily_metrics_from_death_data(local_death_data, forecast_horizon=60, policy_change_dates=[], contain_rate=0.8,
                                      test_rate=0.2):
    """test rate is defined as ratio of confirmed positive cases over all infected cases. A test rate=1 mean
    we can catch all infected case. In this case there is no uncertainty on the infected case, it is exactly
    equal confirmed case. When test rate is smaller than 1 the uncertainty is higher. Test rate is estimated
    through prevalence study when randomly test a sample from the population to get the estimation of the
    infected case and then use the confirmed case to derive the test rate.
    We will assume that all death due to Covid19 has been tested and counted, so there is no extra uncertainty on the
    number of death lower and upper bound.
    For other metrics derive from death, we need to use this test rate to add uncertainty into their bounds.
    Due to the definition, standard deviation of the derived metrics gets inflated by 1 over squareroot of test rate"""

    daily_predicted_death, daily_predicted_death_lb, daily_predicted_death_ub, model_beta = \
        get_daily_predicted_death(local_death_data, forecast_horizon, policy_change_dates, contain_rate)
    upper_length_death = daily_predicted_death_ub - daily_predicted_death
    upper_length_derived = (upper_length_death*1/np.sqrt(test_rate)).astype('int', errors='ignore')
    lower_length_death = daily_predicted_death - daily_predicted_death_lb
    lower_length_derived = (lower_length_death * 1 / np.sqrt(test_rate)).astype('int', errors='ignore')

    daily_local_death_new = local_death_data.diff().fillna(0)
    daily_local_death_new.columns = ['death']
    daily_infected_cases_new = get_infected_cases(daily_predicted_death)
    daily_symptomatic_cases_new = get_symptomatic_cases(daily_predicted_death)
    daily_hospitalized_cases_new = get_hospitalized_cases(daily_predicted_death)
    daily_hospital_beds_need = get_number_hospital_beds_need(daily_predicted_death)
    daily_ICU_need = get_number_ICU_need(daily_predicted_death)

    # daily_infected_cases_new_lb = daily_infected_cases_new - get_infected_cases(lower_length_derived)
    # daily_symptomatic_cases_new_lb = daily_symptomatic_cases_new - get_symptomatic_cases(lower_length_derived)
    # daily_hospitalized_cases_new_lb = daily_hospitalized_cases_new - get_hospitalized_cases(lower_length_derived)
    # daily_hospital_beds_need_lb = daily_hospital_beds_need - get_number_hospital_beds_need(lower_length_derived)
    # daily_ICU_need_lb = daily_hospital_beds_need - get_number_ICU_need(lower_length_derived)

    return pd.concat([daily_local_death_new,
                      daily_predicted_death,
                      daily_predicted_death_lb,
                      daily_predicted_death_ub,
                      daily_infected_cases_new,
                      daily_symptomatic_cases_new,
                      daily_hospitalized_cases_new,
                      daily_hospital_beds_need,
                      daily_ICU_need], axis=1, sort=True), model_beta


def get_cumulative_metrics_from_death_data(local_death_data, forecast_horizon=60, policy_change_dates=[], contain_rate=0.8,
                                           test_rate=0.2):
    daily_metrics, model_beta = get_daily_metrics_from_death_data(local_death_data, forecast_horizon,
                                                                  policy_change_dates, contain_rate, test_rate)
    cumulative_metrics = daily_metrics.drop(columns=['ICU', 'hospital_beds']).cumsum()
    # data_end_date = max(local_death_data.index)
    # cumulative_metrics['lower_bound'] = daily_metrics['lower_bound']
    # cumulative_metrics['lower_bound'].loc[local_death_data.index] = np.nan
    # cumulative_metrics['lower_bound'].loc[data_end_date] = local_death_data.loc[data_end_date][0]
    # cumulative_metrics['lower_bound'] = cumulative_metrics['lower_bound'].cumsum()
    # cumulative_metrics['upper_bound'] = daily_metrics['upper_bound']
    # cumulative_metrics['upper_bound'].loc[local_death_data.index] = np.nan
    # cumulative_metrics['upper_bound'].loc[data_end_date] = local_death_data.loc[data_end_date][0]
    # cumulative_metrics['upper_bound'] = cumulative_metrics['upper_bound'].cumsum()
    cumulative_metrics['ICU'] = daily_metrics['ICU']
    cumulative_metrics['hospital_beds'] = daily_metrics['hospital_beds']

    return cumulative_metrics, model_beta


def get_metrics_by_country(country, forecast_horizon=60, policy_change_dates=[], contain_rate=0.8, test_rate=0.2,
                           back_test=False, last_data_date=dt.date.today()):
    local_death_data = get_data_by_country(country, type='deaths')
    local_death_data_original = local_death_data.copy()
    daily_local_death_data_original = get_daily_data(local_death_data_original)
    if back_test:
        local_death_data = local_death_data[local_death_data.index.date <= last_data_date]
    local_confirmed_data = get_data_by_country(country, type='confirmed')
    daily_local_confirmed_data = get_daily_data(local_confirmed_data)
    daily_metrics, model_beta = get_daily_metrics_from_death_data(local_death_data, forecast_horizon,
                                                                  policy_change_dates, contain_rate, test_rate)
    daily_metrics['confirmed'] = daily_local_confirmed_data
    if back_test:
        daily_metrics['death']= daily_local_death_data_original
    cumulative_metrics = daily_metrics.drop(columns=['ICU', 'hospital_beds']).cumsum()
    cumulative_metrics['ICU'] = daily_metrics['ICU']
    cumulative_metrics['hospital_beds'] = daily_metrics['hospital_beds']
    #cumulative_metrics, model_beta = get_cumulative_metrics_from_death_data(local_death_data, forecast_horizon,
    # lockdown_date)
    return daily_metrics, cumulative_metrics, model_beta


def get_metrics_by_state_US(state, forecast_horizon=60, policy_change_dates=[], contain_rate=0.8, test_rate=0.2,
                            back_test=False, last_data_date=dt.date.today()):
    local_death_data = get_data_by_state(state, type='deaths')
    local_death_data_original = local_death_data.copy()
    daily_local_death_data_original = get_daily_data(local_death_data_original)
    if back_test:
        local_death_data = local_death_data[local_death_data.index.date <= last_data_date]
    local_confirmed_data = get_data_by_state(state, type='confirmed')
    daily_local_confirmed_data = get_daily_data(local_confirmed_data)
    daily_metrics, model_beta = get_daily_metrics_from_death_data(local_death_data, forecast_horizon,
                                                                  policy_change_dates, contain_rate, test_rate)
    daily_metrics['confirmed'] = daily_local_confirmed_data
    if back_test:
        daily_metrics['death']= daily_local_death_data_original
    cumulative_metrics = daily_metrics.drop(columns=['ICU', 'hospital_beds']).cumsum()
    cumulative_metrics['ICU'] = daily_metrics['ICU']
    cumulative_metrics['hospital_beds'] = daily_metrics['hospital_beds']
    return daily_metrics, cumulative_metrics, model_beta


def get_metrics_by_county_and_state_US(county, state, forecast_horizon=60, policy_change_dates=[]):
    local_death_data = get_data_by_county_and_state(county, state, type='deaths')
    local_confirmed_data = get_data_by_county_and_state(county, state, type='confirmed')
    daily_local_confirmed_data = get_daily_data(local_confirmed_data)
    daily_metrics, model_beta = get_daily_metrics_from_death_data(local_death_data, forecast_horizon,
                                                                  policy_change_dates)
    daily_metrics['confirmed'] = daily_local_confirmed_data
    cumulative_metrics = daily_metrics.drop(columns=['ICU', 'hospital_beds']).cumsum()
    cumulative_metrics['ICU'] = daily_metrics['ICU']
    cumulative_metrics['hospital_beds'] = daily_metrics['hospital_beds']
    return daily_metrics, cumulative_metrics, model_beta


def get_log_daily_predicted_death_by_country(country, forecast_horizon=60, policy_change_dates=[], contain_rate=0.8,
                                             back_test=False, last_data_date=dt.date.today()):
    local_death_data = get_data_by_country(country, type='deaths')
    local_death_data.columns = ['death']
    local_death_data_original = local_death_data.copy()
    daily_local_death_data_original = get_daily_data(local_death_data_original)
    log_daily_death_original = np.log(daily_local_death_data_original)
    if back_test:
        local_death_data = local_death_data[local_death_data.index.date <= last_data_date]
    log_predicted_death, log_predicted_death_lb, log_predicted_death_ub, model_beta = \
            get_log_daily_predicted_death(local_death_data, forecast_horizon, policy_change_dates)
    return pd.concat([log_daily_death_original, log_predicted_death, log_predicted_death_lb,
                      log_predicted_death_ub], axis=1).replace([np.inf, -np.inf], np.nan), model_beta


def get_log_daily_predicted_death_by_state_US(state, forecast_horizon=60, policy_change_dates=[], contain_rate=0.8,
                                              back_test=False, last_data_date=dt.date.today()):
    local_death_data = get_data_by_state(state, type='deaths')
    local_death_data.columns = ['death']
    local_death_data_original = local_death_data.copy()
    daily_local_death_data_original = get_daily_data(local_death_data_original)
    log_daily_death_original = np.log(daily_local_death_data_original)
    if back_test:
        local_death_data = local_death_data[local_death_data.index.date <= last_data_date]
    log_predicted_death, log_predicted_death_lb, log_predicted_death_ub, model_beta = \
        get_log_daily_predicted_death(local_death_data, forecast_horizon, policy_change_dates, contain_rate)
    return pd.concat([log_daily_death_original, log_predicted_death, log_predicted_death_lb,
                      log_predicted_death_ub], axis=1).replace([np.inf, -np.inf], np.nan), model_beta


def get_log_daily_predicted_death_by_county_and_state_US(county, state, forecast_horizon=60, policy_change_dates=[],
                                                         contain_rate=0.8, back_test=False,
                                                         last_data_date=dt.date.today()):
    local_death_data = get_data_by_county_and_state(county, state, type='deaths')
    local_death_data.columns = ['death']
    local_death_data_original = local_death_data.copy()
    daily_local_death_data_original = get_daily_data(local_death_data_original)
    log_daily_death_original = np.log(daily_local_death_data_original)
    log_predicted_death, log_predicted_death_lb, log_predicted_death_ub, model_beta = \
        get_log_daily_predicted_death(local_death_data, forecast_horizon, policy_change_dates, contain_rate)
    return pd.concat([log_daily_death_original, log_predicted_death, log_predicted_death_lb,
                      log_predicted_death_ub], axis=1).replace([np.inf, -np.inf], np.nan), model_beta


def append_row_2_logs(row, log_file='logs/model_params_logs.csv'):
    # Open file in append mode
    with open(log_file, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(row)


def get_table_download_link(df, filename="data.csv"):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    import base64
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" >Download csv file</a>'
    return href

