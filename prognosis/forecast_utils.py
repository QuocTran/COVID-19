#!/usr/bin/env python
import argparse
import pandas as pd
import datetime as dt
import epiweeks
import model_utils as mu

mu.DEATH_RATE = 0.36
mu.ICU_RATE = 0.78
mu.HOSPITAL_RATE = 2.18
mu.SYMPTOM_RATE = 10.2
mu.INFECT_2_HOSPITAL_TIME = 11
mu.HOSPITAL_2_ICU_TIME = 4
mu.ICU_2_DEATH_TIME = 4
mu.ICU_2_RECOVER_TIME = 7
mu.NOT_ICU_DISCHARGE_TIME = 5

fips = pd.read_csv('data/locations.csv')
metric_map = {'death': 'predicted_death'}


def get_epiweek_enddate(x):
    return epiweeks.Week.fromdate(pd.to_datetime(x).date()).enddate()


def get_target_str(target_end_date, forecast_date, target_metric, target_aggr):
    forecast_date_week_end = get_epiweek_enddate(forecast_date)
    target = '{week} wk ahead {target_aggr} {target_metric}'\
        .format(week=(target_end_date - forecast_date_week_end).days//7 + 1,
                target_aggr=target_aggr,
                target_metric=target_metric)
    return target


def format_forecast(input_forecast, 
                    location_name, 
                    forecast_date,
                    target_metric,
                    target_aggr):
    forecast_date = pd.to_datetime(forecast_date).date()
    input_forecast['target_end_date'] = input_forecast.date.apply(get_epiweek_enddate)
    input_forecast['target'] = input_forecast.target_end_date.apply(get_target_str, args=(forecast_date,
                                                                                          target_metric,
                                                                                          target_aggr))
    input_forecast['forecast_date'] = forecast_date
    input_forecast['location'] = fips.query('location_name == @location_name').location.iloc[0]
    input_forecast['quantile'] = 'NA'
    input_forecast['type'] = 'point'
    input_forecast.rename(columns={metric_map[target_metric]: 'value'}, inplace=True)
    output = input_forecast[['forecast_date', 'target', 'target_end_date', 'quantile', 'type', 'value', 'location']]
    return output.groupby(['forecast_date', 'target', 'target_end_date', 'quantile', 'type', 'location']).sum()\
        .reset_index().query('target_end_date>forecast_date')


def generate_formatted_forecast(scope,
                                location_name,
                                forecast_date,
                                target_metric='death',
                                target_aggr='inc'):
    forecast_date = pd.to_datetime(forecast_date).date()
    if scope == 'World':
        forecast_fun = mu.get_metrics_by_country
        policy_date_fun = mu.get_policy_change_dates_by_country
    else:
        forecast_fun = mu.get_metrics_by_state_US
        policy_date_fun = mu.get_policy_change_dates_by_state_US
    input_forecast, _, _ = forecast_fun(location_name, 
                                        forecast_horizon=60,
                                        policy_change_dates=policy_date_fun(location_name),
                                        back_test=True, last_data_date=forecast_date)
    input_forecast.index.rename('date', inplace=True)
    input_forecast.reset_index(inplace=True)
    return format_forecast(input_forecast, location_name, forecast_date, target_metric, target_aggr)


def add_cum_forecast(inc_forecast, last_epi_week_cum):
    cum_forecast = inc_forecast.copy()
    cum_forecast['value'] = cum_forecast.value.cumsum()+last_epi_week_cum
    cum_forecast['target'] = cum_forecast.target.str.replace('inc', 'cum')
    return pd.concat([inc_forecast, cum_forecast])


def generate_US_formatted_forecast(forecast_date, target_metric='death', target_aggr='inc'):
    US_forecast = generate_formatted_forecast('World', 'US', forecast_date, target_metric, target_aggr)
    US_forecast = US_forecast.query('target!="9 wk ahead inc death"')
    forecast_date = pd.to_datetime(forecast_date).date()
    last_epiweek_enddate = get_epiweek_enddate(forecast_date+epiweeks.timedelta(-7))
    latest_cum_US = mu.get_data_by_country('US').loc[last_epiweek_enddate][0]
    US_forecast = add_cum_forecast(US_forecast, latest_cum_US)
    US_state_list = mu.get_data(scope='US', type='deaths').State.unique()

    for state in US_state_list:
        try:
            print(state)
            state_forecast = generate_formatted_forecast('US', state, forecast_date)\
                .query('target!="9 wk ahead inc death"')
            latest_cum_state = mu.get_data_by_state(state).loc[last_epiweek_enddate][0]
            state_forecast = add_cum_forecast(state_forecast, latest_cum_state)
            US_forecast = pd.concat([US_forecast, state_forecast])
        except (ValueError, IndexError):
            pass

    US_forecast.to_csv('data_processed/{}-AIpert-pwllnod.csv'.format(forecast_date), index=False)


def generate_world_formatted_forecast(forecast_date, target_metric='death', target_aggr='inc'):
    world_forecast = generate_formatted_forecast('World', 'US', forecast_date, target_metric, target_aggr)
    world_forecast = world_forecast.query('target!="9 wk ahead inc death"')
    forecast_date = pd.to_datetime(forecast_date).date()
    last_epiweek_enddate = get_epiweek_enddate(forecast_date+epiweeks.timedelta(-7))
    latest_cum = mu.get_data_by_country('US').loc[last_epiweek_enddate][0]
    world_forecast = add_cum_forecast(world_forecast, latest_cum)
    country_list = mu.get_data(scope='global', type='deaths').Country.unique()
    top_country_list = ['India', 'Brazil', 'Russia', 'France', 'United Kingdom', 'Turkey', 'Italy', 'Spain',
                        'Germany', 'Colombia', 'Argentina', 'Mexico', 'Poland', 'Iran', 'Iraq', 'Ukraine',
                        'South Africa', 'Peru', 'Netherlands', 'Belgium', 'Chile', 'Romania', 'Canada',
                        'Ecuador', 'Czechia', 'Pakistan', 'Hungary', 'Philippines', 'Switzerland']

    for country in top_country_list:
        try:
            print(country)
            country_forecast = generate_formatted_forecast('World', country, forecast_date)\
                .query('target!="9 wk ahead inc death"')
            latest_cum_country = mu.get_data_by_country(country).loc[last_epiweek_enddate][0]
            country_forecast = add_cum_forecast(country_forecast, latest_cum_country)
            world_forecast = pd.concat([world_forecast, country_forecast])
        except (ValueError, IndexError):
            pass

    world_forecast.to_csv('data_processed/World-{}-AIpert-pwllnod.csv'.format(forecast_date), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate US formatted forecast file')
    parser.add_argument('-d', '--date', default=dt.date.today(), help='date to run forecast, usually Monday,'
                                                                      ' default to today')
    args = parser.parse_args()
    generate_US_formatted_forecast(forecast_date=args.date)
