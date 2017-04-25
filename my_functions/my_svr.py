"""
Fuctions that process data for SVR input and run SVR model.
"""

import pandas as pd
import numpy as np


def add_if_holiday(elec_and_weather, workdays=[], holidays=[]):
    """
    Add a feature column `if_holiday`:
        0: weekday, workday.
        1: weekend, holiday.
    First simply set all weekdays equals to 0 and weekends equals to 1.
    Then modify those special dates according to parameters `workdays` and `holidays`.

    Args:
        elec_and_weather: DataFrame.
        workdays: dates to be modified to 0
        holidays: dates to be modified to 1
    """
    elec_and_weather['if_holiday'] = 0
    elec_and_weather.loc[elec_and_weather.index.dayofweek >= 5,
                         'if_holiday'] = 1

    if workdays:
        for day in workdays:
            elec_and_weather.loc[day, 'if_holiday'] = 0
    if holidays:
        for day in holidays:
            elec_and_weather.loc[day, 'if_holiday'] = 1


def add_hour_of_day(elec_and_weather):
    """
    Add feature column `hour of day`: 0 ~ 23
    """
    for i in range(24):
        elec_and_weather[i] = 0
        elec_and_weather.loc[elec_and_weather.index.hour == i, i] = 1


def add_historical_kwh(elec_and_weather):
    """
    Add t-1, t-2, t-3, t-4, t-5,
    t-6, t-12, t-24, t-48 historical elec usage columns.
    Some columns will have NA values.
    """
    elec_and_weather['kwh_t-1'] = elec_and_weather['kwh'].shift(1)
    elec_and_weather['kwh_t-2'] = elec_and_weather['kwh'].shift(2)
    elec_and_weather['kwh_t-3'] = elec_and_weather['kwh'].shift(3)
    elec_and_weather['kwh_t-4'] = elec_and_weather['kwh'].shift(4)
    elec_and_weather['kwh_t-5'] = elec_and_weather['kwh'].shift(5)
    elec_and_weather['kwh_t-6'] = elec_and_weather['kwh'].shift(6)
    elec_and_weather['kwh_t-12'] = elec_and_weather['kwh'].shift(12)
    elec_and_weather['kwh_t-24'] = elec_and_weather['kwh'].shift(24)
    elec_and_weather['kwh_t-48'] = elec_and_weather['kwh'].shift(48)


def split(elec_and_weather, time_str, drop_col=[]):
    """
    Split the data into training set and test set.

    Args:
        elec_and_weather: DataFrame.
        time_str: list [train_start, train_end, test_start, test_end]
                  example: ['2/15/2017', '3/23/2017', '3/24/2017', '3/31/2017']
        drop_col: unnecessary columns like ['kwh_t-2', 'kwh_t-3', 'kwh_t-4']
    Returns:
        X_train, y_train, X_test, y_test
    """
    train_start, train_end, test_start, test_end = time_str

    big_set = elec_and_weather.dropna().drop(drop_col, axis=1)
    X_train = big_set.drop('kwh', axis=1)[train_start:train_end]
    y_train = np.array(big_set.loc[train_start:train_end, 'kwh'])
    X_test = big_set.drop('kwh', axis=1)[test_start:test_end]
    y_test = np.array(big_set.loc[test_start:test_end, 'kwh'])

    return X_train, y_train, X_test, y_test

