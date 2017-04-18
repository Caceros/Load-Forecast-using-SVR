"""
This module is used for loading weather data and 
electric load consumption data.
"""

import pandas as pd


def weather(path):
    """Read hourly weather data downloaded by scrape_weather_data.py

    Args:
        path (str): csv weather data file path.

    Returns:
        A pandas DataFrame.
    """
    weather = pd.read_csv(path)
    # there may have more than one observation at a given hour
    weather = weather.drop_duplicates(subset='Unnamed: 0', keep='last').set_index('Unnamed: 0')
    weather.index = pd.to_datetime(weather.index)
    weather.head()
    return weather


def elec(path):
    """Read 15min electric consumption data.

    Args:
        path (str): csv elec data file path.

    Returns:
        Hourly elec DataFrame.
    """
    df = pd.read_excel(path, parse_cols='B, D, F:H, J, L, M, O').set_index('stat_time')
    df = df.resample('H').sum()
    return df


def elec_and_weather(df_elec, df_weather, startDate, endDate):
    """
    Merge elec and weather data frame.

    Args:
        df_elec: DataFrame generated by read_elec_data.
        df_weather: DataFrame generated by read_weather_data.
        starDate, endDate: Make two df index corresponding.

    Retuns:
        A merged DataFrame.
    """
    df_elec = df_elec[startDate:endDate]
    df_weather = df_weather[startDate:endDate]
    elec_and_weather = pd.merge(df_elec, df_weather, left_index=True, right_index=True)
    return elec_and_weather