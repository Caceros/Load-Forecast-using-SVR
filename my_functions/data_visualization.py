"""
Some plot functions that can be used to analyze the data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from my_functions import my_errors


def plot_elec_hourly(df):
    """
    Visualize elec data by hour of day.l

    Args:
        Elec DataFrame.
    """
    fig = plt.figure(figsize=(12, 7))
    plt.plot(df.index.hour, df.kwh, '.', color='grey', label='data point')
    plt.xticks(range(24), range(24))
    plt.ylabel('Electricity Usage (kwh)')
    plt.xlabel('Hour of day')
    plt.plot(df.kwh.groupby(df.index.hour).mean(), label='Avg')
    plt.legend()


def plot_elec_daily(df):
    """
    Visualize daily elec data.

    Args:
        Elec DataFrame.
    """
    fig = plt.figure(figsize=(12, 7))
    plt.plot(df['kwh'].resample('D').sum(), marker='.', markersize=7)
    plt.ylabel('Electricity Usage (kwh)')


def compare_weekday_weekend(elec_and_weather):
    """
    Compare the electricity usage on weekdays and weekends.
    Can also compare temperature.

    Args:
        A DataFrame containing electricity usage data.

    Returns:
        A fig. (Only elec usage)
    """
    # First groupby weekday/weekend and hour of day
    WkdayWkend_Hour = elec_and_weather.groupby([elec_and_weather.index.dayofweek >= 5,
                                                elec_and_weather.index.hour])
    # Calculate an average weekday and average weekend by hour (electricity
    # and outdoor temp)
    AvgDay = pd.DataFrame([WkdayWkend_Hour['kwh'].mean(),
                           WkdayWkend_Hour['tempm'].mean()])
    fig = plt.figure(figsize=(8, 5))
    AvgDay[True].ix['kwh'].plot(marker='x', color='b')
    AvgDay[False].ix['kwh'].plot(marker='.', color='g')
    plt.xlim(0, 23)
    plt.xlabel('Hour of Day')
    plt.ylabel('Electricity Usage (kWh)')
    plt.legend(['Weekend', 'Weekday'], loc='upper left')
    plt.title('Average Electricity Usage On Weekdays And Weekends')


def compare_elec_temp(elec_and_weather, if_ols=False):
    """
    Use linear regression to find the relationship between elec usage and temperature.
    It's not accurate, but this visualization might tell you something.

    Args:
        DataFrame that has both elec and weather data.
        if_ols: wether to do an ordinary least regression.

    Returns:
        print out fit result and fig.
    """

    fig = plt.figure(figsize=(10, 10))
    plt.scatter(elec_and_weather['tempm'], elec_and_weather['kwh'],
                marker='x', color='g')
    plt.xlabel('Outdoor Temperature ($^\circ$C)')
    plt.ylabel('Electricity Usage (kwh)')
    plt.title('Electricity Usage (kwh) vs. Outdoor Temperature ($^\circ$C)')

    if if_ols:
        model = sm.OLS(elec_and_weather['kwh'],
                       sm.add_constant(elec_and_weather['tempm']))
        res = model.fit()
        print(res.summary())
        p1, = plt.plot(elec_and_weather['tempm'], res.fittedvalues, color='blue',
                       label='OLS $R^2$=%.3f' % res.rsquared)
        plt.legend(handles=[p1])


def compare_elec_hum(elec_and_weather, if_ols=False):
    """
    Similar to compare_elec_temp, but compare with humidity.
    """

    fig = plt.figure(figsize=(10, 10))
    plt.scatter(elec_and_weather['hum'], elec_and_weather['kwh'],
                marker='x', color='g')
    plt.xlabel('Relative Humidity (%)')
    plt.ylabel('Electricity Usage (kwh)')
    plt.title('Electricity Usage (kwh) vs. Relative Humidity (%)')

    if if_ols:
        model = sm.OLS(elec_and_weather['kwh'],
                       sm.add_constant(elec_and_weather['hum']))
        res = model.fit()
        print(res.summary())
        p1, = plt.plot(elec_and_weather['hum'], res.fittedvalues, color='blue',
                       label='OLS $R^2$=%.3f' % res.rsquared)
        plt.legend(handles=[p1])


def p(ax, col, dropped):
    """
    This function will be used in the function compare_t_t_k.

    Args:
        ax: axes object.
        his: column names like `kwh_t-1`
        dropped: elec_and_weather.dropna()
    """
    gray_light = '#d4d4d4'

    model = sm.OLS(dropped['kwh'], sm.add_constant(dropped[col]))
    res = model.fit()
    # dots
    ax.plot(dropped[col], dropped['kwh'],
            '.', color=gray_light, alpha=1.0)
    # line
    h, = ax.plot([min(dropped[col]), max(dropped[col])],
                 [min(res.fittedvalues), max(res.fittedvalues)])
    ax.set_xlabel('Elec. Usage at $%s$ hour' % col[4:])
    ax.legend([h], ['OLS $R^2$=%.2f' % res.rsquared], loc='lower right')


def compare_t_t_k(elec_and_weather):
    """
    Compare elec usage at t and t-k.
    Args:
        elec_and_weather processed by my_svr.add_historical_kwh

    Returns:
        fig
    """
    dropped = elec_and_weather.dropna()

    f, ax = plt.subplots(3, 3)
    f.set_figheight(15)
    f.set_figwidth(15)
    f.text(0.07, 0.5, 'Elec. Usage at $t$ hour',
           va='center', rotation='vertical', fontsize=14)
    f.text(0.5, 0.9, 'Elec. Usage at $t$ vs. Elec. Usage at $t-k$\nUnit: kwh',
           ha='center', fontsize=14, fontweight='bold')

    p(ax[0, 0], 'kwh_t-1', dropped)
    p(ax[0, 1], 'kwh_t-2', dropped)
    p(ax[0, 2], 'kwh_t-3', dropped)
    p(ax[1, 0], 'kwh_t-4', dropped)
    p(ax[1, 1], 'kwh_t-5', dropped)
    p(ax[1, 2], 'kwh_t-6', dropped)
    p(ax[2, 0], 'kwh_t-12', dropped)
    p(ax[2, 1], 'kwh_t-24', dropped)
    p(ax[2, 2], 'kwh_t-48', dropped)

    plt.setp([a.get_xticklabels() for a in ax[0, :]], visible=False)
    plt.setp([a.get_xticklabels() for a in ax[1, :]], visible=False)
    plt.setp([a.get_yticklabels() for a in ax[:, 1]], visible=False)
    plt.setp([a.get_yticklabels() for a in ax[:, 2]], visible=False)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)


def plot_pred(y_test, pred):
    """
    Plot prediction result.

    Args:
        y_test: test set target values, the actual values.
        pred: prediction made by model
    """
    fig = plt.figure(figsize=(16, 9))
    plt.plot(y_test.index, pred, linestyle='-.', lw=4, label='prediction')
    plt.plot(y_test.index, y_test, lw=3, label='observations')
    plt.ylabel('Electricity Usage (kwh)')
    plt.legend(fontsize=14)
    print(my_errors.errors(y_test, pred))
