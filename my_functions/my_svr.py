"""
Functions that process data for SVR input.
"""

import pandas as pd
import numpy as np
from my_functions import my_errors
from sklearn import svm


def add_if_holiday(df, workdays=[], holidays=[]):
    """
    Add a feature column `if_holiday`:
        0: weekday, workday.
        1: weekend, holiday.
    First simply set all weekdays equals to 0 and weekends equals to 1.
    Then modify those special dates according to parameters `workdays` and `holidays`.

    Args:
        df: DataFrame.
        workdays: dates to be modified to 0
        holidays: dates to be modified to 1
    """
    df['if_holiday'] = 0
    df.loc[df.index.dayofweek >= 5, 'if_holiday'] = 1

    if workdays:
        for day in workdays:
            df.loc[day, 'if_holiday'] = 0
    if holidays:
        for day in holidays:
            df.loc[day, 'if_holiday'] = 1


def delete_if_holiday(elec_and_weather):
    """
    Delete if_holiday features.
    """
    elec_and_weather.drop('if_holiday', axis=1, inplace=True)


def add_hour_of_day(elec_and_weather):
    """
    Add feature column `hour of day`: 0 ~ 23
    """
    for i in range(24):
        elec_and_weather[i] = 0
        elec_and_weather.loc[elec_and_weather.index.hour == i, i] = 1


def delete_hour_of_day(elec_and_weather):
    """
    Delete hour_of_day features.
    """
    elec_and_weather.drop(range(24), axis=1, inplace=True)


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
    Split the data into training set, cross validation set and test set.

    Args:
        elec_and_weather: DataFrame.
        time_str: list [train_start, train_end,
                        cross_start, cross_end,
                        test_start, test_end]
                  example: ['2/15/2017', '3/23/2017',
                            '3/24/2017', '3/31/2017',
                            '4/1/2017', '4/7/2017']
        drop_col: unnecessary columns like ['kwh_t-2', 'kwh_t-3', 'kwh_t-4']
    Returns:
        X_train, y_train, X_cross, y_cross, X_test, y_test
    """
    train_start, train_end, cross_start, cross_end, test_start, test_end = time_str

    big_set = elec_and_weather.dropna().drop(drop_col, axis=1)
    X_train = big_set.drop('kwh', axis=1)[train_start:train_end]
    y_train = big_set.loc[train_start:train_end, 'kwh']
    X_cross = big_set.drop('kwh', axis=1)[cross_start:cross_end]
    y_cross = big_set.loc[cross_start:cross_end, 'kwh']
    X_test = big_set.drop('kwh', axis=1)[test_start:test_end]
    y_test = big_set.loc[test_start:test_end, 'kwh']

    return X_train, y_train, X_cross, y_cross, X_test, y_test


def predict_one_day(model, scaler, df, date, scale):
    """
    I can't directly use model.predict(X_train_scaled) because I use
    historical data (t-1) as input features. If I do so, the information
    will leak to the model as the historical data contains actual observations.
    So I have to predict the kwh hour by hour and use my prediction for the
    next hour prediction. Note that I must scale my prediction for the use
    of next hour prediction.

    The same idea applies to the elec data aggregated by different time scale.
    For example, predict kwh every half hour, or 4 hours.

    Args:
        model: sklearn model.
        scaler: sklearn.preprocessing.StandardScaler().fit(X_train)
        df: df_X_test_scaled, scaled test set.
        date: string like '4/16/2017', the day you want to predict.
        scale: time scale.

    Returns:
        pred: prediction Array.
    """
    if scale == 'H':
        # 'block' the historical value so the model can't not see it
        # the first hour prediction uses the true historical data
        # this will actually overwrite the original data frame
        df.ix[date + ' 01:00:00':date + ' 23:00:00', 'kwh_t-1'] = np.nan
        pred = np.zeros(24)  # store prediction values
        for i in range(24):
            if np.isnan(df.ix[date + ' %s:00:00' % i, 'kwh_t-1']):
                # use prediction value, must be scale
                # scaler.mean_[2] because this is the mean for kwh_t-1 column
                # when you add features, make sure you add historical data
                # first
                df.ix[date + ' %s:00:00' % i, 'kwh_t-1'] = (
                    pred[i - 1] - scaler.mean_[2]
                ) / scaler.scale_[2]
            pred[i] = model.predict(  # reshape(1, -1) if predict one example
                df.ix[date + ' %s:00:00' % i].values.reshape(1, -1))

    if scale == '30min':
        df.ix[date + ' 00:30:00':date + ' 23:30:00', 'kwh_t-1'] = np.nan
        pred = np.zeros(48)
        for hour in range(24):
            for j in range(2):
                minute = j * 30
                if np.isnan(df.ix[date + ' %s:%s:00' % (hour, minute), 'kwh_t-1']):
                    df.ix[date + ' %s:%s:00' % (hour, minute), 'kwh_t-1'] = (
                        pred[hour * 2 + j - 1] - scaler.mean_[2]
                    ) / scaler.scale_[2]
                pred[hour * 2 + j] = model.predict(
                    df.ix[date + ' %s:%s:00' % (hour, minute)].values.reshape(1, -1))

    if scale == '4H':
        # in this case I didn't choose kwh_t-1 but kwh_t-6
        df.ix[date + ' 04:00:00':date + ' 20:00:00', 'kwh_t-6'] = np.nan
        pred = np.zeros(6)
        for i in range(6):
            hour = i * 4
            if np.isnan(df.ix[date + ' %s:00:00' % hour, 'kwh_t-6']):
                df.ix[date + ' %s:00:00' % hour, 'kwh_t-6'] = (
                    pred[i - 1] - scaler.mean_[2]
                ) / scaler.scale_[2]
            pred[i] = model.predict(
                df.ix[date + ' %s:00:00' % hour].values.reshape(1, -1))

    if scale == '2H':
        df.ix[date + ' 02:00:00':date + ' 22:00:00', 'kwh_t-12'] = np.nan
        pred = np.zeros(12)
        for i in range(12):
            hour = i * 2
            if np.isnan(df.ix[date + ' %s:00:00' % hour, 'kwh_t-12']):
                df.ix[date + ' %s:00:00' % hour, 'kwh_t-12'] = (
                    pred[i - 1] - scaler.mean_[2]
                ) / scaler.scale_[2]
            pred[i] = model.predict(
                df.ix[date + ' %s:00:00' % hour].values.reshape(1, -1))

    return pred


def predict_many_days(model, scaler, df_X_test_scaled, y_test, days, scale):
    """
    Iterative version of predict_one_day. Note this is not the 'consecutive'
    prediction. That is, I use the actual day 0 data to predict day 1, use
    actual day 1 data, not the predict day 1 data, to predict day 2 and so on.

    Args:
        model: sklearn model.
        scaler: sklearn.preprocessing.StandardScaler().fit(X_train)
        df_X_test_scaled: scaled test set.
        y_test: test set (or validation set) target values.
        days: list like ['2017-04-16', '2017-04-17'].
        scale: time scale, 'H', '30min'

    Returns:
        result: Prediction result, can be used for plot.
        df_err: Error values of this model.
    """
    result = pd.Series(np.zeros(len(y_test)), index=y_test.index)
    err = []
    for day in days:
        pred = predict_one_day(model, scaler, df_X_test_scaled, day, scale)
        result[day] = pred
        err.append(my_errors.errors(y_test[day], pred))
    df_err = pd.DataFrame(err, index=days)
    return result, df_err


def choose_kernel(X_train, y_train, X_test, y_test,
                  scaler, days, scale, c=1000, g=0.05, d=3):
    """
    When you first have the data, you are not sure which kernel to use.
    This test is simple, and this method is maybe not reliable, but it
    may give you a 'first insight' of what kernel may probabl
    Args:
        X_train, y_train, X_test, y_test: scaled data set.
        days: list like ['2017-04-16', '2017-04-17']
        scale: time scale
        c, g, d: C, gamma, degree in SVR model
    """
    model_linear = svm.SVR(kernel='linear', C=c).fit(X_train, y_train)
    model_rbf = svm.SVR(kernel='rbf', C=c, gamma=g).fit(X_train, y_train)
    model_poly = svm.SVR(kernel='poly', C=c, degree=d).fit(X_train, y_train)

    print('Testing linear kernel...')
    result_linear, err_linear = predict_many_days(
        model_linear, scaler, X_test, y_test, days, scale)
    print('Testing rbf kernel...')
    result_rbf, err_rbf = predict_many_days(
        model_rbf, scaler, X_test, y_test, days, scale)
    print('Testing poly kernel...')
    result_poly, err_poly = predict_many_days(
        model_poly, scaler, X_test, y_test, days, scale)

    d = {'linear': [result_linear, err_linear],
         'rbf': [result_rbf, err_rbf],
         'poly': [result_poly, err_poly]}

    a = {key: value[1].mean() for key, value in d.items()}
    avg = pd.DataFrame(a)
    d.update({'avg': avg})
    print()
    print()
    print('Linear kernel:')
    print(err_linear)
    print()
    print('RBF kernel:')
    print(err_rbf)
    print()
    print('Poly kernel:')
    print(err_poly)
    print('Average:')
    print()
    print(avg)

    return d


def grid_search(parameters, X_train, y_train, X_test, y_test,
                scaler, predict_date, scale):
    """
    Args:
        parameters: dict like {'kernel': ['linear', rbf', 'poly'],
                               'C': [100, 1000],
                               'gamma': [1e-4, 1e-3, 1e-2, 1e-1]}
        X_train, y_train: training set, scaled DataFrame / Series
        X_test, y_test: test set, scaled DataFrame / Series
        scaler: scaler used to scale data
        predict_date: str like '4/16/2017', to test the performance.

    Returns:
        search_result: a list of 3 pivoted DataFrames.
    """
    search_result = {}
    for k in parameters['kernel']:
        print('Testing %s kernel...' % k)

        if k == 'linear':
            err = []
            for c in parameters['C']:
                model = svm.SVR('linear', C=c).fit(
                    X_train, y_train)
                pred = predict_one_day(
                    model, scaler, X_test, predict_date, scale)
                err.append(my_errors.errors(y_test[predict_date], pred))
            df = pd.DataFrame(err, index=parameters['C'], columns=[
                'MAE', 'MAPE', 'RMSE'])
            df.index.names = ['C']
            pivoted = df.transpose()

        elif k == 'rbf':
            err = []
            for c in parameters['C']:
                for g in parameters['gamma']:
                    model = svm.SVR('rbf', C=c, gamma=g).fit(
                        X_train, y_train)
                    pred = predict_one_day(
                        model, scaler, X_test, predict_date, scale)
                    err.append(my_errors.errors(y_test[predict_date], pred))
            # for the purpose of MultiIndex
            cs = [i for i in parameters['C']
                  for b in range(len(parameters['gamma']))]
            gs = parameters['gamma'] * len(parameters['C'])
            df = pd.DataFrame(err, index=[cs, gs], columns=[
                'MAE', 'MAPE', 'RMSE'])
            df.index.names = ['C', 'gamma']
            pivoted = df.reset_index().pivot('gamma', 'C')

        elif k == 'poly':
            err = []
            for c in parameters['C']:
                for d in parameters['degree']:
                    model = svm.SVR('poly', C=c, degree=d).fit(
                        X_train, y_train)
                    pred = predict_one_day(
                        model, scaler, X_test, predict_date, scale)
                    err.append(my_errors.errors(y_test[predict_date], pred))
            # for the purpose of MultiIndex
            cs = [i for i in parameters['C']
                  for b in range(len(parameters['degree']))]
            gs = parameters['degree'] * len(parameters['C'])
            df = pd.DataFrame(err, index=[cs, gs], columns=[
                'MAE', 'MAPE', 'RMSE'])
            df.index.names = ['C', 'degree']
            pivoted = df.reset_index().pivot('degree', 'C')

        else:
            raise Exception('Invalid kernel.')

        search_result.update({k: pivoted})
    print('Done.')
    return search_result
