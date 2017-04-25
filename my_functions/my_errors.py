"""
Functions of metrics usually used to measure the prediction error.
obs: observed values, actual values.
pred: predctions.

RMSE and MAE are usually used to compare models trained with same data.

@Caceros Thu Apr 20 22:39:17 2017
"""
import numpy as np


def RMSE(obs, pred):
    """
    Root Mean Square Error
    """
    return np.sqrt(MSE(obs, pred))


def MSE(obs, pred):
    """
    Mean Square Error
    """
    n = len(obs)
    if n != len(pred):
        raise Exception('Length error.')

    return ((obs - pred)**2).sum() / n


def MAE(obs, pred):
    """
    Mean Absolute Error
    """
    n = len(obs)
    if n != len(pred):
        raise Exception('Length error.')

    return (np.abs(obs - pred)).sum() / n


def MAPE(obs, pred):
    """
    Mean Absolute Percentage Error
    """
    n = len(obs)
    if n != len(pred):
        raise Exception('Length error.')

    return (np.abs((obs - pred) / obs)).sum() / n * 100


def errors(obs, pred):
    """
    Return a dict of RMSE, MAE, MAPE
    """
    return {'RMSE': RMSE(obs, pred),
            'MAE': MAE(obs, pred),
            'MAPE': MAPE(obs, pred)}
