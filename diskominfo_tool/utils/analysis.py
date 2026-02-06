import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from statsmodels.tsa.holtwinters import Holt
import streamlit as st
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

def get_basic_info(df):
    """Returns basic information about the dataframe."""
    return {
        'rows': df.shape[0],
        'columns': df.shape[1],
        'missing_values': df.isnull().sum().sum(),
        'missing_per_column': df.isnull().sum()
    }

def get_descriptive_stats(df):
    """Returns descriptive statistics for numerical columns."""
    return df.describe()

def get_frequency_dist(df, column):
    """Returns frequency distribution for a categorical column."""
    if column in df.columns:
        # value_counts returns Series, reset_index converts to DataFrame
        counts = df[column].value_counts().reset_index()
        counts.columns = [column, 'Count']
        return counts
    return pd.DataFrame()

def get_outliers_iqr(df, column):
    """Detects outliers using IQR method."""
    if column not in df.columns:
        return {'lower_bound': 0, 'upper_bound': 0, 'count': 0, 'data': pd.DataFrame()}
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    return {
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'count': len(outliers),
        'data': outliers
    }

def perform_linear_regression(df, x_col, y_col):
    """Performs simple linear regression."""
    try:
        data = df[[x_col, y_col]].dropna()
        if len(data) < 2:
            return None
            
        X = data[[x_col]]
        y = data[y_col]
        
        model = LinearRegression()
        model.fit(X, y)
        
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        
        return {
            'intercept': model.intercept_,
            'slope': model.coef_[0],
            'r2': r2,
            'X': X, 
            'y': y,
            'y_pred': y_pred
        }
    except Exception:
        return None

def perform_multiple_regression(df, x_cols, y_col):
    """Performs multiple linear regression."""
    try:
        data = df[x_cols + [y_col]].dropna()
        if len(data) < 2:
            return None
            
        X = data[x_cols]
        y = data[y_col]
        
        model = LinearRegression()
        model.fit(X, y)
        
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        
        return {
            'r2': r2,
            'coefficients': dict(zip(x_cols, model.coef_)),
            'y_actual': y,
            'y_pred': y_pred
        }
    except Exception:
        return None

def perform_forecasting(df, time_col, target_col, periods=5, freq_option=None):
    """Performs forecasting using Holt's Linear Trend."""
    try:
        data = df[[time_col, target_col]].dropna()
        
        try:
            data[time_col] = pd.to_datetime(data[time_col])
        except:
            pass
            
        data_indexed = data.set_index(time_col).sort_index()
        
        if freq_option:
            data_indexed = data_indexed.asfreq(freq_option)
            data_indexed = data_indexed.fillna(method='ffill')
        
        model = Holt(data_indexed[target_col], initialization_method="estimated").fit()
        forecast = model.forecast(periods)
        
        history = data_indexed[[target_col]].copy()
        history = history.reset_index()
        if time_col not in history.columns:
            history.rename(columns={'index': time_col}, inplace=True)

        forecast_df = pd.DataFrame({'Forecast': forecast})
        forecast_df = forecast_df.reset_index()
        if time_col not in forecast_df.columns:
            forecast_df.rename(columns={'index': time_col}, inplace=True)
        
        return {
            'history': history,
            'forecast': forecast_df
        }
        
    except Exception as e:
        print(f"Forecasting Error: {e}")
        return None

def perform_backpropagation_forecasting(df, time_col, target_col, periods=5, freq_option=None):
    """
    Performs forecasting using MLPRegressor (Neural Network).
    Uses sliding window approach.
    """
    try:
        data = df[[time_col, target_col]].dropna()
        try:
            data[time_col] = pd.to_datetime(data[time_col])
            data = data.sort_values(by=time_col)
        except:
             pass

        series = data[target_col].values
        
        window_size = min(3, len(series)//2)
        if window_size < 1: window_size = 1
        
        X, y = [], []
        for i in range(len(series) - window_size):
            X.append(series[i:i+window_size])
            y.append(series[i+window_size])
            
        X = np.array(X)
        y = np.array(y)
        
        if len(X) == 0:
            return None

        model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=2000, random_state=42)
        model.fit(X, y)
        
        last_window = series[-window_size:]
        forecast_values = []
        for _ in range(periods):
            pred = model.predict(last_window.reshape(1, -1))[0]
            forecast_values.append(pred)
            last_window = np.append(last_window[1:], pred)
            
        history_df = data.copy() 
        
        forecast_index = None
        if freq_option and hasattr(data[time_col], 'dt'):
             last_date = data[time_col].iloc[-1]
             forecast_index = pd.date_range(start=last_date, periods=periods+1, freq=freq_option)[1:]
        
        if forecast_index is not None:
            forecast_df = pd.DataFrame({'Forecast': forecast_values}, index=forecast_index)
            forecast_df = forecast_df.reset_index().rename(columns={'index': time_col})
        else:
             start_idx = len(series)
             idx = range(start_idx, start_idx + periods)
             forecast_df = pd.DataFrame({'Forecast': forecast_values, time_col: idx})

        return {
            'history': history_df,
            'forecast': forecast_df
        }

    except Exception as e:
        print(f"Backprop Error: {e}")
        return None

def check_assumptions(residuals, X):
    """
    Perform classic assumption tests for regression models.

    Parameters:
        residuals (pd.Series or np.array): Residuals from the regression model.
        X (pd.DataFrame): Independent variables used in the regression model.

    Returns:
        dict: Results of assumption tests (normality, homoscedasticity, autocorrelation, multicollinearity).
    """
    results = {}

    # 1. Normality Test 
    try:
        shapiro_test = shapiro(residuals)
        results['normality'] = {
            'p_value': shapiro_test.pvalue,
            'is_normal': shapiro_test.pvalue > 0.05
        }
    except Exception as e:
        results['normality'] = None
        print(f"Normality Test Error: {e}")

    # 2. Homoscedasticity Test 
    try:
        lm_test = het_breuschpagan(residuals, sm.add_constant(X))
        results['homoscedasticity'] = {
            'p_value': lm_test[1],
            'is_homoscedastic': lm_test[1] > 0.05
        }
    except Exception as e:
        results['homoscedasticity'] = None
        print(f"Homoscedasticity Test Error: {e}")

    # 3. Autocorrelation Test 
    try:
        dw_stat = durbin_watson(residuals)
        results['autocorrelation'] = {
            'statistic': dw_stat,
            'is_correlated': not (1.5 <= dw_stat <= 2.5)
        }
    except Exception as e:
        results['autocorrelation'] = None
        print(f"Autocorrelation Test Error: {e}")

    # 4. Multicollinearity Test 
    try:
        vif_data = pd.DataFrame()
        vif_data['Variable'] = X.columns
        vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        results['multicollinearity'] = {
            'data': vif_data
        }
    except Exception as e:
        results['multicollinearity'] = None
        print(f"Multicollinearity Test Error: {e}")

    return results