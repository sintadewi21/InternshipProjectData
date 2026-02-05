import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from statsmodels.tsa.holtwinters import Holt
import streamlit as st

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
            'X': X, # DataFrame for visualization compatibility
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
        
        # Prepare history with time_col column
        history = data_indexed[[target_col]].copy()
        history = history.reset_index()
        if time_col not in history.columns:
            # Fallback if index name was lost
            history.rename(columns={'index': time_col}, inplace=True)

        # Prepare forecast with time_col column
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
            
        history_df = data.copy() # Contains time_col and target_col
        # Ensure target_col is there (it is)
        
        # Forecast DataFrame setup
        forecast_index = None
        if freq_option and hasattr(data[time_col], 'dt'):
             last_date = data[time_col].iloc[-1]
             forecast_index = pd.date_range(start=last_date, periods=periods+1, freq=freq_option)[1:]
        
        if forecast_index is not None:
            forecast_df = pd.DataFrame({'Forecast': forecast_values}, index=forecast_index)
            forecast_df = forecast_df.reset_index().rename(columns={'index': time_col})
        else:
             # Create a numeric continuation
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