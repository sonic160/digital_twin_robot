import pandas as pd  
import os 


def smooth_data_moving_average(data, window_size):
    """Smooth data by computing moving average."""
    return data.rolling(window=window_size, min_periods=1).mean()


label_columns = ['data_motor_1_label', 'data_motor_2_label', 'data_motor_3_label', 'data_motor_4_label', 'data_motor_5_label', 'data_motor_6_label']

def remove_outliers(data, alpha):
        if all(col in data.columns for col in label_columns):
                # If label columns exist, remove them
                data_without_labels = data.drop(columns=label_columns)
        else:
                # If label columns do not exist, keep the original dataframe
                data_without_labels = data.copy(deep = True)
                
        # Interquartile range method
        Q1 = data_without_labels.quantile(0.25, numeric_only=True)
        Q3 = data_without_labels.quantile(0.75, numeric_only=True)
        IQR = Q3 - Q1
        lower_bound = Q1 - alpha * IQR
        upper_bound = Q3 + alpha * IQR
        mask = ~((data_without_labels.lt(lower_bound, axis=1)) | (data_without_labels.gt(upper_bound, axis=1))).any(axis=1)
        return data[mask]
    
def remove_outliers2(data, alpha):
    if all(col in data.columns for col in label_columns):
        # If label columns exist, remove them
        data_without_labels = data.drop(columns=label_columns)
    else:
        # If label columns do not exist, keep the original dataframe
        data_without_labels = data.copy(deep=True)
    
    # Interquartile range method
    Q1 = data_without_labels.quantile(0.25, numeric_only=True)
    Q3 = data_without_labels.quantile(0.75, numeric_only=True)
    IQR = Q3 - Q1
    lower_bound = Q1 - alpha * IQR
    upper_bound = Q3 + alpha * IQR
    mask = ~((data_without_labels.lt(lower_bound, axis=1)) | (data_without_labels.gt(upper_bound, axis=1))).any(axis=1)
    
    # Return indices of outliers removed
    removed_indices = data.index[~mask]
    
    return data[mask], removed_indices