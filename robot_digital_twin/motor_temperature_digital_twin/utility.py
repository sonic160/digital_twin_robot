import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape
from sklearn.metrics import max_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from pathlib import Path
from pandas.plotting import scatter_matrix
import logging
import sys
import os



# We provide some supporting function for training a data-driven digital twin for predicting the temperature of motors.


def remove_outliers(df: pd.DataFrame):
    ''' # Description
    Remove outliers from the dataframe based on defined valid ranges. 
    Define a valid range of temperature and voltage. 
    Use ffil function to replace the invalid measurement with the previous value.
    '''
    df['temperature'] = df['temperature'].where(df['temperature'] <= 200, np.nan)
    df['temperature'] = df['temperature'].where(df['temperature'] >= 0, np.nan)
    df['temperature'] = df['temperature'].ffill()

    df['voltage'] = df['voltage'].where(df['voltage'] >= 6000, np.nan)
    df['voltage'] = df['voltage'].where(df['voltage'] <= 9000, np.nan)
    df['voltage'] = df['voltage'].ffill()

    df['position'] = df['position'].where(df['position'] >= 0, np.nan)
    df['position'] = df['position'].where(df['position'] <= 1000, np.nan)
    df['position'] = df['position'].ffill()


def read_all_csvs_one_test(folder_path: str, test_id: str = 'unknown', outlier_removal: callable = remove_outliers) -> pd.DataFrame:
    ''' ## Description
    Combine the six CSV files (each for a motor) in a folder into a single DataFrame. The test condition in the input will be recorded as a column in the combined dataframe.
    
    ## Parameters
    - folder_path: Path to the folder containing the six CSV files
    - test_condition: The condition of the test. Should be read from "Test conditions.xlsx". Default is 'unknown'. 
    - outlier_removal: A function to remove outliers from the dataframe. Default is remove_outliers.

    ## Return
    - combined_df: A DataFrame containing all the data from the CSV files.    
    '''

    # Get a list of all CSV files in the folder
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

    # Create an empty DataFrame to store the combined data
    combined_df = pd.DataFrame()

    # Iterate over the CSV files in the folder
    for file in csv_files:
        # Construct the full path to each CSV file
        file_path = os.path.join(folder_path, file)

        # Read each CSV file into a DataFrame
        df = pd.read_csv(file_path)
        # Drop the time. Will add later.
        df = df.drop(labels=df.columns[0], axis=1)

        # Remove outliers by defining ranges.
        outlier_removal(df)        

        # Extract the file name (excluding the extension) to use as a prefix
        file_name = os.path.splitext(file)[0]

        # Add a prefix to each column based on the file name
        df = df.add_prefix(f'{file_name}_')

        # Concatenate the current DataFrame with the combined DataFrame
        combined_df = pd.concat([combined_df, df], axis=1)

    # Add time and test condition
    df = pd.read_csv(file_path)
    combined_df = pd.concat([df['time'], combined_df], axis=1)
    combined_df.loc[:, 'test_condition'] = test_id

    return combined_df


# Sliding the window to create features and response variables.
def prepare_sliding_window(df_x, y, sequence_name_list, window_size):
    ''' ## Description
    Create a new feature matrix X and corresponding y, by sliding a window of size window_size.

    ## Parameters:
    - df_x: The dataframe containing the features. Must have a column named "test_condition".
    - y: The target variable.
    - sequence_name_list: The list of sequence names, each name represents one sequence.
    - window_size: Size of the sliding window. The previous window size points will be used to create a new feature.

    ## Return  
    - X: Dataframe of the new features.
    - y: Series of the response variable.          
    '''
    X_window = []
    y_window = []
    for name in sequence_name_list:
        df_tmp = df_x[df_x['test_condition']==name]
        y_tmp = y[df_x['test_condition']==name]
        for i in range(len(df_tmp)-window_size):
            X_window.append(df_tmp.iloc[i:i+window_size, :-1].values.flatten())
            y_window.append(y_tmp.iloc[i+window_size-1])
    
    X_window = pd.DataFrame(X_window)
    y_window = pd.Series(y_window)

    return X_window, y_window

def run_cross_val(reg_mdl, df_x, y, n_fold=5, threshold=3, window_size=1):
    ''' ## Description
    Run a k-fold cross validation based on the testing conditions. Each test sequence is considered as a elementary part in the data.

    ## Parameters:
    - reg_mdl: The regression model to be trained.
    - df_X: The dataframe containing the features. Must have a column named "test_condition".
    - y: The target variable.
    - n_fold: The number of folds. Default is 5.
    - threshold: The threshold for the exceedance rate. Default is 3.
    - window_size: Size of the sliding window. The previous window size points will be used to create a new feature.

    ## Return
    - perf: A dataframe containing the performance indicators. There are three columns: "Max error", "RMSE", and "Exceed boundary rate".
    '''
   
    # Get the unique test conditions.
    test_conditions = df_x['test_condition'].unique().tolist()

    # Define the cross validator.
    kf = KFold(n_splits=n_fold)

    # Do the cross validation.
    perf = np.zeros((n_fold, 3))
    counter = 0
    for train_index, test_index in kf.split(test_conditions):
        # Get the dataset names.
        names_train = [test_conditions[i] for i in train_index]
        names_test = [test_conditions[i] for i in test_index]

        # Get training and testing data.       
        X_train, y_train = prepare_sliding_window(df_x, y, names_train, window_size)
        X_test, y_test = prepare_sliding_window(df_x, y, names_test, window_size)

        # Fitting and prediction.
        reg_mdl, _, y_pred = run_reg_mdl(reg_mdl, X_train, y_train, X_test, y_test, is_cv=True)

        # Calculate the performance indicators.
        perf[counter, :] = np.array([max_error(y_test, y_pred), 
        mean_squared_error(y_test, y_pred, squared=False), 
        sum(abs(y_pred - y_test)>threshold)/y_test.shape[0]])

        counter += 1

    return pd.DataFrame(data=perf, columns=['Max error', 'RMSE', 'Exceed boundary rate'])
    

def run_reg_mdl(reg_mdl, X_tr, y_tr, X_test, y_test, is_cv=False):  
    ''' ## Description
    This subfunction fits different regression models, and test the performance in both training and testing dataset. 
    
    ## Parameters
    - reg_mdl: The regression model to be fitted. 
    - X_tr: The training data. 
    - y_tr: The training labels. 
    - X_test: The testing data. 
    - y_test: The testing labels. 
    - is_cv: Whether the function is used for cross validation. Default is False.

    ## Returns
    - reg_mdl: The fitted regression model.
    - y_pred_tr: The predicted labels on the training dataset.
    - y_pred: The predicted labels on the testing dataset.
    '''
    # Training the regression model.    
    reg_mdl = reg_mdl.fit(X_tr, y_tr)

    # Prediction
    y_pred_tr = reg_mdl.predict(X_tr)
    y_pred = reg_mdl.predict(X_test)
    
    # Transform back
    # y_pred_tr = scaler_y.inverse_transform(y_pred_tr)
    # y_pred = scaler_y.inverse_transform(y_pred)
    # y_tr = scaler_y.inverse_transform(y_tr)

    # If not in cv mode, draw the performance on the training and testing dataset.
    if not is_cv:
        model_pef(y_tr, y_test, y_pred_tr, y_pred)

    return reg_mdl, y_pred_tr, y_pred


def model_pef(y_tr, y_test, y_pred_tr, y_pred):
    ''' ## Description
    This subfunction visualize the performance of the fitted model on both the training and testing dataset. 
    
    ## Parameters
    - y_tr: The training labels. 
    - y_test: The testing labels. 
    - y_pred_tr: The predicted labels on the training dataset. 
    - y_pred: The predicted labels on the testing dataset. 
    '''

    # Plot the predicted and truth.
    # Training data set.
    fig_1 = plt.figure(figsize = (16,6))
    ax = fig_1.add_subplot(1,2,1) 
    ax.set_xlabel('index of data point', fontsize = 15)
    ax.set_ylabel('y', fontsize = 15)
    ax.set_title('Prediction V.S. the truth on the training dataset', fontsize = 20)
    ax.plot(range(len(y_tr)), y_tr, 'xb', label='Training data')
    ax.plot(range(len(y_pred_tr)), y_pred_tr, 'or', label='Prediction')
    ax.legend()

    # Testing data set.
    ax = fig_1.add_subplot(1,2,2) 
    ax.set_xlabel('index of data points', fontsize = 15)
    ax.set_ylabel('y', fontsize = 15)
    ax.set_title('Prediction V.S. the truth on the testing dataset', fontsize = 20)
    ax.plot(range(len(y_test)), y_test, 'xb', label='Training data')
    ax.plot(range(len(y_pred)), y_pred, 'or', label='Prediction')
    ax.legend()
    
    # Plot the residual errors.
    # Training data set.
    fig = plt.figure(figsize = (16,6))
    ax = fig.add_subplot(1,2,1) 
    ax.set_xlabel('Index of the data points', fontsize = 15)
    ax.set_ylabel('Residual error', fontsize = 15)
    ax.set_title('Residual errors on the training dataset', fontsize = 20)
    ax.plot(y_pred_tr - y_tr, 'o')
    ax.hlines([3, -3], 0, len(y_tr), linestyles='dashed', colors='r')

    # Testing data set.
    ax = fig.add_subplot(1,2,2) 
    ax.set_xlabel('Index of the data points', fontsize = 15)
    ax.set_ylabel('Residual error', fontsize = 15)
    ax.set_title('Residual errors on the testing dataset', fontsize = 20)
    ax.plot(y_pred-y_test, 'o')
    ax.hlines([3, -3], 0, len(y_test), linestyles='dashed', colors='r')

    # Plot the distribution of residual errors.
    # Training data set.
    fig = plt.figure(figsize = (16,6))
    ax = fig.add_subplot(1,2,1) 
    ax.set_xlabel('Residual error', fontsize = 15)
    ax.set_ylabel('Counts', fontsize = 15)
    ax.set_title('Distribution of residual errors (training)', fontsize = 20)
    ax.hist(y_pred_tr - y_tr)
    ax.axvline(x=3, linestyle='--', color='r')
    ax.axvline(x=-3, linestyle='--', color='r')

    # Testing data set.
    ax = fig.add_subplot(1,2,2) 
    ax.set_xlabel('Residual error', fontsize = 15)
    ax.set_ylabel('Counts', fontsize = 15)
    ax.set_title('Distribution of residual errors (testing)', fontsize = 20)
    ax.hist(y_pred-y_test)
    ax.axvline(x=3, linestyle='--', color='r')
    ax.axvline(x=-3, linestyle='--', color='r')

    # Performance indicators
    # Show the model fitting performance.
    print('Training performance, max error is: ' + str(max_error(y_tr, y_pred_tr ) ))
    print('Training performance, mean root square error is: ' + str(mean_squared_error(y_tr, y_pred_tr ,  squared=False)))
    print('Training performance, residual error > 3 (%): ' + str(sum(abs(y_tr - y_pred_tr)>3)/y_tr.shape[0]*100) + '%')

    print('Prediction performance, max error is: ' + str(max_error(y_pred, y_test)))
    print('Prediction performance, mean root square error is: ' + str(mean_squared_error(y_pred, y_test, squared=False)))
    print('Prediction performance, percentage of residual error > 3' + str(sum(abs(y_pred - y_test)>3)/y_test.shape[0]*100) + '%')



if __name__ == '__main__':
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LinearRegression

    base_dictionary = 'robot_digital_twin/condition_monitoring_matlab_ros/matlab_application/collected_data/'
    # Get all the folders in the base_dictionary
    path_list = os.listdir(base_dictionary)
    # Only keep the folders, not the excel file.
    path_list = path_list[:-1]

    # Read the data.
    df_data = pd.DataFrame()
    for tmp_path in path_list:
        path = base_dictionary + tmp_path
        tmp_df = read_all_csvs_one_test(path, tmp_path)
        df_data = pd.concat([df_data, tmp_df])
        df_data = df_data.reset_index(drop=True)

    # Seperate features and the response variable.
    # name of the response variable.
    y_name = 'data_motor_1_temperature'
    # Remove the irrelavent features.
    df_x = df_data.drop(columns=['data_motor_1_label', 'data_motor_2_label', 'data_motor_3_label',
                        'data_motor_4_label', 'data_motor_5_label', 'data_motor_6_label'])
    df_x = df_x.drop(columns=[y_name])
    # Get y.
    y = df_data.loc[:, y_name]

    # Define the steps of the pipeline
    steps = [
        ('standardizer', StandardScaler()),  # Step 1: StandardScaler
        ('regressor', LinearRegression())    # Step 2: Linear Regression
    ]

    # Create the pipeline
    pipeline = Pipeline(steps)

    # Now you can use this pipeline object for fitting and prediction
    df_perf = run_cross_val(pipeline, df_x, y, window_size=3)
    print(df_perf)
 