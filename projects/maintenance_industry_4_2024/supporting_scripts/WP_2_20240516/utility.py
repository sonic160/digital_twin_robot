import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape
from sklearn.metrics import max_error, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from pathlib import Path
from pandas.plotting import scatter_matrix
import logging
import sys
import os
import matplotlib.pyplot as plt
import copy


# We provide some supporting function for training a data-driven digital twin for predicting the temperature of motors.


def extract_selected_feature(df_data: pd.DataFrame, feature_list: list, motor_idx: int, mdl_type: str):
    ''' ### Description
    Extract the selected features and the response variable from the dataframe.

    ### Parameters
    df_data: The dataframe containing the data.
    feature_list: The list of features to be used.
    motor_idx: The index of the motor.
    mdl_type: The type of the model. 'clf' for classification, 'reg' for regression.

    ### Return
    df_x: The dataframe containing the features.
    y: The response variable.
    '''
    
    # Create a copy of feature_list
    feature_list_local = copy.deepcopy(feature_list)
    # Get the name of the response variable.
    if mdl_type == 'clf':
        y_name = f'data_motor_{motor_idx}_label'
    elif mdl_type == 'reg':
        y_name = f'data_motor_{motor_idx}_temperature'
    else:
        raise ValueError('mdl_type must be \'clf\' or \'reg\'.')
    
    # Remove the y from the feature
    if y_name in feature_list_local:
        feature_list_local.remove(y_name)
    
    # Seperate features and the response variable.
    # Remove the irrelavent features.
    feature_list_local.append('test_condition')
    df_x = df_data[feature_list_local]
    # Get y.
    y = df_data.loc[:, y_name]

    return df_x, y


def run_cv_one_motor(motor_idx, df_data, mdl, feature_list, n_fold=5, threshold=3, window_size=1, sample_step=1, prediction_lead_time=1, single_run_result=True, mdl_type='clf'):
    ''' ### Description
    Run cross validation for a given motor and return the performance metrics for each cv run.
    Can be used for both classification and regression models.

    ### Parameters
    - motor_idx: The index of the motor.
    - df_data: The dataframe containing the data. Must contain a column named 'test_condition'.
    - mdl: The model to be trained. Must have a fit() and predict() method.
    - feature_list: The list of features to be used for the model.
    - n_fold: The number of folds for cross validation. Default is 5. The training and testing data are split by sequence.
    So one needs to make sure n_fold <= the number of sequences.
    - threshold: The threshold for the out-of-threshold percentage. Default is 3. Only needed for regression models.
    - window_size: The window size for the sliding window. Default is 0, which means no sliding window.
    - sample_step: We take every sample_step points from the window_size. default is 1.
    - prediction_lead_time: The number of time steps to predict into the future. Only valid for regression model. Default is 0.
    - single_run_result: Whether to return the performance metrics for each cv run. Default is True.
    - mdl_type: The type of the model. Can be 'clf' or 'reg'. Default is 'clf'.

    ### Return
    - df_perf: The dataframe containing the performance metrics for each cv run.
    If mdl_type is 'clf', the performance metrics are accuracy, precision, recall, and f1 score.
    If mdl_type is 'reg', the performance metrics are max error, mean squared error, and out-of-threshold percentage.
    
    '''
    # Extract the selected features.
    df_x, y = extract_selected_feature(df_data, feature_list, motor_idx, mdl_type)

    print(f'Model for motor {motor_idx}:')
    # Run cross validation.
    df_perf = run_cross_val(mdl, df_x, y, n_fold=n_fold, threshold=threshold, window_size=window_size, sample_step=sample_step, prediction_lead_time=prediction_lead_time, single_run_result=single_run_result, mdl_type=mdl_type)
    print(df_perf)
    print('\n')
    # Print the mean performance and standard error.
    print('Mean performance metric and standard error:')
    for name, metric, error in zip(df_perf.columns, df_perf.mean(), df_perf.std()):
        print(f'{name}: {metric:.4f} +- {error:.4f}') 
    print('\n')

    return df_perf


def read_all_test_data_from_path(base_dictionary: str, pre_processing: callable=None, is_plot=True) -> pd.DataFrame:
    ''' ## Description
    Read all the test data from a folder. The folder should contain subfolders for each test. Each subfolder should contain the six CSV files for each motor. 
    The test condition in the input will be recorded as a column in the combined dataframe.
    
    ## Parameters
    - base_dictionary: Path to the folder containing the subfolders for each test.
    - pre_processing: A function handle to the data preprocessing function.Default is None.
    - is_plot: Whether to plot the data. Default is True.

    ## Return
    - df_data: A DataFrame containing all the data from the CSV files.
    '''

    # Get all the folders in the base_dictionary
    path_list = os.listdir(base_dictionary)
    # Only keep the folders, not the excel file.
    path_list_sorted = sorted(path_list)
    path_list = path_list_sorted[:-1]

    # Read the data.
    df_data = pd.DataFrame()
    for tmp_path in path_list:
        path = base_dictionary + tmp_path
        tmp_df = read_all_csvs_one_test(path, tmp_path, pre_processing)
        df_data = pd.concat([df_data, tmp_df])
        df_data = df_data.reset_index(drop=True)

    # Read the test conditions
    df_test_conditions = pd.read_excel(base_dictionary+'Test conditions.xlsx')

    # Visulize the data
    if is_plot:
        for selected_sequence_idx in path_list:
            filtered_df = df_data[df_data['test_condition'] == selected_sequence_idx]

            print('{}: {}\n'.format(selected_sequence_idx, df_test_conditions[df_test_conditions['Test id'] == selected_sequence_idx]['Description']))

            fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))
            for ax, col in zip(axes.flat, ['data_motor_1_position', 'data_motor_2_position', 'data_motor_3_position', 
                'data_motor_1_temperature', 'data_motor_2_temperature', 'data_motor_3_temperature',
                'data_motor_1_voltage', 'data_motor_2_voltage', 'data_motor_3_voltage']):
                
                label_name = col[:13] + 'label'
                tmp = filtered_df[filtered_df[label_name]==0]
                ax.plot(tmp['time'], tmp[col], marker='o', linestyle='None', label=col)
                tmp = filtered_df[filtered_df[label_name]==1]
                ax.plot(tmp['time'], tmp[col], marker='x', color='red', linestyle='None', label=col)
                ax.set_ylabel(col)

            fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))
            for ax, col in zip(axes.flat, ['data_motor_4_position', 'data_motor_5_position', 'data_motor_6_position',
                'data_motor_4_temperature', 'data_motor_5_temperature', 'data_motor_6_temperature',
                'data_motor_4_voltage', 'data_motor_5_voltage', 'data_motor_6_voltage']):
                
                label_name = col[:13] + 'label'
                tmp = filtered_df[filtered_df[label_name]==0]
                ax.plot(tmp['time'], tmp[col], marker='o', linestyle='None', label=col)
                tmp = filtered_df[filtered_df[label_name]==1]
                ax.plot(tmp['time'], tmp[col], marker='x', color='red', linestyle='None', label=col)
                ax.set_ylabel(col)

            plt.show()
    
    return df_data


def read_all_csvs_one_test(folder_path: str, test_id: str = 'unknown', pre_processing: callable = None) -> pd.DataFrame:
    ''' ## Description
    Combine the six CSV files (each for a motor) in a folder into a single DataFrame. The test condition in the input will be recorded as a column in the combined dataframe.
    
    ## Parameters
    - folder_path: Path to the folder containing the six CSV files
    - test_condition: The condition of the test. Should be read from "Test conditions.xlsx". Default is 'unknown'. 
    - pre_processing: A function handle to the data preprocessing function. Default is None.

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

        # Apply the pre-processing.
        if pre_processing:
            pre_processing(df)

        # Extract the file name (excluding the extension) to use as a prefix
        file_name = os.path.splitext(file)[0]

        # Add a prefix to each column based on the file name
        df = df.add_prefix(f'{file_name}_')

        # Concatenate the current DataFrame with the combined DataFrame
        combined_df = pd.concat([combined_df, df], axis=1)

    # Add time and test condition
    df = pd.read_csv(file_path)
    combined_df = pd.concat([df['time'], combined_df], axis=1)

    # Calculate the time difference since the first row
    time_since_first_row = combined_df['time'] - combined_df['time'].iloc[0]
    # Replace the 'time' column with the time difference
    combined_df['time'] = time_since_first_row

    combined_df.loc[:, 'test_condition'] = test_id

    # Drop the NaN values, which represents the first n data points in the original dataframe.
    combined_df.dropna(inplace=True)

    return combined_df


# Subfunction for create the sliding window.
def concatenate_features(df_input, y_input, X_window, y_window, window_size=1, sample_step=1, prediction_lead_time=1, mdl_type='clf'):
    ''' ### Description
    This function takes every sample_step point from a interval window_size, and concatenate the extracted 
    features into a new feature list X_window. It extracts the corresponding y in y_window.

    ### Parameters
    - df_input: The original feature matrix.
    - y_input: The original response variable.
    - X_window: A list containing the existing concatenated features. Each element is a list of the new features.
    - y_window: A list containing the existing concatenated response variable.
    - window_size: Size of the sliding window. The points in the sliding window will be used to create a new feature.
    - sample_step: We take every sample_step points from the window_size. default is 1.
    - prediction_lead_time: We predict y at t using the previous measurement of y up to t-prediction_lead_time. Default is 1.
    - mdl_type: The type of the model. 'clf' for classification, 'reg' for regression. Default is 'clf'.

    ### Return
    - X_window: The X_window after adding the concatenated features from df_input.
    - y_window: The y_window after adding the corresponding y.
    '''
    
    # Get the index of the last element in the dataframe.
    idx_last_element = len(df_input)-1    
    # Get the indexes the sampled feature in the window, from the last element of the dataframe.
    idx_samples = list(reversed(range(idx_last_element, idx_last_element-window_size, -1*sample_step)))
    # Get the sample X features, and concatenate 
    new_features = df_input.iloc[idx_samples].drop(columns=['test_condition']).values.flatten().tolist()
    
    # If mdl_type is 'reg', we need to add the past ys to the new_features.
    if mdl_type == 'reg':
        if prediction_lead_time<=1: # It is meaningless to add the current y as it is what we need to predict. So the prediction_leatime >= 1.
            prediction_lead_time = 1 
        if prediction_lead_time<window_size and window_size>1: # Otherwise no need to add y_prev as they are beyond the window_size.
            tmp_idx_pred = [x for x in idx_samples if x <= idx_last_element-prediction_lead_time]
            new_features.extend(y_input.iloc[tmp_idx_pred].values.flatten().tolist())
    
    # Add the added features and the corresponding ys into X_window and y_window.
    X_window.append(new_features) # New features
    y_window.append(y_input.iloc[idx_last_element]) # Corresponding y

    return X_window, y_window


# Sliding the window to create features and response variables.
def prepare_sliding_window(df_x, y, sequence_name_list=None, window_size=1, sample_step=1, prediction_lead_time=1, mdl_type='clf'):
    ''' ## Description
    Create a new feature matrix X and corresponding y, by sliding a window of size window_size.

    ## Parameters:
    - df_x: The dataframe containing the features. Must have a column named "test_condition".
    - y: The target variable.
    - sequence_name_list: The list of sequence names, each name represents one sequence.
    - window_size: Size of the sliding window. The points in the sliding window will be used to create a new feature.
    - sample_step: We take every sample_step points from the window_size. default is 1.
    - prediction_lead_time: We predict y at t using the previous measurement of y up to t-prediction_lead_time. Default is 1.
    - mdl_type: The type of the model. 'clf' for classification, 'reg' for regression. Default is 'clf'.

    ## Return  
    - X: Dataframe of the new features.
    - y: Series of the response variable.          
    '''
    X_window = []
    y_window = []    

    # If no sequence_list is given, extract all the unique values from 'test_condition'.
    if sequence_name_list is None:
        sequence_name_list = df_x['test_condition'].unique().tolist()

    # Process sequence by sequence.
    for name in sequence_name_list:
        # Extract one sequence.
        df_tmp = df_x[df_x['test_condition']==name]
        y_tmp = y[df_x['test_condition']==name]

        # Do a loop to concatenate features by sliding the window.
        for i in range(window_size, len(df_tmp)+1):
            X_window, y_window = concatenate_features(df_input=df_tmp.iloc[i-window_size:i, :], y_input=y_tmp.iloc[i-window_size:i], 
                X_window=X_window, y_window=y_window, window_size=window_size, sample_step=sample_step, prediction_lead_time=prediction_lead_time, mdl_type=mdl_type)
        
    # Transform into dataframe.
    X_window = pd.DataFrame(X_window)
    y_window = pd.Series(y_window)

    return X_window, y_window


def run_cross_val(mdl, df_x, y, n_fold=5, threshold=3, window_size=1, sample_step=1, prediction_lead_time=1, single_run_result=True, mdl_type='reg'):
    ''' ## Description
    Run a k-fold cross validation based on the testing conditions. Each test sequence is considered as a elementary part in the data.

    ## Parameters:
    - mdl: The model to be trained.
    - df_X: The dataframe containing the features. Must have a column named "test_condition".
    - y: The target variable.
    - n_fold: The number of folds. Default is 5.
    - threshold: The threshold for the exceedance rate. Default is 3. Only needed when mdl_type == 'reg'.
    - window_size: Size of the sliding window. The previous window size points will be used to create a new feature.
    - sample_step: We take every sample_step points from the window_size. default is 1.
    - prediction_lead_time: The number of time steps to predict into the future. Only valid for regression model. Default is 0.
    - single_run_result: Whether to return the single run result. Default is True.
    - mdl_type: The type of the model. Default is 'reg'. Alternately, put 'clf' for classification.

    ## Return
    - perf: A dataframe containing the performance indicators.
    '''
   
    # Get the unique test conditions.
    test_conditions = df_x['test_condition'].unique().tolist()

    # Define the cross validator.
    kf = KFold(n_splits=n_fold)

    # Do the cross validation.
    # Set initial values for perf to store the performance of each run.
    if mdl_type == 'reg':
        perf = np.zeros((n_fold, 3))
    elif mdl_type == 'clf':
        perf = np.zeros((n_fold, 4))
    else:
        TypeError('mdl_type should be either "reg" or "clf".')
    
    counter = 0
    for train_index, test_index in kf.split(test_conditions):
        # Get the dataset names.
        names_train = [test_conditions[i] for i in train_index]
        names_test = [test_conditions[i] for i in test_index]

        # Get training and testing data.       
        X_train, y_train = prepare_sliding_window(df_x, y, names_train, window_size=window_size, sample_step=sample_step, prediction_lead_time=prediction_lead_time, mdl_type=mdl_type)
        X_test, y_test = prepare_sliding_window(df_x, y, names_test, window_size, sample_step=sample_step, prediction_lead_time=prediction_lead_time, mdl_type=mdl_type)

        # Fitting and prediction.
        if mdl_type == 'reg':
            # Train and predict.
            mdl, y_pred_tr, y_pred = run_mdl(mdl, X_train, y_train, X_test)
            # Calculate the performance indicators.
            perf[counter, :] = np.array([max_error(y_test, y_pred), 
            mean_squared_error(y_test, y_pred, squared=False), 
            sum(abs(y_pred - y_test)>threshold)/y_test.shape[0]])
            # If selected, draw the performance on the training and testing dataset.
            if single_run_result:
                show_reg_result(y_train, y_test, y_pred_tr, y_pred, threshold=threshold)
        elif mdl_type == 'clf':
            mdl, y_pred_tr, y_pred = run_mdl(mdl, X_train, y_train, X_test)
            accuracy, precision, recall, f1 = cal_classification_perf(y_test, y_pred)
            perf[counter, :] = np.array([accuracy, precision, recall, f1])
            if single_run_result:
                show_clf_result(y_train, y_test, y_pred_tr, y_pred)

        else:
            TypeError('mdl_type should be either "reg" or "clf".')

        counter += 1

    if mdl_type == 'reg':
        return pd.DataFrame(data=perf, columns=['Max error', 'RMSE', 'Exceed boundary rate'])
    elif mdl_type == 'clf':
        return pd.DataFrame(data=perf, columns=['Accuracy', 'Precision', 'Recall', 'F1 score'])
    else:
        TypeError('mdl_type should be either "reg" or "clf".')


def cal_classification_perf(y_true, y_pred):
    ''' ### Description
    This function calculates the classification performance: Accuracy, Precision, Recall and F1 score.
    It considers different scenarios when divide by zero could occur for Precision, Recall and F1 score calculation.

    ### Parameters:
    - y_true: The true labels.
    - y_pred: The predicted labels.

    ### Return:
    - accuracy: The accuracy.
    - precision: The precision.
    - recall: The recall.
    - f1: The F1 score.
    '''
    accuracy = accuracy_score(y_true, y_pred)
    # Only when y_pred contains no zeros, and y_true contains no zeros, set precision to be 1 when divide by zero occurs.
    if sum(y_true)==0 and sum(y_pred)==0:
        precision = precision_score(y_true, y_pred, zero_division=1)
        recall = recall_score(y_true, y_pred, zero_division=1)
        f1 = f1_score(y_true, y_pred, zero_division=1)
    else:
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

    return accuracy, precision, recall, f1
    
    

def run_mdl(mdl, X_tr, y_tr, X_test):  
    ''' ## Description
    This subfunction fits different ML models, and test the performance in both training and testing dataset. 
    
    ## Parameters
    - mdl: The model to be fitted. Can be regression or classification models.
    - X_tr: The training data. 
    - y_tr: The training labels. 
    - X_test: The testing data.

    ## Returns
    - mdl: The fitted model.
    - y_pred_tr: The predicted labels on the training dataset.
    - y_pred: The predicted labels on the testing dataset.
    '''
    # Training the regression model.    
    mdl = mdl.fit(X_tr, y_tr)

    # Prediction
    y_pred_tr = mdl.predict(X_tr)
    y_pred = mdl.predict(X_test)
    
    # Transform back
    # y_pred_tr = scaler_y.inverse_transform(y_pred_tr)
    # y_pred = scaler_y.inverse_transform(y_pred)
    # y_tr = scaler_y.inverse_transform(y_tr)

    return mdl, y_pred_tr, y_pred


def show_reg_result(y_tr, y_test, y_pred_tr, y_pred, threshold=3):
    ''' ## Description
    This subfunction visualize the performance of the fitted model on both the training and testing dataset. 
    
    ## Parameters
    - y_tr: The training labels. 
    - y_test: The testing labels. 
    - y_pred_tr: The predicted labels on the training dataset. 
    - y_pred: The predicted labels on the testing dataset. 
    - threshold: The threshold for exceeding the boundary.
    '''

    # Plot the predicted and truth.
    # Training data set.
    fig_1 = plt.figure(figsize = (16,6))
    ax = fig_1.add_subplot(1,2,1) 
    ax.set_xlabel('index of data point', fontsize = 15)
    ax.set_ylabel('y', fontsize = 15)
    ax.set_title('Prediction V.S. the truth on the training dataset', fontsize = 20)
    ax.plot(range(len(y_tr)), y_tr, 'xb', label='Truth')
    ax.plot(range(len(y_pred_tr)), y_pred_tr, 'or', label='Prediction')
    ax.legend()

    # Testing data set.
    ax = fig_1.add_subplot(1,2,2) 
    ax.set_xlabel('index of data points', fontsize = 15)
    ax.set_ylabel('y', fontsize = 15)
    ax.set_title('Prediction V.S. the truth on the testing dataset', fontsize = 20)
    ax.plot(range(len(y_test)), y_test, 'xb', label='Truth')
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
    ax.hlines([threshold, -1*threshold], 0, len(y_tr), linestyles='dashed', colors='r')

    # Testing data set.
    ax = fig.add_subplot(1,2,2) 
    ax.set_xlabel('Index of the data points', fontsize = 15)
    ax.set_ylabel('Residual error', fontsize = 15)
    ax.set_title('Residual errors on the testing dataset', fontsize = 20)
    ax.plot(y_pred-y_test, 'o')
    ax.hlines([threshold, -1*threshold], 0, len(y_test), linestyles='dashed', colors='r')

    # Plot the distribution of residual errors.
    # Training data set.
    fig = plt.figure(figsize = (16,6))
    ax = fig.add_subplot(1,2,1) 
    ax.set_xlabel('Residual error', fontsize = 15)
    ax.set_ylabel('Counts', fontsize = 15)
    ax.set_title('Distribution of residual errors (training)', fontsize = 20)
    ax.hist(y_pred_tr - y_tr)
    ax.axvline(x=threshold, linestyle='--', color='r')
    ax.axvline(x=-1*threshold, linestyle='--', color='r')

    # Testing data set.
    ax = fig.add_subplot(1,2,2) 
    ax.set_xlabel('Residual error', fontsize = 15)
    ax.set_ylabel('Counts', fontsize = 15)
    ax.set_title('Distribution of residual errors (testing)', fontsize = 20)
    ax.hist(y_pred-y_test)
    ax.axvline(x=threshold, linestyle='--', color='r')
    ax.axvline(x=-1*threshold, linestyle='--', color='r')

    # Performance indicators
    # Show the model fitting performance.
    print('\n New cv run:\n')
    print('Training performance, max error is: ' + str(max_error(y_tr, y_pred_tr ) ))
    print('Training performance, mean root square error is: ' + str(mean_squared_error(y_tr, y_pred_tr ,  squared=False)))
    print(f'Training performance, residual error > {threshold}: ' + str(sum(abs(y_tr - y_pred_tr)>3)/y_tr.shape[0]*100) + '%')
    print('\n')
    print('Prediction performance, max error is: ' + str(max_error(y_test, y_pred)))
    print('Prediction performance, mean root square error is: ' + str(mean_squared_error(y_test, y_pred, squared=False)))
    print(f'Prediction performance, percentage of residual error > {threshold}：' + str(sum(abs(y_pred - y_test)>3)/y_test.shape[0]*100) + '%')

    plt.show()


def show_clf_result(y_tr, y_test, y_pred_tr, y_pred):
    ''' ## Description
    This subfunction visualize the performance of the fitted model on both the training and testing dataset for the classfication model. 
    
    ## Parameters
    - y_tr: The training labels. 
    - y_test: The testing labels. 
    - y_pred_tr: The predicted labels on the training dataset. 
    - y_pred: The predicted labels on the testing dataset. 
    '''

    # Plot the predicted and truth.
    # Training data set.
    fig_1 = plt.figure(figsize = (16,12))
    ax = fig_1.add_subplot(2,2,1) 
    ax.set_xlabel('index of data point', fontsize = 15)
    ax.set_ylabel('y', fontsize = 15)
    ax.set_title('Training: Truth', fontsize = 20)
    ax.plot(range(len(y_tr)), y_tr, 'xb', label='Truth')
    ax.legend()

    ax = fig_1.add_subplot(2,2,3) 
    ax.set_xlabel('index of data point', fontsize = 15)
    ax.set_ylabel('y', fontsize = 15)
    ax.set_title('Training: Prediction', fontsize = 20)
    ax.plot(range(len(y_pred_tr)), y_pred_tr, 'or', label='Prediction')
    ax.legend()

    # Testing data set.
    ax = fig_1.add_subplot(2,2,2) 
    ax.set_xlabel('index of data points', fontsize = 15)
    ax.set_ylabel('y', fontsize = 15)
    ax.set_title('Testing: Truth', fontsize = 20)
    ax.plot(range(len(y_test)), y_test, 'xb', label='Truth')
    ax.legend()

    ax = fig_1.add_subplot(2,2,4) 
    ax.set_xlabel('index of data points', fontsize = 15)
    ax.set_ylabel('y', fontsize = 15)
    ax.set_title('Testing: Prediction', fontsize = 20)
    ax.plot(range(len(y_pred)), y_pred, 'or', label='Prediction')
    ax.legend()
    
    # Performance indicators
    # Show the model fitting performance.
    accuracy_tr, precision_tr, recall_tr, f1_tr = cal_classification_perf(y_tr, y_pred_tr)
    print('\n New cv run:\n')
    print('Training performance, accuracy is: ' + str(accuracy_tr))
    print('Training performance, precision is: ' + str(precision_tr))
    print('Training performance, recall: ' + str(recall_tr))
    print('Training performance, F1: ' + str(f1_tr))
    print('\n')
    accuracy, precision, recall, f1 = cal_classification_perf(y_test, y_pred)
    print('Prediction performance, accuracy is: ' + str(accuracy))
    print('Prediction performance, precision is: ' + str(precision))
    print('Prediction performance, recall is：' + str(recall))
    print('Prediction performance, F1 is：' + str(f1))

    plt.show()


if __name__ == '__main__':
    # Test extract_selected_features()

    # Define the path to the folder 'collected_data'
    base_dictionary = 'projects/maintenance_industry_4_2024/dataset/training_data/'
    # Read all the data
    df_data = read_all_test_data_from_path(base_dictionary, is_plot=False)

    feature_list_all = ['time', 'data_motor_1_position', 'data_motor_1_temperature', 'data_motor_1_voltage',
                    'data_motor_2_position', 'data_motor_2_temperature', 'data_motor_2_voltage',
                    'data_motor_3_position', 'data_motor_3_temperature', 'data_motor_3_voltage',
                    'data_motor_4_position', 'data_motor_4_temperature', 'data_motor_4_voltage',
                    'data_motor_5_position', 'data_motor_5_temperature', 'data_motor_5_voltage',
                    'data_motor_6_position', 'data_motor_6_temperature', 'data_motor_6_voltage']
    df_x, y = extract_selected_feature(df_data=df_data, feature_list=feature_list_all, motor_idx=6, mdl_type='reg')

    sequence_list = ['20240425_093699', '20240425_094425', '20240426_140055',
                    '20240503_164675', '20240503_165189',
                    '20240503_163963', '20240325_155003']
    X_window, y_window = prepare_sliding_window(df_x, y, sequence_list, window_size=50, sample_step=10, prediction_lead_time=49, mdl_type='reg')