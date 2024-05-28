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
from tqdm import tqdm


# We provide some supporting function for training a data-driven digital twin for predicting the temperature of motors.


class FaultDetectReg():
    ''' ### Description
    This is the class for fault detection based on regression models.

    ### Initialize
    Initialize the class with the following parameters:    
    - reg_mdl: The pre-trained regression model.
    - pre_trained: If the provided reg_mdl is pretrained. Default is True.
    - threshold: threshold for the residual error. If the residual error is larger than the threshold, we consider it as a fault. Default is 1.
    - abnormal_limit: If the model predict a abnormal point and its residual error <= abnormal_limit, the predicted value will be used to replace
      the measured value in the next prediction. Default is 1.
    - window_size: Size of the sliding window. The previous window size points will be used to create a new feature.
    - sample_step: We take every sample_step points from the window_size. default is 1.
    - prediction_lead_time: The number of time steps to predict into the future. Only valid for regression model. Default is 1.   
    
    ### Attributes
    - reg_mdl: The pre-trained regression model.
    - pre_trained: If the provided reg_mdl is pretrained. Default is True.
    - threshold: threshold for the residual error. If the residual error is larger than the threshold, we consider it as a fault. Default is 1.
    - abnormal_limit: If the model predict a abnormal point and its residual error <= abnormal_limit, the predicted value will be used to replace
      the measured value in the next prediction. Default is 1.
    - window_size: Size of the sliding window. The previous window size points will be used to create a new feature.
    - sample_step: We take every sample_step points from the window_size. default is 1.
    - prediction_lead_time: The number of time steps to predict into the future. Only valid for regression model. Default is 1.   
    - residual_norm: The stored residual errors for all the normal samples in the current sequence. 
      It is used to calculate the threshold adpatively.
    - threshold_int: The threshold value set by the initialization function.
    
    ### Methods
    - fit(): This method learns the regression model from the training data. If self.pre_trained is True, it will directly use the pre-trained model.
    - predict(): This method predicts the labels for the input data.
    - predict_label_by_reg_base(): This method is the base function for predicting the label, called from self.predict().
    - run_cross_val(): This method defines a cross validation scheme to test the performance of the model.

    '''
    def __init__(self, reg_mdl, pre_trained: bool=True, threshold: int=1, abnormal_limit: int=3, window_size: int = 1, sample_step: int = 1, pred_lead_time: int = 1):
        ''' ### Description
        Initialization function.        
        '''
        self.reg_mdl = reg_mdl
        self.window_size = window_size
        self.sample_step = sample_step
        self.pred_lead_time = pred_lead_time
        self.threshold = threshold
        self.threshold_int = threshold
        self.pre_trained = pre_trained
        self.abnormal_limit = abnormal_limit
        self.residual_norm = []


    def fit(self, df_x, y_label, y_response):
        ''' ### Description
        Learn the regression model from the training data, and use the trained model to predict the labels and the repsonse variables for the training data.
        If self.pre_trained is True, it will directly use the pre-trained model.     

        ### Parameters
        - df_x: The training features.
        - y_label: The labels in the training dataset.
        - y_response: The response variable in the training dataset.

        ### Returns
        - y_label_pred: The predicted labels using the best regression model learnt from training data.
        - y_response_pred: The predicted response variable using the best regression model learnt from training data.
        '''
        # Train the regression model if not pretrained.
        if not self.pre_trained:
            # Train a regression model first, based on the normal samples.
            # Align indices
            df_x_normal = copy.deepcopy(df_x)
            # Initialize a counter for numbering
            counter = 0
            flag = False
            # Iterate over each row in df
            for i in range(len(y_label)):
                value = y_label.iloc[i]
                if i>0:
                    if value==0 and y_label.iloc[i-1] == 1:
                        flag = True
                        counter += 1                
                    if value==1 and y_label.iloc[i-1] == 0:
                        flag = False                
                    if flag:
                        df_x_normal.at[df_x_normal.index[i], 'test_condition'] += f'_{counter}'               

            df_x_normal = df_x_normal[y_label==0]
            y_response_normal = y_response[y_label==0]

            # Train the regression model.
            x_tr, y_temp_tr = prepare_sliding_window(df_x=df_x_normal, y=y_response_normal, window_size=self.window_size, sample_step=self.sample_step, prediction_lead_time=self.pred_lead_time, mdl_type='reg')
            self.reg_mdl = self.reg_mdl.fit(x_tr, y_temp_tr)

        # Calculate and return the predicted labels and response variable for the training data.
        y_label_pred, y_response_pred = self.predict(df_x, y_response)

        return y_label_pred, y_response_pred


    def predict(self, df_x_test, y_response_test):
        ''' ### Description
        Predict the labels using the trained regression model and the measured response variable.
        Note that if a fault is predicted, the predicted, not measured response variable will be used to concatenate features
        for predicting the values of next response variable.

        ### Parameters
        - df_x_test: The testing features.
        - y_response_test: The measured response variable.

        ### Return
        - y_label_pred: The predicted labels.
        - y_response_pred: The predicted response variable.
        '''
        # Get parameters.
        window_size = self.window_size
        sample_step = self.sample_step
        prediction_lead_time = self.pred_lead_time

        # Get the sequence names.
        sequence_name_list = df_x_test['test_condition'].unique().tolist()

        # Initial values
        y_label_pred = []
        y_response_pred = []

        # Process sequence by sequence.
        for name in tqdm(sequence_name_list):
            # Reset the stored residual errors for normal samples and the threshold value.
            self.residual_norm = []
            self.threshold = self.threshold_int

            # Extract one sequence.
            df_x_test_seq = df_x_test[df_x_test['test_condition']==name]
            y_temp_test_seq = y_response_test[df_x_test['test_condition']==name]        
            y_temp_local = copy.deepcopy(y_temp_test_seq)

            # Initial values of the prediction.
            # Length is len - window_size + 1 because we need to use the sliding window to define features.
            y_label_pred_tmp = np.zeros(len(df_x_test_seq)-window_size+1) # Predicted label.
            y_temp_pred_tmp = np.zeros(len(df_x_test_seq)-window_size+1) # Predicted temperature.
            
            # Making the prediction using a sequential approach.
            for i in range(window_size, len(df_x_test_seq)+1):
                # Get the data up to current moment i-1.
                tmp_df_x = df_x_test_seq.iloc[i-window_size:i, :]
                tmp_y_temp_measure = y_temp_local.iloc[i-window_size:i]

                # Use the same sliding window to generate features.
                feature_x, _ = concatenate_features(df_input=tmp_df_x, y_input=tmp_y_temp_measure, X_window=[], y_window=[], 
                        window_size=window_size, sample_step=sample_step, prediction_lead_time=prediction_lead_time, mdl_type='reg')
                
                # Make prediction.
                tmp_y_label_pred, tmp_y_temp_pred, tmp_residual= self.predict_label_by_reg_base(X=feature_x, y_temp_measure=tmp_y_temp_measure.iloc[-1])
                
                # Save the prediction at the current moment i.
                y_label_pred_tmp[i-window_size] = tmp_y_label_pred[-1]
                y_temp_pred_tmp[i-window_size] = tmp_y_temp_pred[-1]

                # If we predict a failure, we replace the measure with the predicted temperature.
                # This is to avoid accumulation of errors.
                if tmp_y_label_pred[-1] == 1 and tmp_residual <= self.abnormal_limit:
                    y_temp_local.iloc[i-1] = tmp_y_temp_pred[-1]

            # Save the results and proceed to the next sequence.
            y_label_pred.extend(y_label_pred_tmp)
            y_response_pred.extend(y_temp_pred_tmp)

        return y_label_pred, y_response_pred


    def predict_label_by_reg_base(self, X, y_temp_measure):
        ''' ### Description
        Predict the response variable (temperature) using the regression model.
        Then, it saves the residuals of all the normal samples in the current sequence.

        ### Parameters
        - X: The features.
        - y_temp_measure: The temperature measurements.

        ### Return
        - y_label pred: The predicted labels.
        - y_temp_pred: The predicted temperature.
        - residual_error: The residual error.        
        '''
        # Predict the temperature
        mdl = self.reg_mdl # Get the regression model.
        y_temp_pred = mdl.predict(X) # Predict the temperature.
        # Calculate the residual
        residual_error = np.array(abs(y_temp_pred - y_temp_measure)) 
        
        # Predict the label based on the threshold
        y_label_pred = np.where(residual_error <= self.threshold, 0, 1)

        # Update the residual error database.
        self.residual_norm.extend(residual_error[y_label_pred==0])
        threshold_prop = np.mean(self.residual_norm) + 6*np.std(self.residual_norm)
        # threshold_prop = np.percentile(self.residual_norm, 75) + 1.5*(np.percentile(self.residual_norm, 75)-np.percentile(self.residual_norm, 25))

        if self.threshold < threshold_prop and threshold_prop < 1.5:
            self.threshold = threshold_prop

        return y_label_pred, y_temp_pred, residual_error
    

    def run_cross_val(self, df_x, y_label, y_response, n_fold=5, single_run_result=True):
        ''' ## Description
        Run a k-fold cross validation based on the testing conditions. Each test sequence is considered as a elementary part in the data.

        ## Parameters:
        - df_X: The dataframe containing the features. Must have a column named "test_condition".
        - y_label: The target variable, i.e., failure.
        - y_response: The response variable associated with the target variable, e.g., temperature.
        - n_fold: The number of folds. Default is 5.
        - single_run_result: Whether to return the single run result. Default is True.
        
        ## Return
        - perf: A dataframe containing the performance indicators.
        '''
    
        # Get the unique test conditions.
        test_conditions = df_x['test_condition'].unique().tolist()

        # Define the cross validator.
        kf = KFold(n_splits=n_fold)

        # Set initial values for perf to store the performance of each run.
        perf = np.zeros((n_fold, 4))

        counter = 0
        for train_index, test_index in kf.split(test_conditions):
            # Get the dataset names.
            names_train = [test_conditions[i] for i in train_index]
            names_test = [test_conditions[i] for i in test_index]

            # If not pretrained, train the model.
            if not self.pre_trained:
                df_tr = df_x[df_x['test_condition'].isin(names_train)]
                y_tr = y_label[df_x['test_condition'].isin(names_train)]
                y_response_tr = y_response[df_x['test_condition'].isin(names_train)]

                # Train the model.
                y_tr_pred, y_response_tr_pred = self.fit(df_x=df_tr, y_label=y_tr, y_response=y_response_tr)
            
            # Extract the training and testing data.
            df_test = df_x[df_x['test_condition'].isin(names_test)]
            y_test = y_label[df_x['test_condition'].isin(names_test)]
            y_response_test = y_response[df_x['test_condition'].isin(names_test)]

               
            # Predict for the testing data.
            y_pred, y_response_test_pred = self.predict(df_x_test=df_test, y_response_test=y_response_test)

            # Calculate the performance.
            # Truncate the true values for y_test and y_response_test in the same format as the concatenated features.
            _, y_test = prepare_sliding_window(df_x=df_test, y=y_test, window_size=self.window_size, sample_step=self.sample_step, prediction_lead_time=self.pred_lead_time, mdl_type='clf')
            accuracy, precision, recall, f1 = cal_classification_perf(y_test, pd.Series(y_pred))
            perf[counter, :] = np.array([accuracy, precision, recall, f1])
            
            # Show single run results.
            if single_run_result:
                if self.pre_trained: # Only show the testing performance.
                    # Show the results.
                    fig_1 = plt.figure(figsize = (8,18))
                    axes_test = [fig_1.add_subplot(3, 1, 1), fig_1.add_subplot(3, 1, 2), fig_1.add_subplot(3, 1, 3)]
                    _, y_response_test = prepare_sliding_window(df_x=df_test, y=y_response_test, window_size=self.window_size, sample_step=self.sample_step, prediction_lead_time=self.pred_lead_time, mdl_type='reg')
                    show_reg_result_single_run(y_response_test, y_response_test_pred, axes_test, 'testing', self.threshold)

                    fig_2 = plt.figure(figsize = (8,12))
                    ax_test_true = fig_2.add_subplot(2,1,1)
                    ax_test_pred = fig_2.add_subplot(2,1,2)
                    show_clf_result_single_run(y_test, y_pred, ax_test_true, ax_test_pred, suffix='testing')

                    plt.show()
                else:
                    # Truncate the true values for y_test and y_response_test in the same format as the concatenated features.
                    _, y_tr = prepare_sliding_window(df_x=df_tr, y=y_tr, window_size=self.window_size, sample_step=self.sample_step, prediction_lead_time=self.pred_lead_time, mdl_type='clf')
                    _, y_response_tr = prepare_sliding_window(df_x=df_tr, y=y_response_tr, window_size=self.window_size, sample_step=self.sample_step, prediction_lead_time=self.pred_lead_time, mdl_type='reg')
                    _, y_response_test = prepare_sliding_window(df_x=df_test, y=y_response_test, window_size=self.window_size, sample_step=self.sample_step, prediction_lead_time=self.pred_lead_time, mdl_type='reg')
                    
                    # Show the results.
                    show_reg_result(y_tr=y_response_tr, y_test=y_response_test, y_pred_tr=y_response_tr_pred, y_pred=y_response_test_pred, threshold=self.threshold)
                    show_clf_result(y_tr, y_test, y_tr_pred, y_pred)

            counter += 1

        return pd.DataFrame(data=perf, columns=['Accuracy', 'Precision', 'Recall', 'F1 score'])
        

def extract_selected_feature_testing(df_data: pd.DataFrame, feature_list: list, motor_idx: int, mdl_type: str):
    ''' ### Description
    Extract the selected features from the dataframe. Used for testing data.

    ### Parameters
    df_data: The dataframe containing the data.
    feature_list: The list of features to be used.
    motor_idx: The index of the motor.
    mdl_type: The type of the model. 'clf' for classification, 'reg' for regression.

    ### Return
    df_x: The dataframe containing the features.
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
    # Filter out only directories, excluding .xlsx files
    path_list = [item for item in path_list if os.path.isdir(os.path.join(base_dictionary, item))]

    # path_list_sorted = sorted(path_list)
    # path_list = path_list_sorted[:-1]

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

                # If we have labels
                if sum(filtered_df[label_name]==0)>0 or sum(filtered_df[label_name]==1)>0:
                    tmp = filtered_df[filtered_df[label_name]==0]
                    ax.plot(tmp['time'], tmp[col], marker='o', linestyle='None', label=col)
                    tmp = filtered_df[filtered_df[label_name]==1]
                    ax.plot(tmp['time'], tmp[col], marker='x', color='red', linestyle='None', label=col)
                else:
                    ax.plot(filtered_df['time'], filtered_df[col], marker='o', linestyle='None', label=col)
                ax.set_ylabel(col)

            fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))
            for ax, col in zip(axes.flat, ['data_motor_4_position', 'data_motor_5_position', 'data_motor_6_position',
                'data_motor_4_temperature', 'data_motor_5_temperature', 'data_motor_6_temperature',
                'data_motor_4_voltage', 'data_motor_5_voltage', 'data_motor_6_voltage']):
                
                label_name = col[:13] + 'label'
                
                # If we have labels
                if sum(filtered_df[label_name]==0)>0 or sum(filtered_df[label_name]==1)>0:
                    tmp = filtered_df[filtered_df[label_name]==0]
                    ax.plot(tmp['time'], tmp[col], marker='o', linestyle='None', label=col)
                    tmp = filtered_df[filtered_df[label_name]==1]
                    ax.plot(tmp['time'], tmp[col], marker='x', color='red', linestyle='None', label=col)
                else:
                    ax.plot(filtered_df['time'], filtered_df[col], marker='o', linestyle='None', label=col)
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
    try:
        time_since_first_row = combined_df['time'] - combined_df['time'].iloc[0]
        # Replace the 'time' column with the time difference
        combined_df['time'] = time_since_first_row
    except:
        # Determine the number of rows in the dataframe
        num_rows = combined_df.shape[0]
        # Generate the sequence of values starting from 0, increasing by 0.1
        time_values = [i * 0.1 for i in range(num_rows)]
        # Assign the generated sequence to the 'time' column
        combined_df['time'] = time_values

    combined_df.loc[:, 'test_condition'] = test_id

    # Drop the NaN values, which represents the first n data points in the original dataframe.
    # combined_df.dropna(inplace=True)
    # Identify columns that end with '_label'
    label_columns = combined_df.columns[combined_df.columns.str.endswith('_label')]
    # Identify columns that do not end with '_label'
    non_label_columns = combined_df.columns.difference(label_columns)
    # Create a mask to keep rows without NaN in non-label columns
    mask = combined_df[non_label_columns].notna().all(axis=1)
    # Filter the dataframe using the mask
    combined_df = combined_df[mask]

    return combined_df


# Subfunction for create the sliding window.
def concatenate_features(df_input, y_input=None, X_window=[], y_window=[], window_size=1, sample_step=1, prediction_lead_time=1, mdl_type='clf'):
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
            if y_input is not None:
                new_features.extend(y_input.iloc[tmp_idx_pred].values.flatten().tolist())
    
    # Add the added features and the corresponding ys into X_window and y_window.
    X_window.append(new_features) # New features
    if y_input is not None:
        y_window.append(y_input.iloc[idx_last_element]) # Corresponding y
        return X_window, y_window
    else:
        return X_window


# Sliding the window to create features and response variables.
def prepare_sliding_window(df_x, y=None, sequence_name_list=None, window_size=1, sample_step=1, prediction_lead_time=1, mdl_type='clf'):
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

        # Check if we need to slide y:
        if y is not None: # If y is given: We are in the training mode.
            y_tmp = y[df_x['test_condition']==name]
            # Do a loop to concatenate features by sliding the window.
            for i in range(window_size, len(df_tmp)+1):
                X_window, y_window = concatenate_features(df_input=df_tmp.iloc[i-window_size:i, :], y_input=y_tmp.iloc[i-window_size:i], 
                    X_window=X_window, y_window=y_window, window_size=window_size, sample_step=sample_step, prediction_lead_time=prediction_lead_time, mdl_type=mdl_type)
        else: # If y is not given: We are in the testing mode.
            for i in range(1, window_size): # If not enough data in the window
                df_input = df_tmp.iloc[0:i, :]
                n_size = len(df_input)
                first_row = df_input.iloc[0]
                new_rows = pd.DataFrame([first_row] * (window_size-n_size), columns=df_input.columns)
                df_input = pd.concat([new_rows, df_input], ignore_index=True)

                X_window = concatenate_features(df_input=df_input, 
                    X_window=X_window, window_size=window_size, sample_step=sample_step, prediction_lead_time=prediction_lead_time, mdl_type=mdl_type)

            for i in range(window_size, len(df_tmp)+1):
                X_window = concatenate_features(df_input=df_tmp.iloc[i-window_size:i, :], 
                    X_window=X_window, window_size=window_size, sample_step=sample_step, prediction_lead_time=prediction_lead_time, mdl_type=mdl_type)
        
    # Transform into dataframe.
    X_window = pd.DataFrame(X_window)

    if y is not None:
        y_window = pd.Series(y_window)
        return X_window, y_window
    else:
        return X_window


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
    - prediction_lead_time: The number of time steps to predict into the future. Only valid for regression model. Default is 1.
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

    # # Plot the predicted and truth.
    # # Training data set.
    # fig_1 = plt.figure(figsize = (16,6))
    # ax = fig_1.add_subplot(1,2,1) 
    # ax.set_xlabel('index of data point', fontsize = 15)
    # ax.set_ylabel('y', fontsize = 15)
    # ax.set_title('Prediction V.S. the truth on the training dataset', fontsize = 20)
    # ax.plot(range(len(y_tr)), y_tr, 'xb', label='Truth')
    # ax.plot(range(len(y_pred_tr)), y_pred_tr, 'or', label='Prediction')
    # ax.legend()

    # # Testing data set.
    # ax = fig_1.add_subplot(1,2,2) 
    # ax.set_xlabel('index of data points', fontsize = 15)
    # ax.set_ylabel('y', fontsize = 15)
    # ax.set_title('Prediction V.S. the truth on the testing dataset', fontsize = 20)
    # ax.plot(range(len(y_test)), y_test, 'xb', label='Truth')
    # ax.plot(range(len(y_pred)), y_pred, 'or', label='Prediction')
    # ax.legend()
    
    # # Plot the residual errors.
    # # Training data set.
    # fig = plt.figure(figsize = (16,6))
    # ax = fig.add_subplot(1,2,1) 
    # ax.set_xlabel('Index of the data points', fontsize = 15)
    # ax.set_ylabel('Residual error', fontsize = 15)
    # ax.set_title('Residual errors on the training dataset', fontsize = 20)
    # ax.plot(y_pred_tr - y_tr, 'o')
    # ax.hlines([threshold, -1*threshold], 0, len(y_tr), linestyles='dashed', colors='r')

    # # Testing data set.
    # ax = fig.add_subplot(1,2,2) 
    # ax.set_xlabel('Index of the data points', fontsize = 15)
    # ax.set_ylabel('Residual error', fontsize = 15)
    # ax.set_title('Residual errors on the testing dataset', fontsize = 20)
    # ax.plot(y_pred-y_test, 'o')
    # ax.hlines([threshold, -1*threshold], 0, len(y_test), linestyles='dashed', colors='r')

    # # Plot the distribution of residual errors.
    # # Training data set.
    # fig = plt.figure(figsize = (16,6))
    # ax = fig.add_subplot(1,2,1) 
    # ax.set_xlabel('Residual error', fontsize = 15)
    # ax.set_ylabel('Counts', fontsize = 15)
    # ax.set_title('Distribution of residual errors (training)', fontsize = 20)
    # ax.hist(y_pred_tr - y_tr)
    # ax.axvline(x=threshold, linestyle='--', color='r')
    # ax.axvline(x=-1*threshold, linestyle='--', color='r')

    # # Testing data set.
    # ax = fig.add_subplot(1,2,2) 
    # ax.set_xlabel('Residual error', fontsize = 15)
    # ax.set_ylabel('Counts', fontsize = 15)
    # ax.set_title('Distribution of residual errors (testing)', fontsize = 20)
    # ax.hist(y_pred-y_test)
    # ax.axvline(x=threshold, linestyle='--', color='r')
    # ax.axvline(x=-1*threshold, linestyle='--', color='r')

    # # Performance indicators
    # # Show the model fitting performance.
    # print('\n New cv run:\n')
    # print('Training performance, max error is: ' + str(max_error(y_tr, y_pred_tr ) ))
    # print('Training performance, mean root square error is: ' + str(mean_squared_error(y_tr, y_pred_tr ,  squared=False)))
    # print(f'Training performance, residual error > {threshold}: ' + str(sum(abs(y_tr - y_pred_tr)>3)/y_tr.shape[0]*100) + '%')
    # print('\n')
    # print('Prediction performance, max error is: ' + str(max_error(y_test, y_pred)))
    # print('Prediction performance, mean root square error is: ' + str(mean_squared_error(y_test, y_pred, squared=False)))
    # print(f'Prediction performance, percentage of residual error > {threshold}：' + str(sum(abs(y_pred - y_test)>3)/y_test.shape[0]*100) + '%')

    # plt.show()

    fig_1 = plt.figure(figsize = (16,18))
    axes_tr = [fig_1.add_subplot(3, 2, 1), fig_1.add_subplot(3, 2, 3), fig_1.add_subplot(3, 2, 5)]
    axes_test = [fig_1.add_subplot(3, 2, 2), fig_1.add_subplot(3, 2, 4), fig_1.add_subplot(3, 2, 6)]

    show_reg_result_single_run(y_tr, y_pred_tr, axes_tr, 'training', threshold)
    show_reg_result_single_run(y_test, y_pred, axes_test, 'testing', threshold)

    plt.show()
    


def show_reg_result_single_run(y_true, y_pred, axes, suffix='training', threshold=3):
    # Plot the regression results.
    axes[0].set_xlabel('index of data point', fontsize = 15)
    axes[0].set_ylabel('y', fontsize = 15)
    axes[0].set_title(f'Regression results on the {suffix} dataset', fontsize = 20)
    axes[0].plot(range(len(y_true)), y_true, 'xb', label='Truth')
    axes[0].plot(range(len(y_pred)), y_pred, 'or', label='Prediction')
    axes[0].legend()

    # Residual errors.
    axes[1].set_xlabel('Index of the data points', fontsize = 15)
    axes[1].set_ylabel('Residual error', fontsize = 15)
    axes[1].set_title(f'Residual errors on the {suffix} dataset', fontsize = 20)
    axes[1].plot(y_pred - y_true, 'o')
    axes[1].hlines([threshold, -1*threshold], 0, len(y_true), linestyles='dashed', colors='r')

    # Plot the distribution of residual errors.
    axes[2].set_xlabel('Residual errors', fontsize = 15)
    axes[2].set_ylabel('Counts', fontsize = 15)
    axes[2].set_title(f'Distribution of residual errors ({suffix})', fontsize = 20)
    axes[2].hist(y_pred - y_true)
    axes[2].axvline(x=threshold, linestyle='--', color='r')
    axes[2].axvline(x=-1*threshold, linestyle='--', color='r')

    # Performance indicators
    # Show the model fitting performance.
    print('\n New run:\n')
    print(f'{suffix} performance, max error is: ' + str(max_error(y_true, y_pred) ))
    print(f'{suffix} performance, mean root square error is: ' + str(mean_squared_error(y_true, y_pred ,  squared=False)))
    print(f'{suffix} performance, residual error > {threshold}: ' + str(sum(abs(y_true - y_pred)>threshold)/y_true.shape[0]*100) + '%')


def show_clf_result(y_tr, y_test, y_pred_tr, y_pred):
    ''' ## Description
    This subfunction visualize the performance of the fitted model on both the training and testing dataset for the classfication model. 
    
    ## Parameters
    - y_tr: The training labels. 
    - y_test: The testing labels. 
    - y_pred_tr: The predicted labels on the training dataset. 
    - y_pred: The predicted labels on the testing dataset. 
    '''
    fig_1 = plt.figure(figsize = (16,12))
    ax_tr_true = fig_1.add_subplot(2,2,1)
    ax_tr_pred = fig_1.add_subplot(2,2,3)
    ax_test_true = fig_1.add_subplot(2,2,2)
    ax_test_pred = fig_1.add_subplot(2,2,4)

    show_clf_result_single_run(y_true=y_tr, y_pred=y_pred_tr, ax_tr=ax_tr_true, ax_pred=ax_tr_pred, suffix='training')
    show_clf_result_single_run(y_true=y_test, y_pred=y_pred, ax_tr=ax_test_true, ax_pred=ax_test_pred, suffix='testing')

    plt.show()


def show_clf_result_single_run(y_true, y_pred, ax_tr, ax_pred, suffix='training'):
    ''' ### Description
    This function plot the predictin results for a classifier, and print the performance metrics.    
    '''
    # Plots
    ax_tr.set_xlabel('index of data point', fontsize = 15)
    ax_tr.set_ylabel('y', fontsize = 15)
    ax_tr.set_title(f'{suffix}: Truth', fontsize = 20)
    ax_tr.plot(range(len(y_true)), y_true, 'xb', label='Truth')
    ax_tr.legend()

    ax_pred.set_xlabel('index of data points', fontsize = 15)
    ax_pred.set_ylabel('y', fontsize = 15)
    ax_pred.set_title(f'{suffix}: Prediction', fontsize = 20)
    ax_pred.plot(range(len(y_pred)), y_pred, 'ro', label='Prediction')
    ax_pred.legend()

    # Performance indicators
    # Show the model fitting performance.
    acc, pre, recall, f1 = cal_classification_perf(y_true, y_pred)
    print('\n New run:\n')
    print(f'{suffix} performance, accuracy is: ' + str(acc))
    print(f'{suffix} performance, precision is: ' + str(pre))
    print(f'{suffix} performance, recall: ' + str(recall))
    print(f'{suffix} performance, F1: ' + str(f1))
    print('\n')


if __name__ == '__main__':
    # Test the class FaultDetectReg

    import numpy as np
    import pandas as pd
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression


    def remove_outliers(df: pd.DataFrame):
        ''' # Description
        Remove outliers from the dataframe based on defined valid ranges. 
        Define a valid range of temperature and voltage. 
        Use ffil function to replace the invalid measurement with the previous value.
        '''
        df['temperature'] = df['temperature'].where(df['temperature'] <= 100, np.nan)
        df['temperature'] = df['temperature'].where(df['temperature'] >= 0, np.nan)
        df['temperature'] = df['temperature'].ffill()
        df['temperature'] = df['temperature'] - df['temperature'].iloc[0]

        df['voltage'] = df['voltage'].where(df['voltage'] >= 6000, np.nan)
        df['voltage'] = df['voltage'].where(df['voltage'] <= 9000, np.nan)
        df['voltage'] = df['voltage'].ffill()
        df['voltage'] = df['voltage'] - df['voltage'].iloc[0]

        df['position'] = df['position'].where(df['position'] >= 0, np.nan)
        df['position'] = df['position'].where(df['position'] <= 1000, np.nan)
        df['position'] = df['position'].ffill()
        df['position'] = df['position'] - df['position'].iloc[0]


    # Read data.
    base_dictionary = 'C:/Users/Zhiguo/OneDrive - CentraleSupelec/Code/Python/digital_twin_robot/projects/maintenance_industry_4_2024/dataset/training_data/'
    df_data = read_all_test_data_from_path(base_dictionary, remove_outliers, is_plot=False)

    # Pre-train the model.
    # Get all the normal data.
    normal_test_id = ['20240105_164214', 
        '20240105_165300', 
        '20240105_165972', 
        '20240320_152031', 
        '20240320_153841', 
        '20240320_155664', 
        '20240321_122650', 
        '20240325_135213', 
        '20240426_141190', 
        '20240426_141532', 
        '20240426_141602', 
        '20240426_141726', 
        '20240426_141938', 
        '20240426_141980', 
        '20240503_164435']
    
    df_tr = df_data[df_data['test_condition'].isin(normal_test_id)]

    feature_list_all = ['time', 'data_motor_1_position', 'data_motor_1_temperature', 'data_motor_1_voltage',
                    'data_motor_2_position', 'data_motor_2_temperature', 'data_motor_2_voltage',
                    'data_motor_3_position', 'data_motor_3_temperature', 'data_motor_3_voltage',
                    'data_motor_4_position', 'data_motor_4_temperature', 'data_motor_4_voltage',
                    'data_motor_5_position', 'data_motor_5_temperature', 'data_motor_5_voltage',
                    'data_motor_6_position', 'data_motor_6_temperature', 'data_motor_6_voltage']

    # Prepare feature and response of the training dataset.
    x_tr_org, y_temp_tr_org = extract_selected_feature(df_data=df_tr, feature_list=feature_list_all, motor_idx=6, mdl_type='reg')

    # Enrich the features based on the sliding window.
    window_size = 10
    sample_step = 1
    prediction_lead_time = 1
    threshold = .8
    abnormal_limit = 3

    x_tr, y_temp_tr = prepare_sliding_window(df_x=x_tr_org, y=y_temp_tr_org, window_size=window_size, sample_step=sample_step, prediction_lead_time=prediction_lead_time, mdl_type='reg')

    # Define the steps of the pipeline
    steps = [
        ('standardizer', StandardScaler()),  # Step 1: StandardScaler
        ('regressor', LinearRegression())    # Step 2: Linear Regression
    ]

    # Create the pipeline
    mdl_linear_regreession = Pipeline(steps)
    # Fit the model
    mdl = mdl_linear_regreession.fit(x_tr, y_temp_tr)

    # Test data.
    test_id = [
        '20240325_155003',
        '20240425_093699',
        # '20240425_094425',
        # '20240426_140055',
        # '20240503_163963',
        # '20240503_164675',
        # '20240503_165189'
    ]
    df_test = df_data[df_data['test_condition'].isin(test_id)]

    # Define the fault detector.
    detector_reg = FaultDetectReg(reg_mdl=mdl, threshold=threshold, abnormal_limit=abnormal_limit, window_size=window_size, sample_step=sample_step, pred_lead_time=prediction_lead_time)

    # Test
    _, y_label_test_org = extract_selected_feature(df_data=df_test, feature_list=feature_list_all, motor_idx=6, mdl_type='clf')
    x_test_org, y_temp_test_org = extract_selected_feature(df_data=df_test, feature_list=feature_list_all, motor_idx=6, mdl_type='reg')

    # Predict the temperature
    # y_label_pred_tr, y_temp_pred_tr = detector_reg.predict(df_x_test=x_tr_org, y_response_test=y_temp_tr_org)
    y_label_pred_tmp, y_temp_pred_tmp = detector_reg.predict(df_x_test=x_test_org, y_response_test=y_temp_test_org)

    # Get the true values.
    _, y_label_test = prepare_sliding_window(df_x=x_test_org, y=y_label_test_org, sequence_name_list=test_id, window_size=window_size, sample_step=sample_step, prediction_lead_time=prediction_lead_time, mdl_type='clf')
    _, y_temp_test_seq = prepare_sliding_window(df_x=x_test_org, y=y_temp_test_org, sequence_name_list=test_id, window_size=window_size, sample_step=sample_step, prediction_lead_time=prediction_lead_time, mdl_type='reg')

    # show_reg_result(y_tr=y_temp_tr, y_test=y_temp_test_seq, y_pred_tr=y_temp_pred_tr, y_pred=y_temp_pred_tmp, threshold=detector_reg.threshold)
    # show_clf_result(y_tr=np.zeros(len(y_label_pred_tr)), y_test=y_label_test, y_pred_tr=y_label_pred_tr, y_pred=y_label_pred_tmp)
    show_reg_result(y_tr=y_temp_test_seq, y_test=y_temp_test_seq, y_pred_tr=y_temp_pred_tmp, y_pred=y_temp_pred_tmp, threshold=detector_reg.threshold)
    show_clf_result(y_tr=y_label_test, y_test=y_label_test, y_pred_tr=y_label_pred_tmp, y_pred=y_label_pred_tmp)



    # # # Run cross validation
    # n_fold = 7
    # _, y_label_test_org = extract_selected_feature(df_data=df_test, feature_list=feature_list_all, motor_idx=6, mdl_type='clf')
    # x_test_org, y_temp_test_org = extract_selected_feature(df_data=df_test, feature_list=feature_list_all, motor_idx=6, mdl_type='reg')

    # motor_idx = 6
    # print(f'Model for motor {motor_idx}:')
    # # Run cross validation.
    # df_perf = detector_reg.run_cross_val(df_x=x_test_org, y_label=y_label_test_org, y_response=y_temp_test_org, 
    #                                      n_fold=n_fold)
    # print(df_perf)
    # print('\n')
    # # Print the mean performance and standard error.
    # print('Mean performance metric and standard error:')
    # for name, metric, error in zip(df_perf.columns, df_perf.mean(), df_perf.std()):
    #     print(f'{name}: {metric:.4f} +- {error:.4f}') 
    # print('\n')