import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape
from sklearn.metrics import max_error
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from pathlib import Path
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import logging
import sys
import os
from sklearn.model_selection import KFold

''' This subfuction performs k-fold cross validation. '''
def run_cross_val(n_fold, X, y, reg_mdl):
    # Define the cross validator.
    kf = KFold(n_splits=n_fold)
    kf.get_n_splits(X)

    # Do the cross validation.
    perf = np.zeros((n_fold, 3))
    counter = 0
    for train_index, test_index in kf.split(X):
        # Get training and testing data.
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Fitting and prediction.
        reg_mdl, _, y_pred = run_reg_mdl(reg_mdl, X_train, y_train, X_test, y_test, is_cv=True)

        # Calculate the performance indicators.
        perf[counter, :] = np.array([max_error(y_test, y_pred), 
        mean_squared_error(y_test, y_pred, squared=False), 
        sum(abs(y_pred - y_test)>3)/y_test.shape])

        counter += 1

    return pd.DataFrame(data=perf, columns=['Max error', 'RMSE', 'Exceed boundary rate'])
    

''' This subfunction fits different regression models, and test the performance in both training and testing dataset. '''
def run_reg_mdl(reg_mdl, X_tr, y_tr, X_test, y_test, is_cv = False):  
    # Training the regression model.    
    reg_mdl = reg_mdl.fit(X_tr, y_tr)

    # Prediction
    y_pred_tr = reg_mdl.predict(X_tr)
    y_pred = reg_mdl.predict(X_test)
    # Transform back
    # y_pred_tr = scaler_y.inverse_transform(y_pred_tr)
    # y_pred = scaler_y.inverse_transform(y_pred)
    # y_tr = scaler_y.inverse_transform(y_tr)

    if not is_cv:
        model_pef(y_tr, y_test, y_pred_tr, y_pred)

    return reg_mdl, y_pred_tr, y_pred


''' This subfunction checks the performance of the fitted model on both the training and testing dataset. '''
def model_pef(y_tr, y_test, y_pred_tr, y_pred):
    # Plot the predicted and truth.
    # Training data set.
    fig_1 = plt.figure(figsize = (16,6))
    ax = fig_1.add_subplot(1,2,1) 
    ax.set_xlabel('True', fontsize = 15)
    ax.set_ylabel('Prediction', fontsize = 15)
    ax.set_title('Prediction V.S. the truth on the training dataset', fontsize = 20)
    ax.plot(y_tr[np.abs(y_tr-y_pred_tr) > 3], y_pred_tr[np.abs(y_tr-y_pred_tr) > 3], 'xr')
    ax.plot(y_tr[np.abs(y_tr-y_pred_tr) <= 3], y_pred_tr[np.abs(y_tr-y_pred_tr) <= 3], 'ob')

    # Testing data set.
    ax = fig_1.add_subplot(1,2,2) 
    ax.set_xlabel('True', fontsize = 15)
    ax.set_ylabel('Prediction', fontsize = 15)
    ax.set_title('Prediction V.S. the truth on the testing dataset', fontsize = 20)
    ax.plot(y_test[np.abs(y_test-y_pred) > 3], y_pred[np.abs(y_test-y_pred) > 3], 'xr')
    ax.plot(y_test[np.abs(y_test-y_pred) <= 3], y_pred[np.abs(y_test-y_pred) <= 3], 'ob')
    
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
    print('Training performance, residual error > 3psi (%): ' + str(sum(abs(y_tr - y_pred_tr)>3)/y_tr.shape[0]*100) + '%')

    print('Prediction performance, max error is: ' + str(max_error(y_pred, y_test)))
    print('Prediction performance, mean root square error is: ' + str(mean_squared_error(y_pred, y_test, squared=False)))
    print('Prediction performance, residual error > 3psi: (%) ' + str(sum(abs(y_pred - y_test)>3)/y_test.shape[0]*100) + '%')

    return


''' This subfunction transform from the original dataframe to X and y. '''
def seperate_X_y(df_engine, features):
    X = df_engine.loc[:, features].values
    y = df_engine.loc[:, 'EOP'].values

    # Standardization.
    scalar_X = StandardScaler()
    # scaler_X_engine_2 = StandardScaler()
    X = scalar_X.fit_transform(X)
    
    return X, y, scalar_X


''' This subfuction reads and preprocess the orginal data. '''
def data_preprocess(dataDirectory, df_nomenclature, save_name):    
    # we get the list of files for the aircraft
    dataPath = Path(dataDirectory)
    list_files = list(dataPath.glob('**/*.gz'))
    
    # Extract the flights in the dataset.
    l_extract = [extractFrom1Flight(file, df_nomenclature) for file in list_files]
    df_extract = pd.concat(l_extract, axis=0)
    # good idea to save the result to avoir reprocess
    df_extract.to_csv(Path(dataDirectory, save_name))
    
    print('the end')

    return df_extract



''' This subfuction extract data for a single flight. '''
def extractFrom1Flight(file_name, df_nomenclature):
    # get num of flight and num of day
    nameSplitted = str(file_name).split('_')
    flightNum = int(nameSplitted[2])
    dayNum = int(nameSplitted[3][1:5])


    # create log here if parallelized (each process has its own logger)
    # create logger with 'LEAP_appairage'
    logger = logging.getLogger('ST4_groupx')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages

    if not logger.hasHandlers():
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        # ch.setLevel(logging.ERROR)
        ch.setLevel(logging.DEBUG)
        fh = logging.FileHandler('ST4.log')
        fh.setLevel(logging.INFO)

        # create formatter and add it to the handlers
        formatter = logging.Formatter(
            '%(asctime)s - %(process)d - %(name)s - %(funcName)s - %(lineno)d - %(levelname)s - %(message)s')

        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        # add the handlers to the logger
        logger.addHandler(ch)
        logger.addHandler(fh)

    # put treatment in try bloc to avoid interrupting treatment on unexpected dirty data
    df_extract = pd.DataFrame()
    try:
        logger.debug(f'commence le traitement du vol {flightNum}')  # f-string works only with python >=3.4
        # we read one flight (later, we will iterate, but we start simple
        df_flight = pd.read_csv(file_name, compression='gzip', index_col=0)

        # We will first perform the range check clean step because it will add Nans in the dataset
        # we start reading the nomenclature file for min;max values

        # the iterate over each column
        # here I don't use apply function because only a few values are supposed to be changed and there are a few columns
        for col_name in df_flight.columns[1:]:
            signal = col_name.split('_')[0]
            _min = df_nomenclature.at[signal, 'min value']
            _max = df_nomenclature.at[signal, 'max value']
            # None is prefered instead of np.nan in order to not change the columns data type (forced to float with np.nan)
            df_flight.loc[~df_flight.loc[:,col_name].between(_min, _max), col_name] = None

            # we create data in the rows EOP is not nan
            # simplest way here because we have low frequency on context data and no signal processing will be performed
            # otherwise some filtering or interpolation would have been necessary
            if signal is not 'EOP_1':
                df_flight[col_name].ffill(inplace=True)

        # we decimate context data were EOP is nan
        df_flight.dropna(inplace=True)

        # we treat both engines separatly
        l_extracted = []  # temporary storage for extracted data, one dataframe per engine
        for engPos in [1, 2]:
            eng_x_cols = [x + '_' + str(engPos) for x in df_nomenclature.index[0:-3]]

            # create names without eng indication for later uniform treatment between eng1 and eng2
            cols = [x for x in df_nomenclature.index[0:-3]] + ['P0', 'TAT']

            df_flight_engx = df_flight.loc[:, eng_x_cols + ['P0_' + str(engPos), 'TAT']]

            # rename cols to allow uniform treatment
            rename_dic = {x + '_' + str(engPos): x for x in df_nomenclature.index[0:-3]}
            rename_dic['P0_' + str(engPos)] = 'P0'
            df_flight_engx.rename(columns=rename_dic, inplace=True)

            # at this stage, we can notice that T49 data is still not clean at the beginning and end of the clean despite it is in the range
            # this is due to recorder limitations when engine is not turning
            # since N1 is null during this dirty passage, we can force the value during this time
            # it is a risk to delete rows because both engines are not started simultaneously
            default_value_1 = df_flight_engx.loc[df_flight_engx.loc[:, 'N1'] >= 10. , ['T49', 'EOT']].iloc[0]
            df_flight_engx.loc[df_flight_engx.loc[:, 'N1']<10. , 'T49'] = default_value_1['T49']
            df_flight_engx.loc[df_flight_engx.loc[:, 'N1']<10. , 'EOT'] = default_value_1['EOT']

            # then reduce the data to a flight phase
            # for cruise, lets take minimum ambiant temperature or minimum ambiant pressure
            # then define a range of N2 close to this domain
            min_TAT = df_flight_engx.loc[:, 'TAT'].min()
            min_TAT_idx = df_flight_engx.loc[:, 'TAT'].idxmin()
            ref_N2 = df_flight_engx.loc[min_TAT_idx, 'N2']
            # 2 and 10 are arbitrary here, and can be improved
            df_flight_engx_cruise = df_flight_engx.loc[df_flight_engx.loc[:, 'N2'].between(ref_N2-2, ref_N2+2) \
                                                     & df_flight_engx.loc[:, 'TAT'].between(min_TAT, min_TAT+10), :].copy()
            # now EOP and EOT relatively constant, scatter_matrix becomes interesting only over several flights segments
            # quantification of recording is visible, some treatment could be done before (smoothing, dequantification, ...)
            # for instance, we can work only with value at instants the EOP change of value:
            df_flight_engx_cruise['dEOP'] = df_flight_engx_cruise['EOP'].diff()
            df_flight_engx_cruise_lt = df_flight_engx_cruise.loc[df_flight_engx_cruise.loc[:, 'dEOP'] != 0, :].copy()
            df_flight_engx_cruise_lt.loc[:, 'tempDeot'] = np.abs(df_flight_engx_cruise_lt.loc[:, 'EOT'] - df_flight_engx_cruise['EOT'].median())
            df_flight_engx_cruise_lt.index = pd.to_datetime(df_flight_engx_cruise_lt.index) + pd.DateOffset(days=dayNum)

            idx_medEOT = df_flight_engx_cruise_lt.loc[:, 'tempDeot'].idxmin()
            # to facilitate potential parallel processing, eng position is added in a column for selection purpose
            # and extracted data for both engnes will be stored in the same dataframe
            _df_extract = df_flight_engx_cruise_lt.loc[idx_medEOT, :]
            _df_extract['flight_ID'] = flightNum
            _df_extract['engine position'] = engPos
            _df_extract.drop(labels=['tempDeot', 'dEOP'], inplace=True)
            l_extracted.append(_df_extract)
        df_extract = pd.concat(l_extracted, axis=1).T
    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        logger.exception(f"exception {exc_type.__name__} lors du vol {flightNum}")   # f-string works only with python >=3.4
    return df_extract


if __name__ == '__main__':
    # Path of the traning data.
    dataDirectory = r'C:\Users\Zhiguo\OneDrive - CentraleSupelec\Paris\Teaching\Risk\2021_ST4\Challenge week\Data challenge\Dataset\Testing data'
    df_nomenclature = pd.read_excel(Path(dataDirectory, 'nomenclature.xlsx'), index_col=0)
    # Read also the nomenclature.
    # Saved file name of the data after preprocessing.
    save_name = 'WP_testing.csv'

    df_test = data_preprocess(dataDirectory, df_nomenclature, save_name)