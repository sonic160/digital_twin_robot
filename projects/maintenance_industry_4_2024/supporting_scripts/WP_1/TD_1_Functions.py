from utility import read_all_test_data_from_path
from utility import read_all_csvs_one_test
import pandas as pd
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,confusion_matrix, roc_curve, auc
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

def plot_pca(X, data):
    # PCA 
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    #PCA with normalization
    X_train_normalized = MinMaxScaler().fit_transform(X)
    pca_normalized = PCA(n_components=2)
    X_pca_normalized = pca_normalized.fit_transform(X_train_normalized)

    # PCA with standardization
    X_train_standard = StandardScaler().fit_transform(X)
    pca_standard = PCA(n_components=2)
    X_pca_standard = pca_standard.fit_transform(X_train_standard)

    fig, axs = plt.subplots(6, 3, figsize=(18, 24))

    for i in range(1, 7):    
        # PCA 
        y = data[f'data_motor_{i}_label']  

        axs[i-1, 0].scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], color='red', marker='^', alpha=.5, label='Class 0')
        axs[i-1, 0].scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], color='blue', marker='o', alpha=.5, label='Class 1')
        axs[i-1, 0].set_title(f'PCA Result for label {i}')
        axs[i-1, 0].set_xlabel('Principal Component 1')
        axs[i-1, 0].set_ylabel('Principal Component 2')
        axs[i-1, 0].legend()
    
        #PCA with normalization

        axs[i-1, 1].scatter(X_pca_normalized[y == 0, 0], X_pca_normalized[y == 0, 1], color='red', marker='^', alpha=.5, label='Class 0')
        axs[i-1, 1].scatter(X_pca_normalized[y == 1, 0], X_pca_normalized[y == 1, 1], color='blue', marker='o', alpha=.5, label='Class 1')
        axs[i-1, 1].set_title(f'PCA Result with Normalization for label {i}')
        axs[i-1, 1].set_xlabel('Principal Component 1')
        axs[i-1, 1].set_ylabel('Principal Component 2')
        axs[i-1, 1].legend()

        # PCA with standardization

        axs[i-1, 2].scatter(X_pca_standard[y == 0, 0], X_pca_standard[y == 0, 1], color='red', marker='^', alpha=.5, label='Class 0')
        axs[i-1, 2].scatter(X_pca_standard[y == 1, 0], X_pca_standard[y == 1, 1], color='blue', marker='o', alpha=.5, label='Class 1')
        axs[i-1, 2].set_title(f'PCA Result with Standardization for label {i}')
        axs[i-1, 2].set_xlabel('Principal Component 1')
        axs[i-1, 2].set_ylabel('Principal Component 2')
        axs[i-1, 2].legend()

    plt.tight_layout()
    plt.show()
    

def smooth_data_moving_average(data, window_size):
    """Smooth data by computing moving average."""
    return data.rolling(window=window_size, min_periods=1).mean()

def remove_outliers(data, label_columns, alpha):
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

def plot_violin(label, label_columns, smoothed_data_normalized):
    #first 9 features and label 1
    data_violin = smoothed_data_normalized.drop(columns=label_columns)
    data_violin = pd.concat([data_violin.iloc[:,0:9], smoothed_data_normalized[f'data_motor_{label}_label']], axis=1)
    data_violin = pd.melt(data_violin, id_vars=f'data_motor_{label}_label', var_name="features", value_name='value')

    plt.figure(figsize=(20,20))
    plt.subplot(211)
    sns.violinplot(x="features", y="value", hue=f'data_motor_{label}_label', data=data_violin, split=True, inner="quart")
    plt.xticks(rotation=90)

    #Last 9 features and label 1
    data_violin = smoothed_data_normalized.drop(columns=label_columns)
    data_violin = pd.concat([data_violin.iloc[:,9:17], smoothed_data_normalized[f'data_motor_{label}_label']], axis=1)
    data_violin = pd.melt(data_violin, id_vars=f'data_motor_{label}_label', var_name="features", value_name='value')

    plt.subplot(212)
    sns.violinplot(x="features", y="value", hue=f'data_motor_{label}_label', data=data_violin, split=True, inner="quart")
    plt.xticks(rotation=90)

    plt.tight_layout()  
    plt.show()


def remove_outliers_2(data, label_columns, alpha):
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