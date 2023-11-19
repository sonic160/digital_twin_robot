import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def combine_csv(folder_path, seq_idx, label):
    """
    Combine all CSV files in a folder into a single DataFrame.
    :param folder_path: Path to the folder containing the CSV files
    :param seq_idx: Sequence index
    :param label: Label of the sequence (Normal - 0, Abnormal - 1)
    :return: A single DataFrame containing all the data from the CSV files
    """

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

        # Extract the file name (excluding the extension) to use as a prefix
        file_name = os.path.splitext(file)[0]

        # Add a prefix to each column based on the file name
        df = df.add_prefix(f'{file_name}_')

        # Concatenate the current DataFrame with the combined DataFrame
        combined_df = pd.concat([combined_df, df], axis=1)

    df = pd.read_csv(file_path)
    combined_df = pd.concat([df['time'], combined_df], axis=1)
    combined_df.loc[:, 'sequence_idx'] = seq_idx
    combined_df.loc[:, 'label'] = label

    return combined_df


def combine_all_csv(path, label, seq_idx=0):
    """
    Combine all CSV files in a folder into a single DataFrame.
    :param path: Path to the folder containing the CSV files
    :param label: Label of the sequence (Normal - 0, Abnormal - 1)
    :return: A single DataFrame containing all the data from the CSV files
    """
    # Get all directories in the given path
    folders = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    df = pd.DataFrame()
    for folder in folders:
        folder_path = path+'\\'+folder
        tmp_df = combine_csv(folder_path, seq_idx, label)
        seq_idx += 1
        df = pd.concat([df, tmp_df])
        df = df.reset_index(drop=True)

    return  df


def read_data(path_normal, path_failure):
    df = pd.concat([combine_all_csv(path_normal, 0), combine_all_csv(path_failure, 1, seq_idx=4)]).reset_index(drop=True)

    return df


if __name__ == '__main__':
    # Root path to the folder containing the CSV files
    path_normal = r'Data collection_20231109\Normal sequence'
    path_failure = r'Data collection_20231109\Failure sequence'
    df = read_data(path_normal, path_failure)

    # Separate the features (X) and the target variable (y)
    X = df.drop(['label', 'sequence_idx', 'time'], axis=1)
    y = df['label']
    # Normalize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform PCA to reduce the dimensionality to 2 dimensions
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], color='red', marker='^', label='Class 0')
    plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], color='blue', marker='o', label='Class 1')
    plt.title('2D PCA Result')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.show()