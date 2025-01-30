from sklearn.metrics import confusion_matrix
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns   
import numpy as np
import pandas as pd



def get_hist(df, hist_x_row=4, figsize=(20,40)):
    """
    diagnostic tool to inspect the distribution of the data.
    It returns a histogram of each column in the dataframe.
    """
    n_cols = np.ceil(len(df.columns)/hist_x_row).astype(int)
    plt.figure(figsize=figsize)
    for i, col in enumerate(df.columns):
        plt.subplot(n_cols, hist_x_row, i+1)
        df[col].hist()
        plt.title(col + ' - ' + str(df[col].dtype)) 

    plt.subplots_adjust(hspace=0.5, wspace=0.3)


def convert_to_object(df):
    """
    It converts the binary variables into objects. This should be run after data loading
    """
    for col in df.columns:
        if len(df[col].unique()) == 2:
            df[col] = df[col].astype('object')
    return df 


def print_confusion_matrix(pred, true):
    cm = confusion_matrix(true, pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot the confusion matrix using a heatmap
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", 
                xticklabels=['Class 0', 'Class 1', 'Class 2'],
                yticklabels=['Class 0', 'Class 1', 'Class 2']) 

    plt.xlabel('Predicted labels')
    plt.ylabel('True labels') 
    plt.title('Confusion Matrix')


def upsample_data(df_X, df_y):
    """
    Upsamples the minority class (class 1) to match the size of the majority class (class 3).
    """
    df_all = pd.concat([df_X, df_y['damage_grade']], axis=1)
    df_1 = df_all[df_all['damage_grade'] == 1]
    df_rest = df_all[df_all['damage_grade'] != 1]
    size_to_match = int(len(df_1) * 2) #df_rest[df_rest['damage_grade'] == 3].shape[0] 

    print(f"Before upsampling - Class 1 size: {df_1.shape[0]}, Class 3 size: {df_rest[df_rest['damage_grade'] == 3].shape[0]}")

    df_1_upsampled = resample(df_1, random_state=42, n_samples=size_to_match, replace=True)
    df_all_upsampled = pd.concat([df_1_upsampled, df_rest]).reset_index(drop=True)

    print(f"After upsampling - New Class 1 size: {df_1_upsampled.shape[0]}, New dataset size: {df_all_upsampled.shape[0]}")

    df_X = df_all_upsampled.drop(columns=['damage_grade'])
    df_y = df_all_upsampled[['building_id', 'damage_grade']]
    
    return df_X, df_y


def resample_data(df_X, df_y, resample_type='upsample'):
    """
    Resamples data based on the chosen type: 'upsample' or 'both'.
    """
    if resample_type == 'upsample':
        return upsample_data(df_X, df_y)
    elif resample_type == 'both':
        return both_resample_data(df_X, df_y)
    else:
        raise ValueError("Choose 'upsample', or 'both'.")


def both_resample_data(df_X, df_y):
    """
    Performs both upsampling for class 1 and downsampling for class 2 to match the size of class 3.
    """
    df_all = pd.concat([df_X, df_y['damage_grade']], axis=1)
    df_1 = df_all[df_all['damage_grade'] == 1]
    df_3 = df_all[df_all['damage_grade'] == 3]
    df_rest = df_all[df_all['damage_grade'] == 2]
    size_to_match_1 = int(len(df_1) * 2)
    size_to_match_3 = int(len(df_3) * 1.2) 

    print(f"Before resampling - Class 1 size: {df_1.shape[0]}, Class 2 size: {df_3.shape[0]}, Class 3 size: {df_rest.shape[0]}")

    df_1_upsampled = resample(df_1, random_state=42, n_samples=size_to_match_1, replace=True)
    df_3_upsampled = resample(df_3, random_state=42, n_samples=size_to_match_3, replace=True)
    
    df_all_resampled = pd.concat([df_1_upsampled, df_3_upsampled, df_rest]).reset_index(drop=True)

    print(f"After resampling - New Class 1 size: {df_1_upsampled.shape[0]}, New Class 3 size: {df_3_upsampled.shape[0]}, New dataset size: {df_all_resampled.shape[0]}")

    df_X = df_all_resampled.drop(columns=['damage_grade'])
    df_y = df_all_resampled[['building_id', 'damage_grade']]

    return df_X, df_y
