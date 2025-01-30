from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns   
import numpy as np


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