
import pandas as pd 
import numpy as np 



def load_data (path) :
    '''
    This function loads the data f

    it loads both train 

    '''
    df = pd.read_csv(path)
    return df



def load_data_train () :
    '''
    This function loads the train data and outputs two dataframes 

    df_x : the values of the train data
    df_y : the labels of the train data
    '''
    
    return load_data("../../data/raw/train_values.csv") , load_data('../../data/raw/train_labels.csv')


def drop_row ( id_list, df_X , df_Y) :
    '''
    This function drops the rows based on the building id in both dataframes

    since the labels and values are stored in two different dataframes, 
    we need to be able to apply changes that we apply on  the rows in values data frame , also to the labels data frame


    '''
    df_X = df_X[~df_X.building_id.isin(id_list)]
    df_Y = df_Y[~df_Y.building_id.isin(id_list)]
    return df_X, df_Y

def drop_duplicates (df_X, df_Y) :
    '''
    This function drops the duplicates in both dataframes

    duplicates are calculated based on the df_X , 
    then the list of duplicated values should be used to drop the rows in the df_Y and df_X,
    we will use the drop_row function to achieve this . 

    first we need to chack for the duplicates in df_X, 
    then extract a list of building_id of the duplicates,

    '''

    df_no_id = df_X.drop('building_id', axis=1)
    ids = df_X[df_no_id.duplicated()].building_id.tolist()
    clean_x, clean_y = drop_row(ids, df_X, df_Y)
    return clean_x, clean_y
    



