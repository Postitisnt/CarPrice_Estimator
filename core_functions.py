import pandas as pd
import numpy as np
from numpy import isnan
import missingno as msno
import tensorflow as tf
from keras.layers import Input, Dense, concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras import initializers
from tensorflow import keras
from keras import callbacks
import matplotlib.pyplot as plt
import seaborn as sns
from numba import jit
import re
from sklearn import preprocessing

"""
Inside this .py you can find all the functions defined in order
to carry out the project.
"""

############################################################
################ Functions for DL models ###################
############################################################

def plot_training_history(model_history):
    """
    Plots the training and validation loss and metrics over epochs.

    Args:
    ----------
    model_history: tensorflow training record object
        History object returned by model.fit()
    """
    # Get the loss and metric values for the training and validation sets
    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']
    mse = model_history.history['mse']
    val_mse = model_history.history['val_mse']

    # Create a plot with two y-axes
    fig, ax1 = plt.subplots()

    # Plot the loss values
    ax1.plot(loss, label='training loss')
    ax1.plot(val_loss, label='validation loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper left')

    # Create a second y-axis and plot the metric values
    ax2 = ax1.twinx()
    ax2.plot(mse, label='training mse', color='orange')
    ax2.plot(val_mse, label='validation mse', color='green')
    ax2.set_ylabel('MSE')
    ax2.legend(loc='upper right')
    plt.show()


#########################################################################################
############################ Data Cleaning and Preprocessing ############################
#########################################################################################


"""
In data (Consume column) I have seen elements like the following:
- â‰ˆ 4,2 l / 100 km
- â‰ˆâ€‰4,0â€‰l/100km

The following function, in this snippet, is aimed at removing and cleaning these data
"""

def consume_numerizer(consume_str):
    """
    This function aims at preprocessing the strings inside
    the Consume column in order to transform them into float
    type objects. 
    
    NOTE: there are some unknown elements inside the cells
    of the Consume column, so the function is aimed also at removing
    these elements and at returning the float version of the consume.

    Args:
    ----------
    consume_str: float or str
        Cell value of the Consume column
    """
    if type(consume_str) == float or type(consume_str) == int:
        return float(consume_str)
    else:
        # The following line emoves whitespaces at the beginning and at the end,
        # replaces ',' with '.' (if possible)
        consume_str = consume_str.strip().replace(',', '.')
        try:
            # Try to turn the string into a float
            return float(consume_str)
        except:
            # If it is not possible, it means that there are some unknown elements, so in this part I remove them and maintain the relevant part (e.g. 4.0, or 5.2...)
            consume_str = consume_str.split('/')[0] # divide in two parts and take the one containing the consume
            to_remove_pattern = r"[^0-9.]" # Match with regex whatever is neither a number or a dot in order to remove these additional elements
            final_consume = re.sub(to_remove_pattern, '', consume_str)
            return float(final_consume)
        
#########################################################################################

@jit(nopython=True)
def check_consume_row(g,u,n):
    """
    This function aims at checking the values of the three columns
    and at returning the mode of the row. The mode is the type of
    consumption that is present in the row. The possible modes are:
    - GENERAL: returned when the only value in the row is the value in 
    the 'Consume' column, or when there are two values. In this second 
    scenario, the general 'Consume' has the priority (as stated at the
    beginning of the consume relative part);
    - URBAN: returned when the only value in the row is the value in 
    the 'ConsumeUrban' column;
    - NOT_URBAN: returned when the only value in the row is the value in 
    the 'ConsumeNotUrban' column;
    - MEAN: returned when there are only values for the two columns 
    'ConsumeUrban' and 'ConsumeNotUrban';
    - ADDITIONAL_CHECK: returned when both the three values are present.
    In this situation, some additional checks will be done by the 
    'compute_consume' function;
    - EMPTY_ROW: returned when no value is present in the row.

    Args:
    ----------
    g: Boolean value
        Boolean describinng whether in the 'Consume' column we have a non-null value (True) or a NaN (False)
    u: Boolean value
        Boolean describinng whether in the 'ConsumeFuel' column we have a non-null value (True) or a NaN (False)
    n: Boolean value
        Boolean describinng whether in the 'ConsumeNotUrban' column we have a non-null value (True) or a NaN (False)
    """
    if g==True and u==True and n==True:
        return 'ADDITIONAL_CHECK'
    elif g==True and u==True and n==False:
        return 'GENERAL'
    elif g==True and u==False and n==True:
        return 'GENERAL'
    elif g==False and u==True and n==True:
        return 'MEAN'
    elif g==True and u==False and n==False:
        return 'GENERAL'
    elif g==False and u==True and n==False:
        return 'URBAN'
    elif g==False and u==False and n==True:
        return 'NOT_URBAN'
    elif g==False and u==False and n==False:
        return 'EMPTY_ROW'

#########################################################################################

def compute_consume(values_list, mode):
    """
    This function aims at returning the value of the 'MeanConsume' column
    based on the mode of the row, previously extracted thanks to the 
    'check_consume_row' function. The value returned could be a single yet
    present value, or the mean of some values. 
    
    NOTE: if the mode is 'ADDITIONAL_CHECK', the function will check whether 
    the values in the 'Consume' and 'ConsumeUrban' columns are the same. 
    If so, it will return the value in the 'Consume' column. If not, it will 
    return the mean of the three values. This is done in order to avoid the 
    presence of two different values for the same car. 
    
    NOTE: if the mode is 'EMPTY_ROW', the function will return a NaN value.

    Args:
    ----------
    values_list: list 
        This contains any value from the three consume-related columns. Could contain only one value, until a maximum of 3.
    mode: str
        Mode of the row
    """ 
    if mode=='GENERAL' or mode=='URBAN' or mode=='NOT_URBAN':
        return values_list[0]
    elif mode=='MEAN':
        return np.mean(values_list)
    elif mode=='ADDITIONAL_CHECK':
        if values_list[0] == values_list[1]:
            return values_list[0]
        elif values_list[0] == values_list[2]:
            return values_list[0]
        else:
            return np.mean(values_list)
        
#########################################################################################

@jit(nopython=True)
def check_emission_class(emission_class, matriculation_year):
    """
    This function is a shorthand to check whether the emission class is missing
    or not. If it is missing, it returns the emission class based on the matriculation
    year. If it is not missing, it raises an exception. 

    Args:
    ----------
    emission_class: cell value
        Represents the emission class of the car or a NaN
    matriculation_year: int
        Matriculation year of the car
    """
    if isnan(emission_class):
        if matriculation_year < 1993:
            return 'Euro 0'
        elif matriculation_year < 1997:
            return 'Euro 1'
        elif matriculation_year < 2001:
            return 'Euro 2'
        elif matriculation_year < 2006:
            return 'Euro 3'
        elif matriculation_year < 2011:
            return 'Euro 4'
        elif matriculation_year < 2015:
            return 'Euro 5'
        elif matriculation_year >= 2015:
            return 'Euro 6'
    else:
        raise Exception("Unknown type")
    
#########################################################################################

def outliers_deletion(df, feature, drop_NaN=True, threshold=3):
    """
    This function aims at deleting the outliers of a given feature.
    The outliers are deleted by computing the IQR and the upper and
    lower bounds. The outliers are then removed from the dataframe.

    NOTE: If drop_NaN is set as False, the function has the aim to 
    maintain null values during the outliers deletion thanks to check
    inside the for-loop. The same exact check is useless in the case
    in which NaN have been previously deleted with the argument 
    drop_NaN=True.

    Args:
    ----------
    df: dataframe
        The dataframe from which we want to remove outliers
    feature: str
        Feature column to be cleaned from outliers
    drop_NaN: boolean value 
        Indicates whether to drop or not the NaN values inside the input dataframe
    threshold: int
        Threshold to be used to compute the upper and lower bounds
    """
    
    no_nan_df = df.copy(deep=False).dropna(subset=[feature])
    deep_df = df.copy(deep=False)
    check_missing_values(df, visualization=False)
    
    # Compute the IQR
    Q1 = np.percentile(no_nan_df[feature], 25,
                    interpolation = 'midpoint')
    
    Q3 = np.percentile(no_nan_df[feature], 75,
                    interpolation = 'midpoint')
    IQR = Q3 - Q1
    
    # Printing the old shape 
    print("Old Shape: ", df.shape)
    sns.boxplot(df[feature])
    plt.show()

    # Compute the Upper bound
    upper=Q3+threshold*IQR
    # Compute the Lower bound
    lower=Q1-threshold*IQR

    # Remove outliers
    for idx, row in deep_df.iterrows():
        if pd.notnull(row[feature]):
            if row[feature] > upper or row[feature] < lower:
                deep_df.drop(idx, inplace=True)

    if drop_NaN:
        deep_df.dropna(subset=[feature], inplace=True)
    
    outlier_indices = df.index.difference(deep_df.index)
    df.drop(index=outlier_indices, inplace=True)

    # Printing the new shape 
    print("New Shape: ", df.shape)
    sns.boxplot(df[feature])
    plt.show()

    check_missing_values(df, visualization=False)
    check_lost_data(df)
    
#########################################################################################
###################### Check status and general controls functions ######################
#########################################################################################

def features(df):
    """
    This function is a shorthand to print the number of features, 
    as well as the titles of all the columns in the dataframe at specific
    time. Inside the code is used just to check whether additions or deletions
    have been successfully accomplished.

    Args:
    ----------
    df: dataframe
        The dataframe of which we want to check the features
    """
    print("The dataframe now contains {} columns.\n The header is: {}.".format(len(df.columns), df.columns))

#########################################################################################

def check_missing_values(df, visualization=False):
    """
    This function is a shorthand to print the percentage of missing values
    for each column, as well as the matrix and barplot showing the missing
    values for each column. It is used to check whether the missing values
    have been successfully filled or dropped.

    Args:
    ----------
    df: dataframe
        The dataframe of which we want to check the missing values
    visualization: boolean value
        If True, the matrix and barplot are shown.
    """
    missing_percentages = df.isna().mean() * 100
    print("The percentage of missing values is the following: \n\n", missing_percentages.round(2), '\n\n')
    if visualization:
        print("This is the matrix showing missing values for each column: \n")
        msno.matrix(df)
        plt.show()
        print("This is the barplot showing the amount of filled values for each column: \n")
        msno.bar(df)
        plt.show()

#########################################################################################

def plot_correlation(df, log_scale=False):
    """
    This function is a shorthand to plot the correlation matrix of the dataframe.
    It is used to check whether the features are correlated with each other.
    
    NOTE: The log_scale parameter does not influence the original dataframe, it creates
    a copy of it and applies the logarithm to the 'price' column, so that the correlation
    matrix is plotted with the logarithm of the 'price' column. The default value of
    the log_scale parameter is False.

    Args:
    ----------
    df: dataframe
        The dataframe of which we want to check the plot the correlation matrix
    log_scale: boolean value 
        Specifies whether you want to see the log-scaled correlation matrix or not
    """
    if log_scale:
        copy = df.copy()
        copy['price']=np.log(copy['price'])
        dataframe = copy
    else:
        dataframe = df
    correlation_matrix = dataframe.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.show()

#########################################################################################

def check_lost_data(df):
    """
    This function is a shorthand to print the percentage of lost data
    after the cleaning process. It is used to check whether the cleaning
    process has been successful. It is based on a fixed number, which is
    the initial dataset length in my specific case (178248). 
    
    NOTE: to re-use this function, just change the fixed number with a 
    variable containing your initial dataset length.

    Args:
    ----------
    df: dataframe
        The dataframe of which we want to check the lost data
    """

    print('We lost the {}% of the initial observations.'.format(round((178248 - len(df.index)) / 178248 * 100, 2)))