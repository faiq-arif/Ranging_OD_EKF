import pandas as pd
import numpy as np
from datetime import datetime
from astropy import units as u
from astropy.time import Time

# Convert 'Date' and 'Time' to datetime and drop original columns
def process_dataframe(df, epoch):
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df.drop(columns=['Date', 'Time'])

    # Calculate the time difference between the initial date time and the first row in seconds
    initial_time_difference = (Time(df['DateTime'].iloc[0]) - epoch).sec
    
    # Calculate the time difference in seconds
    df['Time Elapsed'] = (Time(df['DateTime']) - epoch).sec
    
    # Convert Range in km to m
    df['Value'] = df['Value'].multiply(1000)
    
    return df
    
# Detect Median Bias and Subtract it from Measurements
def subtract_median(df, column_name):
    """
    This function calculates the median of a specified column in a pandas DataFrame
    and subtracts the median from each value in that column.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column_name (str): The name of the column to process.
    
    Returns:
    pd.DataFrame: A DataFrame with the updated column.
    """
    median_value = df[column_name].median()  # Calculate the median of the column
    print(f"Bias '{column_name}': {median_value} m")
    df[column_name] = df[column_name] - median_value  # Subtract the median from each value in the column

def subtract_mean(df, column_name):
    """
    This function calculates the median of a specified column in a pandas DataFrame
    and subtracts the median from each value in that column.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column_name (str): The name of the column to process.
    
    Returns:
    pd.DataFrame: A DataFrame with the updated column.
    """
    mean_value = df[column_name].mean()  # Calculate the mean of the column
    print(f"Bias '{column_name}': {mean_value} m")
    df[column_name] = df[column_name] - mean_value  # Subtract the mean from each value in the column 

# @title (FUNCTION compute_velocity) Finite Difference to Compute Velocity from Measurements

def compute_velocity(df, position_column):
    """
    Computes the velocity from position data using the central finite difference method,
    taking into account variable time steps from the 'DateTime' column.

    Parameters:
    - df: DataFrame containing the position and 'DateTime' data.
    - position_column: String, the name of the column containing the position data.

    Returns:
    - The velocity values as a numpy array.
    """
    # Ensure the DateTime column is in pandas datetime format
    df['DateTime'] = pd.to_datetime(df['DateTime'])

    # Calculate time differences in seconds
    time_diffs = df['DateTime'].diff().dt.total_seconds().bfill().values

    # Extract the position data
    positions = df[position_column].values

    # Compute velocities using central finite differences
    velocities = np.zeros_like(positions, dtype=float)

    # Use central difference for interior points
    for i in range(1, len(df) - 1):
        velocities[i] = (positions[i + 1] - positions[i - 1]) / (time_diffs[i] + time_diffs[i - 1])

    # Use forward difference for the first point
    velocities[0] = (positions[1] - positions[0]) / time_diffs[0]

    # Use backward difference for the last point
    velocities[-1] = (positions[-1] - positions[-2]) / time_diffs[-1]

    # Return the velocities
    return velocities