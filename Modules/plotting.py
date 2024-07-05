import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

def format_yaxis(value, tick_number):
    return f'{value / 1000:.1f}'

def plot_column(df, title, column_name, y_label, unit):
    """
    Plots a specified column of a DataFrame with 'Time Elapsed' as the x-axis.
    
    Parameters:
    df (DataFrame): The DataFrame containing the data.
    title (str): The title of the plot.
    column_name (str): The name of the column to plot.
    y_label (str): The label for the y-axis.
    unit (str): The unit for the y-axis label.
    """
    plt.figure(figsize=(14, 8))  # Adjust the size for horizontal space
    plt.plot(df['Time Elapsed'], df[column_name], marker='x', linestyle='-')
    plt.title(title)
    plt.xlabel('Time Elapsed (seconds)')
    plt.ylabel(f'{y_label} ({unit})')
    plt.grid(True)
    
    # Format the y-axis values
    formatter = FuncFormatter(format_yaxis)
    plt.gca().yaxis.set_major_formatter(formatter)
    
    plt.show()

def plot_ekf_results(mu, Sigma, times):
    T = len(times)
    x = mu[:, 0]
    y = mu[:, 1]
    z = mu[:, 2]
    vx = mu[:, 3]
    vy = mu[:, 4]
    vz = mu[:, 5]
    P = Sigma[:, 0, 0]
    Q = Sigma[:, 1, 1]
    R = Sigma[:, 2, 2]

    fig, axs = plt.subplots(3, 2, figsize=(15, 10))
    fig.suptitle('State Estimates Over Time')

    axs[0, 0].plot(times, x, label='x')
    axs[0, 0].set_title('x Position')

    axs[0, 1].plot(times, y, label='y')
    axs[0, 1].set_title('y Position')

    axs[1, 0].plot(times, z, label='z')
    axs[1, 0].set_title('z Position')

    axs[1, 1].plot(times, vx, label='vx')
    axs[1, 1].set_title('x Velocity')

    axs[2, 0].plot(times, vy, label='vy')
    axs[2, 0].set_title('y Velocity')

    axs[2, 1].plot(times, vz, label='vz')
    axs[2, 1].set_title('z Velocity')

    for ax in axs.flat:
        ax.legend()
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.grid(True)

    plt.tight_layout()
    plt.show()

    # Find the last index where the change in variance is more than 0.001
    def find_cutoff_index(variance):
        for i in range(1, len(variance)):
            if abs(variance[i] - variance[i - 1]) <= 0.001:
                return i
        return len(variance)

    P_cutoff = find_cutoff_index(P)
    Q_cutoff = find_cutoff_index(Q)
    R_cutoff = find_cutoff_index(R)

    cutoff_index = min(P_cutoff, Q_cutoff, R_cutoff)

    plt.figure(figsize=(10, 6))
    plt.plot(times[:cutoff_index], P[:cutoff_index], label='P (x variance)')
    plt.plot(times[:cutoff_index], Q[:cutoff_index], label='Q (y variance)')
    plt.plot(times[:cutoff_index], R[:cutoff_index], label='R (z variance)')
    plt.title('Covariance Matrix Elements Over Time')
    plt.xlabel('Time')
    plt.ylabel('Covariance')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def plot_gs_comparison(df1, df2, title, label, unit, y_axis_multiple, frame_type):
    """
    Plot individual axes (X, Y, Z) from two dataframes against Time Elapsed.
    
    Parameters:
    df1 (pd.DataFrame): First dataframe (e.g., jah10_data)
    df2 (pd.DataFrame): Second dataframe (e.g., ssr_jah10_df)
    title (str): Title of the plot
    label (str): Plot label
    unit (str): Unit of measurement
    y_axis_multiple (float): Multiple for y-axis scaling
    frame_type (str): 'RM', 'RS', or 'TOD'
    
    Returns:
    None
    """
    frame_columns = {
        'RM': ['X_RM_Site', 'Y_RM_Site', 'Z_RM_Site'],
        'RS': ['X_RS_Site', 'Y_RS_Site', 'Z_RS_Site'],
        'TOD': ['X_TOD_Site', 'Y_TOD_Site', 'Z_TOD_Site']
    }
    
    if frame_type not in frame_columns:
        raise ValueError("Invalid frame_type. Choose 'RM', 'RS', or 'TOD'.")
    
    df2_columns = frame_columns[frame_type]
    
    # Extract relevant columns
    df1_time = df1['Time Elapsed']
    df2_time = df2['Time Elapsed']
    
    df1_x = df1['X']
    df1_y = df1['Y']
    df1_z = df1['Z']
    
    df2_x = df2[df2_columns[0]]
    df2_y = df2[df2_columns[1]]
    df2_z = df2[df2_columns[2]]
    
    # Plot
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    fig.suptitle(title, fontsize=16)
    
    axs[0].plot(df1_time, df1_x, label=f'{label} X', linestyle='-', marker='o')
    axs[0].plot(df2_time, df2_x, label=f'{frame_type} X', linestyle='--', marker='x')
    axs[0].set_ylabel(f'X ({unit})')
    axs[0].legend()
    axs[0].grid(True)
    
    axs[1].plot(df1_time, df1_y, label=f'{label} Y', linestyle='-', marker='o')
    axs[1].plot(df2_time, df2_y, label=f'{frame_type} Y', linestyle='--', marker='x')
    axs[1].set_ylabel(f'Y ({unit})')
    axs[1].legend()
    axs[1].grid(True)
    
    axs[2].plot(df1_time, df1_z, label=f'{label} Z', linestyle='-', marker='o')
    axs[2].plot(df2_time, df2_z, label=f'{frame_type} Z', linestyle='--', marker='x')
    axs[2].set_xlabel('Time Elapsed (s)')
    axs[2].set_ylabel(f'Z ({unit})')
    axs[2].legend()
    axs[2].grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    
def plot_gs_comparison_with_error(df1, df2, title, label, unit, frame_type):
    """
    Plot individual axes (X, Y, Z) from two dataframes against Time Elapsed,
    and plot the difference (error) between the dataframes.
    
    Parameters:
    df1 (pd.DataFrame): First dataframe (e.g., jah10_data)
    df2 (pd.DataFrame): Second dataframe (e.g., ssr_jah10_df)
    title (str): Title of the plot
    label (str): Plot label
    unit (str): Unit of measurement
    frame_type (str): 'RM', 'RS', or 'TOD'
    
    Returns:
    None
    """
    frame_columns = {
        'RM': ['X_RM_Site', 'Y_RM_Site', 'Z_RM_Site'],
        'RS': ['X_RS_Site', 'Y_RS_Site', 'Z_RS_Site'],
        'TOD': ['X_TOD_Site', 'Y_TOD_Site', 'Z_TOD_Site']
    }
    
    if frame_type not in frame_columns:
        raise ValueError("Invalid frame_type. Choose 'RM', 'RS', or 'TOD'.")
    
    df2_columns = frame_columns[frame_type]
    
    # Convert time elapsed to float for interpolation
    df1.loc[:, 'Time Elapsed'] = df1['Time Elapsed'].astype(float)
    df2.loc[:, 'Time Elapsed'] = df2['Time Elapsed'].astype(float)
    
    # Merge dataframes on 'Time Elapsed' with interpolation
    combined_df = pd.merge_asof(df1.sort_values('Time Elapsed'), 
                                df2[['Time Elapsed'] + df2_columns].sort_values('Time Elapsed'), 
                                on='Time Elapsed')
    
    combined_df = combined_df.infer_objects()
    combined_df.interpolate(inplace=True)

    # Extract relevant columns
    df1_time = combined_df['Time Elapsed']
    
    df1_x = combined_df['X']
    df1_y = combined_df['Y']
    df1_z = combined_df['Z']
    
    df2_x = combined_df[df2_columns[0]]
    df2_y = combined_df[df2_columns[1]]
    df2_z = combined_df[df2_columns[2]]
    
    # Calculate errors
    error_x = df1_x - df2_x
    error_y = df1_y - df2_y
    error_z = df1_z - df2_z
    
    # Calculate mean errors
    mean_error_x = error_x.mean()
    mean_error_y = error_y.mean()
    mean_error_z = error_z.mean()
    
    # Plot
    fig, axs = plt.subplots(6, 1, figsize=(10, 30))
    fig.suptitle(title, fontsize=16)
    
    # X plot
    axs[0].plot(df1_time, df1_x, label=f'{label} X', linestyle='-', marker='o')
    axs[0].plot(df1_time, df2_x, label=f'{frame_type} X', linestyle='--', marker='x')
    axs[0].set_ylabel(f'X ({unit})')
    axs[0].legend()
    axs[0].grid(True)
    
    # X error plot
    axs[1].plot(df1_time, error_x, label=f'Error X', linestyle='-', marker='o')
    axs[1].set_ylabel(f'Error X ({unit})')
    axs[1].legend()
    axs[1].grid(True)
    axs[1].text(0.95, 0.01, f'Mean Error X: {mean_error_x:.2f}', verticalalignment='bottom', horizontalalignment='right', transform=axs[1].transAxes, color='green', fontsize=15)
    
    # Y plot
    axs[2].plot(df1_time, df1_y, label=f'{label} Y', linestyle='-', marker='o')
    axs[2].plot(df1_time, df2_y, label=f'{frame_type} Y', linestyle='--', marker='x')
    axs[2].set_ylabel(f'Y ({unit})')
    axs[2].legend()
    axs[2].grid(True)
    
    # Y error plot
    axs[3].plot(df1_time, error_y, label=f'Error Y', linestyle='-', marker='o')
    axs[3].set_ylabel(f'Error Y ({unit})')
    axs[3].legend()
    axs[3].grid(True)
    axs[3].text(0.95, 0.01, f'Mean Error Y: {mean_error_y:.2f}', verticalalignment='bottom', horizontalalignment='right', transform=axs[3].transAxes, color='green', fontsize=15)
    
    # Z plot
    axs[4].plot(df1_time, df1_z, label=f'{label} Z', linestyle='-', marker='o')
    axs[4].plot(df1_time, df2_z, label=f'{frame_type} Z', linestyle='--', marker='x')
    axs[4].set_ylabel(f'Z ({unit})')
    axs[4].legend()
    axs[4].grid(True)
    
    # Z error plot
    axs[5].plot(df1_time, error_z, label=f'Error Z', linestyle='-', marker='o')
    axs[5].set_xlabel('Time Elapsed (s)')
    axs[5].set_ylabel(f'Error Z ({unit})')
    axs[5].legend()
    axs[5].grid(True)
    axs[5].text(0.95, 0.01, f'Mean Error Z: {mean_error_z:.2f}', verticalalignment='bottom', horizontalalignment='right', transform=axs[5].transAxes, color='green', fontsize=15)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    
def plot_transformation_process(df, title="Transformation Process"):
    """
    Plot transformation process (ECEF, RM, RS) against Time Elapsed.
    
    Parameters:
    df (pd.DataFrame): Dataframe containing the relevant columns
    title (str): Title of the plot
    
    Returns:
    None
    """
    columns = [
        'X_ECEF', 'Y_ECEF', 'Z_ECEF',
        'X_RM', 'Y_RM', 'Z_RM',
    ]
    
    # Ensure all columns are in the dataframe
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe")
    
    df_time = df['Time Elapsed']
    
    # Plot
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    fig.suptitle(title, fontsize=16)
    
    # X plot
    axs[0].plot(df_time, df['X_ECEF'], label='X_ECEF', linestyle='-', marker='o')
    axs[0].plot(df_time, df['X_RM'], label='X_RM', linestyle='--', marker='x')
    axs[0].set_ylabel('X (m)')
    axs[0].legend()
    axs[0].grid(True)
    
    # Y plot
    axs[1].plot(df_time, df['Y_ECEF'], label='Y_ECEF', linestyle='-', marker='o')
    axs[1].plot(df_time, df['Y_RM'], label='Y_RM', linestyle='--', marker='x')
    axs[1].set_ylabel('Y (m)')
    axs[1].legend()
    axs[1].grid(True)
    
    # Z plot
    axs[2].plot(df_time, df['Z_ECEF'], label='Z_ECEF', linestyle='-', marker='o')
    axs[2].plot(df_time, df['Z_RM'], label='Z_RM', linestyle='--', marker='x')
    axs[2].set_xlabel('Time Elapsed (s)')
    axs[2].set_ylabel('Z (m)')
    axs[2].legend()
    axs[2].grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# Example usage:
# plot_transformation_process(ssr_jah10_df)

def create_2d_plots(X_kepler=None, measurements=None, elapsed_time=None, time_range=None, show_model=True, show_measurements=None):
    components = ['rx', 'ry', 'rz', 'vx', 'vy', 'vz']
    indices = range(6)
    
    if time_range:
        start_time, end_time = time_range
        mask = (elapsed_time >= start_time) & (elapsed_time <= end_time)
        elapsed_time = elapsed_time[mask]
        if X_kepler is not None:
            X_kepler = X_kepler[mask]
        if measurements is not None:
            for name in measurements.keys():
                measurements[name] = measurements[name][mask]

    figs = []

    for component, index in zip(components[:3], range(3)):
        fig, ax = plt.subplots(figsize=(12, 6))  # Increase figure size

        if show_model and X_kepler is not None:
            ax.plot(elapsed_time, X_kepler[:, index], label=f'X_kepler_{component}', marker='o', markersize=1, linewidth=0.5)

        if show_measurements is not None:
            for name in show_measurements:
                if name in measurements:
                    ax.plot(elapsed_time, measurements[name][:, index], label=f'{name}_{component}', marker='o', markersize=1, linewidth=0.5)

        ax.set_title(f'2D Plot for {component}')
        ax.set_xlabel('Elapsed Time (seconds)')
        ax.set_ylabel(f'{component} (meters)')
        ax.legend()
        plt.grid(True)
        plt.tight_layout()  # Adjust subplots to fit into figure area.
        plt.show()

        figs.append(fig)

    for component, index in zip(components[3:], range(3)):
        fig, ax = plt.subplots(figsize=(12, 6))  # Increase figure size

        if show_model and X_kepler is not None:
            ax.plot(elapsed_time, X_kepler[:, index + 3], label=f'X_kepler_{component}', marker='o', markersize=1, linewidth=0.5)

        if show_measurements is not None:
            for name in show_measurements:
                if name in measurements:
                    ax.plot(elapsed_time, measurements[name][:, index + 3], label=f'{name}_{component}', marker='o', markersize=1, linewidth=0.5)

        ax.set_title(f'2D Plot for {component}')
        ax.set_xlabel('Elapsed Time (seconds)')
        ax.set_ylabel(f'{component} (meters/second)')
        ax.legend()
        plt.grid(True)
        plt.tight_layout()  # Adjust subplots to fit into figure area.
        plt.show()

        figs.append(fig)

    return figs