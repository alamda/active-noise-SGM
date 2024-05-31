import numpy as np
import sys
import os

input_path = sys.argv[1] # currently assumes a relative image path
image_path = sys.argv[2]

xy_data = np.load(os.getcwd()+'/'+input_path)

def filter_outliers(xy_array, threshold=1.5):
    """
    Filter out huge outliers from a NumPy array of XY positions.

    :param xy_array: NumPy array of shape (N, 2) containing XY positions
    :param threshold: IQR multiplier for outlier detection (default: 1.5)
    :return: Filtered NumPy array without outliers
    """
    # Ensure the input is a NumPy array
    xy_array = np.asarray(xy_array)

    # Check if the array has the correct shape
    if xy_array.shape[1] != 2:
        raise ValueError("Input array must have shape (N, 2) for XY positions")

    # Separate X and Y coordinates
    x = xy_array[:, 0]
    y = xy_array[:, 1]

    # Function to calculate IQR and identify outliers
    def get_outliers(data):
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - (threshold * iqr)
        upper_bound = q3 + (threshold * iqr)
        return (data >= lower_bound) & (data <= upper_bound)

    # Identify non-outliers in both X and Y coordinates
    x_mask = get_outliers(x)
    y_mask = get_outliers(y)

    # Combine masks to keep points that are not outliers in either X or Y
    combined_mask = x_mask & y_mask

    # Apply the mask to filter out outliers
    filtered_array = xy_array[combined_mask]

    return filtered_array

filtered_data = filter_outliers(xy_data)

print("Original points:")
print(xy_data.shape)
print("\nFiltered points:")
print(filtered_data.shape)


plt.scatter(filtered_data[:,0],filtered_data[:,1], s=3)
plt.savefig(image_path)
