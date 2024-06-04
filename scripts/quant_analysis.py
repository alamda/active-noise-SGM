import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from scipy.stats import multivariate_normal
from scipy.special import kl_div
from util.toy_data import inf_data_gen
import sys
import glob
import pandas as pd

# example:  python kl_div.py swissroll 512 /scratch/midway3/nandy/redone/active-noise-SGM/simulations/reweightedv1_active_swissroll_tau0.01/work_dir/dataset_seed_0/samples/

dataset = sys.argv[1]
batch = int(sys.argv[2])
iter_path = sys.argv[3]
tau = sys.argv[4]

data = np.array(inf_data_gen(dataset,batch))
while data.shape[0] != 512:
    data = np.array(inf_data_gen(dataset,batch))
print(data.shape,'SHAPE')

def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def bhattacharyya_coefficient(p, q):
    return np.sum(np.sqrt(p * q))

def filter_outliers(xy_array, threshold=2):
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

    return filtered_array,combined_mask

from sklearn.neighbors import KernelDensity

def calculate_metrics(points_original, points_reconstruction, grid_size=500, bandwidth=0.01):
    # Determine the range for the grid
    x_min, x_max = min(np.min(points_original[:, 0]), np.min(points_reconstruction[:, 0])), max(np.max(points_original[:, 0]), np.max(points_reconstruction[:, 0]))
    y_min, y_max = min(np.min(points_original[:, 1]), np.min(points_reconstruction[:, 1])), max(np.max(points_original[:, 1]), np.max(points_reconstruction[:, 1]))

    # Create a grid of points
    x, y = np.mgrid[x_min:x_max:grid_size*1j, y_min:y_max:grid_size*1j]
    positions = np.vstack([x.ravel(), y.ravel()])

    # Estimate PDFs using KDE
    kde_original = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde_original.fit(points_original)
    pdf_original = np.exp(kde_original.score_samples(positions.T))

    kde_reconstruction = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde_reconstruction.fit(points_reconstruction)
    pdf_reconstruction = np.exp(kde_reconstruction.score_samples(positions.T))

    # Normalize PDFs to ensure they sum to 1
    pdf_original /= np.sum(pdf_original)
    pdf_reconstruction /= np.sum(pdf_reconstruction)

    # Add a small constant to avoid log(0)
    epsilon = 1e-10
    pdf_original = np.maximum(pdf_original, epsilon)
    pdf_reconstruction = np.maximum(pdf_reconstruction, epsilon)

    # Calculate KL divergence: KL(P || Q) = sum(P * log(P / Q))
    kl_divergence = np.sum(pdf_original * np.log(pdf_original / pdf_reconstruction))

    # Calculate Bhattacharyya coefficient: BC(P, Q) = sum(sqrt(P * Q))
    bhattacharyya_coef = np.sum(np.sqrt(pdf_original * pdf_reconstruction))

    return kl_divergence, bhattacharyya_coef

files = glob.glob(iter_path+'/iter_*/sample_x.npy')

dictlist = []
for i, data_file in enumerate(files):
    print(data_file)
    print(data,'DATA')
    iter_number = int(data_file.split('iter_')[1].split('/')[0])
    reconstructed = np.load(data_file)
    reconstructed, mask = filter_outliers(reconstructed)
    print(data.shape, mask.shape, reconstructed.shape)
    data_copy = data[mask]

    kl_divergence, bhattacharyya_coef = calculate_metrics(data_copy, reconstructed)
    print(kl_divergence,bhattacharyya_coef)
    dictlist.append({'iter':iter_number,'kldiv':kl_divergence,'bc':bhattacharyya_coef})
    #swiss_roll_reconstructed = multivariate_normal.pdf(reconstructed, mean=np.mean(reconstructed, axis=0), cov=np.cov(reconstructed,rowvar=False))
    #swiss_roll = multivariate_normal.pdf(data_copy, mean=np.mean(data_copy, axis=0), cov=np.cov(data_copy,rowvar=False))
    # Ensure densities are non-zero to avoid log(0) errors in KL divergence
    #swiss_roll[swiss_roll == 0] = 1e-10
    #swiss_roll_reconstructed[swiss_roll_reconstructed == 0] = 1e-10
    
    #kl_div_val = kl_divergence(swiss_roll, swiss_roll_reconstructed)
    #bhattacharyya_coeff = bhattacharyya_coefficient(swiss_roll, swiss_roll_reconstructed)
    print(data_file,'file')
    #print(kl_div_val,'kl')
    #print(bhattacharyya_coeff,'bc')

df = pd.DataFrame(dictlist)
sort_df = df.sort_values(by='iter')
print(sort_df)
sort_df.to_csv('tau'+tau+'.csv',index=False)
