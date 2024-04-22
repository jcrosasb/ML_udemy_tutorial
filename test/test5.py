import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# def sum_square_residual(y, average):
#     return sum((y[i] - average)**2 for i in range(len(y)))


df = pd.read_csv('drug.csv')

min_number_split = 2
x = df['dosage'].values
y = df['effectiveness'].values

splits = []
avgs = []


def left_right_avg(obs, split_index):
    '''split_index is the index of the first data point on right side'''
    avg_left = np.mean(obs[:split_index])
    avg_right = np.mean(obs[split_index:])
    return avg_left, avg_right

def sum_square_residual(obs, split_index):
    avg_left, avg_right = left_right_avg(obs, split_index)
    left_ssr, right_ssr = 0, 0
    for index in range(len(obs)):
        if index < split_index:
            left_ssr += (obs[index] - avg_left)**2 
        else:
            right_ssr += (obs[index] - avg_right)**2
    return left_ssr + right_ssr

def find_split(data, obs, min_number):
    if 2 * min_number >= len(obs):
        return None
    min_ssr = sum_square_residual(obs, min_number)
    split_index = min_number
    split = (data[min_number] + data[min_number-1])/2
    for index in range(min_number+1, len(obs)-min_number):
        ssr = sum_square_residual(obs, index)
        if ssr < min_ssr:
            min_ssr = ssr
            split_index = index
            split = (data[index] + data[index-1])/2
    splits.append(split)
    return split_index


def recursive_split(data, obs, min_number):
    # Base case
    split_index = find_split(data, obs, min_number)
    if split_index is None:
        return None
    left_data = data[:split_index]
    right_data = data[split_index:]
    
    # Recursively call recursive_split on left and right sections
    recursive_split(left_data, obs[:split_index], min_number)
    recursive_split(right_data, obs[split_index:], min_number)


recursive_split(data=x, obs=y, min_number=min_number_split)

print(splits)

plt.scatter(df['dosage'],  df['effectiveness'])
plt.vlines(splits, color='red', ymin=min(y), ymax=max(y), linestyle='--')
plt.xlabel('dosage')
plt.ylabel('effectiveness')
plt.show()