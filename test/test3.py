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
    return left_ssr + right_ssr, avg_left, avg_right

def find_split(data, obs, min_number):
    if 2 * min_number >= len(obs):
        return None
    min_ssr, avg_left, avg_right = sum_square_residual(obs, min_number)
    split_index = min_number
    split = (data[min_number] + data[min_number-1])/2
    for index in range(min_number+1, len(obs)-min_number):
        ssr, avg_left_tmp, avg_right_tmp = sum_square_residual(obs, index)
        if ssr < min_ssr:
            min_ssr = ssr
            avg_left, avg_right = avg_left_tmp, avg_right_tmp
            split_index = index
            split = (data[index] + data[index-1])/2
    return split, split_index, avg_left, avg_right

# =========================
split, split_index, avg_left, avg_right = find_split(x,y,2)

splits.append(split)
avgs.extend([avg_left, avg_right])
# =========================

# LEFT =========================
split, split_index, avg_left, avg_right = find_split(x[:split_index],y[:split_index],2)

splits.append(split)
avgs.extend([avg_left, avg_right])

# RIGHT =========================
split, split_index, avg_left, avg_right = find_split(x[split_index:],y[split_index:],2)

splits.append(split)
avgs.extend([avg_left, avg_right])


print(splits)




plt.scatter(df['dosage'],  df['effectiveness'])
plt.vlines(splits, color='red', ymin=min(y), ymax=max(y), linestyle='--')
plt.xlabel('dosage')
plt.ylabel('effectiveness')
plt.show()

