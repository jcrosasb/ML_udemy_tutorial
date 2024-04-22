import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# def sum_square_residual(y, average):
#     return sum((y[i] - average)**2 for i in range(len(y)))


df = pd.read_csv('drug.csv')

min_number_split = 2
x = df['dosage'].values
y = df['effectiveness'].values

# possible_split = 9
# left_index_split = 3
# right_index_split = left_index_split + 1


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
    min_ssr = sum_square_residual(obs, min_number)
    split_index = min_number
    split = (data[min_number] + data[min_number-1])/2
    for index in range(min_number+1, len(obs)-min_number):
        ssr = sum_square_residual(obs, index)
        if ssr < min_ssr:
            min_ssr = ssr
            split_index = index
            split = (data[index] + data[index-1])/2
    return split, split_index



plt.scatter(df['dosage'],  df['effectiveness'])
plt.axvline(find_split(x,y,8)[0], color='red', linestyle='--')
plt.xlabel('dosage')
plt.ylabel('effectiveness')
plt.show()

#
    

# print(left_right_avg(y, 4))
# print(sum_square_residual(y, 4))

# print((0-0)**2 + (5-47.1)**2 + (20-47.1)**2 + 4*((100-47.1)**2) + (62.5-47.1)**2 + (58-47.1)**2 + (53-47.1)**2 + (50-47.1)**2 \
#       + (48-47.1)**2 + (10-47.1)**2 + 3*((0-47.1)**2))

# def sum_square_residual(initial_index, final_index, split_left_index, split_right_index):
#     avg_left, avg_right = left_right_avg(split_right_index)
#     for index in range(initial_index, final_index+1):
#         if index <= split_left_index:
#             left_ssr = 
            


#print(left_right_avg(right_index_split))

