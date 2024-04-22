import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sum_square_residual(y, average):
    return sum((y[i] - average)**2 for i in range(len(y)))

df = pd.read_csv('drug.csv')

min_number_split = 2
x = df['dosage'].values
y = df['effectiveness'].values

initial_index = min_number_split -  1
final_index = len(x) - min_number_split

# min_avg = np.mean(y[min_number_split: len(x)-min_number_split])
# print(min_avg)

#possible_split = [(x[index+1] + x[index])/2 for index in range(min_number_split-1, len(x)-min_number_split)] 

avg = np.mean(y[initial_index:])
min_ssr = sum_square_residual(y[initial_index:], avg)
for index in range(initial_index + 1, final_index):
    avg = np.mean(y[index:])
    ssr = sum_square_residual(y[index:], avg)
    print(index, avg, ssr, min_ssr)
    if ssr < min_ssr:
        min_ssr = ssr

#print(y[initial_index:])
print(avg, min_ssr)


# possible_split = []
# for index in range(min_number_split-1, len(x)-min_number_split):
#     possible_split.append((x[index+1] + x[index])/2)

# print(x.iloc[min_number_split-1: len(x)-min_number_split])

#print(possible_split)

# plt.scatter(df['dosage'],  df['effectiveness'])
# plt.xlabel('dosage')
# plt.ylabel('effectiveness')
# plt.show()