import numpy as np
import matplotlib.pyplot as plt

p = np.linspace(0.0001, 0.9999, 100)
x = np.log(p / (1 - p))

# # Create the first subplot
# plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
# plt.plot(x, p)
# plt.xlabel('x')
# plt.ylabel('Probability')
# plt.grid()

# # Create the second subplot

# plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
# plt.semilogy(x,p)
# #plt.xlabel()
# #plt.ylabel()
# plt.grid()

# plt.tight_layout()  # Adjust layout to prevent overlapping
# plt.show()

x = np.array([0.00001])
y = np.log(x/(1-x))

print(y)

# plt.plot(x,y)
# plt.grid()
# plt.show()