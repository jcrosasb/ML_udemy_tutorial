import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('heights_2.csv')
y = dataset.iloc[:, :-1].values  # The gender (1 is male; 0 is female)
X = dataset.iloc[:, -1].values  # The height (in inches?)

male = pd.read_csv('male.csv')
y_male = male.iloc[:, :-1].values 
X_male = male.iloc[:, -1].values  

female = pd.read_csv('female.csv')
y_female = female.iloc[:, :-1].values 
X_female = female.iloc[:, -1].values

male_30 = pd.read_csv('male_30.csv')
y_male_30 = male_30.iloc[:, :-1].values 
X_male_30 = male_30.iloc[:, -1].values  

female_30 = pd.read_csv('female_30.csv')
y_female_30 = female_30.iloc[:, :-1].values 
X_female_30 = female_30.iloc[:, -1].values


# Define the sigmoid function
def sigmoid_function(x, a, b, c):
    return c / (1 + np.exp(-a * (x - b)))

# Define the logit function from sigmoid
def logit_function(sigmoid):
    return np.log(sigmoid/(1-sigmoid))

# Define your parameters
a = 1.0  # Slope parameter
b = np.mean(X)  # Midpoint parameter
c = 1.0  # Scaling parameter

# Calculate probabilities using the sigmoid function
p = sigmoid_function(np.sort(X), a, b, c)

# Some straight line in logit space
a0, a1 = -65.5743332913415, 1.000000000000254  # lk = -377.31
#a0, a1 = -65.5743332913415, 0.9

a1_p = -0.4
a0 = a1_p * (a0/a1)
a1 = a1_p

some_logit = a0 + a1 * np.sort(X)
#plt.plot(np.sort(X), some_logit)

# Sigmoid calculated from some_logit
my_sig = 1/(1+np.exp(-some_logit))

#p_m = sigmoid_function(np.sort(X_male), a, b, c)
p_m = 1/(1+np.exp(-(a0 + a1 * np.sort(X_male))))
log_likelihood_m = np.sum(np.log(p_m))

#p_f = sigmoid_function(np.sort(X_female), a, b, c)
p_f = p_m = 1/(1+np.exp(-(a0 + a1 * np.sort(X_female))))
log_likelihood_f = np.sum(np.log(1-p_f))

print(log_likelihood_m)
print(log_likelihood_f)

log_likelihood = log_likelihood_f + log_likelihood_m


# lh = np.prod(1/(1+np.exp(-(a0 + a1 * np.sort(X_male))))) * np.prod(1-(1/(1+np.exp(-(a0 + a1 * np.sort(X_male))))))
# print(lh)

# log_lh = np.sum(np.log(1/(1+np.exp(-(a0 + a1 * np.sort(X_male))))))

my_color = 'orange'
plt.subplot(1, 2, 1)
plt.scatter(X_male, y_male, edgecolors='black', facecolors='none')
plt.scatter(X_female, y_female, edgecolors='black', facecolors='none')
plt.plot(np.sort(X), my_sig, color=my_color)
plt.ylim(-0.1, 1.1)  
plt.yticks([0, 1], ['F', 'M'])
plt.xlabel('Height')
plt.ylabel('Probability ($p$)')
ax = plt.gca()
bbox_props = dict(boxstyle="round,pad=0.3", fc="white", alpha=0.5)
textstr = f'log(likelihood)={round(log_likelihood,2)}'
ax.text(0.05, 0.5, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='center', horizontalalignment='left', color=my_color, bbox=bbox_props)
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(np.sort(X), some_logit, color=my_color)
plt.scatter(np.sort(X_male_30), a0 + a1 * np.sort(X_male_30), edgecolors='black', facecolors='none')
plt.scatter(np.sort(X_female_30), a0 + a1 * np.sort(X_female_30), edgecolors='black', facecolors='none')
plt.xlabel('Height')
plt.ylabel(r'$log(\frac{p}{1-p})$')
plt.xlim(55,78)
plt.ylim(-10,12)
ax = plt.gca()
bbox_props = dict(boxstyle="round,pad=0.3", fc="white", alpha=0.5)
textstr = f'$a_0={round(a0,2)}$, $a_1={round(a1,2)}$'
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='left', color=my_color, bbox=bbox_props)
plt.grid()

# print(1/(1+np.exp(-(a0 + a1 * np.sort(X_male_30)))))

# print(np.sum(np.log(1/(1+np.exp(-(a0 + a1 * np.sort(X_male)))))) + np.sum(np.log(1-(1/(1+np.exp(-(a0 + a1 * np.sort(X_female))))))))



# plt.subplot(1, 2, 2)
# plt.scatter(np.sort(X), my_sig)
# #plt.plot(np.sort(X), p, color='red')
# plt.plot(np.sort(X), my_sig, color='red')
# # plt.plot(np.sort(X), p2, color='green')
# # plt.plot(np.sort(X), sp3, color='black')
# plt.ylim(-0.1, 1.1)  
# plt.yticks([0, 1], ['F', 'M'])
# plt.xlabel('Height')
# plt.ylabel('Probability ($p$)')
# plt.grid()

plt.show()


