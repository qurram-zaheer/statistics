import pandas as pd

# pandas displaying options. Not necessary, but needed for Data Exploration
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

number_of_data_points = 10000                # Increase this to have a better model
train_df_total = pd.read_csv("train_V2.csv")     # This might take a few seconds
print(train_df_total.shape)
train_df = train_df_total.sample(n=number_of_data_points)    # Sampling 10,000 rows at random
train_df.head()

# Write your code here
train_df = train_df.drop(['Id', 'groupId', 'matchId', 'matchType'], axis = 1)
train_df.head()

target = train_df.iloc[:,-1].values
train_df = train_df.drop(['winPlacePerc'], axis = 1).values

import numpy as np
normalized_train = (train_df - np.mean(train_df, axis = 0))/np.std(train_df, axis = 0)

from numpy import cov            # To calculate covraiance matrix
from numpy.linalg import eig     # To calculate eigenvalues and eigenvectors

train_array = train_df  # Returns numpy array of values of the dataframe

ones = np.ones((10000,1))
ones
train_std = np.append(normalized_train,ones, axis = 1)

m = 7000
x_train = train_std[:m,:]
x_test = train_std[m:,:]
y_train = target[:m]
y_test = target[m:]

n = x_train.shape[1]

m = 7000

theta = np.ones(n)

cost = (1/(2*m))*np.transpose((x_train@theta - y_train))@(x_train@theta - y_train)

array_a = [0.0005,0.01,0.1,0.5,1]

cost_lists = []
thetas = []
for a in array_a:
    print(a)
    theta = np.ones(n)
    cost_list = []
    for i in range(m):
        theta = theta - a*(1/m)*np.transpose(x_train)@(x_train@theta - y_train)
        cost_val = (1/(2*m))*np.transpose((x_train@theta - y_train))@(x_train@theta - y_train)
        cost_list.append(cost_val)
    cost_lists.append(cost_list)
    thetas.append(theta)

import matplotlib.pyplot as plt
plt.axis([0,5000,0,40])
x_axis = [i for i in range(7000)]
for i in range(3):
    plt.plot(x_axis, cost_lists[i], label = 'a = ' + str(array_a[i]))
    plt.legend()
plt.show()

final_theta = thetas[2]
cost_val = (1/(2*m))*np.transpose((x_test@final_theta - y_test))@(x_test@final_theta - y_test)



