import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv('IndiaPopulation_2021.csv')

print(df.describe())
x = np.array(df['Year']).reshape(-1,1)

y = np.array(df['Population'])

model = LinearRegression().fit(x,y)
X = np.array([x for x in range(1950, 2051)]).reshape(-1,1)
Y = model.predict(X)

# print(model.predict([[2050]]))
plt.plot(x,y, color='blue')
plt.plot(X, Y, color='red')
plt.show()