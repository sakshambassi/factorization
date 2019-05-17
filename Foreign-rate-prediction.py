from scipy.stats import skew, kurtosis
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
from scipy.stats import entropy
import math
import keras


def entropy1(labels, base=None):
  value,counts = np.unique(labels, return_counts=True)
  return entropy(counts, base=base)

df = pd.read_csv("/home/saksham/TIFR-CAM/Finance/Foreign Exchange/USA-AUS-exchange-rates.csv")
# ND in USA-AUS, '.' in csv downloaded from FRED
df = df[df.iloc[:, 1] != 'ND']
# df = df.convert_objects(convert_numeric=True)
df = df.iloc[:, 1].astype('float64')
df = df.values
df = 1/df
print(df)

train = df[0: int(math.floor(0.8 * len(df)))]
valid = df[int(math.floor(0.8 * len(df))):]

print(df.shape)


print(train.shape, valid.shape)

num_days = 7
x_train, y_train = [], []
for i in range(num_days,len(train)):
	x_train.append(train[i-num_days:i])
	y_train.append(train[i])
x_train, y_train = np.array(x_train), np.array(y_train)

print(x_train[0].dtype)
# In[28]:

y_train.shape


# In[29]:

inputs = df[len(train) - num_days:]
x_test = []

for i in range(num_days,inputs.shape[0]):
	x_test.append(inputs[i-num_days:i])
x_test = np.array(x_test)
print(x_test.shape)


# In[30]:
def return_features(a):
	Feature_array = []
	for row in range(0,a.shape[0]):
		mean = np.mean(a[row])
		f = np.array([mean])
		skewness = skew(a[row], axis=0, bias=True)
		f = np.append(f, skewness)
		kurt = kurtosis(a[row], axis=0, fisher=True, bias=True)
		f= np.append(f, kurt)
		variance = np.var(a[row])
		f = np.append(f, variance)
		std = np.std(a[row])
		f = np.append(f, std)
		x = np.arange(0,len(a[row]),1)
		slope, intercept, r_value, p_value, std_err = stats.linregress(x,a[row])
		f = np.append(f, slope)
		f = np.append(f, entropy1(a[row]))
		df = pd.DataFrame({'A' : a[row]})
		f = np.append(f,df.ewm(alpha = 0.6).mean().iloc[-1,0])
		Feature_array.append(f)

a = x_train
Feature_array = return_features(a)
Feature_train = np.asarray(Feature_array)

a = x_test
Feature_array = return_features(a)
Feature_test = np.asarray(Feature_array)


# In[33]:

# x_train = np.reshape(Feature_train, (Feature_train.shape[0],Feature_train.shape[1]))
# x_test = np.reshape(Feature_test, (Feature_test.shape[0],Feature_test.shape[1]))

print(x_train.shape)
# create and fit the LSTM network
def CNN():
	model = keras.models.Sequential()
	model.add(keras.layers.Dense(units=128, activation='relu',input_dim=x_train.shape[1]))
	model.add(keras.layers.Dense(units=64, activation='relu'))
	model.add(keras.layers.Dense(units=32, activation='relu'))
	model.add(keras.layers.Dense(units=16, activation='relu'))
	model.add(keras.layers.Dense(1))
	model.summary()
	model.compile(loss=keras.losses.mse, optimizer=keras.optimizers.Adam())
	model.fit(x_train, y_train, epochs=200, batch_size=8, verbose=1, validation_data=(x_test, valid) )
	return model

model = CNN()

closing_price = model.predict(x_test)
rms=np.sqrt(np.mean(np.power((valid-closing_price),2)))
print("RMS:",rms)

least_abs_mean = np.mean(valid-closing_price)
print("Least abs mean:", least_abs_mean)

import seaborn as sns
sns.set(font_scale = 2.2)


# In[49]:

train = y_train
closing_price_x = np.arange(len(y_train), len(y_train) + len(closing_price), 1)
valid_x = np.arange(len(y_train), len(y_train) + len(valid), 1)
plt.figure(figsize=(16,8))
plt.ylabel("Rate")
plt.xlabel("Days")
plt.plot(train, label = 'Train Data')
plt.plot(valid_x, valid, label = 'Test Data')
plt.plot(closing_price_x, closing_price, label = 'Prediction')
plt.legend()
plt.savefig("Pred-on-test-with-train.pdf")
plt.show()


# In[50]:

# y_pred = model.predict(x_train)
plt.figure(figsize=(20,10))
plt.ylabel("Rate")
plt.xlabel("Days")
plt.plot(closing_price.ravel(), label = 'Predicted Rate')
plt.plot(valid.ravel(), label = 'Actual Rate')
plt.legend()	
plt.savefig("Pred-on-test.pdf")
plt.show()