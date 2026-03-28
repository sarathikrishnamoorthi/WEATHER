WEATHER DATA SCIENCE PROJECT
# =========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score
from scipy.stats import norm

np.random.seed(7)

# =========================================
# EXP 1 - WEATHER CORRELATION
# =========================================
print("\nEXP 1: Weather Correlation")

data = np.random.randint(10, 100, (50, 5))
features = ["Temp", "Humidity", "Rainfall", "Wind", "Pressure"]

corr = np.corrcoef(data, rowvar=False)

plt.imshow(corr)
plt.colorbar()
plt.xticks(range(5), features)
plt.yticks(range(5), features)
plt.title("Weather Correlation Heatmap")
plt.show()

# =========================================
# EXP 2 - RAIN PREDICTION
# =========================================
print("\nEXP 2: Rain Prediction")

df = pd.DataFrame(np.random.rand(200, 4), columns=["Temp","Humidity","Wind","Pressure"])
df["Rain"] = np.random.randint(0, 2, 200)

X = df.iloc[:, :-1]
y = df["Rain"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

# =========================================
# EXP 3 - TEMPERATURE TREND
# =========================================
print("\nEXP 3: Temperature Trend")

years = np.arange(2000, 2020)
temp = 25 + 0.1*(years-2000) + np.random.randn(20)

model = LinearRegression()
model.fit(years.reshape(-1,1), temp)

plt.scatter(years, temp)
plt.plot(years, model.predict(years.reshape(-1,1)))
plt.title("Temperature Trend")
plt.show()

# =========================================
# EXP 4 - DATA SAMPLING
# =========================================
print("\nEXP 4: Sampling")

df = pd.DataFrame({
    "Day": np.arange(1,101),
    "Weather": np.random.choice(["Sunny","Rainy","Cloudy"],100)
})

print(df.sample(frac=0.25))

# =========================================
# EXP 5 - Z TEST
# =========================================
print("\nEXP 5: Z-Test")

temps = np.random.normal(30,5,40)
z = (np.mean(temps) - 28) / (np.std(temps)/np.sqrt(40))

print("Z-score:", z)
print("Decision:", "Reject H0" if z > norm.ppf(0.95) else "Accept H0")

# =========================================
# EXP 6 - NUMPY OPERATIONS
# =========================================
print("\nEXP 6: Operations")

rain = np.random.randint(0,50,12)

print("Mean:", np.mean(rain))
print("Sum:", np.sum(rain))

reshaped = rain.reshape(12,1)
print("Transpose:\n", reshaped.T)

# =========================================
# EXP 7 - CLEANING
# =========================================
print("\nEXP 7: Cleaning")

data = np.array([[20,30,np.nan],[25,35,45]])
data[np.isnan(data)] = np.nanmean(data)
print(data)

# =========================================
# EXP 8 - WEATHER ANALYSIS
# =========================================
print("\nEXP 8: Analysis")

weather = np.random.normal(50,10,(100,3))
print("Average values:", weather.mean(axis=0))

# =========================================
# EXP 9 - PATTERN ANALYSIS
# =========================================
print("\nEXP 9: Pattern Analysis")

data = np.random.randint(10,100,(10,5))
print("Avg:", data.mean(axis=0))

plt.imshow(np.corrcoef(data, rowvar=False))
plt.colorbar()
plt.title("Weather Pattern Correlation")
plt.show()

# =========================================
# EXP 10 - RAIN ANALYSIS
# =========================================
print("\nEXP 10: Rain Analysis")

df = pd.DataFrame({
    "Rainfall": np.random.randint(0,100,50),
    "Humidity": np.random.randint(20,100,50)
})

plt.scatter(df["Humidity"], df["Rainfall"])
plt.title("Humidity vs Rainfall")
plt.xlabel("Humidity")
plt.ylabel("Rainfall")
plt.show()
