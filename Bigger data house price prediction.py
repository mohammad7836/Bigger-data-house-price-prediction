import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = {
    "Area": [60, 70, 80, 90, 100, 110, 120],
     "bedrooms" : [1, 2, 3, 4, 3, 2, 1],
     "bathrooms" : [1, 1, 2, 2, 3, 3, 1],
     "floors" : [1, 1, 2, 2, 3, 3, 4],
     "age" : [5, 10, 15, 20, 25, 30, 3],
     "parking" : [1, 1, 2, 3, 4, 2, 1],
     "price" : [50000, 60000, 70000, 80000, 90000, 100000, 10000]
}

df = pd.DataFrame(data)

x = df[["Area", "bedrooms", "bathrooms", "floors", "age", "parking"]]
y = df["price"]

model = LinearRegression()
model.fit(x, y)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
features = ["Area", "bedrooms", "bathrooms", "floors", "age", "parking"]

for ax, feature in zip(axes.flatten(), features):
    ax.scatter(df[feature], df["price"])
    ax.set_xlabel(feature)
    ax.set_ylabel("price")
    ax.set_title(f"{feature} vs Price")
plt.tight_layout()
plt.suptitle("scatter plots of features vs price", fontsize=16, y=1.02)
plt.show()
print("مدل آموزش دید")
print(f"ضرب مساحت:{model.coef_[0]:.2f}")
print(f"ضرب تعداد اتاق:{model.coef_[1]:.2f}")
print(f"ضرب تعداد حمام:{model.coef_[2]:.2f}")
print(f"ضرب تعداد طبقات:{model.coef_[3]:.2f}")
print(f"ضرب سن:{model.coef_[4]:.2f}")
print(f"ضرب پارکینگ:{model.coef_[5]:.2f}")
print(f"مقدار ثابت:{model.intercept_:.2f}")

