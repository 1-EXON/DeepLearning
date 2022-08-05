import tensorflow as tf
import pandas as pd
import numpy as np

data = pd.read_csv('train.csv')
data = data.dropna()

yData = data['Survived'].values
xData = []

for i, rows in data.iterrows():
    sex = 0
    if rows['Sex'] == 'male':
        sex = 1
    xData.append([ rows['Pclass'], sex ])

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='sigmoid'),
    tf.keras.layers.Dense(256, activation='sigmoid'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(np.array(xData), np.array(yData), epochs=1000)

test = pd.read_csv('test.csv')
test = test.dropna()

for i, rows in test.iterrows():
    sex = 0
    if rows['Sex'] == 'male':
        sex = 1
    result = model.predict([[ rows['Pclass'], sex ]])
    r = 0
    if result[0][0] > 0.5:
        r = 1
    
    print(r)