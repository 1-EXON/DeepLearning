# 2. 성적으로 대학 입학여부 예측하기 (데이터 여러개)

## 데이터 전처리
```python
data = pd.read_csv('gpascore.csv')
data = data.dropna() # 데이터의 빈 공간이 있으면 그 빈공간의 행을 삭제하는 함수

yData = data['admit'].values # admit 열의 값들 (배열)
xData = []

for i, rows in data.iterrows():
    xData.append([ rows['gre'], rows['gpa'], rows['rank'] ])
```

## 모델 디자인, 학습
```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='sigmoid'), # Layer 만들기 (128개)
    tf.keras.layers.Dense(256, activation='sigmoid'), # activation: 활성 함수
    tf.keras.layers.Dense(512, activation='sigmoid'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 모델 완성
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# optimizer: 경사 하강법을 진행 할 "때, 기울기만큼 빼게 되는데 그 기울기의 learning weight를 곱해서 뺀다.
# 이 때 균등하게 빼게 되면 학습이 잘 안될 수 있다고 한다. 이 learning witght 를 효과적으로 적용해주는 시스템이다.
# loss: 손실 함수. 0~1인지 예측을 하고 싶을 때 효율적인 손실 함수는 binary_crossentropy 라고 한다.
# metrics: 

model.fit(np.array(xData), np.array(yData), epochs=1000) # 학습시키기
# epochs: 반복
```

## 결과 예측
```python
result = model.predict([[750, 3.70, 3], [400, 2.2, 1]])
print(result)
```


