import tensorflow as tf

xData = [1, 2, 3]
yData = [3, 5, 7]
x = 0
y = 0
# weight = height * a + b

a = tf.Variable(1)
b = tf.Variable(2)

def loss():
    return tf.square(y - (x * a + b))

opt = tf.keras.optimizers.Adam(learning_rate=0.1)

t = 0
for i in range(300):
    x = xData[t % len(xData)]
    y = yData[t % len(yData)]
    opt.minimize(loss, var_list=[a, b])
    t += 1
    print(a.numpy(), b.numpy())

n = int(input('자신의 키를 입력하세요: '))
print(n * a.numpy() + b.numpy())