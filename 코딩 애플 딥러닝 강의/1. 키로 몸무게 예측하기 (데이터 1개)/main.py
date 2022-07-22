import tensorflow as tf

height = 176
weight = 54
# weight = height * a + b

a = tf.Variable(0.1)
b = tf.Variable(0.2)

def loss():
    return tf.square(weight - (height * a + b))

opt = tf.keras.optimizers.Adam(learning_rate=0.1)

for i in range(300):
    opt.minimize(loss, var_list=[a, b])
    print(a.numpy(), b.numpy())

n = int(input('자신의 키를 입력하세요: '))
print(n * a.numpy() + b.numpy())