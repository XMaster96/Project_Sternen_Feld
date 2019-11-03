import numpy as np
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Dropout, Input, LSTM, Reshape
from tensorflow.python.keras.layers.merge import Add, Multiply
from tensorflow.python.keras.optimizers import Adam

from tensorflow.python.keras import backend as K

import tensorflow as tf


sess = tf.Session()
K.set_session(sess)

input = Input(shape=(1, ), batch_size=6)

re = Reshape(target_shape=(1, 1))(input)

rnn_1 = LSTM(128, stateful=True, return_sequences=True)(re)
rnn_2 = LSTM(128, stateful=True, return_sequences=False)(rnn_1)

output_1 = Dense(1, activation='linear')(rnn_2)
output_2 = Dense(1, activation='sigmoid')(rnn_2)
output_3 = Dense(3, activation='softmax')(rnn_2)

model = Model(inputs=input, outputs=[output_1, output_2, output_3])
adam = Adam(lr=0.001)
model.compile(loss="mse", optimizer=adam)

raw_weights = model.get_weights()
new_weights = []

for raw in raw_weights:

    new_weights.append(np.random.uniform(-5, 5, raw.shape))

model.set_weights(np.array(new_weights))

one, tow, three = model.predict([1, 2, 3, 4, 5, 6])

one = one.tolist()
tow = tow.tolist()
three = three.tolist()

print('predict: [1, 2, 3, 4, 5, 6]')
for i in range(len(one)):
    print([one[i][0], tow[i][0], three[i]])

model.reset_states()
print('')

one, tow, three = model.predict([0, 0, 0, 1, 2, 3])

one = one.tolist()
tow = tow.tolist()
three = three.tolist()

print('predict: [0, 0, 0, 1, 2, 3]')
for i in range(len(one)):
    print([one[i][0], tow[i][0], three[i]])


model.reset_states()
print('')

one, tow, three = model.predict([0, 0, 1, 2, 0, 3])

one = one.tolist()
tow = tow.tolist()
three = three.tolist()

print('predict: [0, 0, 1, 2, 0, 3]')
for i in range(len(one)):
    print([one[i][0], tow[i][0], three[i]])


model.reset_states()
print('')


one, tow, three = sess.run([output_1, output_2, output_3], feed_dict={input: [[1], [2], [3], [4], [5], [6]]})

one = one.tolist()
tow = tow.tolist()
three = three.tolist()

print('sess.run: [1, 2, 3, 4, 5, 6]')
for i in range(len(one)):
    print([one[i][0], tow[i][0], three[i]])


model.reset_states()
print('')
