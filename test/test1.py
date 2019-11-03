import numpy as np
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Dropout, Input
from tensorflow.python.keras.layers.merge import Add, Multiply
from tensorflow.python.keras.optimizers import Adam

from tensorflow.python.keras import backend as K

import tensorflow as tf


sess = tf.Session()
K.set_session(sess)


input = Input(shape=(1,))
dens_1 = Dense(128, activation='relu')(input)
dens_2 = Dense(128, activation='relu')(dens_1)
dens_3 = Dense(128, activation='relu')(dens_2)
dens_4 = Dense(128, activation='relu')(dens_3)

output_1 = Dense(1, activation='linear')(dens_4)
output_2 = Dense(1, activation='sigmoid')(dens_4)
output_3 = Dense(3, activation='softmax')(dens_4)

model = Model(inputs=input, outputs=[output_1, output_2, output_3])
adam = Adam(lr=0.001)
model.compile(loss="mse", optimizer=adam)

one, tow, three = model.predict([1, 2, 3, 4, 5, 6])

one = one.tolist()
tow = tow.tolist()
three = three.tolist()

for i in range(len(one)):
    print([one[i][0], tow[i][0], three[i]])


local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

print('')
print('')
print('')

vars = sess.run(local_vars)
print(vars[0])

print('')

_, _, vars = sess.run([output_1, output_2, local_vars], feed_dict={input: [[1], [2], [3], [4], [5], [6]]})
print(vars[0])






