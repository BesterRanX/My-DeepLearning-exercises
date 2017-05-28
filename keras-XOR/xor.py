from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
import numpy as np
# X is input. y is the desired output
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
sgd = SGD(lr=0.2)

model = Sequential()
# input layer "2" unit because 'X' has 2 shape
model.add(Dense(4, input_dim=2, kernel_initializer='uniform', activation='sigmoid', use_bias=0.5))
# output layer. "1" unit because 'y' has 1 shape
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=sgd)
model.fit(X, y, batch_size=1, epochs=1000)
# prediction
print(model.predict_proba([[0, 1], [0, 0], [0, 0], [1, 0]]))