import pandas as pd
import matplotlib as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense


from df import X, Y
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=42, test_size=0.3, stratify=Y)

model = Sequential()
model.add(Flatten(input_shape=(37,)))
model.add(Dense(50, activation='sigmoid'))
model.add(Dense(25, activation='sigmoid'))
model.add(Dense(1, activation='relu'))
model.summary()

model.compile(optimizer=SGD(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history1 = model.fit(x_train, y_train, epochs=15, batch_size=128, validation_split=0.2)
#Doing one layer with 100 inputs, I have a horrible accuracy with 0.27, so instead try two layers with 50 and 25 nodes. At first I got 0.98 accuracy but after rerunning im getting .70 and im not sure why. I do notice that adding too many epoch brings the accuracy down so much at the end.

model = Sequential()
model.add(Flatten(input_shape=(37,)))
# model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer=SGD(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history2 = model.fit(x_train, y_train, epochs=15, batch_size=128, validation_split=0.2)
#Instead I tried sigmoid at the output and the results became so much better. I will save it jsut in case rerunning it makes it go away:

pd.DataFrame(history2.history).plot(figsize=(8, 5))
model.evaluate(x_test, y_test)
#It seems like even with three layers we were not overfitted and was able to perform well on both a training and testing set
