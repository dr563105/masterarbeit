import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, Dense, Activation, Flatten, Lambda, Dropout
from keras.layers import MaxPooling2D, LSTM, TimeDistributed, concatenate, BatchNormalization, GRU
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.optimizers import Adam
import tensorflow.keras.backend as K
from keras.utils import plot_model, to_categorical
from keras.losses import binary_crossentropy, mean_squared_error, categorical_crossentropy
from utils import load_multi_dataset, mkdir_p, HDF5_PATH_DS3, MODEL_PATH
from datetime import datetime
import time
import errno
from sklearn.model_selection import train_test_split
import functools

time_steps = 15
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def w_categorical_crossentropy(y_true, y_pred, sample_weight=None):
	#x=tf.keras.losses.CategoricalCrossentropy(y_pred, y_true)
	
	cce = tf.keras.losses.CategoricalCrossentropy()
	x=cce(y_true, y_pred)
	y=K.mean(x)
	print("Shape of cce",x)
	print("Shape of cce after mean",y)
	return y*1 


def w_mse(y_true, y_pred, sample_weight=None):
	x=K.square(y_true - y_pred)
	y=K.mean(x)
	print("Shape of mse",x)
	print("Shape of mse after mean ",y)
	return y*0.001 

def st_mse(y_true, y_pred, sample_weight=None):
	x=K.square(y_true - y_pred)
	y=K.mean(x)
	print("Shape of st_mse",x)
	print("Shape of st_mse after mean ",y)
	return y*10

ncce = functools.partial(w_categorical_crossentropy,sample_weight=None)
ncce.__name__ = 'cce'

nmse = functools.partial(w_mse,sample_weight=None)
nmse.__name__ = 'mse'

nstmse = functools.partial(st_mse,sample_weight=None)
nstmse.__name__ = 'stmse'


print(f'Loading data from HDF5... at {time.ctime()}')

X_data, Y_data = load_multi_dataset(os.path.join(HDF5_PATH_DS3, f'train_ts{time_steps}_ds3_5_h5_list.txt'))

print('Number of images:', X_data.shape)
print('Number of labels:', Y_data.shape)

print(f'Splitting data into training set and testing set....at {time.ctime()}')
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)
print(f'Splitting data into training set and testing end....at {time.ctime()}')

print('X_train shape:', X_train.shape)
print('Y_train shape:', Y_train.shape)
print('X_test shape:', X_test.shape)
print('Y_test shape:', Y_test.shape)

input_layer = Input(shape = (15, 70, 160, 1))

input_normalisation= Lambda(lambda x: x / 255.0) (input_layer)
#24, 36, 48, 64, 80, 96
conv2d_layer_1 = TimeDistributed(Conv2D(24, (5, 5), strides=(2, 2), padding='same', activation='relu')) (input_normalisation) 
conv2d_layer_2 = TimeDistributed(Conv2D(36, (5, 5), strides=(2, 2), padding='same', activation='relu')) (conv2d_layer_1) 
conv2d_layer_3 = TimeDistributed(Conv2D(48, (5, 5), strides=(2, 2), padding='same', activation='relu')) (conv2d_layer_2)
conv2d_layer_4 = TimeDistributed(Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu')) (conv2d_layer_3)
conv2d_layer_5 = TimeDistributed(Conv2D(80, (3, 3), strides=(2, 2), padding='same', activation='relu')) (conv2d_layer_4) 
conv2d_layer_6 = TimeDistributed(Conv2D(96, (3, 3), strides=(2, 2), padding='same', activation='relu')) (conv2d_layer_5) 
flatten_layer = TimeDistributed(Flatten()) (conv2d_layer_6)

lstm_layer_common = LSTM(200, activation='tanh',return_sequences=True) (flatten_layer)

lstm_layer = LSTM(100, activation='tanh') (lstm_layer_common)
lstm_layer2 = LSTM(100, activation='tanh') (lstm_layer_common)
lstm_layer3 = LSTM(100, activation='tanh') (lstm_layer_common)

dense_layer_1 = Dense(100, activation='relu') (lstm_layer)
dense_layer_2 = Dense(100, activation='relu') (lstm_layer2)
dense_layer_3 = Dense(100, activation='relu') (lstm_layer3)

dense_layer_4 = Dense(10, activation='relu') (dense_layer_1)
dense_layer_5 = Dense(10, activation='relu') (dense_layer_2)
dense_layer_6 = Dense(10, activation='relu') (dense_layer_3)

output_steering = Dense(1, activation = 'tanh', name='st') (dense_layer_6)
output_velocity = Dense(1, activation ='relu', name='velocity') (dense_layer_5)
output_classification = Dense(3, activation = 'softmax', name='classification') (dense_layer_4)

#output_combo = concatenate([output_accel, output_steering], name = 'acc_st')

model = Model(inputs=input_layer, outputs=[output_classification, output_velocity, output_steering] , name='evaluate_ts15_ds3_Class5_conv1')

model.summary()
plot_model(model, to_file='evaluate_ts15_ds3_Class5_conv1.png', show_shapes=True)

model.compile(optimizer=Adam(lr=1e-04, decay = 0.0), loss={'classification':ncce, 'st':nstmse,'velocity':nmse})




#'classification':'categorical_crossentropy', 'st':'mse',
#print("Loading model_criteria68....")
#model = load_model(os.path.join(MODEL_PATH, 'evaluate_ts15_ds3_Class4_6.h5'))

#callbacks
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
mc = ModelCheckpoint(os.path.join(MODEL_PATH, 'evaluate_ts15_ds3_Class5_conv1.h5'), 
	monitor='val_loss', mode='min', verbose=1, save_best_only=True)
logdir = "tb_logs/evaluation/" + "evaluate_ts15_ds3_Class5_conv1_" +  datetime.now().strftime("%y%m%d-%H%M")
tbc = TensorBoard(log_dir=logdir)

#
#
t0 = time.time()
model.fit(X_train, {'classification': Y_train[:,0:3],'st': Y_train[:,-2],'velocity': Y_train[:,-1] } , 
	validation_data=(X_test, {'classification': Y_test[:,0:3],'st': Y_test[:,-2],'velocity': Y_test[:,-1]}), shuffle=True, epochs=150, batch_size=128, 
	verbose=1, callbacks=[mc, tbc])
#'classification': Y_train[:,0:3],'st': Y_train[:,-2], 'classification': Y_test[:,0:3],'st': Y_test[:,-2],
t1 = time.time()
print('Total training time:', t1 - t0, 'seconds')

mkdir_p(MODEL_PATH)
model_id = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
model_file = os.path.join(MODEL_PATH, f'{model_id}.h5')
model.save(model_file)
print(f"Training done successfully and model has been saved: {model_file}")
print("Drive safely!")


'''
evaluate_ts15_ds3_Class5_conv1.hd5

Number of images: (100923, 15, 70, 160, 1)
Number of labels: (100923, 5)
Splitting data into training set and testing set....at Sun Oct 11 20:59:03 2020
Splitting data into training set and testing end....at Sun Oct 11 20:59:05 2020
X_train shape: (80738, 15, 70, 160, 1)
Y_train shape: (80738, 5)
X_test shape: (20185, 15, 70, 160, 1)
Y_test shape: (20185, 5)
Model: "evaluate_ts15_ds3_Class5_conv1"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 15, 70, 160,  0                                            
__________________________________________________________________________________________________
lambda_1 (Lambda)               (None, 15, 70, 160,  0           input_1[0][0]                    
__________________________________________________________________________________________________
time_distributed_1 (TimeDistrib (None, 15, 35, 80, 2 624         lambda_1[0][0]                   
__________________________________________________________________________________________________
time_distributed_2 (TimeDistrib (None, 15, 18, 40, 3 21636       time_distributed_1[0][0]         
__________________________________________________________________________________________________
time_distributed_3 (TimeDistrib (None, 15, 9, 20, 48 43248       time_distributed_2[0][0]         
__________________________________________________________________________________________________
time_distributed_4 (TimeDistrib (None, 15, 5, 10, 64 27712       time_distributed_3[0][0]         
__________________________________________________________________________________________________
time_distributed_5 (TimeDistrib (None, 15, 3, 5, 80) 46160       time_distributed_4[0][0]         
__________________________________________________________________________________________________
time_distributed_6 (TimeDistrib (None, 15, 2, 3, 96) 69216       time_distributed_5[0][0]         
__________________________________________________________________________________________________
time_distributed_7 (TimeDistrib (None, 15, 576)      0           time_distributed_6[0][0]         
__________________________________________________________________________________________________
lstm_1 (LSTM)                   (None, 15, 200)      621600      time_distributed_7[0][0]         
__________________________________________________________________________________________________
lstm_2 (LSTM)                   (None, 100)          120400      lstm_1[0][0]                     
__________________________________________________________________________________________________
lstm_3 (LSTM)                   (None, 100)          120400      lstm_1[0][0]                     
__________________________________________________________________________________________________
lstm_4 (LSTM)                   (None, 100)          120400      lstm_1[0][0]                     
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 100)          10100       lstm_2[0][0]                     
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 100)          10100       lstm_3[0][0]                     
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 100)          10100       lstm_4[0][0]                     
__________________________________________________________________________________________________
dense_4 (Dense)                 (None, 10)           1010        dense_1[0][0]                    
__________________________________________________________________________________________________
dense_5 (Dense)                 (None, 10)           1010        dense_2[0][0]                    
__________________________________________________________________________________________________
dense_6 (Dense)                 (None, 10)           1010        dense_3[0][0]                    
__________________________________________________________________________________________________
classification (Dense)          (None, 3)            33          dense_4[0][0]                    
__________________________________________________________________________________________________
velocity (Dense)                (None, 1)            11          dense_5[0][0]                    
__________________________________________________________________________________________________
st (Dense)                      (None, 1)            11          dense_6[0][0]                    
==================================================================================================
Total params: 1,224,781
Trainable params: 1,224,781
Non-trainable params: 0
___________________________
'''