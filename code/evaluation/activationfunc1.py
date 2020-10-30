# import numpy as np
# import matplotlib.pylab as plt

# def tanh(x):
# 	return np.tanh(x)

# def step(x):
#     return np.array(x > 0, dtype=np.int)

# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# def relu(x):
#     return np.maximum(0.0, x)


# # rectified linear function
# def rectified(x):
# 	return max(0.0, x)

# # # define a series of inputs
# series_in = [x for x in range(-6, 6)]
# # # calculate outputs for our inputs
# series_out1 = [relu(x) for x in series_in]


# s = np.arange(-6.0, 6.0, 0.1)
# y_sigmoid = sigmoid(s)
# y_tanh = tanh(s)
# y_relu = relu(s)
# # # line plot of raw inputs to rectified outputs
# #plt.plot(series_in, series_out1)
# plt.plot(series_in, series_out1, label='ReLU')

# # x = np.arange(-5.0, 5.0, 0.1)
# # y_step = step(x)
# # y_sigmoid = sigmoid(x)
# # y_relu = relu(x)
# # y_tanh = tanh(x)

# # #plt.plot(x, y_step, label='Step', color='k', lw=1, linestyle=None)
# # plt.plot(x, y_sigmoid, label='Sigmoid', color='k', lw=1, ls='--')
# #plt.plot(x, y_relu, label='ReLU', color='k', lw=1, linestyle='-.')
# # plt.plot(x, y_tanh, label='Tanh', color='k', lw=1, linestyle='solid')
# plt.xlim(-6.0, 6.0)
# plt.ylim(-1.5, 1.5)
# # plt.legend()
# plt.show()


