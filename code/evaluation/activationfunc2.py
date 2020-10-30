
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0.0, x)

def tanh(x):
	return np.tanh(x)

tanh = np.vectorize(tanh) #vectorize function
values=np.linspace(-6, 6) #generate values between -10 and 10
fig = plt.figure()
ax = fig.add_subplot(221)

#spine placement data centered
ax.spines['left'].set_position(('center' ))
#ax.spines['bottom'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')# Create and show plot



ax.plot(values, tanh(values), linewidth=2.5, color ='#B65256')
ax.plot(values, relu(values), linewidth=2.5, color ='#B65256')
#plt.savefig("tanh.jpg")
plt.show()


# import numpy as np
# import matplotlib.pyplot as plt

# z = np.arange(-5, 5, .1)
# t = np.(tanh(z)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(z, t, linewidth=3)
# ax.set_ylim([-1.0, 1.0])
# ax.set_xlim([-5,5])
# #ax.grid(True)
# ax.set_xlabel('z')
# ax.set_title('tanh function')

# plt.show()