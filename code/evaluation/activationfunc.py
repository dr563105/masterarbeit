import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    s=1/(1+np.exp(-x))
    ds=s*(1-s)  
    return s,ds

def tanh(x):
    t=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    dt=1-t**2
    return t,dt

def relu(x):
	zero = np.zeros(len(z))
	return np.max([zero, z], axis=0)
	#return np.maximum(x, 0)

def softmax(X):
    expo = np.exp(X)
    expo_sum = np.sum(np.exp(X))
    return expo/expo_sum


z=np.arange(-2,2,0.1) #np.arange(-4,4,0.01)
tanh(z)
#x=np.arange(-6,6,0.01)
sigmoid(z)
relu(z)

# Setup centered axes
#fig, ax = plt.subplots(figsize=(5, 5))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')# Create and show plot
ax.set_ylim([0.0, 1.0])
ax.set_xlim([-2,2])
ax.plot(z,relu(z), color="#BDCD00", linewidth=3, label="relu")
#ax.plot(z,sigmoid(z), color="#A11035", linewidth=3, label="sigmoid")
#ax.plot(z,tanh(z), color="#307EC7", linewidth=3, label="tanh")
ax.legend(loc="lower right", frameon=True)

plt.show()
#plt.savefig('act.png')