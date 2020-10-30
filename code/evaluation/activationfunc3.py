
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

fig = plt.figure(figsize=(6,4))
ax0 = fig.add_subplot(221)
ax1 = fig.add_subplot(222)
ax2 = fig.add_subplot(223)
ax0.vlines(0,0,6,linewidth=0.5)
ax1.vlines(0,0,1,linewidth=0.5)
ax2.vlines(0,-1.0,1.0,linewidth=0.5)

sigma_fn = np.vectorize(lambda values: 1/(1+np.exp(-values)))
sigma = sigma_fn(values)
#ax1.set_ylim([0.0, 1.0])
#ax1.set_xlim([-5,5])
#ax1.spines['left'].set_position(('center' ))
#ax1.spines['bottom'].set_position('center')
#ax0.spines['right'].set_color('none')
#ax0.spines['top'].set_color('none')
#ax0.xaxis.set_ticks_position('bottom')
#ax0.yaxis.set_ticks_position('left')# Create and show plot

#ax1.spines['right'].set_color('none')
#ax1.spines['top'].set_color('none')
#ax1.xaxis.set_ticks_position('bottom')
#ax1.yaxis.set_ticks_position('left')# Create and show plot

#ax2.spines['left'].set_position(('center' ))
#ax2.spines['bottom'].set_position('center')
#ax2.spines['right'].set_color('none')
#ax2.spines['top'].set_color('none')
#ax2.xaxis.set_ticks_position('bottom')
#ax2.yaxis.set_ticks_position('left')# Create and show plot


ax0.plot(values, relu(values), linewidth=2.5, color ='#B65256')
ax2.plot(values, tanh(values), linewidth=2.5, color ='#F6A800')
ax1.plot(values, sigma, linewidth=2.5, color ='#57AB27')
plt.tight_layout()
#plt.savefig("tanh.jpg")
plt.show()

