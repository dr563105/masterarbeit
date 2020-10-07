import numpy as np
import matplotlib.pyplot as plt

N = 3
notraffic = (0.1, 1, 0.1)
traffic = (1, 2, 1)

ind = np.arange(N) 
width = 0.35       
plt.bar(ind, notraffic, width, label='Without Traffic')
plt.bar(ind + width, traffic, width,
    label='With Traffic')


plt.ylabel('Number of Collisions per 30s episode')
plt.ylim([0, 5])
plt.title('Afternoon vs Traffic vs Collisions')

plt.xticks(ind + width / 2, ('Dataset 1', 'Dataset 2', 'Dataset 3'))
plt.legend(loc='best')
plt.show()