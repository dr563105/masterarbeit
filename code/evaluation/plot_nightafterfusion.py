import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

#style.use("seaborn")

N = 1
notraffic = (2)
traffic = (3)

ind = np.arange(N) 
width = 0.35       
plt.bar(ind, notraffic, width,color='#646567', label='Without Traffic')
plt.bar(ind + width, traffic, width,color='#D85C41',
   label='With Traffic')


plt.ylabel('Number of Collisions per 30s episode')
plt.ylim([0, 5])
#plt.title('Night - Dataset 3, RGB-G and Seg fusion - Traffic vs Collisions')

plt.xlabel('Dataset 3')
plt.xticks(ind + width / 2, (''))
plt.legend(loc='best')
plt.show()