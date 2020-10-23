from matplotlib import pyplot as plt
from matplotlib import style
import numpy as np
import csv
import pandas as pd
import seaborn as sn

#style.use("ggplot")

distribution = {'labels':['Acceleration', 'Braking', 'No Action'],
             'slices':[3491,513, 96954]}

df = pd.DataFrame(distribution, columns=['labels','slices'])

slices = ['3491','513','96954']
#slices = ['11370','9080','79784']
labels = ['Acceleration', 'Braking', 'No Action']
colours = ['#D85C41', '#57AB27','#646567']
my_explode = (0.1, 0.1, 0)
##
plt.figure()
plt.pie(df['slices'],  labels = df['labels'],autopct='%1.1f%%', colors=colours, shadow=True, wedgeprops={'linewidth': 0.5, 'edgecolor':'black'}, startangle = 30, explode = my_explode)

plt.title('Control Commands distribution in dataset 3')
plt.tight_layout()
#plt.savefig('dataset3_control_cmds.png')
plt.show()


# Dataset 1
# slices = ['3491','513','96954']
# labels = ['Acceleration', 'Braking', 'No Action']

# Datset 3
# slices = ['11370','9080','79784']
# labels = ['Acceleration', 'Braking', 'No Action']