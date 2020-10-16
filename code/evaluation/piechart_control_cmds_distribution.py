from matplotlib import pyplot as plt
from matplotlib import style
import numpy as np
import csv
import pandas as pd
import seaborn as sn

style.use("ggplot")

slices = ['3491','513','96954']
#slices = ['11370','9080','79784']
labels = ['Acceleration', 'Braking', 'No Action']
colours = ['#D85C41', '#57AB27','#646567']
##
plt.pie(slices, labels=labels, colors=colours, shadow=False, wedgeprops={'linewidth': 0.5, 'edgecolor':'black'})

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