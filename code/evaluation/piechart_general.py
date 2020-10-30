from matplotlib import pyplot as plt
from matplotlib import style
import numpy as np
import csv
import pandas as pd
import seaborn as sn

#style.use("fivethiryeight")

slices = ['109800','103805','269376']
labels = ['Dataset1', 'Dataset2','Dataset3']
colours = ['#D85C41', '#57AB27','#407FB7']

plt.pie(slices, labels=labels, colors=colours, shadow=False, wedgeprops={'linewidth': 0.5, 'edgecolor':'black'})

plt.title('Dataset distribution')
plt.tight_layout()
#plt.savefig('datasets_general.png')
plt.show()


# Dataset 1
# slices = ['3491','513','96954']
# labels = ['Acceleration', 'Braking', 'No Action']

# Datset 3
# slices = ['11370','9080','79784']
# labels = ['Acceleration', 'Braking', 'No Action']