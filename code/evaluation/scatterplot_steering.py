from matplotlib import pyplot as plt
from matplotlib import style
import numpy as np
import csv
import pandas as pd
import seaborn as sn

#df  = pd.read_csv("/home/kaladin/Documents/arbeit/real/code/evaluation/training_data_ds1.csv")
#print(df)

#style.use("fivethirtyeight")
images=[]
labels=[]
with open('/home/kaladin/Documents/arbeit/real/code/evaluation/training_data_ds1.csv', 'r') as  csvfile:
	lines = csvfile.readlines()
	reader = csv.reader(lines, delimiter=',')
	for line in reader:
		images.append(line[0].split()) #changes a list of strings to string list
		value = [float(item) for item in line[1][1:-1].split(",")]
		value[1], value[2] = value[2], value[1]
		value=value[1]
		labels.append(value)
counts = []
fig, axes = plt.subplots(1,1, figsize=(8,6))
for count, index in enumerate(labels[:10000]):
	axes.bar(count+1, index)
plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt
# df  = pd.read_csv("data.csv")
# df.plot()  # plots all columns against index
# df.plot(kind='scatter',x='x',y='y') # scatter plot
# df.plot(kind='density')  # estimate density function
# df.plot(kind='hist')  # histogram

