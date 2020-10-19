from matplotlib import pyplot as plt
from matplotlib import style
import numpy as np
import csv
import pandas as pd
import seaborn as sn

#df  = pd.read_csv("/home/kaladin/Documents/arbeit/real/code/evaluation/training_data_ds1.csv")
#print(df)

#style.use("fivethirtyeight")
radar = []
images=[]
labels=[]

with open('/home/kaladin/Documents/arbeit/real/code/evaluation/training_data_ds3.csv', 'r') as  csvfile:
	lines = csvfile.readlines()
	reader = csv.reader(lines, delimiter=',')
	for line in reader:
		images.append(line[0].split()) #changes a list of strings to string list
		#value = [(y) for item in line[1][1:-1].split(",") for y in item]
		value = [item for item in line[1][1:-1].split(",")]
		labels.append(value)
		#[[float(y) for y in x] for x in l]

[item for sublist in [[item] if type(item) is not list else item for item in list1] for item in sublist]

def flatten(foo):
    for x in foo:
        if hasattr(x, '__iter__') and not isinstance(x, str):
            for y in flatten(x):
                yield y
        else:
            yield x

images = np.array(images)
labels = np.array(labels)
print((labels[73]))
#print(flatten_matrix)
#flatten_matrix = [val for sublist in value for val in sublist] 
		#radarvalue = [(item) for item in value[7]]
		#value1 = [float(item) for item in value[0:6]]
		#value[1], value[2] = value[2], value[1]
		#value=value[1]

# data_size = 10000#images.shape[0]
# print('Total data size:', data_size)

# indices = np.arange(data_size)
# train_size = int(round(data_size * 1))
# train_idx, test_idx = indices[:train_size], indices[train_size:]

# X_train = images[train_idx, :]
# Y_train = labels[train_idx, :]
# X_test = images[test_idx, :]
# Y_test = labels[test_idx, :]

# if X_train.shape[0] > 0:
# 	with open( 'radartry.txt', 'w+') as f:
# 		for i in range(len(X_train)):
# 			f.write('{}|{}\n'.format(X_train[i][0], list(Y_train[i][:])))

#counts = []
#fig, axes = plt.subplots(1,1, figsize=(8,6))
#for count, index in enumerate(labels[:10000]):
#axes.bar(count+1, index)
#plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt
# df  = pd.read_csv("data.csv")
# df.plot()  # plots all columns against index
# df.plot(kind='scatter',x='x',y='y') # scatter plot
# df.plot(kind='density')  # estimate density function
# df.plot(kind='hist')  # histogram

