import numpy as np
import matplotlib.pyplot as plt


# Datasets = ['Dataset 1', 'Dataset 2', 'Dataset 3']
# Morning = np.array([2, 2, 1])
# Afternoon = np.array([0, 1, 0])
# Lateevening = np.array([7, 6, 7])
# ind = [x for x, _ in enumerate(Datasets)]

# plt.bar(ind, Morning, width=0.8, label='Early Morning', color='gold', bottom=Afternoon+Lateevening)
# plt.bar(ind, Afternoon, width=0.8, label='Afternoon/ early evening', color='silver', bottom=Lateevening)
# plt.bar(ind, Lateevening, width=0.8, label='Late evening/ night', color='#CD853F')

# plt.xticks(ind, Datasets)
# plt.ylabel("Number of Collisions per 30s episode")
# plt.xlabel("Datasets")
# plt.legend(loc="upper right")
# plt.title("DS vs LC vs Collisions")

# plt.show()



# N = 3


# Morning = np.array([2, 2, 1])
# Afternoon = np.array([0, 1, 0])
# Lateevening = np.array([7, 6, 7])
# Datasets = ['Dataset 1', 'Dataset 2', 'Dataset 3']
# ind1 = [x for x, _ in enumerate(Datasets)]


# #data = [Morning, Afternoon, Lateevening]
# ind = np.arange(N) 
# width = 0.35       
# plt.bar(ind, Morning, width, label='Early Morning')
# plt.bar(ind + width, Afternoon, width,
#     label='Afternoon/ early evening')
# plt.bar(ind + (width*2), Lateevening, width,
#     label='Late evening/ night')

# plt.ylabel('Number of Collisions per 30s episode')
# plt.title('DS vs LC vs Collisions')

# plt.xticks(ind + width , ('Dataset 1', 'Dataset 2', 'Dataset 3'))
# plt.legend(loc='best')
# plt.show()


# # color_list = ['b', 'g', 'r']
# # gap = .8 / len(data)
# # for i, row in enumerate(data):
# #   X = np.arange(len(row))
# #   plt.bar(X + i * gap, row,
# #     width = gap,
# #     color = color_list[i % len(color_list)])
  
# # plt.xticks(ind, Datasets)

# # plt.ylabel('Number of Collisions per 30s episode')
# # plt.title('DS vs LC vs Collisions')
# # plt.legend(loc='best')
# # plt.show()



labels = ['Dataset 1', 'Dataset 3']
Morning = [2, 2, 1]
Afternoon = [4, 0.1]
Lateevening = [7, 6, 7]

x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots()
#rects1 = ax.bar(x - width/2, Morning, width, label='Early Morning')
rects2 = ax.bar(x + width/2, Afternoon, width, label='Afternoon/ early evening')
#rects3 = ax.bar(x + 1.5*width, Lateevening, width, label='Late evening/ night')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Number of Collisions per 30s episode')
ax.set_ylim([0, 8])
ax.set_title('Afternoon Tanh Activation - Datasets vs Collisions')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='best')


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


#autolabel(rects1)
#autolabel(rects2)
#autolabel(rects3)

fig.tight_layout()

plt.show()