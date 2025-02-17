import numpy as np
import matplotlib.pyplot as plt

labels = ['Dataset 1', 'Dataset 2', 'Dataset 3']
Morning = [0.286,	0.286,	0.143]
Afternoon = [0.01, 0.143, 0.01]
Lateevening = [1,	0.85, 1]


x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, Morning, width, label='Early Morning')
rects2 = ax.bar(x + width/2, Afternoon, width, label='Afternoon/ early evening')
rects3 = ax.bar(x + 1.5*width, Lateevening, width, label='Late evening/ night')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Percentage of Collisions per 30s episode')
ax.set_ylim([0, 1.1])
ax.set_title('Traffic ON - Datasets vs Light Conditions vs Average Collisions in %')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='best')


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

fig.tight_layout()

plt.show()