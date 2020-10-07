import numpy as np
import matplotlib.pyplot as plt

labels = ['Dataset 1', 'Dataset 2', 'Dataset 3']
Morning = [28.57,	28.57,	14.29]
Afternoon = [1, 14.29, 1]
Lateevening = [100,	85.71, 100]


x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, Morning, width, label='Early Morning')
rects2 = ax.bar(x + width/2, Afternoon, width, label='Afternoon/ early evening')
rects3 = ax.bar(x + 1.5*width, Lateevening, width, label='Late evening/ night')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Percentage of Collisions per 30s episode')
ax.set_ylim([0, 110])
ax.set_title('Datasets vs Light Conditions vs Average Collisions in %')
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