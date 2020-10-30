import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.gridspec import GridSpec

style.use("fast")

def format_axes(fig):
    for i, ax in enumerate(fig.axes):
        #ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
        ax.grid()
        ax.tick_params(labelbottom=True, labelleft=True)

df_3_loss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/velocitycsv/run-evaluate_ts15_ds3_Class4_61_201010-1634-tag-loss.csv')
df_3_class_loss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/velocitycsv/run-evaluate_ts15_ds3_Class4_61_201010-1634-tag-classification_loss.csv')
df_3_st_loss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/velocitycsv/run-evaluate_ts15_ds3_Class4_61_201010-1634-tag-st_loss.csv')
df_3_vel_loss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/velocitycsv/run-evaluate_ts15_ds3_Class4_61_201010-1634-tag-velocity_loss.csv')

df_3_vloss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/velocitycsv/run-evaluate_ts15_ds3_Class4_61_201010-1634-tag-val_loss.csv')
df_3_class_vloss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/velocitycsv/run-evaluate_ts15_ds3_Class4_61_201010-1634-tag-val_classification_loss.csv')
df_3_st_vloss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/velocitycsv/run-evaluate_ts15_ds3_Class4_61_201010-1634-tag-val_st_loss.csv')
df_3_vel_vloss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/velocitycsv/run-evaluate_ts15_ds3_Class4_61_201010-1634-tag-val_velocity_loss.csv')

fig = plt.figure(figsize=(8,6))

gs = GridSpec(2, 2, figure=fig)
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(df_3_loss.Step+1,df_3_loss.Value, label='Dataset 3 4LSTMS Loss')
ax1.plot(df_3_vloss.Step+1,df_3_vloss.Value, label='Dataset 3 4LSTMS Loss')
ax1.set_ylabel('Loss')
ax1.set_xlabel('Epochs')
ax1.legend()

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(df_3_class_loss.Step+1,df_3_class_loss.Value, label='Dataset 3 Training Classification Loss')
ax2.plot(df_3_class_vloss.Step+1,df_3_class_vloss.Value, label='Dataset 3 Validation Classification Loss')
ax2.set_title('Classification Loss')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')

ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(df_3_st_loss.Step+1,df_3_st_loss.Value, label='Dataset 3 Training Steering Loss')
ax3.plot(df_3_st_vloss.Step+1,df_3_st_vloss.Value, label='Dataset 3 Validation Steering Loss')
ax3.set_title('Steering Loss')
ax3.set_xlabel('Epochs')
ax3.set_ylabel('Loss')

ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(df_3_vel_loss.Step+1,df_3_vel_loss.Value, label='Dataset 3 Training Velocity Loss')
ax4.plot(df_3_vel_vloss.Step+1,df_3_vel_vloss.Value, label='Dataset 3 Validation Velocity Loss')
ax4.set_title('Velocity Loss')
ax4.set_xlabel('Epochs')
ax4.set_ylabel('Loss')
plt.show()