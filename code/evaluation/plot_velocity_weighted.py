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

# gridspec inside gridspec


df_1_vloss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/convtweakscsv/run-evaluate_ts15_ds3_Class5_conv12_201012-1311-tag-val_loss.csv')
df_1_class_vloss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/convtweakscsv/run-evaluate_ts15_ds3_Class5_conv12_201012-1311-tag-val_classification_loss.csv')
df_1_st_vloss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/convtweakscsv/run-evaluate_ts15_ds3_Class5_conv12_201012-1311-tag-val_st_loss.csv')
df_1_vel_vloss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/convtweakscsv/run-evaluate_ts15_ds3_Class5_conv12_201012-1311-tag-val_velocity_loss.csv')


# df_2_vloss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/convtweakscsv/run-evaluate_ts15_ds3_Class4_5_201003-1416-tag-val_loss.csv')
# df_2_class_vloss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/convtweakscsv/run-evaluate_ts15_ds3_Class4_5_201003-1416-tag-val_classification_loss.csv')
# df_2_st_vloss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/convtweakscsv/run-evaluate_ts15_ds3_Class4_5_201003-1416-tag-val_st_loss.csv')
# df_2_vel_vloss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/convtweakscsv/run-evaluate_ts15_ds3_Class4_5_201003-1416-tag-val_velocity_loss.csv')

# df_3_vloss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/convtweakscsv/run-evaluate_ts15_ds3_Class4_61_201010-1634-tag-val_loss.csv')
# df_3_class_vloss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/convtweakscsv/run-evaluate_ts15_ds3_Class4_61_201010-1634-tag-val_classification_loss.csv')
# df_3_st_vloss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/convtweakscsv/run-evaluate_ts15_ds3_Class4_61_201010-1634-tag-val_st_loss.csv')
# df_3_vel_vloss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/convtweakscsv/run-evaluate_ts15_ds3_Class4_61_201010-1634-tag-val_velocity_loss.csv')

fig, ax = plt.subplots(2,2, figsize=(8,6))

#gs = GridSpec(2, 2, figure=fig)
#ax1 = fig.add_subplot(gs[0, 0])

ax[0,0].plot(df_1_vloss.Step+1,df_1_vloss.Value,color='#CC071E', label='adjusted feature maps depth')
#ax[0,0].plot(df_2_vloss.Step+1,df_2_vloss.Value,color='#57AB27', label='b')
#ax[0,0].plot(df_3_vloss.Step[:49]+1,df_3_vloss[:49].Value,color='#646567', label='c')
ax[0,0].set_title('Loss')
ax[0,0].set_ylabel('Loss')
#ax[0,0].set_xlabel('Epochs')
ax[0,0].legend()
ax[0,0].grid()

#ax3 = fig.add_subplot(gs[1, 0])
ax[0,1].plot(df_1_class_vloss.Step+1,df_1_class_vloss.Value,color='#CC071E', label='Classification Loss')
#ax[1,0].plot(df_2_class_vloss.Step+1,df_2_class_vloss.Value,color='#57AB27', label='Dataset 3 Validation Classification Loss')
#ax[1,0].plot(df_3_class_vloss.Step[:49]+1,df_3_class_vloss[:49].Value,color='#646567', label='Dataset 3 Validation Classification Loss')
ax[0,1].set_title('Classification Loss')
#ax[1,0].set_xlabel('Epochs')
#ax[0,1].set_ylabel('Loss')
ax[0,1].grid()

#ax4 = fig.add_subplot(gs[1, 1])

ax[1,0].plot(df_1_st_vloss.Step+1,df_1_st_vloss.Value,color='#CC071E', label='Steering Loss')
#ax[1,1].plot(df_2_st_vloss.Step+1,df_2_st_vloss.Value,color='#57AB27', label='Dataset 3 Validation Steering Loss')
#ax[1,1].plot(df_3_st_vloss.Step[:49]+1,df_3_st_vloss[:49].Value,color='#646567', label='Dataset 3 Validation Steering Loss')
ax[1,0].set_title('Steering Loss')
ax[1,0].set_xlabel('Epochs')
ax[1,0].set_ylabel('Loss')
ax[1,0].grid()

#ax2 = fig.add_subplot(gs[0, 1])

ax[1,1].plot(df_1_vel_vloss.Step+1,df_1_vel_vloss.Value,color='#CC071E', label='Velocity Loss')
#ax[0,1].plot(df_2_vel_vloss.Step+1,df_2_vel_vloss.Value,color='#57AB27', label='Dataset 3 Validation Velocity Loss')
#ax[0,1].plot(df_3_vel_vloss.Step[:49]+1,df_3_vel_vloss[:49].Value,color='#646567', label='Dataset 3 Validation Velocity Loss')
ax[1,1].set_title('Velocity Loss')
ax[1,1].set_xlabel('Epochs')
#ax[0,1].set_ylabel('Loss')
ax[1,1].grid()





plt.show()

24
36
48
64
64
24, (5, 5), strides=(2, 2), 
36, (5, 5), strides=(2, 2), 
48, (5, 5), strides=(2, 2), 
64, (5, 5), strides=(2, 2), 
80, (3, 3), strides=(1, 1), 
96, (3, 3), strides=(1, 1), 
112, (3, 3), strides=(1, 1), 
128, (3, 3), strides=(1, 1), 
144, (3, 3), strides=(1, 1), 
160, (3, 3), strides=(1, 1),
176, (3, 3), strides=(1, 1),