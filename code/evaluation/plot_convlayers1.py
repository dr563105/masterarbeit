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


df_1_vloss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/convtweakscsv/run-evaluate_ts15_ds3_Class5_conv11_201011-2330-tag-val_loss.csv')
df_1_class_vloss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/convtweakscsv/run-evaluate_ts15_ds3_Class5_conv11_201011-2330-tag-val_classification_loss.csv')
df_1_st_vloss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/convtweakscsv/run-evaluate_ts15_ds3_Class5_conv11_201011-2330-tag-val_st_loss.csv')
df_1_vel_vloss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/convtweakscsv/run-evaluate_ts15_ds3_Class5_conv11_201011-2330-tag-val_velocity_loss.csv')

df_2_vloss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/convtweakscsv/run-evaluate_ts15_ds3_Class5_conv14_201012-1754-tag-val_loss.csv')
df_2_class_vloss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/convtweakscsv/run-evaluate_ts15_ds3_Class5_conv14_201012-1754-tag-val_classification_loss.csv')
df_2_st_vloss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/convtweakscsv/run-evaluate_ts15_ds3_Class5_conv14_201012-1754-tag-val_st_loss.csv')
df_2_vel_vloss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/convtweakscsv/run-evaluate_ts15_ds3_Class5_conv14_201012-1754-tag-val_velocity_loss.csv')

df_3_vloss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/convtweakscsv/run-evaluate_ts15_ds3_Class5_conv13_201012-1459-tag-val_loss.csv')
df_3_class_vloss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/convtweakscsv/run-evaluate_ts15_ds3_Class5_conv13_201012-1459-tag-val_classification_loss.csv')
df_3_st_vloss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/convtweakscsv/run-evaluate_ts15_ds3_Class5_conv13_201012-1459-tag-val_st_loss.csv')
df_3_vel_vloss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/convtweakscsv/run-evaluate_ts15_ds3_Class5_conv13_201012-1459-tag-val_velocity_loss.csv')

fig, ax = plt.subplots(2,2, figsize=(8,6))

#gs = GridSpec(2, 2, figure=fig)
#ax1 = fig.add_subplot(gs[0, 0])
#ax[0,0].plot(df_1_loss.Step+1,df_1_loss.Value, label='Dataset 3 3LSTMs Loss')
ax[0,0].plot(df_1_vloss.Step[:49]+1,df_1_vloss[:49].Value,color='#CC071E', label='a model')
#ax[0,0].plot(df_2_loss.Step+1,df_2_loss.Value, label='Dataset 3 2LSTMS Loss')
ax[0,0].plot(df_2_vloss.Step+1,df_2_vloss.Value,color='#57AB27', label='adjust feature maps')
#ax[0,0].plot(df_3_loss.Step+1,df_3_loss.Value, label='Dataset 3 4LSTMS Loss')
ax[0,0].plot(df_3_vloss.Step[:49]+1,df_3_vloss[:49].Value,color='#646567', label='adjust stride')
#ax[0,0].plot(df_4_vloss.Step[:49]+1,(df_4_vloss[:49].Value*10),color='#00549F', label='weightedx10')
#ax[0,0].plot(df_4_vloss.Step[:49]+1,(df_4_vloss[:49].Value*10),color='#00549F', label='weightedx10')
ax[0,0].set_title('Loss')
ax[0,0].set_ylabel('Loss')
#ax[0,0].set_xlabel('Epochs')
ax[0,0].legend()
ax[0,0].grid()



#ax3 = fig.add_subplot(gs[1, 0])
#ax[1,0].plot(df_1_class_loss.Step+1,df_1_class_loss.Value, label='Dataset 3 Training Classification Loss')
ax[0,1].plot(df_1_class_vloss[:49].Step+1,df_1_class_vloss[:49].Value,color='#CC071E', label='a model')
#ax[1,0].plot(df_2_class_loss.Step+1,df_2_class_loss.Value, label='Dataset 3 Training Classification Loss')
ax[0,1].plot(df_2_class_vloss.Step+1,df_2_class_vloss.Value,color='#57AB27', label='adjust feature maps')
#ax[1,0].plot(df_3_class_loss.Step+1,df_3_class_loss.Value, label='Dataset 3 Training Classification Loss')
ax[0,1].plot(df_3_class_vloss.Step[:49]+1,df_3_class_vloss[:49].Value,color='#646567', label='adjust stride')
#ax[1,0].plot(df_4_class_vloss[:49].Step+1,df_4_class_vloss[:49].Value,color='#00549F', label='weighted')
ax[0,1].set_title('Classification Loss')
#ax[0,1].set_xlabel('Epochs')
ax[0,1].set_ylabel('Loss')
ax[0,1].grid()
ax[0,1].legend()

#ax2 = fig.add_subplot(gs[0, 1])
#ax[1,0].plot(df_1_vel_loss.Step+1,df_1_vel_loss.Value, label='Dataset 3 Training Velocity Loss')
ax[1,0].plot(df_1_vel_vloss.Step[:49]+1,df_1_vel_vloss[:49].Value,color='#CC071E', label='a model')
#ax[1,0].plot(df_2_vel_loss.Step+1,df_2_vel_loss.Value, label='Dataset 3 Training Velocity Loss')
ax[1,0].plot(df_2_vel_vloss.Step+1,df_2_vel_vloss.Value,color='#57AB27', label='adjust feature maps')
#ax[1,0].plot(df_3_vel_loss.Step+1,df_3_vel_loss.Value, label='Dataset 3 Training Velocity Loss')
ax[1,0].plot(df_3_vel_vloss.Step[:49]+1,df_3_vel_vloss[:49].Value,color='#646567', label='adjust stride')
#ax[1,0].plot(df_4_vel_vloss[:49].Step+1,(df_4_vel_vloss[:49].Value*1000),color='#00549F', label='weighted/1000')
#ax[1,0].plot(df_4_vel_vloss[:49].Step+1,(df_4_vel_vloss[:49].Value*1000),color='#00549F', label='weightedx1000')
ax[1,0].set_title('Velocity Loss')
ax[1,0].set_xlabel('Epochs')
#ax[1,0].set_ylabel('Loss')
ax[1,0].grid()
ax[1,0].legend()
#ax4 = fig.add_subplot(gs[1, 1])
#ax[1,1].plot(df_1_st_loss.Step+1,df_1_st_loss.Value, label='Dataset 3 Training Steering Loss')
ax[1,1].plot(df_1_st_vloss.Step[:49]+1,df_1_st_vloss[:49].Value,color='#CC071E', label='a model')
#ax[1,1].plot(df_2_st_loss.Step+1,df_2_st_loss.Value, label='Dataset 3 Training Steering Loss')
ax[1,1].plot(df_2_st_vloss.Step+1,df_2_st_vloss.Value,color='#57AB27', label='adjust feature maps')
#ax[1,1].plot(df_3_st_loss.Step+1,df_3_st_loss.Value, label='Dataset 3 Training Steering Loss')
ax[1,1].plot(df_3_st_vloss.Step[:49]+1,df_3_st_vloss[:49].Value,color='#646567', label='adjust stride')
#ax[1,1].plot(df_4_st_vloss[:49].Step+1,(df_4_st_vloss[:49].Value/5),color='#00549F', label='weighted*5')
#ax[1,1].plot(df_4_st_vloss[:49].Step+1,(df_4_st_vloss[:49].Value/5),color='#00549F', label='weighted/5')
ax[1,1].set_title('Steering Loss')
ax[1,1].set_xlabel('Epochs')
#ax[1,1].set_ylabel('Loss')
ax[1,1].grid()
ax[1,1].legend()


plt.show()

