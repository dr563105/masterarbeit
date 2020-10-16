import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.gridspec import GridSpec


style.use("fast")

# gridspec inside gridspec


df_1_loss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/classification csv/binary/run-evaluate_ts15_ds1_Class1_200920-2341-tag-loss.csv')
df_1_accel = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/classification csv/binary/run-evaluate_ts15_ds1_Class1_200920-2341-tag-accel_loss.csv')
df_1_st = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/classification csv/binary/run-evaluate_ts15_ds1_Class1_200920-2341-tag-st_loss.csv')

df_2_val = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/classification csv/binary/run-evaluate_ts15_ds1_Class1_200920-2341-tag-val_loss.csv')
df_2_accel = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/classification csv/binary/run-evaluate_ts15_ds1_Class1_200920-2341-tag-val_accel_loss.csv')
df_2_st = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/classification csv/binary/run-evaluate_ts15_ds1_Class1_200920-2341-tag-val_st_loss.csv')


df_3_loss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/classification csv/binary/run-evaluate_ts15_ds3_Class1_200921-0312-tag-loss.csv')
df_3_accel = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/classification csv/binary/run-evaluate_ts15_ds3_Class1_200921-0312-tag-accel_loss.csv')
df_3_st = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/classification csv/binary/run-evaluate_ts15_ds3_Class1_200921-0312-tag-st_loss.csv')

df_4_val = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/classification csv/binary/run-evaluate_ts15_ds3_Class1_200921-0312-tag-val_loss.csv')
df_4_accel = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/classification csv/binary/run-evaluate_ts15_ds3_Class1_200921-0312-tag-val_accel_loss.csv')
df_4_st = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/classification csv/binary/run-evaluate_ts15_ds3_Class1_200921-0312-tag-val_st_loss.csv')

# f = plt.figure(figsize=(8,6))

# gs0 = gridspec.GridSpec(1, 2, figure=f)

# gs00 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs0[0])

# ax1 = f.add_subplot(gs00[:-1, :])
# ax1.plot(df_1_loss.Step,df_1_loss.Value)
# ax2 = f.add_subplot(gs00[-1, :-1])
# ax2.plot(df_1_accel.Step,df_1_accel.Value)
# ax3 = f.add_subplot(gs00[-1, -1])
# ax3.plot(df_1_st.Step,df_1_st.Value)

# # the following syntax does the same as the GridSpecFromSubplotSpec call above:
# gs01 = gs0[1].subgridspec(2, 2)

# ax4 = f.add_subplot(gs01[:-1, :])
# ax4.plot(df_2_val.Step,df_2_val.Value)
# ax5 = f.add_subplot(gs01[-1, :-1])
# ax5.plot(df_2_accel.Step,df_2_accel.Value)
# ax6 = f.add_subplot(gs01[-1, -1])
# ax6.plot(df_2_st.Step,df_2_st.Value)



def format_axes(fig):
    for i, ax in enumerate(fig.axes):
        #ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
        ax.grid()
        ax.tick_params(labelbottom=True, labelleft=True)

fig = plt.figure(figsize=(8,6))

gs = GridSpec(2, 2, figure=fig)
ax1 = fig.add_subplot(gs[0, :-1])
#ax1.plot(df_1_loss.Step,df_1_loss.Value, label='Dataset 1 Training Loss')
ax1.plot(df_2_val.Step+1,df_2_val.Value, label='Dataset 1')
ax1.plot(df_4_val.Step+1,df_4_val.Value, label='Dataset 3')
ax1.set_title('Overall Validation Loss')
ax1.set_ylabel('Loss')
ax1.legend()

ax2 = fig.add_subplot(gs[1, :-1])
#ax2.plot(df_1_accel.Step,df_1_accel.Value, label='Dataset 1 Training Acceleration Loss')
ax2.plot(df_2_accel.Step+1,df_2_accel.Value, label='Dataset 1')
ax2.plot(df_4_accel.Step+1,df_4_accel.Value, label='Dataset 3')
ax2.set_title('Acceleration Loss')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
#ax2.legend()

ax3 = fig.add_subplot(gs[1:, -1])
#ax3.plot(df_1_st.Step,df_1_st.Value, label='Dataset 1 Training Steering Loss')
ax3.plot(df_2_st.Step+1,df_2_st.Value, label='Dataset 1')
ax3.plot(df_4_st.Step+1,df_4_st.Value, label='Dataset 3')
ax3.set_title('Steering Loss')
ax3.set_xlabel('Epochs')
#ax3.legend()

#ax4 = fig.add_subplot(gs[-1, 0])
#ax5 = fig.add_subplot(gs[-1, -2])

#fig.suptitle("GridSpec")
format_axes(fig)
#plt.savefig("BinaryCross2.svg")
plt.show()

#plt.suptitle("GridSpec Inside GridSpec")
#format_axes(f)


# df_4_loss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/classification csv/binary/.csv')
# df_4_val = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/classification csv/binary/.csv')

# df_5_loss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/classification csv/binary/.csv')
# df_5_val = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/classification csv/binary/.csv')

#print(df_1_val['Value'])
#fig, axes = plt.subplots(1,1, figsize=(8,6))

# x1_loss = df_1_loss.Step
# y1_loss = df_1_loss.Value

# x1_val = df_1_val.Step
# y1_val = df_1_val.Value

# l1 = axes[0].plot(x1_loss,y1_loss, label='Dataset 1 Loss')
# l2 = axes[0].plot(x1_val,y1_val, label='Dataset 1 Val Loss')

# x2_loss = df_2_loss.Step
# y2_loss = df_2_loss.Value

# x2_val = df_2_val.Step
# y2_val = df_2_val.Value

# l3 = axes[1].plot(x2_loss,y2_loss, color='#D85C41', label='Dataset 3 Loss')
# l4 = axes[1].plot(x2_val,y2_val,color='#57AB27', label='Dataset 3 Val Loss')


# x3_loss = df_3_loss.Step
# y3_loss = df_3_loss.Value

# x3_val = df_3_val.Step
# y3_val = df_3_val.Value

# #axes[0].plot(x3_loss,y3_loss, label='Dataset 1 training Loss')
# #axes[0].plot(x3_val,y3_val, label='Dataset 1 Validation Loss')
# #plt.plot(x3_loss,y3_loss)
# #plt.plot(x3_val,y3_val)


# x4_loss = df_4_loss.Step
# y4_loss = df_4_loss.Value

# x4_val = df_4_val.Step
# y4_val = df_4_val.Value

# axes.plot(x4_loss,y4_loss, color='#D85C41', label='Dataset 3 Training Loss')
# axes.plot(x4_val,y4_val,color='#57AB27', label='Dataset 3 Validation Loss')
# #plt.plot(x4_loss,y4_loss)
# #plt.plot(x4_val,y4_val)

# x5_loss = df_5_loss.Step
# y5_loss = df_5_loss.Value

# x5_val = df_5_val.Step
# y5_val = df_5_val.Value

# axes.plot(x5_loss,y5_loss,  label='Dataset 3 Training Loss dense layers adjusted')
# axes.plot(x5_val,y5_val, label='Dataset 3 Validation Loss dense layers adjusted')
# #plt.plot(x5_loss,y5_loss)
# #plt.plot(x5_val,y5_val)
# axes.legend()
# #axes[1].legend()
# axes.grid()
# #axes[1].grid()
# axes.set_ylabel('Loss')
# axes.set_xlabel('Epochs')
# #axes[1].set_ylabel('Validation Loss')
# #axes[1].set_xlabel('Epochs')

# plt.tight_layout()
#plt.savefig("BinaryCross1.svg")
#plt.show()

# (l1, l3) ,('dataset 1','dataset 3'), loc='best'
# (l2, l4),('dataset 1','dataset 3'), loc='best'