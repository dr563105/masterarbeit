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


df_1_loss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/classification csv/binary/run-evaluate_ts15_ds3_Class1_200921-0312-tag-loss.csv')
df_1_accel = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/classification csv/binary/run-evaluate_ts15_ds3_Class1_200921-0312-tag-accel_loss.csv')
df_1_st = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/classification csv/binary/run-evaluate_ts15_ds3_Class1_200921-0312-tag-st_loss.csv')

df_2_val = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/classification csv/binary/run-evaluate_ts15_ds3_Class1_200921-0312-tag-val_loss.csv')
df_2_accel = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/classification csv/binary/run-evaluate_ts15_ds3_Class1_200921-0312-tag-val_accel_loss.csv')
df_2_st = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/classification csv/binary/run-evaluate_ts15_ds3_Class1_200921-0312-tag-val_st_loss.csv')




fig = plt.figure(figsize=(8,6))

gs = GridSpec(2, 2, figure=fig)
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(df_1_loss.Step,df_1_loss.Value, label='Dataset 3 Training Loss')
ax1.plot(df_2_val.Step,df_2_val.Value, label='Dataset 3 Validation Loss')
ax1.set_ylabel('Loss')
ax1.legend()

ax2 = fig.add_subplot(gs[1, :-1])
ax2.plot(df_1_accel.Step,df_1_accel.Value, label='Dataset 3 Training Acceleration Loss')
ax2.plot(df_2_accel.Step,df_2_accel.Value, label='Dataset 3 Validation Acceleration Loss')
ax2.set_title('Acceleration Loss')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')

ax3 = fig.add_subplot(gs[1:, -1])
ax3.plot(df_1_st.Step,df_1_st.Value, label='Dataset 3 Training Steering Loss')
ax3.plot(df_2_st.Step,df_2_st.Value, label='Dataset 3 Validation Steering Loss')
ax3.set_title('Steering Loss')
ax3.set_xlabel('Epochs')

format_axes(fig)
plt.savefig("BinaryCross2ds3.svg")
plt.show()

