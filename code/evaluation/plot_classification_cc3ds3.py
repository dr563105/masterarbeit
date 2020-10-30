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


df_1_loss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/classification csv/categorical/2lstm/run-evaluate_ts15_ds3_Class2_2_200921-2052(2LSTMs split))-tag-loss.csv')
df_1_accel = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/classification csv/categorical/2lstm/run-evaluate_ts15_ds3_Class2_2_200921-2052(2LSTMs split))-tag-classification_loss.csv')
df_1_st = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/classification csv/categorical/2lstm/run-evaluate_ts15_ds3_Class2_2_200921-2052(2LSTMs split))-tag-st_loss.csv')

df_2_val = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/classification csv/categorical/2lstm/run-evaluate_ts15_ds3_Class2_2_200921-2052(2LSTMs split))-tag-val_loss.csv')
df_2_accel = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/classification csv/categorical/2lstm/run-evaluate_ts15_ds3_Class2_2_200921-2052(2LSTMs split))-tag-val_classification_loss.csv')
df_2_st = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/classification csv/categorical/2lstm/run-evaluate_ts15_ds3_Class2_2_200921-2052(2LSTMs split))-tag-val_st_loss.csv')




fig = plt.figure(figsize=(8,6))

gs = GridSpec(2, 2, figure=fig)
ax1 = fig.add_subplot(gs[0, :-1])
ax1.plot(df_1_loss.Step+1,df_1_loss.Value,color='#57AB27', label='Dataset 3 Training Loss')
ax1.plot(df_2_val.Step+1,df_2_val.Value,  color='#CC071E', label='Dataset 3 Validation Loss')
ax1.set_title('Overall Loss')
ax1.set_ylabel('Loss')
ax1.legend()

ax2 = fig.add_subplot(gs[1, :-1])
ax2.plot(df_1_accel.Step+1,df_1_accel.Value,color='#57AB27', label='Training Classification Loss')
ax2.plot(df_2_accel.Step+1,df_2_accel.Value, color='#CC071E',label='Validation Classification Loss')
ax2.set_title('Classification Loss')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
#ax2.legend()

ax3 = fig.add_subplot(gs[1:, -1])
ax3.plot(df_1_st.Step+1,df_1_st.Value, color='#57AB27',label='Training Steering Loss')
ax3.plot(df_2_st.Step+1,df_2_st.Value, color='#CC071E',label='Validation Steering Loss')
ax3.set_title('Steering Loss')
ax3.set_xlabel('Epochs')
#ax3.legend()

format_axes(fig)
#plt.savefig("categoricalcrossds3lstm.svg")
plt.show()

