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


df_1_loss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/lstmvsnonlstm/run-evaluate_ds3_nolstm_200921-1230-tag-loss.csv')
df_1_val = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/lstmvsnonlstm/run-evaluate_ds3_nolstm_200921-1230-tag-val_loss.csv')

df_2_loss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/lstmvsnonlstm/run-evaluate_ts15_ds3_2_200919-1528-tag-loss.csv')
df_2_val = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/lstmvsnonlstm/run-evaluate_ts15_ds3_2_200919-1528-tag-val_loss.csv')

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,6))

#gs = GridSpec(2, 2, figure=fig)
#ax1 = fig.add_subplot(gs[0, :])
ax1.plot(df_1_loss.Step+1,df_1_loss.Value, label='Dataset 3 Training Loss')
ax1.plot(df_1_val.Step+1,df_1_val.Value, label='Dataset 3 Validation Loss')
ax1.set_title('Without LSTM')
ax1.set_ylabel('Loss')
ax1.legend()

#ax2 = fig.add_subplot(gs[1, :-1])
ax2.plot(df_2_loss.Step+1,(df_2_loss.Value), label='Dataset 3 Training Loss')
ax2.plot(df_2_val.Step+1,(df_2_val.Value), label='Dataset 3 Validation Loss')
ax2.set_title('With LSTM')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')

# ax3 = fig.add_subplot(gs[1:, -1])
# ax3.plot(df_1_st.Step,df_1_st.Value, label='Dataset 3 Training Steering Loss')
# ax3.plot(df_2_st.Step,df_2_st.Value, label='Dataset 3 Validation Steering Loss')
# ax3.set_title('Steering Loss')
# ax3.set_xlabel('Epochs')

format_axes(fig)
#plt.savefig("categoricalcrossds32nn.svg")
plt.show()

