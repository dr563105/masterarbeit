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


df_1_basic_loss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/classification csv/categorical/run-evaluate_ts15_ds3_Class2_200921-1004-tag-st_loss.csv')
df_1_dense_loss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/classification csv/categorical/dense/run-evaluate_ts15_ds3_Class2_1_200921-1712-tag-st_loss.csv')
df_1_lstm_loss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/classification csv/categorical/2lstm/run-evaluate_ts15_ds3_Class2_2_200921-2052(2LSTMs split))-tag-st_loss.csv')
df_1_2nn_loss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/classification csv/categorical/2nn/run-evaluate_ts15_ds3_Class3_200922-2159(2NNs)-tag-st_loss.csv')


df_2_basic_vloss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/classification csv/categorical/run-evaluate_ts15_ds3_Class2_200921-1004-tag-val_st_loss.csv')
df_2_dense_vloss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/classification csv/categorical/dense/run-evaluate_ts15_ds3_Class2_1_200921-1712-tag-val_st_loss.csv')
df_2_lstm_vloss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/classification csv/categorical/2lstm/run-evaluate_ts15_ds3_Class2_2_200921-2052(2LSTMs split))-tag-val_st_loss.csv')
df_2_2nn_vloss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/classification csv/categorical/2nn/run-evaluate_ts15_ds3_Class3_200922-2159(2NNs)-tag-val_st_loss.csv')


#categoricalcrossds3steeringCompare

#fig = plt.figure(figsize=(8,6))

fig, ax1 = plt.subplots(1,1, figsize=(8,6))#gs = GridSpec(2, 2, figure=fig)
#ax[0,0] = fig.add_subplot(gs[1, :-1])
# ax[0].plot(df_1_basic_loss.Step+1,df_1_basic_loss.Value, label='Basic model ')
# ax[0].plot(df_1_dense_loss.Step+1,df_1_dense_loss.Value, label='Separte dense layers')
# ax[0].plot(df_1_lstm_loss.Step+1,df_1_lstm_loss.Value, label='Separate LSTM layers')
# ax[0].plot(df_1_2nn_loss.Step+1,df_1_2nn_loss.Value, label='Separate NNs')
# ax[0].set_title('Dataset 3 Training Steering loss')
# ax[0].set_ylabel('Loss')
# ax[0].set_xlabel('Epochs')
# ax[0].legend()
# ax[0].grid()

#ax2 = fig.add_subplot(gs[1:, -1])
ax1.plot(df_2_basic_vloss.Step+1,df_2_basic_vloss.Value,color='#57AB27', label='Basic model')
ax1.plot(df_2_dense_vloss.Step+1,df_2_dense_vloss.Value, color='#006165',label='Separte dense layers')
ax1.plot(df_2_lstm_vloss.Step+1,df_2_lstm_vloss.Value,color ='#000000', label='Separate LSTM layers')
ax1.plot(df_2_2nn_vloss.Step+1,df_2_2nn_vloss.Value,color='#CC071E', label='Separate NNs')
ax1.set_title('Comparison of steering losses')
ax1.set_xlabel('Epochs')
ax1.legend()
ax1.grid()
#Dataset 3 Validation Steering loss -




#ax3 = fig.add_subplot(gs[1, :-1])
# ax[1,0].plot(df_1_accel.Step,df_1_accel.Value, label='Dataset 3 Training Classification Loss')
# ax[1,0].plot(df_2_accel.Step,df_2_accel.Value, label='Dataset 3 Validation Classification Loss')
# #ax[1,0].set_title('Classification Loss')
# ax[1,0].set_xlabel('Epochs')
# ax[1,0].set_ylabel('Loss')

# #ax4 = fig.add_subplot(gs[1:, -1])
# ax[1,1].plot(df_1_st.Step,df_1_st.Value, label='Dataset 3 Training Steering Loss')
# ax[1,1].plot(df_2_st.Step,df_2_st.Value, label='Dataset 3 Validation Steering Loss')
# #ax[1,1].set_title('Steering Loss')
# ax[1,1].set_xlabel('Epochs')

#format_axes(fig)
#plt.savefig("categoricalcrossds3steeringCompare.svg")
plt.show()

