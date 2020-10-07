import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

style.use("fast")


df_1_loss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/regression csv/run-evaluate_ts15_ds1_lstm100_200917-1903-tag-loss.csv')

df_1_val = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/regression csv/run-evaluate_ts15_ds1_lstm100_200917-1903-tag-val_loss.csv')

df_2_loss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/regression csv/run-evaluate_ts15_ds3_2_200919-1528-tag-loss.csv')
df_2_val = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/regression csv/run-evaluate_ts15_ds3_2_200919-1528-tag-val_loss.csv')

df_3_loss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/regression csv/run-evaluate_ts15_ds1_lstm100Sig1_200919-1200-tag-loss.csv')
df_3_val = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/regression csv/run-evaluate_ts15_ds1_lstm100Sig1_200919-1200-tag-val_loss.csv')

df_4_loss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/regression csv/run-evaluate_ts15_ds3_sig1_200919-2204-tag-loss.csv')
df_4_val = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/regression csv/run-evaluate_ts15_ds3_sig1_200919-2204-tag-val_loss.csv')

df_5_loss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/regression csv/run-evaluate_ts15_ds3_sig2_200920-2106-tag-loss.csv')
df_5_val = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/regression csv/run-evaluate_ts15_ds3_sig2_200920-2106-tag-val_loss.csv')

#print(df_1_val['Value'])
fig, axes = plt.subplots(1,1, figsize=(8,6))

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


x3_loss = df_3_loss.Step
y3_loss = df_3_loss.Value

x3_val = df_3_val.Step
y3_val = df_3_val.Value

#axes[0].plot(x3_loss,y3_loss, label='Dataset 1 training Loss')
#axes[0].plot(x3_val,y3_val, label='Dataset 1 Validation Loss')
#plt.plot(x3_loss,y3_loss)
#plt.plot(x3_val,y3_val)


x4_loss = df_4_loss.Step
y4_loss = df_4_loss.Value

x4_val = df_4_val.Step
y4_val = df_4_val.Value

axes.plot(x4_loss,y4_loss, color='#D85C41', label='Dataset 3 Training Loss')
axes.plot(x4_val,y4_val,color='#57AB27', label='Dataset 3 Validation Loss')
#plt.plot(x4_loss,y4_loss)
#plt.plot(x4_val,y4_val)

x5_loss = df_5_loss.Step
y5_loss = df_5_loss.Value

x5_val = df_5_val.Step
y5_val = df_5_val.Value

axes.plot(x5_loss,y5_loss,  label='Dataset 3 Training Loss dense layers adjusted')
axes.plot(x5_val,y5_val, label='Dataset 3 Validation Loss dense layers adjusted')
#plt.plot(x5_loss,y5_loss)
#plt.plot(x5_val,y5_val)
axes.legend()
#axes[1].legend()
axes.grid()
#axes[1].grid()
axes.set_ylabel('Loss')
axes.set_xlabel('Epochs')
#axes[1].set_ylabel('Validation Loss')
#axes[1].set_xlabel('Epochs')

plt.tight_layout()
#plt.savefig("regressionModelsSigActivation.svg")
plt.show()

# (l1, l3) ,('dataset 1','dataset 3'), loc='best'
# (l2, l4),('dataset 1','dataset 3'), loc='best'