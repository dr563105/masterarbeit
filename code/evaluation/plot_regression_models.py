import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

#style.use("fast")


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
#regressionModelsTanhActivation
#print(df_1_val['Value'])
fig, axes = plt.subplots(1,2, figsize=(8,6))

x1_loss = df_1_loss.Step
y1_loss = df_1_loss.Value

x1_val = df_1_val.Step
y1_val = df_1_val.Value

l1 = axes[0].plot(df_1_loss.Step+1,df_1_loss.Value,color='#006165', label='Dataset 1 Training Loss')
l2 = axes[0].plot(df_1_val.Step+1,df_1_val.Value, color='#7A6FAC',label='Dataset 1 Validation Loss')

x2_loss = df_2_loss.Step
y2_loss = df_2_loss.Value

x2_val = df_2_val.Step
y2_val = df_2_val.Value

l3 = axes[1].plot(df_2_loss.Step+1,df_2_loss.Value, color='#57AB27', label='Dataset 3 Training Loss')
l4 = axes[1].plot(df_2_val.Step+1,df_2_val.Value,color='#CC071E', label='Dataset 3 Validation Loss')


x3_loss = df_3_loss.Step
y3_loss = df_3_loss.Value

x3_val = df_3_val.Step
y3_val = df_3_val.Value

#plt.plot(x3_loss,y3_loss)
#plt.plot(x3_val,y3_val)


x4_loss = df_4_loss.Step
y4_loss = df_4_loss.Value

x4_val = df_4_val.Step
y4_val = df_4_val.Value

#plt.plot(x4_loss,y4_loss)
#plt.plot(x4_val,y4_val)

x5_loss = df_5_loss.Step
y5_loss = df_5_loss.Value

x5_val = df_5_val.Step
y5_val = df_5_val.Value

#plt.plot(x5_loss,y5_loss)
#plt.plot(x5_val,y5_val)
axes[0].legend()
axes[1].legend()
axes[0].grid()
axes[1].grid()
axes[0].set_ylabel('Loss')
axes[0].set_xlabel('Epochs')
#axes[1].set_ylabel('Validation Loss')
axes[1].set_xlabel('Epochs')

plt.tight_layout()
#plt.savefig("regressionModelsTanhActivation.svg")
plt.show()

# (l1, l3) ,('dataset 1','dataset 3'), loc='best'
# (l2, l4),('dataset 1','dataset 3'), loc='best'