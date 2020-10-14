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

#only rgb
df_1_vloss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/convtweakscsv/run-evaluate_ts15_ds3_Class5_conv11_201011-2330-tag-val_loss.csv')
df_1_class_vloss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/convtweakscsv/run-evaluate_ts15_ds3_Class5_conv11_201011-2330-tag-val_classification_loss.csv')
df_1_st_vloss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/convtweakscsv/run-evaluate_ts15_ds3_Class5_conv11_201011-2330-tag-val_st_loss.csv')
df_1_vel_vloss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/convtweakscsv/run-evaluate_ts15_ds3_Class5_conv11_201011-2330-tag-val_velocity_loss.csv')

#only depth
df_2_vloss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/depthcsv/run-evaluate_ts15_ds3_Class6_depth1_201013-1215-tag-val_loss.csv')
df_2_class_vloss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/depthcsv/run-evaluate_ts15_ds3_Class6_depth1_201013-1215-tag-val_classification_loss.csv')
df_2_st_vloss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/depthcsv/run-evaluate_ts15_ds3_Class6_depth1_201013-1215-tag-val_st_loss.csv')
df_2_vel_vloss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/depthcsv/run-evaluate_ts15_ds3_Class6_depth1_201013-1215-tag-val_velocity_loss.csv')

#rgb+depth(EF)
df_3_vloss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/depthcsv/run-evaluate_ts15_ds3_Class6_cdepth1_201013-1704-tag-val_loss.csv')
df_3_class_vloss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/depthcsv/run-evaluate_ts15_ds3_Class6_cdepth1_201013-1704-tag-val_classification_loss.csv')
df_3_st_vloss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/depthcsv/run-evaluate_ts15_ds3_Class6_cdepth1_201013-1704-tag-val_st_loss.csv')
df_3_vel_vloss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/depthcsv/run-evaluate_ts15_ds3_Class6_cdepth1_201013-1704-tag-val_velocity_loss.csv')

#rgb+seg(EF)
df_4_vloss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/segcsv/run-evaluate_ts15_ds3_Class6_cseg1_201013-1932-tag-val_loss.csv')
df_4_class_vloss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/segcsv/run-evaluate_ts15_ds3_Class6_cseg1_201013-1932-tag-val_classification_loss.csv')
df_4_st_vloss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/segcsv/run-evaluate_ts15_ds3_Class6_cseg1_201013-1932-tag-val_st_loss.csv')
df_4_vel_vloss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/segcsv/run-evaluate_ts15_ds3_Class6_cseg1_201013-1932-tag-val_velocity_loss.csv')

#rgb+depth(LF)
df_5_vloss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/depthcsv/run-evaluate_ts15_ds3_Class6_cdepth2_201013-2229-tag-val_loss.csv')
df_5_class_vloss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/depthcsv/run-evaluate_ts15_ds3_Class6_cdepth2_201013-2229-tag-val_classification_loss.csv')
df_5_st_vloss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/depthcsv/run-evaluate_ts15_ds3_Class6_cdepth2_201013-2229-tag-val_st_loss.csv')
df_5_vel_vloss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/depthcsv/run-evaluate_ts15_ds3_Class6_cdepth2_201013-2229-tag-val_velocity_loss.csv')

#rgb+seg(LF)
df_6_vloss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/convtweakscsv/run-evaluate_ts15_ds3_Class5_conv16_201013-0857-tag-val_loss.csv')
df_6_class_vloss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/convtweakscsv/run-evaluate_ts15_ds3_Class5_conv16_201013-0857-tag-val_classification_loss.csv')
df_6_st_vloss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/convtweakscsv/run-evaluate_ts15_ds3_Class5_conv16_201013-0857-tag-val_st_loss.csv')
df_6_vel_vloss = pd.read_csv('/home/kaladin/Documents/arbeit/real/code/evaluation/convtweakscsv/run-evaluate_ts15_ds3_Class5_conv16_201013-0857-tag-val_velocity_loss.csv')
fig, ax = plt.subplots(2,2, figsize=(8,6))

#gs = GridSpec(2, 2, figure=fig)
#ax1 = fig.add_subplot(gs[0, 0])
ax[0,0].plot(df_1_vloss.Step[:49]+1,df_1_vloss[:49].Value,color='#D85C41',ls='--', label='Only RGB-G')
ax[0,0].plot(df_2_vloss.Step+1,df_2_vloss.Value,color='#57AB27', label='Only Depth')
ax[0,0].plot(df_3_vloss.Step[:49]+1,df_3_vloss[:49].Value,color='#646567',ls=':', label='RGB-G+Depth(EF)')
ax[0,0].plot(df_4_vloss.Step[:49]+1,(df_4_vloss[:49].Value),color='#7A6FAC', label='RGB-G+Seg(EF)')
ax[0,0].plot(df_5_vloss.Step[:49]+1,(df_5_vloss[:49].Value),color='#000000', label='RGB-G+Depth(LF)')
#ax[0,0].plot(df_6_vloss.Step[:49]+1,(df_6_vloss[:49].Value),color='#00549F', label='RGB-G+Seg(LF)')
ax[0,0].set_title('Loss')
ax[0,0].set_ylabel('Loss')
#ax[0,0].set_xlabel('Epochs')
ax[0,0].legend()
ax[0,0].grid()



#ax3 = fig.add_subplot(gs[1, 0])

ax[0,1].plot(df_1_class_vloss[:49].Step+1,df_1_class_vloss[:49].Value,color='#D85C41',ls='--', label='Only RGB-G')
ax[0,1].plot(df_2_class_vloss.Step+1,df_2_class_vloss.Value,color='#57AB27', label='Only Depth')
ax[0,1].plot(df_3_class_vloss.Step[:49]+1,df_3_class_vloss[:49].Value,color='#646567',ls=':', label='RGB-G+Depth(EF)')
ax[0,1].plot(df_4_class_vloss[:49].Step+1,df_4_class_vloss[:49].Value,color='#7A6FAC', label='RGB-G+Seg(EF)')
ax[0,1].plot(df_5_class_vloss[:49].Step+1,df_5_class_vloss[:49].Value,color='#000000', label='RGB-G+Depth(LF)')
#ax[0,1].plot(df_6_class_vloss[:49].Step+1,df_6_class_vloss[:49].Value,color='#00549F', label='RGB-G+Seg(LF)')
ax[0,1].set_title('Classification Loss')
#ax[0,1].set_xlabel('Epochs')
#ax[0,1].set_ylabel('Loss')
ax[0,1].grid()
ax[0,1].legend()

#ax2 = fig.add_subplot(gs[0, 1])
ax[1,0].plot(df_1_vel_vloss.Step[:49]+1,df_1_vel_vloss[:49].Value,color='#D85C41',ls='--', label='Only RGB-G')
ax[1,0].plot(df_2_vel_vloss.Step+1,df_2_vel_vloss.Value,color='#57AB27', label='Only Depth')
ax[1,0].plot(df_3_vel_vloss.Step[:49]+1,df_3_vel_vloss[:49].Value,color='#646567',ls=':', label='RGB-G+Depth(EF)')
ax[1,0].plot(df_4_vel_vloss[:49].Step+1,(df_4_vel_vloss[:49].Value),color='#7A6FAC', label='RGB-G+Seg(EF)')
ax[1,0].plot(df_5_vel_vloss[:49].Step+1,(df_5_vel_vloss[:49].Value),color='#000000', label='RGB-G+Depth(LF)')
#ax[1,0].plot(df_6_vel_vloss[:49].Step+1,(df_6_vel_vloss[:49].Value),color='#00549F', label='RGB-G+Seg(LF)')
ax[1,0].set_title('Velocity Loss')
ax[1,0].set_xlabel('Epochs')
ax[1,0].set_ylabel('Loss')
ax[1,0].grid()
ax[1,0].legend()
#ax4 = fig.add_subplot(gs[1, 1])

ax[1,1].plot(df_1_st_vloss.Step[:49]+1,df_1_st_vloss[:49].Value,color='#D85C41',ls='--', label='Only RGB-G')
ax[1,1].plot(df_2_st_vloss.Step+1,df_2_st_vloss.Value,color='#57AB27', label='Only Depth')
ax[1,1].plot(df_3_st_vloss.Step[:49]+1,df_3_st_vloss[:49].Value,color='#646567',ls=':', label='RGB-G+Depth(EF)')
ax[1,1].plot(df_4_st_vloss[:49].Step+1,(df_4_st_vloss[:49].Value),color='#7A6FAC', label='RGB-G+Seg(EF)')
ax[1,1].plot(df_5_st_vloss[:49].Step+1,(df_5_st_vloss[:49].Value),color='#000000', label='RGB-G+Depth(LF)')
#ax[1,1].plot(df_6_st_vloss[:49].Step+1,(df_6_st_vloss[:49].Value),color='#00549F', label='RGB-G+Seg(LF)')
ax[1,1].set_title('Steering Loss')
ax[1,1].set_xlabel('Epochs')
#ax[1,1].set_ylabel('Loss')
ax[1,1].grid()
ax[1,1].legend()


plt.show()

