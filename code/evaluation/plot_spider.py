import pandas as pd
from math import pi
import matplotlib.pyplot as plt
from soccerplots.radar_chart import Radar
#basic
  # Criteria(Softmax/Categorical crossentropy) & Dataset 1 & Dataset 3 \\\midrule
  #   Lane keeping/Drive straight  & 1 & 3  \\
  #   Gradual acceleration increase & 3 & 3\\
  #   Smooth braking behaviour observed & 1 & 3 \\
  #   Smooth steering control at high speed(10m/s) & 1 & 1 \\
  #   Smooth steering control at turnings & 1 & 1\\
  #   Avoids colliding into static objects & 1 & 1 \\
  #   Detects traffic as dynamic objects & 5 & 5\\
  #   Navigates traffic smoothly & 1 & 1\\
  #   Stops at random places & 5 & 5 \\\bottomrule

  
#dense
 # Criteria(Softmax/Categorical crossentropy) & Dataset 3 \\\midrule
 #    Lane keeping/Drive straight  & 4  \\
 #    Gradual acceleration increase  & 3 \\
 #    Smooth braking behaviour observed & 3 \\
 #    Smooth steering control at high speed(10m/s) & 2 \\
 #    Smooth steering control at turnings & 1\\
 #    Avoids colliding into static objects & 1 \\
 #    Detects vehicles as dynamic objects & 5 \\
 #    Navigates traffic smoothly & 3\\
 #    Stops at random places & 5 \\\bottomrule

#2LSTM
 # Criteria(Softmax/Categorical crossentropy) & Dataset 3 \\\midrule
 #    Lane keeping/Drive straight  & 4  \\
 #    Gradual acceleration increase  & 3\\
 #    Smooth braking behaviour observed & 3 \\
 #    Smooth steering control at high speed(10m/s) & 4 \\
 #    Smooth steering control at turnings & 1\\
 #    Avoids colliding with static objects & 2 \\
 #    Detects vehicles as dynamic objects & 5 \\
 #    Navigates traffic smoothly & 4\\
 #    Stops at random places & 5 \\
 #    Smooth evaluation experience & 3 \\\bottomrule

    

#2NN
 # Criteria(Softmax/Categorical crossentropy) & Dataset 3 \\\midrule
 #    Lane keeping/Drive straight  & 4  \\
 #    Gradual acceleration increase  & 4\\
 #    Smooth braking behaviour observed & 4 \\
 #    Smooth steering control at high speed(10m/s) & 4 \\
 #    Smooth steering control at turnings & 2\\
 #    Avoids colliding with static objects & 3 \\
 #    Detects vehicles as dynamic objects & 5 \\
 #    Navigates traffic smoothly & 3\\
 #    Stops at random places & 5 \\
 #    Smooth evaluation experience & 4 \\\bottomrule


## parameter names
params = ['Lane keeping', 'Acceleration change', 'Smooth braking', 'High speed control', 'Control at turnings', 'Detects vehicles', 'Detects static objects', 'Traffic navigation', 'No random braking', 'Sunlight glare', 'Low-light driving', 'Averse to weather','Overall experience']

## range values
ranges = [(1, 5), (1, 5), (1, 5), (1, 5), (1, 5), (1, 5), (1, 5), (1, 5), (1, 5), (1, 5), (1, 5), (1, 5), (1, 5)]

## parameter value
values = [
      [3, 3, 2, 1, 1, 5, 1, 1, 1, 1, 1, 1, 2],    ## for basic
    [4, 3, 3, 2, 1, 5, 1, 3, 1, 1, 1, 1, 3]    ## for dense 
]
values1 = [
    [4, 3, 3, 4, 1, 5, 2, 4, 1, 1, 1, 1, 3],    ## for 2LSTM
    [4, 4, 4, 4, 1, 5, 3, 2, 1, 1, 1, 1, 4]     ## for 2NN
]

## title
title1 = dict( 
    title_name='Separate LSTM layers',
    title_color='#CC071E',
    subtitle_name='CCE DS3',
    subtitle_color='#000000',
    title_name_2='Separate NN',
    title_color_2='#407FB7',
    subtitle_name_2='CCE DS3',
    subtitle_color_2='#000000',
    title_fontsize=18,
    subtitle_fontsize=15
)
## endnote 
#endnote = "Visualization made by: Anmol Durgapal(@slothfulwave612)\nAll units are in per90"

fig, (ax2) = plt.subplots(1,1, figsize=(12, 10))

## instantiate object 
radar = Radar()

## plot radar -- alphas
fig, ax2 = radar.plot_radar(ranges=ranges, params=params, values=values1, 
                                 radar_color=['#CC071E', '#407FB7'], 
                                 alphas=[0.6, 0.7 ], title=title1,
                                 compare=True,figax=(fig,ax2), dpi=300)
#, '#00549F','#F6A800'
plt.show()
plt.tight_layout()

