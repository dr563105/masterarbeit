import pandas as pd
from math import pi
import matplotlib.pyplot as plt
from soccerplots.radar_chart import Radar

 # Criteria & Rating \\\midrule
 #    Lane keeping/Drive straight  & 4  \\
 #    Gradual acceleration increase  & 4\\
 #    Smooth braking behaviour observed & 2 \\
 #    Smooth steering control at high speed(10m/s) & 3 \\
 #    Smooth steering control at turnings & 4\\
 #    Avoids colliding with static objects & 3 \\
 #    Detects vehicles as dynamic objects & 5 \\
 #    Navigates traffic smoothly & 2\\
 #    Stops at random places& 5 \\
 #    Smooth evaluation experience & 4 \\\bottomrule

# # Lane keeping/Drive straight  & 5  
# # Gradual acceleration increase  & 3
# # Smooth braking behaviour observed & 2 
# # Smooth steering control at high speed(10m/s) & 2
# # Smooth steering control at turnings & {5}
# # Avoids colliding with static objects & {5}
# # Detects vehicles dynamic objects & 4
# # Navigates traffic smoothly & 2
# # Affected by sunlight & {4}
# # Affected by low-light or night driving & {4}
# # Stops at random places & 5 
# Smooth evaluation experience & 5 

## parameter names
params = ['Lane keeping', 'Acceleration change', 'Smooth braking', 'High speed control', 'Control at turnings', 'Detects vehicles', 'Detects static objects', 'Traffic navigation', 'Stops at random places', 'Sunlight glare', 'Low-light driving', 'Experience']

## range values
ranges = [(1, 5), (1, 5), (1, 5), (1, 5), (1, 5), (1, 5), (1, 5), (1, 5), (1, 5), (1, 5), (1, 5), (1, 5)]

## parameter value
values = [
    [4, 4, 2, 3, 4, 3, 5, 2, 5, 3, 1, 4],   ## for 100k
    [5, 3, 2, 2, 5, 5, 4, 2, 5, 4, 4, 5]    ## for 224k
]

## title
title = dict(
    title_name='RGB-G+Seg(100k)',
    title_color='#B6282F',
    subtitle_name='Early Fusion',
    subtitle_color='#B6282F',
    title_name_2='RGB-G+Seg(224K)',
    title_color_2='#344D94',
    subtitle_name_2='Early Fusion',
    subtitle_color_2='#344D94',
    title_fontsize=18,
    subtitle_fontsize=15,
)

## endnote 
#endnote = "Visualization made by: Anmol Durgapal(@slothfulwave612)\nAll units are in per90"

fig, ax = plt.subplots(figsize=(8, 6))

## instantiate object 
radar = Radar()

## plot radar -- alphas
fig, ax = radar.plot_radar(ranges=ranges, params=params, values=values, 
                                 radar_color=['#B6282F', '#F6A800'], 
                                 alphas=[0.8, 0.6], title=title,
                                 compare=True,figax=(fig,ax), dpi=300)

plt.show()


