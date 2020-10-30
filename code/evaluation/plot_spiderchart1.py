import pandas as pd
from math import pi
import matplotlib.pyplot as plt
from soccerplots.radar_chart import Radar

def createRadar(player, data):
    Attributes = ["Lane keeping","Night driving","Pace","Passing","Physical","Shooting"]
    
    data += data [:1]
    
    angles = [n / 6 * 2 * pi for n in range(6)]
    angles += angles [:1]
    
    ax = plt.subplot(111, polar=True)

    plt.xticks(angles[:-1],Attributes)
    ax.plot(angles,data)
    ax.fill(angles, data, 'blue', alpha=0.1)

    ax.set_title(player)
    plt.show()


createRadar("224k RGB-G+Seg(EF)",[2,1,4,1,5,3])

