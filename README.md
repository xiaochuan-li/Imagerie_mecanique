# Imagerie_mecanique
Imagerie dans l'analyse d'essai mecanique  


## 0. Quick Start  
It's for a experiment mecanic, and each time we have to set different configuration for different experiment(we'd like to invent an automatic project which turns out to be complexe and we'll add it to our todo list). For this time, we have provided 2 configs(in config.py), which work on the data provided. You can create your own config (also in config.py) and import it in utils.py. The structure of config is shown in config.py  
Once the config is set, you can use main.py to analyse your data, while all the functions principle are preserved in utils.py.

## 1. Result
### 1.1 Location of points
<center><img src="https://github.com/xiaochuan-li/Imagerie_mecanique/blob/master/pic/sans%20bulle.gif" /></center>

### 1.2 Deformation - Time
<center><img src='https://github.com/xiaochuan-li/Imagerie_mecanique/blob/master/data/test%20sans%20bull_defome-temps.png' /></center>  

### 1.3 Contrainte - Deformation
<center><img src='https://github.com/xiaochuan-li/Imagerie_mecanique/blob/master/data/sans%20bullefin.png' /></center>

## 2. TODO LIST
#### 2.1 Try to create an automatic one. In this project, there are several constant set through experiment, which is not cool, we'd like to get them automaticly.
#### 2.2 The calculate of boundaries is simple, directe and a little violent, would it be possible to come up with a more elegent method? For example, what if we add a little bit of convolution when searching for the mask? What if we come up with a neural net to detect the points? And what if we use the NMS to remove those repetitions?
