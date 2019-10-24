import keras
import pandas as pd
import tensorflow as tf
from keras import layers, models

df = pd.read_csv("https://github.com/bgweber/Twitch/raw/master/Recommendations/games-expand.csv")

df.head()
