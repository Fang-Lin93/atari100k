

"""
liu yiyuan defence plot +
"""
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

data = pd.read_csv('total_count.csv')

# fig, ax = plt.subplots(1, 1, figsize=(20, 10))
# data['count_x'] = data['count_x'] /data['count_x'].sum()
# data['count_y'] = data['count_y'] /data['count_y'].sum()
# sns.lineplot(x=range(len(data['count_x'])), y=data['count_x'], ax=ax, alpha=0.7, label='alive')
# sns.lineplot(x=range(len(data['count_y'])), y=data['count_y'], ax=ax, color='orange', alpha=0.7, label='dead')

fig, ax = plt.subplots(5, 1, figsize=(20, 20))


for i in range(5):
    alive = data['count_x'][i*200:(i+1)*200]
    dead = data['count_y'][i*200:(i+1)*200]
    alive, dead = alive/sum(alive), dead /sum(dead)

    # x = data['drug'][i*200:(i+1)*200]
    x = range(len(alive))
    sns.lineplot(x=x, y=alive.values, ax=ax[i], alpha=0.7, label='alive')
    sns.lineplot(x=x, y=dead.values, ax=ax[i], color='orange', alpha=0.7, label='dead')


fig.show()

#####

