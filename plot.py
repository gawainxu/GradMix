# visualize the feature ensemble results
#https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot

import matplotlib.pyplot as plt
import numpy as np

values_wo = {"accuracy": np.array([ 95.73,	96.2, 96.23, 96.78]),
          "auroc": np.array([86.62,	87.48, 87.8, 88.41,])} 


values_w = {"accuracy": np.array([ 96.62,	96.75,	97.1,	97.32]),
          "auroc": np.array([89.15,	89.41,	90.83,	91.57])} 

beta = ["0.2", "0.4", "0.6", "0.8"]


# Plotting both curves
plt.plot(values_w["auroc"], values_w["accuracy"], label='w GradMix', color='blue', marker='o')
plt.plot(values_wo["auroc"],  values_wo["accuracy"], label='wo GradMix', color='red', marker='o')


for i, (x, y) in enumerate(zip(values_w["auroc"], values_w["accuracy"])):
    plt.annotate(
        beta[i],           # The label text
        xy=(x, y),               # The point to annotate
        textcoords="offset points", # How to position the text
        xytext=(0, 10),          # Distance from text to point (x,y)
        ha='center',             # Horizontal alignment
        fontsize=8,
        color='k'
    )


for i, (x, y) in enumerate(zip(values_wo["auroc"], values_wo["accuracy"])):
    plt.annotate(
        beta[i],           # The label text
        xy=(x, y),               # The point to annotate
        textcoords="offset points", # How to position the text
        xytext=(0, 10),          # Distance from text to point (x,y)
        ha='center',             # Horizontal alignment
        fontsize=8,
        color='k'
    )


plt.xlabel('AUROC (%)')
plt.ylabel('Accuracy (%)')
plt.title(r'AUROC vs. Accuracy with varying $\beta$')
plt.legend()

plt.savefig("./plots/preto_cifar.pdf")