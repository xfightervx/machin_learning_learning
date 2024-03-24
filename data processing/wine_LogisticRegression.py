from sklearn.linear_model import LogisticRegression
from wine_dataset import standardscalewine, load_wine_dataset
import matplotlib.pyplot as plt
import numpy as np
X_train_std,X_test_std,y_train,y_test = standardscalewine()
df_wine = load_wine_dataset()
fig = plt.figure()
ax = plt.subplot(111)
colors = ['blue', 'green', 'red', 'cyan','magenta', 'yellow', 'black','pink', 'lightgreen', 'lightblue','gray', 'indigo', 'orange']
weights, prams = [],[]
for c in np.arange(-4,6):
    lr = LogisticRegression(solver='liblinear', penalty='l1', C=10.**c, random_state=0)
    lr.fit(X_train_std,y_train)
    weights.append(lr.coef_[1])
    prams.append((10.**c))

weights = np.array(weights)
for columns , color in zip(range(weights.shape[1]), colors):
    plt.plot(prams, weights[:,columns], label=df_wine.columns[columns +1] , color = color)

plt.axhline(0, color='black', linestyle='--', linewidth=3)
plt.xlim([10**(-5), 10**5])
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center',bbox_to_anchor=(1.38, 1.03),ncol=1, fancybox=True)
plt.show()

