from sklearn.ensemble import RandomForestClassifier
from wine_dataset import load_wine_dataset,load_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel


df_wine = load_wine_dataset()
feat_labels = df_wine.columns[1:]
forest = RandomForestClassifier(criterion='gini', n_estimators=500 , random_state=1)
X_train, X_test , y_train , y_test = load_data()
forest.fit(X_train,y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
sfm = SelectFromModel(forest,threshold=0.1 , prefit=True)
X_selected = sfm.transform(X_test)
for f in range(X_train.shape[1]):
    print ("%2d) %-*s %f" % (f + 1, 30,feat_labels[indices[f]],importances[indices[f]]))

plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]),importances[indices],align='center')
plt.xticks(range(X_train.shape[1]),feat_labels, rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()