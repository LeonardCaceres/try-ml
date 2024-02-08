import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

############# https://github.com/mwaskom/seaborn-data/blob/master/penguins.csv #############

data = pd.read_csv('penguins.csv')

data = data.dropna()

labels = data['species']
features = data[['bill_length_mm', 'bill_depth_mm']]

train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.2, random_state=123)

best_accuracy = 0
worst_accuracy = 1
best_params = {'n_neighbors': 0, 'weights': ''}
worst_params = {'n_neighbors': 0, 'weights': ''}

for n in range(1, 11):  # 20
    for weights in ['uniform', 'distance']:
        model = KNeighborsClassifier(n_neighbors=n, weights=weights)
        model.fit(train_features, train_labels)
        predictions = model.predict(test_features)
        accuracy = accuracy_score(test_labels, predictions)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params['n_neighbors'] = n
            best_params['weights'] = weights
        if accuracy < worst_accuracy:
            worst_accuracy = accuracy
            worst_params['n_neighbors'] = n
            worst_params['weights'] = weights

print('Best accuracy: {:.6f}'.format(best_accuracy))
print('Worst accuracy: {:.6f}'.format(worst_accuracy))
