import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler

#Function optimal K

def find_optimal_k(X_std, y, visualize):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    k_values = np.arange(2, 50)  # initiallized Possible values of k from 2 to 50
    sse_values = []  # List to store the SSE for each value of k

    for k in k_values:
        sse_k_fold = []

        for train_index, test_index in kf.split(X_std):
            X_train, X_test = X_std[train_index], X_std[test_index]
            y_train, y_test = y[train_index], y[test_index]

            knn_model = KNeighborsRegressor(n_neighbors=k)
            knn_model.fit(X_train, y_train)

            y_pred = knn_model.predict(X_test)

            # Calculate SSE
            sse = np.sum((y_test - y_pred) ** 2)
            sse_k_fold.append(sse)

        #calculating total SSE across all folds for this particular k
        total_sse = np.sum(sse_k_fold)
        sse_values.append(total_sse)

    k_best_index = np.argmin(sse_values)
    k_best = k_values[k_best_index]

    # Visualizing
    if visualize:
        plt.plot(k_values, sse_values)
        plt.axvline(x=k_best, color='r', linestyle='--', label=f'k = {k_best}')  # Vertical line at k_best with legend
        plt.xlabel('k')
        plt.ylabel('Total SSE')
        plt.title('k vs. Total SSE of 5-fold CV')
        plt.legend()  # Show legend
        plt.savefig(visualize)
        plt.close()

    return k_best


#function knn from library sklearn

def knn_sklearn(X, y, X_pred):
    scaler = MinMaxScaler()
    X_std = scaler.fit_transform(X)

    k_best = find_optimal_k(X_std, y,'img.png')

    knn_model = KNeighborsRegressor(n_neighbors=k_best)
    knn_model.fit(X_std, y)

    X_pred_std = scaler.transform(X_pred)

    y_pred = knn_model.predict(X_pred_std)

    return y_pred, k_best






#Euclidian Distance function

def euclidean_distance(p, q):
    distance = 0.0
    for i in range(len(p)):
        distance += (p[i] - q[i]) ** 2
    return distance ** 0.5

#knn without any libraries

def knn(k, X, y, X_pred):
    #standardization of data, calculating mean and standard deviation
    num_features = len(X[0])
    means_X = [sum([X[j][i] for j in range(len(X))]) / len(X) for i in range(num_features)]
    std_devs_X = [((sum([(X[j][i] - means_X[i]) ** 2 for j in range(len(X))]) / len(X)) ** 0.5) for i in
                  range(num_features)]

    #standardize the training data using Min-Max scaling without any libraries
    X_min = [min([X[j][i] for j in range(len(X))]) for i in range(num_features)]
    X_max = [max([X[j][i] for j in range(len(X))]) for i in range(num_features)]
    X_std = [[(X[j][i] - X_min[i]) / (X_max[i] - X_min[i]) for i in range(num_features)] for j in range(len(X))]

    #standardize the prediction data using the same scaling factors as the training data
    X_pred_std = [[(X_pred[j][i] - X_min[i]) / (X_max[i] - X_min[i]) for i in range(num_features)] for j in
                  range(len(X_pred))]

    preds = []
    for pred_point in X_pred_std:
        distances = []
        for i, train_point in enumerate(X_std):
            distance = euclidean_distance(train_point, pred_point)
            distances.append((distance, y[i]))

        distances.sort()
        top_k = distances[:k]

        total = 0
        for dist, label in top_k:
            total += label / k
        pred = total
        preds.append(pred)

    return preds

#Testing
df = pd.read_csv('knn_from_scratch.csv', usecols=['price', 'mileage', 'year'])
df['age'] = 2015 - df.pop('year')
X = df[['mileage', 'age']].to_numpy()
y = df['price'].to_numpy()

X_pred = [[100000, 10], [50000, 3]]
y_pred, k_best = knn_sklearn(X, y, X_pred)

print(y_pred, k_best)

y_pred2 = knn(19, X, y, X_pred)
print(y_pred2)

