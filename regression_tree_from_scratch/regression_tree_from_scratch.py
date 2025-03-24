import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def calc_sse(yval1, yval2):
    mean_yval1 = yval1.mean()
    mean_yval2 = yval2.mean()

    sse_yval1 = ((yval1 - mean_yval1) ** 2).sum()
    sse_yval2 = ((yval2 - mean_yval2) ** 2).sum()

    final_sse = sse_yval1 + sse_yval2
    return final_sse


def split(X, y):
    fin_split_val = []
    final_sse_value = []
    for col in X.columns:
        sse_lst = []
        split_lst = []
        X[col] = np.sort(X[col])
        for i in range(len(X) - 1):
            split_val = 0
            split_val = (X[col][i] + X[col][i + 1]) / 2
            split_lst.append(split_val)
            yval1 = y[X[col] <= split_val]  # store the y values in based on the split condition
            yval2 = y[X[col] > split_val]
            final_sse = calc_sse(yval1, yval2)  # functions which calculates sse for each split_value
            sse_lst.append(final_sse)

        min_index = np.argmin(sse_lst)  # index of the min sse value from the list
        min_sse_value = min(sse_lst)
        final_split_val = split_lst[min_index]  # split value of column based on min sse
        fin_split_val.append(final_split_val)
        final_sse_value.append(min_sse_value)

        plt.plot(split_lst, sse_lst)
        plt.axvline(x=final_split_val, color='r', linestyle='--', label=f'min_split_value = {final_split_val}')
        plt.xlabel('splits')
        plt.ylabel('SSE')
        plt.title(f'splits vs. SSE of column {col}'.format(col))
        plt.legend()  # Show legend
        plt.show()
    sse_col_idx = np.argmin(final_sse_value)  # min index value among two columns having lower sse

    var = X.columns[sse_col_idx]  # column with lower sse
    val = fin_split_val[sse_col_idx]  # split value having lower sse
    return var, val


df = pd.read_csv("susedcars.csv", usecols=['price', 'mileage', 'year'])
df['age'] = 2015 - df.pop('year')
X = df.drop(columns=['price'])
y = df['price']

var, value = split(X, y)
print(var, value)

