 def read_column(filename, col): 
   with open(filename) as file:
        columname = []
        next(file) # used to skip the column header
        for line in file:
            # Seperating numbers in each line
            line = line.strip().split(',')
            # Converting the above list numbers to integer data type
            line = list(map(int,line))
            columname.append(line[col])
        return list(columname)
        

def ordinary_least_squares(x, y):
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    numerator = 0
    denominator = 0
    
    for i in range(n):
        numerator +=(x[i] - mean_x) * (y[i] - mean_y)
        denominator +=(x[i] - mean_x) ** 2
    
    #calculating slope
    beta = numerator / denominator
    
    #calculating the intercept(alpha)
    alpha = mean_y - beta * mean_x
    
    return alpha, beta
    
sizes = read_column('ols_from_scratch_data.csv', 0)
prices = read_column('ols_from_scratch_data.csv', 2)
#print(sizes,prices)
a, b = ordinary_least_squares(sizes, prices)
print(f'The OLS estimate of regression of Price on Size is price = {a:.4f} + {b:.4f} * size.')