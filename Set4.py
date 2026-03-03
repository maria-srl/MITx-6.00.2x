import os
os.environ["OPENBLAS_NUM_THREADS"]="1"
import numpy as np
def generate_models(x, y, degs):
    """
    Generate regression models by fitting a polynomial for each degree in degs
    to points (x, y).
    Args:
        x: a list with length N, representing the x-coords of N sample points
        y: a list with length N, representing the y-coords of N sample points
        degs: a list of degrees of the fitting polynomial
    Returns:
        a list of numpy arrays, where each array is a 1-d array of coefficients
        that minimizes the squared error of the fitting polynomial
    """
    models=[]
    xVals=np.array(x)
    yVals=np.array(y)
    for degree in degs:
        coeff=np.polyfit(xVals,yVals,degree)
        models.append(coeff)
    return models
def r_squared(y, estimated):
    """
    Calculate the R-squared error term.
    Args:
        y: list with length N, representing the y-coords of N sample points
        estimated: a list of values estimated by the regression model
    Returns:
        a float for the R-squared error term
    """
    mean=0
    for coord in y:
        mean+=coord
    mean/=len(y)
    sum1=0
    sum2=0
    for i in range(len(y)):
        aux1=(y[i]-estimated[i])**2
        aux2=(y[i]-mean)**2
        sum1+=aux1
        sum2+=aux2
    return 1-(sum1/sum2)
# Problem 3
y = []
x = INTERVAL_1
for year in INTERVAL_1:
    y.append(raw_data.get_daily_temp('BOSTON', 1, 10, year))
models = generate_models(x, y, [1])
evaluate_models_on_training(x, y, models)
# Problem 4
x1 = INTERVAL_1
x2 = INTERVAL_2
y = []
for year in INTERVAL_1:
    y.append(numpy.mean(raw_data.get_yearly_temp('BOSTON', year)))
models = generate_models(x1, y, [1])    
evaluate_models_on_training(x1, y, models)
