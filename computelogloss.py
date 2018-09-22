import numpy as np

def compute_log_loss(predicted, actual, eps=1e-14):

    #Computes log loss between predicteds and actuals for 1D arrays
    #predicted: predicted probabilities as floats from 0-1
    #actual: actual data 0 or 1
    #eps optional:how close to 0 to get for log (0)=-inf

    predicted = np.clip(predicted, eps, 1-eps)
    loss=-1*np.mean(actual*np.log(predicted))+(1-actual)*np.log(1-predicted)

    return loss