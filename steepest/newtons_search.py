import numpy as np
from autograd import grad
from autograd import hessian

def newtons_search(func,X0,B=0.5,steps=1000, COUNTING_LIMIT=100):
    
    # initializing variables used in method 
    q = func                    # function to optimize
    grad_fct = grad(q)          # gradeint of function
    hessian_fct = hessian(q)    # hessian of function
    X = X0                      # initial point


    # initializing lists used in method or in returned params
    points = [X]
    new_X = []
    for i in X: new_X.append(0.0)
    

    # performing algorythm
    for i in range(0,steps):

        # calculate new_X value
        gradient = grad_fct(X)
        hes = hessian_fct(X)

        d_mat = (np.linalg.inv(np.matrix(hes))*np.matrix(gradient).transpose()).tolist()
        flat_list = []
        for row in d_mat:
            flat_list.extend(row)
        d = np.array(flat_list)

        new_X = X + B*d

        # checking limits
        for j,x in enumerate(new_X):
            if x > COUNTING_LIMIT:
                new_X[j] = COUNTING_LIMIT
            elif x < -COUNTING_LIMIT:
                new_X[j] = -COUNTING_LIMIT

        # assigning new_X to X variable to use in next iteration
        X = new_X

        # adding current point to point list
        points.append(X)
    
    return X, q(X), points
