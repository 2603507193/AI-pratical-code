'''

2018 Assigment One : Differential Evolution
    
Scafolding code

Complete the missing code at the locations marked 
with 'INSERT MISSING CODE HERE'

To run task_2 you will need to download an unzip the file dataset.zip

If you have questions, drop by in one of the pracs on Wednesday 
     11am-1pm in S503 or 3pm-5pm in S517
You can also send questions via email to f.maire@qut.edu.au


'''

import numpy as np

import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier

from sklearn import preprocessing

from sklearn import model_selection

# ----------------------------------------------------------------------------

def differential_evolution(fobj, 
                           bounds, 
                           mut=2, 
                           crossp=0.7, 
                           popsize=20, 
                           maxiter=100,
                           verbose = True):
    '''
    This generator function yields the best solution x found so far and 
    its corresponding value of fobj(x) at each iteration. In order to obtain 
    the last solution,  we only need to consume the iterator, or convert it 
    to a list and obtain the last value with list(differential_evolution(...))[-1]    
    
    
    @params
        fobj: function to minimize. Can be a function defined with a def 
            or a lambda expression.
        bounds: a list of pairs (lower_bound, upper_bound) for each 
                dimension of the input space of fobj.
        mut: mutation factor
        crossp: crossover probability
        popsize: population size
        maxiter: maximum number of iterations
        verbose: display information if True    
    '''
    #    This generates our initial population of 10 random vectors. 
    #    Each component x[i] is normalized between [0, 1]. 
    #    We will use the bounds to denormalize each component only for 
    #    evaluating them with fobj.
    # dimension of the input space of 'fobj'
    n_dimensions = len(bounds) 
  
    #initialize all individuals w of the initial population with random points from the search-space.
    #TO use w to generate the 4 hyper parameters  
    pop = np.random.rand(popsize, n_dimensions)
    
    #normalization by bound range
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff
  
    #Get the all -accurary value from  eval_hyper() method
    cost = np.asarray([fobj(ind) for ind in pop_denorm])
    # Find the index of the smallest -accuracy (the index of best accuracy)
    best_idx = np.argmin(cost)
    #based on the index, find the corresponding four hyper parameters 
    best = pop_denorm[best_idx]
    #calculate the initial cost and best
    
    if verbose:
        print('** Lowest cost in initial population = {} '.format(cost[best_idx]))        
    for i in range(maxiter):
        if verbose:
            print('** Starting generation {}, '.format(i))        
        
        #for each individual w in the population do
        for w in range(popsize):
            #Pick three distinct individuals a, b and c from the current population at random, distinct from w as well.
            idxs = [idx for idx in range(popsize) if idx != w]
            a, b, c = pop[np.random.choice(idxs, 3, replace = False)]
            #Create a mutant vector a + mut * (b – c) and clip the mutant entries to the interval [0, 1]
            mutant = np.clip(a + mut * (b - c), 0, 1)
            #Create a trial vector by assigning to trial[k] with probability crossp the value mutant[k]
            #and probability 1-crossp the value of w[k]
            cross_points = np.random.rand(n_dimensions) < crossp
            # if cross_points all return false, change one false value to true
            if not np.any(cross_points):
                cross_points[np.random.randint(0, n_dimensions)] = True
            #based on the each element of cross_points value, 
            #if the element value is true,the current vector will be taken from the mutant
            #if the element values is fales, the current vector will be taken from the pop
            trial = np.where(cross_points, mutant, pop[w])
            #normalization the trial vector based on the bound
            trial_denorm = min_b + trial * diff
            # evaluate the trial vector 
            f = fobj(trial_denorm)
            # Compare trial vector and target vector and keep the best vector 
            if f < cost[w]:
                cost[w] = f
                pop[w] = trial
            #compare the trial vector and the best vector of initial vector
                if f < cost[best_idx]:
            #obtain the index of best vector and the best cost 
                    best_idx = w
                    best = trial_denorm
            
        yield best, cost[best_idx]

# ----------------------------------------------------------------------------

def task_1():
    '''
    Our goal is to fit a curve (defined by a polynomial) to the set of points 
    that we generate. 
    '''

    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
    def fmodel(x, w):
        '''
        Compute and return the value y of the polynomial with coefficient 
        vector w at x.  
        For example, if w is of length 5, this function should return
        w[0] + w[1]*x + w[2] * x**2 + w[3] * x**3 + w[4] * x**4 
        The argument x can be a scalar or a numpy array.
        The shape and type of x and y are the same (scalar or ndarray).
        '''
        if isinstance(x, float) or isinstance(x, int):
            y = 0
        else:
            assert type(x) is np.ndarray
            y = np.zeros_like(x)
            
        for i in range(len(w)):
            y = y + w[i] * x**i
            
        return y

    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        
    def rmse(w):
        '''
        Compute and return the root mean squared error (RMSE) of the 
        polynomial defined by the weight vector w. 
        The RMSE is is evaluated on the training set (X,Y) where X and Y
        are the numpy arrays defined in the context of function 'task_1'.        
        '''
        Y_pred = fmodel(X, w)
        return np.sqrt(sum((Y - Y_pred)**2)/len(Y))


    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .  
    
    # Create the training set
    X = np.linspace(-5, 5, 500)
    Y = np.cos(X) + np.random.normal(0, 0.2, len(X))
    
    # Create the DE generator
    de_gen = differential_evolution(rmse, [(-5, 5)] * 6, mut=1, maxiter=2000)
    
    # We'll stop the search as soon as we found a solution with a smaller
    # cost than the target cost
    target_cost = 0.5
    
    # Loop on the DE generator
    for i , p in enumerate(de_gen):
        w, c_w = p
        # w : best solution so far
        # c_w : cost of w 
        # p: result of method de_gen
        # Stop when solution cost is less than the target cost
        if c_w< target_cost:
            break
        
    # Print the search result
    print('Stopped search after {} generation. Best cost found is {}'.format(i,c_w))
    #    result = list(differential_evolution(rmse, [(-5, 5)] * 6, maxiter=1000))    
    #    w = result[-1][0]
        
    # Plot the approximating polynomial
    plt.scatter(X, Y, s=2)
    plt.plot(X, np.cos(X), 'r-',label='cos(x)')
    plt.plot(X, fmodel(X, w), 'g-',label='model')
    plt.legend()
    plt.title('Polynomial fit using DE')
    plt.show()    
    

# ----------------------------------------------------------------------------

def task_2():
    '''
    Goal : find hyperparameters for a MLP
    
       w = [nh1, nh2, alpha, learning_rate_init]
    '''
    
    
    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .  
    def eval_hyper(w):
        '''
        Return the negative of the accuracy of a MLP with trained 
        with the hyperparameter vector w
        
        alpha : float, optional, default 0.0001
                L2 penalty (regularization term) parameter.
        '''
        
        nh1, nh2, alpha, learning_rate_init  = (
                int(1+w[0]), # nh1
                int(1+w[1]), # nh2
                10**w[2], # alpha on a log scale
                10**w[3]  # learning_rate_init  on a log scale
                )


        clf = MLPClassifier(hidden_layer_sizes=(nh1, nh2), 
                            max_iter=100, 
                            alpha=alpha, #1e-4
                            learning_rate_init=learning_rate_init, #.001
                            solver='sgd', verbose=10, tol=1e-4, random_state=1
                            )
        
        clf.fit(X_train_transformed, y_train)
        # compute the accurary on the test set
        mean_accuracy = clf.score(X_test_transformed, y_test)
 
        return -mean_accuracy
    
    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .  

    # Load the dataset
    X_all = np.loadtxt('dataset_inputs.txt', dtype=np.uint8)[:1000]
    y_all = np.loadtxt('dataset_targets.txt',dtype=np.uint8)[:1000]    
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
                                X_all, y_all, test_size=0.4, random_state=42)
       
    # Preprocess the inputs with 'preprocessing.StandardScaler'
    
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_transformed = scaler.transform(X_train)
    X_test_transformed = scaler.transform(X_test)


    
    bounds = [(1,100),(1,100),(-6,2),(-6,1)]  # bounds for hyperparameters
    
    de_gen = differential_evolution(
            eval_hyper, 
            bounds, 
            mut = 1,
            popsize=10, 
            maxiter=20,
            verbose=True)
    
    for i, p in enumerate(de_gen):
        w, c_w = p
        print('Generation {},  best cost {}'.format(i,abs(c_w)))
        # Stop if the accuracy is above 90%
        if abs(c_w)>0.90:
            break
 
    # Print the search result
    print('Stopped search after {} generation. Best accuracy reached is {}'.format(i,abs(c_w)))   
    print('Hyperparameters found:')
    print('nh1 = {}, nh2 = {}'.format(int(1+w[0]), int(1+w[1])))          
    print('alpha = {}, learning_rate_init = {}'.format(10**w[2],10**w[3]))
    
# ----------------------------------------------------------------------------

def task_3():
    '''
    To run experiments to compare the following (population_size, max_iter) allocations
    in the list [(5,40),(10,20),(20,10),(40,5)], we only need to iterate the allocations
    one by one. In each iteration, it works the same way as task_2 with only popsize and
    maxiter different in the differential_evolution function.
    '''
    
    # Same as task_2
    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .  
    def eval_hyper(w):
        '''
        Return the negative of the accuracy of a MLP with trained 
        with the hyperparameter vector w
        
        alpha : float, optional, default 0.0001
                L2 penalty (regularization term) parameter.
        
        During whole process, the sklearn.MLPClassifier is used to build the 
        neural network.
        MPClassifier: 
            hidden_layer: two hiddn_layer is set, which is nh1 and nh2.
            activation function: 'tanh' return f(x) = tanh(x).
            max_iter: maximum number of iteration is 100 times.
            alpha: default values is 1e-4.
            learning_rate_init: learning rate schedule for weight updates.
            solver: To solver for weight optimization.
            verbose: Whether to print progress messages to stdout.
            tol: Tolerance for the optimization.when the loss is not improving by 
            at lest tol for two consecutive iterations, conergence is considered 
            to be reached and training stops.
            random_state: int, RandomState instance.                         
        '''
        
        nh1, nh2, alpha, learning_rate_init  = (
                int(1+w[0]), # nh1
                int(1+w[1]), # nh2
                10**w[2], # alpha on a log scale
                10**w[3]  # learning_rate_init  on a log scale
                )

        # set verbose to False to omit the unnecessary information
        clf = MLPClassifier(hidden_layer_sizes=(nh1, nh2),
                            activation = 'tanh',
                            max_iter=100, 
                            alpha=alpha, #1e-4
                            learning_rate_init=learning_rate_init, #.001
                            solver='sgd', verbose=False, tol=1e-4, random_state=1
                            )
        
        clf.fit(X_train_transformed, y_train)
        # compute the accurary on the test set
        mean_accuracy = clf.score(X_test_transformed, y_test)
 
        return -mean_accuracy
    
    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .  

    # Load the dataset
    X_all = np.loadtxt('dataset_inputs.txt', dtype=np.uint8)[:1000]
    y_all = np.loadtxt('dataset_targets.txt',dtype=np.uint8)[:1000]    
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
                                X_all, y_all, test_size=0.4, random_state=42)
       
    # Preprocess the inputs with 'preprocessing.StandardScaler'
    
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_transformed = scaler.transform(X_train)
    X_test_transformed = scaler.transform(X_test)


    
    bounds = [(1,100),(1,100),(-6,2),(-6,1)]  # bounds for hyperparameters
    
    popiter = [(5,40),(10,20),(20,10),(40,5)] # the allocation list
    
    bestcost = [[] for i in range(4)]
    
    exp_time = 10 #the experiment run times, set to 10 to run times.
    for t in range(exp_time):
        # iterate the list
        for j,k in enumerate(popiter):
            print('Results for population_size={}, max_iter={}'.format(k[0],k[1]))
            de_gen = differential_evolution(
                    eval_hyper, 
                    bounds, 
                    mut = 1,
                    popsize=k[0], # popsize in an allocation
                    maxiter=k[1], # generation in an allocation
                    verbose=False) # no verbose to omit the unnecessary information
            
            for i, p in enumerate(de_gen):
                w, c_w = p             
                # Stop if the accuracy is above 88.5%
                if abs(c_w)>0.885:
                    break
 
            # Print the search result
            bestcost[j].append(abs(c_w))
            print('Stopped search after {} generation.'.format(i))  
            print('Best accuracy reached is {}'.format(abs(c_w)))
            print('Hyperparameters found:')
            print('nh1 = {}, nh2 = {}'.format(int(1+w[0]), int(1+w[1])))          
            print('alpha = {}, learning_rate_init = {}'.format(10**w[2],10**w[3]))
            print('---------------------------------------------------------')
            
    # display the final accuracy of four type of structure  
    for j,k in enumerate(popiter):        
        print('The best costs for population_size={},max_tier={} in the experiments are {}'. \
              format(k[0],k[1],bestcost[j]))
                        
# ----------------------------------------------------------------------------


if __name__ == "__main__":
      task_1()    
      task_2()    
      task_3()    

