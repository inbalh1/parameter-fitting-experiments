# Trying to understand how to use SMAC
from ConfigSpace import Configuration, ConfigurationSpace
from smac import BlackBoxFacade
from smac import Scenario

import sys
#sys.path.append("../src")
import models
import parameters
# target function - I suppose this is this "evaluation function". They say that we want to minimize the value returned from this function
# facade - there are several options, and they say its important


# TODO: can write the code in a more general way, to combine with the other experiments (the smac should be just the fitter I guess)

model_class = models.ErdosRenyi
    
def target1(config: Configuration, seed: int):
	# sample about 20/50 sanpmles based on the parameters, reduce to biggest CC. 
	# Calculate mean of target features?
    # params = list(in_param.input_guess(target_param) for in_param, target_param in zip(
    #        self.model.input_parameters(), self.target_parameters))
    n = parameters.NumberOfVertices(config["n"])
    d = parameters.AverageDegree(config["d"])
    model = model_class(n, d)
    g = model.generate()
    return (g.numberOfNodes() - 1500) ** 2# TODO: should find diff from original graph

def target_function(config: Configuration, seed: int):
    n = parameters.NumberOfVertices(config["n"])
    d = parameters.AverageDegree(config["d"])
    model = model_class(n, d)
    g = model.generate()
    avg_deg = 2 * g.numberOfEdges() / g.numberOfNodes()
    target1 = (g.numberOfNodes() - 1500) ** 2
    target2 = (avg_deg - 9) ** 2
    return [target1, target2]# TODO: should find diff from original graph


# TODO: should check whether we have something like that
# Configuration space for Erdos renyi
configspace = ConfigurationSpace({
    "n": (1000, 10000),
    "d": (2, 10)
})

def uniObjective(n_trials):
    # Scenario object specifying the optimization environment
    scenario = Scenario(configspace, deterministic=True, n_trials=n_trials)
    # Use SMAC to find the best configuration/hyperparameters
    smac = BlackBoxFacade(scenario, target_function=target1)
    incumbent = smac.optimize()
    return incumbent
    
def multiObjective(n_trials)->list[Configuration]:
    scenario = Scenario(configspace, deterministic=True, n_trials=n_trials, objectives=["n", "d"])
    smac = BlackBoxFacade(scenario, target_function=target_function)
    incumbent = smac.optimize()
    print(incumbent)
    return incumbent




#uniObjective(n_trials=2)
multiObjective(n_trials=5)

# Questions:
# The configuration space
# Target function - what should we minimize - compare to a single input graph?
# Facade - make should to choose the right one...
# number of trials / mean over a batch of samples - how many??? Whats the connection?
