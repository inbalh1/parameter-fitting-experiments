from typing import Protocol, Dict
import numpy as np
import math
from ConfigSpace.hyperparameters import Hyperparameter, UniformIntegerHyperparameter, UniformFloatHyperparameter
from models import GraphModel

# Smac configuration for the parameters

class ParamSpec(Protocol):
    name: str # Parameter identifier 
    termination_threshold: float | np.floating

    # The value is the value of corresnding target feature
    @staticmethod
    def generate_config(value: float, model: GraphModel, features_dict: Dict[str, float]) -> Hyperparameter: ...

    @staticmethod
    def generate_weight(value: float, model: GraphModel, features_dict: Dict[str, float]) -> float: ...

    # Transformation goal is to change the density of the sampling distribution for the parameter
    @staticmethod
    def transform(value: float) -> float: ...


class NumberOfVertices:
    name = "n"
    termination_threshold = 100
    
    @staticmethod
    def base_max_bound(value: float):
        max_value = int(math.floor(value + 10 * math.sqrt(value))) + 1
        return max_value

    @staticmethod
    def generate_config(value: float, model: GraphModel, features_dict: Dict[str, float]) -> UniformIntegerHyperparameter:
        # This is a rough heuristics to estimate the range for n
        if model.name() == 'Erdos-Renyi':
            max_value = NumberOfVertices.base_max_bound(value)
        else:
            if features_dict['d'] < 5:
                max_value = 2 * math.floor(value)
            else:
                max_value = NumberOfVertices.base_max_bound(value)
            
        min_value = math.floor(value)
        #config['n'] = (min_value , max_value)
            
        return UniformIntegerHyperparameter(
                "n", lower=min_value, upper=max_value)

    @staticmethod
    def generate_weight(value: float, model: GraphModel, features_dict: Dict[str, float]) -> float:
        if model.name() == 'Erdos-Renyi':
            max_value = NumberOfVertices.base_max_bound(value)
        else:
            if features_dict['d'] < 5:
                max_value = 2 * math.floor(value)
            else:
                max_value = NumberOfVertices.base_max_bound(value)
            
        min_value = math.floor(value)
        return 1 / (max_value - min_value)

    @staticmethod
    def transform(value: float) -> float:
        # No transformation
        return value



class AverageDegree:
    name = "d"
    termination_threshold = 0.1

    @staticmethod
    def generate_config(value: float, model: GraphModel, features_dict: Dict[str, float]) -> UniformFloatHyperparameter:
        # config['d'] = (1, 15)
        return UniformFloatHyperparameter("d", lower=1, upper=15)

    @staticmethod
    def generate_weight(value: float, model: GraphModel, features_dict: Dict[str, float]) -> float:
        #'config['d'] = (1, 15)
        return (1 / 14)

    @staticmethod
    def transform(value: float) -> float:
        # No transformation
        return value


class PowerlawBeta:
    name = "beta"
    termination_threshold = np.inf

    @staticmethod
    def generate_config(value: float, model: GraphModel, features_dict: Dict[str, float]) -> UniformFloatHyperparameter:
        # This will go through transformation in the objective function
        # config['beta'] = (2, 28)
        return UniformFloatHyperparameter(
                name="beta",
                lower=2.0,
                upper=28.0,
                log=True,
                default_value=2.5
                )
        # return UniformFloatHyperparameter("beta", lower=1.5, upper=15)

    @staticmethod
    def generate_weight(value: float, model: GraphModel, features_dict: Dict[str, float]) -> float:
        return 1/ 2

    @staticmethod
    def transform(value: float) -> float:
        return value


class Temperature:
    name = "t"
    termination_threshold = np.inf

    @staticmethod
    def generate_config(value: float, model: GraphModel, features_dict: Dict[str, float]) -> UniformFloatHyperparameter:
        # This will go through exponential transformation
        # The values after transformation: config['t'] = (0, 0.99)
        # uppper_bound is approximately ln(1/(1-0.99))
        return UniformFloatHyperparameter("t", lower=0, upper=5)

    @staticmethod
    def generate_weight(value: float, model: GraphModel, features_dict: Dict[str, float]) -> float:
        return 1

    @staticmethod
    def transform(value: float) -> float:
        new_value =  1 - math.exp(-value)
        assert 0 <= new_value < 1
        return new_value


PARAMS_SPEC: dict[str, type[ParamSpec]] = {
    "n": NumberOfVertices,
    "d": AverageDegree,
    "beta": PowerlawBeta,
    "t": Temperature,
}