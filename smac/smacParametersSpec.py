from typing import Protocol
import numpy as np
import math
from ConfigSpace.hyperparameters import Hyperparameter, UniformIntegerHyperparameter, UniformFloatHyperparameter

# Smac configuration for the parameters

class ParamSpec(Protocol):
    name: str # Parameter identifier 
    termination_threshold: float | np.floating

    # The value is the value of corresnding target feature
    @staticmethod
    def generate_config(value: float) -> Hyperparameter: ...

    @staticmethod
    def generate_weight(value: float) -> float: ...

    # Transformation goal is to change the density of the sampling distribution for the parameter
    @staticmethod
    def transform(value: float) -> float: ...


class NumberOfVertices:
    name = "n"
    termination_threshold = 100

    @staticmethod
    def generate_config(value: float) -> UniformIntegerHyperparameter:
        max_value = int(math.floor(value + 10 * math.sqrt(value))) + 1
        min_value = math.floor(value)
        #config['n'] = (min_value , max_value)
        
        return UniformIntegerHyperparameter(
            "n", lower=min_value, upper=max_value)

    @staticmethod
    def generate_weight(value: float) -> float:
        # TODO: Is this the correct way to give weight??
        max_value = int(math.floor(value + 10 * math.sqrt(value))) + 1
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
    def generate_config(value: float) -> UniformFloatHyperparameter:
        # config['d'] = (1, 15)
        return UniformFloatHyperparameter("d", lower=1, upper=15)

    @staticmethod
    def generate_weight(value: float) -> float:
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
    def generate_config(value: float) -> UniformFloatHyperparameter:
        # This will go through transformation in the objective function
        # config['beta'] = (1.5, 15)
        return UniformFloatHyperparameter("beta", lower=1.5, upper=15)

    @staticmethod
    def generate_weight(value: float) -> float:
        return 1/ 2

    @staticmethod
    def transform(value: float) -> float:
        assert 1.5 <= value <= 15
        
        # map [2,10] -> [2,3]
        if 2 <= value <= 10:
            # Linear mapping [x0, x1] => [a, b]
            #   formula: new = a + (x - x0) * (b - a) / (x1 - x0)
            return 2 + (value - 2) / 8
    
        return value


class Temperature:
    name = "t"
    termination_threshold = np.inf

    @staticmethod
    def generate_config(value: float) -> UniformFloatHyperparameter:
        # This will go through exponential transformation
        # The values after transformation: config['t'] = (0, 0.999)
        return UniformFloatHyperparameter("t", lower=0, upper=6.9)

    @staticmethod
    def generate_weight(value: float) -> float:
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