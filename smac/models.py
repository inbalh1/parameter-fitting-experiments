from abc import ABC, abstractmethod
import random

import networkit


from helpers import shrink_to_giant_component
from parameters import NumberOfVertices, NumberOfEdges, AverageDegree, Temperature


class GraphModel(ABC):
    def __init__(self, *parameters, reduce_to_largest_cc=True, seed: int | None = None):
        super().__init__()
        self.parameters = parameters
        self.reduce_to_largest_cc = reduce_to_largest_cc
        self.seed = seed

    @staticmethod
    @abstractmethod
    def input_parameters():
        pass

    @classmethod
    def measure_output_parameters(cls, g):
        params = []
        for input_param in cls.input_parameters():
            output_param = input_param.output_parameter()
            params.append(output_param.measure(g))
        return params

    def generate(self):
        g = self._generate()
        if self.reduce_to_largest_cc:
            return shrink_to_giant_component(g)
        return g

    @abstractmethod
    def _generate(self):
        pass

    @staticmethod
    @abstractmethod
    def name():
        pass


class ErdosRenyi(GraphModel):
    @staticmethod
    def input_parameters():
        return NumberOfVertices, AverageDegree

    def _generate(self):
        n, d = map(lambda param: param.value, self.parameters)
        n = int(n)
        if n < 2:
            p = 0
        else:
            p = d / (n - 1)
        if self.seed is not None:
            networkit.setSeed(seed=self.seed, useThreadId=False)
        return networkit.generators.ErdosRenyiGenerator(n, p).generate()

    @staticmethod
    def name():
        return 'Erdos-Renyi'

# Erdos renyi that provides also m (for MLE)
class ErdosRenyiExtended(GraphModel):
    @staticmethod
    def input_parameters():
        return NumberOfVertices, AverageDegree, NumberOfEdges

    def _generate(self):
        print('all parameters are: ', self.parameters)
        n, d = map(lambda param: param.value, self.parameters)
        n = int(n)
        if n < 2:
            p = 0
        else:
            p = d / (n - 1)
        if self.seed is not None:
            networkit.setSeed(seed=self.seed, useThreadId=False)
        return networkit.generators.ErdosRenyiGenerator(n, p).generate()

    @staticmethod
    def name():
        return 'Erdos-Renyi-Extended'



ALL_MODELS = [ErdosRenyi, ErdosRenyiExtended]
