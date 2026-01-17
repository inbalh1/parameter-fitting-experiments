from abc import ABC, abstractmethod
import random

import networkit
from pygirgs import hypergirgs, girgs

from helpers import shrink_to_giant_component, powerlaw_generate
from parameters import NumberOfVertices, NumberOfEdges, AverageDegree, PowerlawBeta, Temperature


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


class ChungLuPowerlaw(GraphModel):
    @staticmethod
    def input_parameters():
        return NumberOfVertices, AverageDegree, PowerlawBeta

    def _generate(self):
        n, d, beta = map(lambda param: param.value, self.parameters)
        d = min(d, n - 1)

        if d == 0 or n == 1:
            return networkit.Graph(1)

        degree_sequence = powerlaw_generate(n, d, beta)
        if self.seed is not None:
            networkit.setSeed(seed=self.seed, useThreadId=False)
        return networkit.generators.ChungLuGenerator(degree_sequence).generate()

    @staticmethod
    def name():
        return 'Chung-Lu-PL'


class GIRG(GraphModel):
    dimension = NotImplementedError

    def __init_subclass__(cls, /, dimension, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls.dimension = dimension

    @staticmethod
    def input_parameters():
        return NumberOfVertices, AverageDegree, PowerlawBeta, Temperature

    def _generate(self):
        if self.seed is not None:
            random.seed(self.seed)
        wseed = random.randrange(10000)
        pseed = random.randrange(10000)
        sseed = random.randrange(10000)

        n, deg, beta, t = map(lambda param: param.value, self.parameters)
        n = int(n)
        deg = min(deg, n - 1)
        alpha = 1 / t

        # Handle special case of empty graph
        if deg == 0.0:
            return networkit.Graph(1)

        weights = girgs.generate_weights(n, beta, wseed, False)
        positions = girgs.generate_positions(n, self.dimension, pseed, False)
        scaling = girgs.scale_weights(weights, deg, self.dimension, alpha)
        weights = [scaling * weight for weight in weights]
        edges = girgs.generate_edges(weights, positions, alpha, sseed)

        g = networkit.Graph(n)

        for u, v in edges:
            g.addEdge(u, v)

        return g

    @classmethod
    def name(cls):
        return f'GIRG-{cls.dimension}d'


class GIRG1D(GIRG, dimension=1):
    pass


class GIRG2D(GIRG, dimension=2):
    pass


class GIRG3D(GIRG, dimension=3):
    pass


ALL_MODELS = [ErdosRenyi, ChungLuPowerlaw, GIRG1D]
