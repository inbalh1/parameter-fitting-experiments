from abc import ABC, abstractmethod
import math

import networkit


class Parameter(ABC):
    def __init__(self, value):
        super().__init__()
        self.value = value

    @staticmethod
    @abstractmethod
    def name():
        pass

    def __repr__(self):
        return f"{self.name()}({self.value})"


class InputParameter(Parameter):
    @abstractmethod
    def clip(self):
        pass

    def loss(self, target):
        assert type(self) is type(
            target), 'Can only calculate loss with same param type!'
        return abs(target.value - self.value) / target.value

    @classmethod
    def input_guess(cls, output_param):
        return cls(output_param.value)

    @classmethod
    @abstractmethod
    def output_parameter(cls):
        pass

    @classmethod
    def io_relation(cls):
        # +1 = positive correlation
        # -1 = negative correlation
        return +1


class OutputParameter(Parameter):
    @classmethod
    @abstractmethod
    def measure(cls, g):
        pass


class NumberOfVertices(InputParameter, OutputParameter):
    MAX_VERTEX_COUNT = 6 * 10**6

    def __init__(self, value):
        super().__init__(value)
        self.value = float(self.value)

    @staticmethod
    def name():
        return 'n'

    def clip(self):
        self.value = max(self.value, 1)
        self.value = min(self.value, self.MAX_VERTEX_COUNT)

    @classmethod
    def measure(cls, g):
        return cls(g.numberOfNodes())

    @classmethod
    def output_parameter(cls):
        return cls


class NumberOfEdges(InputParameter, OutputParameter):
    def __init__(self, value):
        super().__init__(value)
        self.value = float(self.value)

    @staticmethod
    def name():
        return 'm'

    def clip(self):
        self.value = max(self.value, 0)

    @classmethod
    def measure(cls, g):
        return cls(g.numberOfEdges())

    @classmethod
    def output_parameter(cls):
        return cls


class AverageDegree(InputParameter, OutputParameter):
    def __init__(self, value):
        super().__init__(value)
        self.value = float(self.value)
        self.clip()

    @staticmethod
    def name():
        return 'd'

    def clip(self):
        self.value = max(self.value, 1.0)

    @classmethod
    def measure(cls, g):
        return cls(2 * g.numberOfEdges() / g.numberOfNodes())

    @classmethod
    def output_parameter(cls):
        return cls


class Heterogeneity(OutputParameter):
    def __init__(self, value):
        super().__init__(value)
        self.value = float(self.value)

    @staticmethod
    def name():
        return 'heterogeneity'

    @classmethod
    def measure(cls, g):
        degrees = networkit.centrality.DegreeCentrality(g).run().scores()
        avg_deg = sum(degrees) / len(degrees)
        if avg_deg == 0:
            return cls(0)
        variance = sum((d - avg_deg)**2 for d in degrees) / len(degrees)
        # Lowest possible value
        if variance == 0:
            return cls(-0.5)
        return cls(math.log10(math.sqrt(variance) / avg_deg))


class PowerlawBeta(InputParameter):
    def __init__(self, value):
        super().__init__(value)
        self.value = float(self.value)
        self.clip()

    @staticmethod
    def name():
        return 'beta'

    def clip(self):
        self.value = max(self.value, 2.1)
        self.value = min(self.value, 25.0)

    @classmethod
    def input_guess(cls, output_param):
        return cls(3.0)

    @classmethod
    def output_parameter(cls):
        return Heterogeneity

    @classmethod
    def io_relation(cls):
        # +1 = positive correlation
        # -1 = negative correlation
        return -1


class Temperature(InputParameter):
    def __init__(self, value):
        super().__init__(value)
        self.value = float(self.value)

    @staticmethod
    def name():
        return 't'

    def clip(self):
        self.value = max(self.value, 0.01)
        self.value = min(self.value, 0.9999)

    @classmethod
    def input_guess(cls, output_param):
        return cls(0.5)

    @classmethod
    def io_relation(cls):
        return -1

    @classmethod
    def output_parameter(cls):
        return ClusteringCoefficient


class ClusteringCoefficient(OutputParameter):
    def __init__(self, value):
        super().__init__(value)
        self.value = float(self.value)

    @staticmethod
    def name():
        return 'cc'

    def loss(self, target):
        assert type(self) is type(
            target), 'Can only calculate loss with same param type!'
        return abs(target.value - self.value)

    @classmethod
    def measure(cls, g):
        clustering = networkit.centrality.LocalClusteringCoefficient(
            g, turbo=True)
        clustering.run()
        clustering_vals = clustering.scores()
        return cls(sum(clustering_vals) / len(clustering_vals))
