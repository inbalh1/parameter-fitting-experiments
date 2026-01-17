import networkit
import powerlaw
import math


def shrink_to_giant_component(g):
    return networkit.components.ConnectedComponents.extractLargestConnectedComponent(g, compactGraph=True)


# Return a power-law distribution
def powerlaw_generate(n, d, beta):
    import numpy as np

    n = int(n)

    degrees = np.array([0.]*n)
    for i in range(n):
        deg = (i+1) ** (1 / (-beta+1))
        degrees[i] = deg

    factor = d * n / sum(degrees)

    degrees *= factor
    degrees = np.around(degrees)

    # print("powerlaw: min {}, max {}".format(generator.getMinimumDegree(), generator.getMaximumDegree()), file=sys.stderr)
    # print("powerlaw: wanted k={}, will get {}".format(d, sum(degrees) / n, file=sys.stderr))

    return list(degrees)
