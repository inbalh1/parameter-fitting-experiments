import networkit
import math


def shrink_to_giant_component(g):
    return networkit.components.ConnectedComponents.extractLargestConnectedComponent(g, compactGraph=True)
