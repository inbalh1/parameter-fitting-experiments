from typing import Protocol
class Statistics(Protocol):
    averaging_iterations: int
    total_iterations: int
    flips: list[int]