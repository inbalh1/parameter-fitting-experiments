from abc import ABC, abstractmethod
import multiprocessing
import numpy as np
from models import GraphModel
from parameters import Parameter
from statistics import Statistics
from typing import Optional


class ParameterFitter(ABC):
    def __init__(self, model: type[GraphModel], target_parameters: list[Parameter]):
        super().__init__()
        self.model = model
        self.target_parameters = target_parameters

    @abstractmethod
    def run(self)->list[Parameter]:
        pass

    @staticmethod
    @abstractmethod
    def name():
        pass

    @abstractmethod
    def statistics()->Optional[Statistics]:
        pass


class RobbinsMonroFinal(ParameterFitter):
    # Maximum number of iterations for the initial phase
    MAX_INITIAL_ITERATIONS = 30
    # Maximum number of iterations for the smoothing phase
    MAX_SMOOTHING_ITERATIONS = 200
    # Minimum number of flips after which we stop the initial phase
    SMOOTHING_MIN_FLIPS = 1
    # Window size for checking whether the algorithm has settled
    SMOOTHING_WINDOW_SIZE = 10

    def __init__(self, model, target_parameters, alpha=0, threshold=0.01):
        super().__init__(model, target_parameters)
        self.parameter_update_callbacks = []
        self.alpha = alpha
        # The maximum relative difference between two iterations that has to hold for the algorithm to count as settled
        self.threshold = threshold

    def add_parameter_update_callback(self, callback):
        self.parameter_update_callbacks.append(callback)

    def _call_parameter_update_callbacks(self, iteration, parameters, flips):
        for callback in self.parameter_update_callbacks:
            callback(iteration, parameters, flips)

    def statistics(self)->Statistics:
        return self.averaging_start, self.total_iterations, self.flips

    def run(self):
        logger = multiprocessing.get_logger()

        params = list(in_param.input_guess(target_param) for in_param, target_param in zip(
            self.model.input_parameters(), self.target_parameters))

        param_values_history = [[] for _ in range(len(params))]
        for i, param in enumerate(params):
            param_values_history[i].append(param.value)

        # We start at -1 flips since the first iteration will always be counted as a flip
        flips = [-1] * len(self.target_parameters)
        prev_signs = [0] * len(self.target_parameters)

        self._call_parameter_update_callbacks(0, params, flips)

        iteration_count = 1
        in_averaging_phase = False
        averaging_start = None
        while True:
            logger.info(f"Iteration {iteration_count}: Now sampling graph")
            param_str = ", ".join(map(repr, params))
            logger.warn(f"Params: {param_str}")

            mo = self.model(*params)
            g = mo.generate()

            # Generate new params
            for pos, (prev_sign, flip, param, target) in enumerate(zip(prev_signs, flips, params, self.target_parameters)):
                measured = target.measure(g)
                measured_str = repr(measured)
                logger.warn(f"Measured param: {measured_str}")

                step_size = 1 / (iteration_count ** self.alpha)
                relation = param.io_relation()
                old_param_value = param.value
                param.value += step_size * relation * \
                    (target.value - measured.value)
                param.clip()

                new_sign = np.sign(param.value - old_param_value)
                # We count unchanged parameter as flip (since this means we settled)
                if new_sign == 0 or new_sign != prev_sign:
                    flips[pos] += 1
                    prev_signs[pos] = new_sign

                param_values_history[pos].append(param.value)

            if not in_averaging_phase:
                logger.info(f"Still in initial phase, flips={flips}")

            # Check for end of phase 1
            if not in_averaging_phase and (min(flips) >= self.SMOOTHING_MIN_FLIPS or iteration_count >= self.MAX_INITIAL_ITERATIONS):
                in_averaging_phase = True
                averaging_start = iteration_count
                averaged_params_values_history = [
                    [] for _ in range(len(params))]
                small_relative_change_history = [[]
                                                 for _ in range(len(params))]
                self.flips = flips[:]

            # Generate current guess based on phase
            averaged_params = []
            for pos, param in enumerate(params):
                if in_averaging_phase:
                    prev_values = param_values_history[pos][averaging_start:]
                    param_val = sum(prev_values) / len(prev_values)
                    averaged_params.append(param.__class__(param_val))
                    averaged_params_values_history[pos].append(param_val)

                    if len(averaged_params_values_history[pos]) > 1:
                        prev_param_val = averaged_params_values_history[pos][-2]
                        if prev_param_val != 0:
                            small_change = abs(
                                1 - param_val / prev_param_val) <= self.threshold
                        else:
                            small_change = (prev_param_val == param_val)
                        small_relative_change_history[pos].append(small_change)
                else:
                    averaged_params.append(param.__class__(param.value))

            self._call_parameter_update_callbacks(
                iteration_count, averaged_params, flips)

            if in_averaging_phase and iteration_count - averaging_start >= self.SMOOTHING_WINDOW_SIZE and all(all(history[-self.SMOOTHING_WINDOW_SIZE:]) for history in small_relative_change_history):
                break

            if in_averaging_phase and iteration_count - averaging_start >= self.MAX_SMOOTHING_ITERATIONS:
                break

            iteration_count += 1

        self.averaging_start = averaging_start
        self.total_iterations = iteration_count

        return averaged_params

    @staticmethod
    def name():
        return 'Robbins-Monro Final'
