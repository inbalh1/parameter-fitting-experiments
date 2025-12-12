import argparse
import csv
import sys
from pathlib import Path
import networkit
from parameters import NumberOfVertices, NumberOfEdges, AverageDegree, PowerlawBeta, Temperature

class TargetParameterMeasurer:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        self._results_initialized = False

        networkit.engineering.setNumberOfThreads(1)

    def execute(self):
        try:
            g = networkit.readGraph(
                str(self.input_file),
                networkit.Format.NetworkitBinary)
        except Exception as e:
            print(e, file=sys.stderr)
            return

        if not g:
            return

        input_params = NumberOfVertices, NumberOfEdges, AverageDegree, PowerlawBeta, Temperature
        output_params = [input_param.output_parameter()
                         for input_param in input_params]
        params = [param.measure(g) for param in output_params]
        output = {}
        for param in params:
            output[param.name()] = param.value

        with open(self.output_file, "w") as results_file:
            fieldnames = sorted(set(output.keys()))
            dict_writer = csv.DictWriter(results_file, fieldnames)
            dict_writer.writeheader()
            dict_writer.writerow(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str)
    parser.add_argument('output_file', type=str)

    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file

    runner = TargetParameterMeasurer(
        input_file=input_file, output_file=output_file)
    runner.execute()
