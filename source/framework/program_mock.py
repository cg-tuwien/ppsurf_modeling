import argparse
from os import path
from sys import exit
import random
from time import sleep


class Mock:
    def __init__(self, input_file, output_file, error, min_time, max_time):
        self._input = input_file
        self._output = output_file
        self._error = error
        self._min = min_time
        self._max = max_time

    def run(self):
        # Throw error if input file does not exist
        if not path.exists(self._input):
            exit("Input file does not exist!")

        # Simulate processing time
        sleep(random.randrange(self._min, self._max+1) / 1000)

        # Simulate processing errors with given probability
        if self._error and random.random() < self._error:
            exit("Error during processing!")

        # If no output file name is given use input name and append ".out"
        if self._output is None or self._output == "":
            self._output = self._input + ".out"

        # Create dummy output file if it doesn't exist already
        if not path.exists(self._output):
            open(self._output, "a").close()


if __name__ == '__main__':
    # Create new argument parser
    parser = argparse.ArgumentParser(prog="program_mock", description="Mock program call which takes an input file and "
                                                                      "process it into output file")
    # Set options and arguments for parser
    parser.add_argument("-e", "--error", type=int, help="probability in percent to throw error while processing a file")
    parser.add_argument("--min", type=int, help="minimum processing time given in ms (default: 0)", default=0)
    parser.add_argument("--max", type=int, help="maximum processing time given in ms (default: 1000)", default=1000)
    parser.add_argument("input", type=str, help="input file")
    parser.add_argument("output", nargs="?", type=str, help="output file", default="")

    # Parse program arguments
    args = parser.parse_args()

    # Create mock object with given parameters
    mockProgram = Mock(str(args.input), str(args.output), None if not args.error else float(args.error/100),
                       int(args.min), int(args.max))

    # Run mocked program
    print("Starting mock...")
    mockProgram.run()
    print("Mock finished...")
