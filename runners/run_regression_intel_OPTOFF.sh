#!/bin/bash

mkdir -p ./benchmark_results
python runner.py --no-intel-optimized --configs ./experiments/regression.json --output-file ./benchmark_results/regression_OPTOFF.json --verbose INFO --report --device none
