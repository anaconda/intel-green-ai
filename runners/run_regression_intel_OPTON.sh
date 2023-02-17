#!/bin/bash

mkdir -p ./benchmark_results
python runner.py --configs ./experiments/regression.json --output-file ./benchmark_results/regression_OPTON.json --verbose INFO --report --device none
