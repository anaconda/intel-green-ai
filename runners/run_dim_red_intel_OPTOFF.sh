#!/bin/bash

mkdir -p ./benchmark_results
python runner.py --no-intel-optimized --configs ./experiments/dimension_reduction.json --output-file ./benchmark_results/dimension_reduction_OPTOFF.json --verbose INFO --report --device none
