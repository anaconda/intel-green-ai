#!/bin/bash

mkdir -p ./benchmark_results
python runner.py --configs ./experiments/dimension_reduction.json --output-file ./benchmark_results/dimension_reduction_OPTON.json --verbose INFO --report --device none
