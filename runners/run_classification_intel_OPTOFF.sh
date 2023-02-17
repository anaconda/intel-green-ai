#!/bin/bash

mkdir -p ./benchmark_results
python runner.py --no-intel-optimized --configs ./experiments/classification.json --output-file ./benchmark_results/classification_OPTOFF.json --verbose INFO --report --device none
