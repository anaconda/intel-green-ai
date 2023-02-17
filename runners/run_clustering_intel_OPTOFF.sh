#!/bin/bash

mkdir -p ./benchmark_results
python runner.py --no-intel-optimized --configs ./experiments/clustering.json --output-file ./benchmark_results/clustering_OPTOFF.json --verbose INFO --report --device none
