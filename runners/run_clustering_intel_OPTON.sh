#!/bin/bash

mkdir -p ./benchmark_results
python runner.py --configs ./experiments/clustering.json --output-file ./benchmark_results/clustering_OPTON.json --verbose INFO --report --device none
