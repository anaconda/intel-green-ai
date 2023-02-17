#!/bin/bash

mkdir -p ./benchmark_results
python runner.py --configs ./experiments/classification.json --output-file ./benchmark_results/classification_OPTON.json --verbose INFO --report --device none
