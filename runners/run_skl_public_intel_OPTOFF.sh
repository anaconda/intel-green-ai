#!/bin/bash

mkdir -p ./benchmark_results
python runner.py --no-intel-optimized --configs ./experiments/skl_public_config.json --output-file ./benchmark_results/skl_public_config_OPTOFF.json --verbose INFO --report --device none
