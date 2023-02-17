#!/bin/bash

mkdir -p ./benchmark_results
python runner.py --no-intel-optimized --configs ./experiments/svc_config.json --output-file ./benchmark_results/svc_config_OPTOFF.json --verbose INFO --report --device none
