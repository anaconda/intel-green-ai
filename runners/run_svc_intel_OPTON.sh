#!/bin/bash

mkdir -p ./benchmark_results
python runner.py --configs ./experiments/svc_config.json --output-file ./benchmark_results/svc_config_OPTON.json --verbose INFO --report --device none
