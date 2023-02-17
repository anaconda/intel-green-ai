#!/bin/bash

mkdir -p ./benchmark_results
python runner.py --configs ./experiments/skl_public_config.json --output-file ./benchmark_results/skl_public_config_OPTON.json --verbose INFO --report --device none
