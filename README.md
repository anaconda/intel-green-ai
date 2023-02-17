## Greener Machine Learning Computing with Intel AI Acceleration

This repository contains the experimental package attached to the article "Greener Machine Learning Computing with Intel AI Acceleration" [_link online version here_]

This documents provides detailed instructions on how to replicate the experimental settings used to measure the **energy consumption** of Machine Learning pipelines as defined in the [IntelPython/Machine Learning Benchmark](https://github.com/IntelPython/scikit-learn_bench).

These experiments compare the performance of stock (i.e. non-optimized) `scikit-learn` algorithms with the corresponding Intel-accelerated ones included in [`scikit-learn-intelex`](https://intel.github.io/scikit-learn-intelex/), the free and open source extension package designed by Intel® to accelerate the Scikit-learn library.

Please make sure that your machine architecture/operating system configuration is [supported](https://intel.github.io/scikit-learn-intelex/system-requirements.html) by Intel®   Acceleration before proceeding.

### Sections

1. [Conda environment](#set-up-the-conda-environment)
2. [Data and Code Availability](#data-and-code-availability)
    2.1 [Download Code](#download-the-machine-learning-benchmark-code)
    2.2 [Download Data](#downloading-the-benchmark-datasets)
3. [Measuring Energy Consumption](#measuring-energy-consumption)
    3.1 [Configure RAPL access](#configure-access-to-rapl) 
    3.2 [Tool to monitor energy consumption](#tool-to-monitor-energy-consumption)
4. [Running the Benchmark](#running-the-benchmark)
### Set up the `conda` environment

The repository provides an  [`environment.yml`](./environment.yml) file that can be used to easy replicate the `conda` environment used in the experiments.

To do so, it is just necessary to have a working version of `conda` installed on your machine. 

Alternatively, it is recommended to  [download](https://www.anaconda.com/products/distribution) and install the **Anaconda Distribution** specific for your architecture, and operating system.

Once conda is available, please run the following command in the Terminal:

```shell
$ conda env create -f environment.yml
```

This will create a new `green-ai` conda environment. To double check this, please execute:

```shell
$ conda info --envs | grep "green-ai"
```

You should get an output similar to: 

```shell
green-ai                 /Users/leriomaggio/anaconda3/envs/green-ai
```

The **last step** is to _activate_ the new conda environment:
```shell
$ conda activate green-ai
```

**Note** 
Please also make sure to run the latest version of `conda` on your machine:

```shell
$ conda update -n base conda
```

### Data and Code Availability

All the experiments reported in the article use the [IntelPython/Machine Learning Benchmark](https://github.com/IntelPython/scikit-learn_bench): a public and open source benchmark for machine learning experiments that supports several machine learning algorithms across multiple data analytics frameworks (e.g. `scikit-learn`, `cuML`, `XGBoost`).

The benchmark expects experiments to be set up via [configuration files](https://github.com/IntelPython/scikit-learn_bench#running-python-benchmarks-with-runner-script) in JSON format.

The [experiments](./experiments) folder in this repository contains the configuration files used to run all the experiments described in the article. 

In more details:

| Experiment  | Configuration File | Description |
| ----------- | ------------------ | ----------- |
| `Scikit-Learn Public Datasets`  | [skl_public_config.json](./experiments/skl_public_config.json)  | Scikit-learn default benchmark on publicly available datasets |
| `Classification` models | [classification.json](./experiments/classification.json)  | Classification tasks using `LogisticRegression` and `RandomForestClassifier` models
| `Regression` models |[regression.json](./experiments/regression.json)  | Regression tasks using `LinearRegression`, `Ridge`, and `RandomForestRegressor` models|
| `Clustering` models |[clustering.json](./experiments/clustering.json)  | Clustering tasks using `KMeans`, and `DBScan` models|
| `Dimension Reduction` models| [dimension_reduction.json](./experiments/dimension_reduction.json) | Dimension Reduction experiments using `PCA` and `t-SNE` models|
| `Support Vector Machines` | [svc_config.json](./experiments/svc_config.json) | Classification and Regression tasks with `SVC`, and `SVR` models|

#### Download the Machine Learning Benchmark Code

To download the code necessary to run the experiments, it is just necessary to clone the reference repository from GitHub: 

```shell
$ git clone https://github.com/leriomaggio/scikit-learn_bench.git -b anaconda-intel-green-ai ./anaconda-intel-green-ai
```

**Note**: Please note that we will be downloading and use a specific _tagged_ version of teh benchmark, gathered from my _fork_ of the original `scikit-learn_bench` project. 
Reasons for this are two:

1. The _tag_ considers the _exact_ same version of code and data used in the experiments. 
2. This version of the benchmark also includes an extra utility script that can be used to download all the necessary benchmark data (see next section). This script currently part of a [PR](https://github.com/IntelPython/scikit-learn_bench/pull/129) not yet merged into the official `main` branch.

#### Downloading the Benchmark datasets

It is highly recommended to **download** all the necessary datasets **before** running the experiments. This would avoid any issue during the execution derived by connectivity failures during the download. Moreover, this would also favour a fairer comparisong between multiple experiments, by not including any accidental overhead caused by network failures during the execution.

To download all the dataset, run the following command:
```shell
$ cd anaconda-intel-green-ai
$ DATASETSROOT=./data python -m datasets.load_datasets --configs ../intel-green-ai/experiments/skl_public_config.json
```

This will download the `25` publicly available datasets used in the benchmark experiments. For further information, please refer to the official [documentation](https://github.com/leriomaggio/scikit-learn_bench/blob/anaconda-intel-green-ai/datasets/README.md).

⚠️ **Please be aware** that this may take several minutes to complete, depending on your Internet connectivity, and it will occupy around `18 GB` of disk space.

## Measuring Energy Consumption

To measure energy consumption of machine learning experiments, we need (A) to make sure that information about consumed energy is reliably provided by our operating system; and (B) download all the necessary software to read this information during the execution.

To measure the energy consumed by each experiment we used `RAPL`, that is `R`unning `A`verage `P`ower `L`imit. This technology has been available on any **Intel CPU** since the _Sandy Bridge generation_ (2010, ed.), and it is supported by any operating system (e.g. the [Power Capping](https://www.kernel.org/doc/html/latest/power/powercap/powercap.html) framework on Linux). RAPL allows to estimates the power consumption of the CPU, RAM and any integrated GPU (if any) in real time.

In our experiments, we used a `c5.metal` [Amazon EC2 instance](https://aws.amazon.com/ec2/instance-types/c5/)  with [Ubuntu 20.04.5 LTS](https://releases.ubuntu.com/focal/) operating system. Therefore, all the following instructions will consider this configuration as the reference to enable RAPL readings.

#### Configure access to RAPL

The `powercap` framework is **not** enabled by default on Ubuntu OS running on AWS instances. Therefore, the first thing to do is to install all the required _kernel modules_:

```shell
$ sudo apt install linux-modules-extra-$(uname -r)
$ sudo update-initramfs -c -k $(uname -r)
```

Once installed, the next step is to dynamically `probe` (i.e. load) these modules into the kernel. To get the names of these modules, we could search for any kernel module including `rapl` in their name by running:

```shell
$ find /lib/modules/$(uname -r) -name *rapl*
```
The output of the `find` command should look similar to:

```shell
/lib/modules/5.15.0-1030-aws/kernel/drivers/thermal/intel/int340x_thermal/processor_thermal_rapl.ko
/lib/modules/5.15.0-1030-aws/kernel/drivers/powercap/intel_rapl_msr.ko
/lib/modules/5.15.0-1030-aws/kernel/drivers/powercap/intel_rapl_common.ko
/lib/modules/5.15.0-1030-aws/kernel/arch/x86/events/rapl.ko
```

To load these modules into the kernel run the following commands:

```shell
$ sudo modprobe rapl
$ sudo modprobe intel_rapl_common
$ sudo modprobe intel_rapl_msr
$ sudo modprobe processor_thermal_rapl
```

To verify that all went well, we should now be able to see listed the `powercap` folder under `/sys/class/`:

```shell
$ ls /sys/class/powercap/
intel-rapl  intel-rapl:0  intel-rapl:0:0  intel-rapl:1  intel-rapl:1:0
```

That's all! Well done! 🎉 This was the hardest part. The rest from now on will be _piece of cake_. 

#### Tool to monitor energy consumption

We used [`jouleit`](https://powerapi-ng.github.io/jouleit.html) to monitor the energy consumption of our experiments.  `jouleit` does not require any installation, and it is very easy to use. 

To download `jouleit` it is just necessary to clone its GitHub repository:

```shell
$ git clone https://github.com/powerapi-ng/jouleit.git
$ cd jouleit
```

**To verify** that `jouleit` works and that it's able to read data from RAPL, we could run the script using the `-l` option:

```shell
$ ./jouleit.sh -l
CPU;DRAM;DURATION;EXIT_CODE
```

The output correspond to the list of headers that the script is able to access, and read from. In this case, `CPU`, `DRAM`, execution time, and `exit_code`of the executed program.

**Note**: If `CPU` and `DRAM` won't be present in the output of the previous command, this would mean that the access to RAPL has not been properly configured.

⚠️ **A few notes on privilegies and permissions**

All the RAPL _virtual device_ files require **root privilegies** to be accessed for security reasons. This means that the `jouleit.sh` script requires _root_ to be used:

As a quick workaround  - **to use for the sole sake of this benckmark** - ownership to those files could be changed to allow access. 

## Running the Benchmark

The [`runners`](./runners) folder contains all the script required to execute the multiple benchmark configurations, _with_ (i.e. `xxx_intel_OPTON.sh`) and _without_ (`xxx_intel_OPTOFF.sh`) Intel AI Acceleration. 

These scripts will be used in conjuction with `jouleit` to also monitor energy consumption of each experiment.

First, I would recommend to copy the `experiments` folder with all the configuration files in the `anaconda-intel-green-ai` folder where the code has been downloaded:

```shell
$ cp -r ./experiments <path to>/anaconda-intel-green-ai
```

Similarly, we could copy all the runner-scripts in the `anacodna-intel-green-ai` main folder:

```shell
$ cp ./runners/run_* <path to>/anaconda-intel-green-ai
```

To run benchmark, whilst also monitoring energy consumption:

```shell
$ ../jouleit/jouleit.sh ./run_XXX.sh
```