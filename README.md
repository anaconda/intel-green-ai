## Greener Machine Learning Computing with Intel AI Acceleration

This repository contains the experimental package attached to the article "Greener Machine Learning Computing with Intel AI Acceleration"

This documents provides detailed instructions on how to replicate the experimental settings used to measure the **energy consumption** of Machine Learning pipelines as defined in the [IntelPython/Machine Learning Benchmark](https://github.com/IntelPython/scikit-learn_bench).

These experiments compare the performance of stock (i.e. non-optimized) `scikit-learn` algorithms with the corresponding Intel-accelerated ones included in [`scikit-learn-intelex`](https://intel.github.io/scikit-learn-intelex/), the free and open source extension package designed by Intel® to accelerate the Scikit-learn library.

Please make sure that your machine architecture/operating system configuration is [supported](https://intel.github.io/scikit-learn-intelex/system-requirements.html) by Intel®   Acceleration before proceeding.

To quickly check that your computer is supported by Intel Acceleration technology, you could run
the following commands in your terminal:

(On Linux)
```
lscpu | grep -e "sse2\|avx"
```

(on Mac OSX)
```
sysctl -a | grep cpu.feat
```

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
conda env create -f environment.yml
```

This will create a new `green-ai` conda environment. To double check this, please execute:

```shell
conda info --envs | grep "green-ai"
```

You should get an output similar to: 

```shell
green-ai                 /Users/leriomaggio/anaconda3/envs/green-ai
```

The **last step** is to _activate_ the new conda environment:
```shell
conda activate green-ai
```

**Note** 
Please also make sure to run the latest version of `conda` on your machine:

```shell
conda update -n base conda
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
git clone https://github.com/leriomaggio/scikit-learn_bench.git -b anaconda-intel-green-ai ./anaconda-intel-green-ai
```

**Note**: Please note that we will be downloading and use a specific _tagged_ version of the benchmark:
1. The _tag_ considers the _exact_ same version of code and data used in the experiments.
2. This version of the benchmark also includes an extra utility script that can be used to download all the necessary benchmark data (see next section). This script is currently part of a [PR](https://github.com/IntelPython/scikit-learn_bench/pull/129) not yet merged into the official `main` branch.

#### Downloading the Benchmark datasets

It is highly recommended to **download** all the necessary datasets **before** running the experiments. This would avoid any issue during the execution derived by connectivity failures during the download. Moreover, this would also favour a fairer comparisong between multiple experiments, by not including any accidental overhead caused by network failures during the execution.

Assuming that you have cloned the benchmark code directly within the `intel-green-ai` main folder, 
to download all the dataset, run the following command:
```shell
cd anaconda-intel-green-ai
DATASETSROOT=./data python -m datasets.load_datasets --configs ../experiments/skl_public_config.json
```

This will download the `25` publicly available datasets used in the benchmark experiments. For further information, please refer to the official [documentation](https://github.com/leriomaggio/scikit-learn_bench/blob/anaconda-intel-green-ai/datasets/README.md).

⚠️ **Please be aware** that this may take several minutes to complete, depending on your Internet connectivity, and it will require around `18 GB` of disk space.
Moreover, if you are executing the download on a remote machine, it is **higly** recommended to run the previous command within a [`tmux`](https://github.com/tmux/tmux/wiki) session.

## Measuring Energy Consumption

To measure energy consumption of machine learning experiments, we need (A) to make sure that information about consumed energy is reliably provided by our operating system; and (B) download all the necessary software to read this information during the execution.

To measure the energy consumed by each experiment we used `RAPL`, that is `R`unning `A`verage `P`ower `L`imit. This technology has been available on any **Intel CPU** since the _Sandy Bridge generation_ (2010, ed.), and it is supported by any operating system (e.g. the [Power Capping](https://www.kernel.org/doc/html/latest/power/powercap/powercap.html) framework on Linux). RAPL allows to estimates the power consumption of the CPU, RAM and any integrated GPU (if any) in real time.

In our experiments, we used a `c5.metal` [Amazon EC2 instance](https://aws.amazon.com/ec2/instance-types/c5/)  with [Ubuntu 20.04.5 LTS](https://releases.ubuntu.com/focal/) operating system. Therefore, all the following instructions will consider this configuration as the reference to enable RAPL readings.

#### Configure access to RAPL

The `powercap` framework is **not** enabled by default on Ubuntu OS running on AWS instances. Therefore, the first thing to do is to install all the required _kernel modules_.

**Note**: Please make sure you are running the following commands using a user account with `sudo` permissions.

```shell
sudo apt install linux-modules-extra-$(uname -r)
sudo update-initramfs -c -k $(uname -r)
```

Afterwards, we need to dynamically `probe` (i.e. load) these modules into the kernel. 
To get the names of these modules, we could search for any kernel module that includes `rapl` in their name:

```shell
find /lib/modules/$(uname -r) -name *rapl*
```
The output of the `find` command should look similar to:

```shell
/lib/modules/5.15.0-1030-aws/kernel/drivers/thermal/intel/int340x_thermal/processor_thermal_rapl.ko
/lib/modules/5.15.0-1030-aws/kernel/drivers/powercap/intel_rapl_msr.ko
/lib/modules/5.15.0-1030-aws/kernel/drivers/powercap/intel_rapl_common.ko
/lib/modules/5.15.0-1030-aws/kernel/arch/x86/events/rapl.ko
```

Therefore, we can load these modules into the kernel by running the following commands:

```shell
sudo modprobe rapl
sudo modprobe intel_rapl_common
sudo modprobe intel_rapl_msr
sudo modprobe processor_thermal_rapl
```

To verify that all went well, we should now be able to see listed the `powercap` folder under the `/sys/class/` folder:

```shell
ls /sys/class/powercap/
intel-rapl  intel-rapl:0  intel-rapl:0:0  intel-rapl:1  intel-rapl:1:0
```

That's all! Well done! 🎉 This was the hardest part. From now on, the rest will be _piece of cake_, I promise. 

#### Tool to monitor energy consumption

We used [`jouleit`](https://powerapi-ng.github.io/jouleit.html) to monitor the energy consumption of our experiments.  `jouleit` does not require any installation, and it is very easy to use. 

To download `jouleit` it is just necessary to clone its GitHub repository:

```shell
git clone https://github.com/powerapi-ng/jouleit.git <path to>/jouleit
cd jouleit
```

**To verify** that `jouleit` works and that it's able to read data from RAPL, we could run the script using the `-l` option:

```shell
./jouleit.sh -l
CPU;DRAM;DURATION;EXIT_CODE
```

The output correspond to the list of headers that the script is able to access, and read from. In this case, `CPU`, `DRAM`, execution time, and `exit_code`of the executed program.

**Note**: If `CPU` and `DRAM` won't be present in the output of the previous command, this would mean that the access to RAPL has not been properly configured.

## Running the Benchmark

The [`runners`](./runners) folder contains all the script required to execute the multiple benchmark configurations, _with_ (i.e. `xxx_intel_OPTON.sh`) and _without_ (`xxx_intel_OPTOFF.sh`) Intel AI Acceleration. 

These scripts will be used in conjuction with `jouleit` to also monitor energy consumption of each experiment.

First, I would recommend copying the `experiments` folder in the main benchmark folder, i.e. the `anaconda-intel-green-ai` folder created in step 2.1:

```shell
cp -r ./experiments <path to>/anaconda-intel-green-ai
```

Similarly, we should copy all the runner-scripts in the `anaconda-intel-green-ai` main folder, as well:

```shell
cp ./runners/run_* <path to>/anaconda-intel-green-ai
```

This is required to make sure that all the paths and dependencies will be available when starting the benchmark execution.

Now the last step: executing a single benchmark experiment, whilst also monitoring energy consumption using `jouleit`:

```shell
cd <path to>/anaconda-intel-green-ai
<path to>/jouleit/jouleit.sh ./run_XXX.sh
```

⚠️ **A few notes on privilegies and permissions**

All the RAPL _virtual device_ files require **root privilegies** to be accessed for security reasons. This means that the `jouleit.sh` script requires _root_ to be used. 
However, if we would run all the commands using directly the `root` user (e.g. using `sudo`), we would also need to setup 
the whole conda environment for root, and that would be unpractical. 
Therefore, as a _workaround_  **for the sole sake of this benckmark experiments** - we could change the ownership of those files to grant access:


```shell
sudo chown ubuntu:ubuntu /sys/class/powercap/intel-rapl*
```

##### Running `Support Vector Machines` experiments

The optimised version of the Support Vector machine included in `sklearnex` requires `OpenCL` to be installed and configured. 
To do so, please make sure that all the required libraries are linked in the right location:

```shell
sudo mkdir -p /etc/OpenCL/vendors/
sudo bash -c "echo libintelocl.so > /etc/OpenCL/vendors/intel64.icd"
```
