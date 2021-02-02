# Introduction
HingeTreeForTorch is a C++ PyTorch extension of RandomHingeForest.

# Tested Environments
HingeTreeForTorch has been built and tested in the following environments
- Ubuntu 18.04
  - Python 3.8.0
  - GCC 7.5
  - CUDA 10.1
  - CUDNN 7.6.3
- CentOS 7.5
  - Python 3.8.5
  - GCC 8.3.0 and 9.2.0
  - CUDA 10.2 and 11.0.3
  - CUDNN 7.6.5 and 8.0.3
- OS X 10.15.7
  - Python 3.8.2
  - XCode 12.4 (Clang 12.0.0)
- Windows 10
  - Python 3.8.5
  - Visual Studio 2017 Community Edition
  - CUDA 10.2
  - CUDNN 7.6.5

# Compiling HingeTree
The HingeTree PyTorch extension can be compiled using setup.py
```shell
python setup.py build
python setup.py install
```

**NOTE**: Ensure that you are using Python 3.6 or later.

## Mac OS X
If you experience any compiler errors, try the following
```shell
export ARCHFLAGS="-arch x86_64"
```
Then try again. Some recent version OS X sets this environment variable to simultaneously build for arm64 and x86_64!

## Windows 10
I have tested building HingeTreeForTorch with Visual Studio 2017 Community Edition and CUDA 10.2 against both PyTorch 1.7.1 and PyTorch 1.8.0 Nightly. I could not build HingeTreeForTorch in PyTorch 1.7.1 owing to uninitialized constexpr variables in PyTorch headers (nvcc error). If you experience this error, install PyTorch 1.8.0 Nightly and try again.

# Testing HingeTree
Once compiled and installed, you should be able to run the speed test
```py
import torch
from HingeTree import *
device = "cuda:0" if torch.cuda.is_available() else "cpu"
x = torch.rand([100, 1000]).to(device)
timings = HingeTree.speedtest(x)
```

# Basic Usage
There are two variants of tree-based learning machines, the `RandomHingeForest` and `RandomHingeFern`. They differ in their decision structure and parameter count, but otherwise behave identically

```py
from RandomHingeForest import RandomHingeForest, RandomHingeFern

forest = RandomHingeForest(in_channels=numFeatures, out_channels=numTrees, depth=treeDepth)
fern = RandomHingeFern(in_channels=numFeatures, out_channels=numTrees, depth=fernDepth)

# Process a batch of size 32
x = torch.rand([32, numFeatures])
y = forest(x)
```
By default, the `RandomHingeForest` and `RandomHingeFern` take in \[batchSize,numFeatures,d0,d1,...\] tensors and produce \[batchSize,numTrees,d0,d1...\] tensors. You can change the number of predictions per leaf with `extra_outputs=numOutputs`
```py
forest = RandomHingeForest(in_channels=numFeatures, out_channels=numTrees, depth=treeDepth, extra_outputs=numOutputs) # Predict numOutputs values per tree
```
In which case the `RandomHingeForest` and `RandomHingeFern` take in \[batchSize,numFeatures,d0,d1,...\] tensors and produce \[batchSize,numTrees,d0,d1...,numOutputs\] tensors.

# Reproducibility
Forward passes are always deterministic on both the CPU and GPU. Backward passes are always deterministic on the CPU and, **by default**, deterministic on the GPU. The deterministic GPU backward pass can be very slow!

You can use `deterministic=False` to ensure gradient calculations on the GPU are calculated as fast as possible, though the result is not deterministic.
```py
forest = RandomHingeForest(in_channels=numFeatures, out_channels=numTrees, depth=treeDepth, deterministic=False)
```

# Initialization
`RandomHingeForest` and `RandomHingeFern` support two types of initialization: random, sequential.

## `random`
Random initialization picks tree/fern decision thresholds on Uniform(-3,3), weights as Gaussian(0,1) and splitting feature indices on U\[0,numFeatures-1\]. This is the default initialization behavior.

```py
forest = RandomHingeForest(in_channels=numFeatures, out_channels=numTrees, depth=treeDepth, init_type="random")
```

## `sequential`
Sequential initialization picks tree/fern decision thresholds on Uniform(-3,3), weights as Gaussian(0,1). The spitting features are instead assigned sequentially in breadth-first-traversal fashion as:
```
vertexFeatureIndex = vertexIndex mod numFeatures
```
For example, you can use this to give each vertex a unique linear combination feature for decisions.
```py
import torch.nn as nn

numTrees=100
treeDepth=7
numFeatures=numTrees*(2**treeDepth - 1) # Give each vertex a unique linear combination feature

fc = nn.Linear(in_features=inputFeatures, out_features=numFeatures, bias=False)
bn = nn.BatchNorm1d(in_features=numFeatures, affine=False)
forest = RandomHingeForest(in_channels=numFeatures, out_channels=numTrees, depth=treeDepth, init_type="sequential")

# Evaluate a batch
x = forest(bn(fc(x)))
```

# Experiment Scripts
Set the `PYTHONPATH` environment variable to reflect the git root folder (the one with `HingeTree.py`, `RandomHingeForest.py`). Then you can run the experiments, for example, in this fashion:

```shell
cd experiments
python run_iris.py
```

And if you have a GPU
```shell
cd experiments
python run_iris.py --device cuda:0
```

## PYTHONPATH on Windows
```
cd C:\Path\To\HingeTreeForTorch
set pythonpath=%pythonpath%;%CD%
```

## PYTHONPATH Unix-like Systems
```shell
cd /path/to/HingeTreeForTorch
export PYTHONPATH="${PYTHONPATH}:${PWD}"
```

## Dependencies for Experiment Scripts
You will need `requests`, `scikit-learn`, `opencv-python`, `xgboost` and `lightgbm` to run these experiments.
```shell
pip install requests scikit-learn opencv-python xgboost lightgbm
```

All data sets are acquired automatically (usually downloaded from UCI machine learning repository).

