# Deep Learning in HPC 

[![Build Status](https://travis-ci.com/eskilj/ml.svg?token=XGRP376xWsF66tXxMV16&branch=master)](https://travis-ci.com/eskilj/ml)

MSc project on Deep Learning

### Prerequisites

What things you need to install the software and how to install them

```
Python
TensorFlow
```

The code is modified to easily fit the Google Cloud ML runtime environment, local execution is still possible.    

### Installing

Use pip to install dependencies. It is recommended to first create a virtual environment. 

```
pip install -r "requirements.txt"
```

## Running the tests
The tests were written using Python's Unittest. 

The main purpose of the tests is to check the validity of the ouput of the key methods in the code. In other cases the format of the tensors are verified.    
```
python -m unittest
```

Flake8 is used to check code formatting and consistency, and is checked when the test command is executed.

## Project Structure

The `config` directory contains example scripts used to define Google Cloud ML job configurations, like custom machine scale and hyperparameter training settings. 

`scripts` contains examples of job submission scripts. These specify job parameters such as code path, runtime version and training data locations.

The `data` directory holds training and validation data, in addition to a utility script for downloading and building images from ImageNet to TFRecords.

`trainer2` contains the deep learning code.  

## Built With

* [PyCharm](https://www.jetbrains.com/pycharm/) - Python IDE
