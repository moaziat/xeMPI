[![xeMPI CI](https://github.com/moaziat/xeMPI/actions/workflows/ci.yml/badge.svg)](https://github.com/moaziat/xeMPI/actions/workflows/ci.yml)

## Overview

Hands on GPU programming with PyOpenCL 


### Installation
Install all dependecies
```bash 
          sudo apt-get update
          sudo apt-get install -y ocl-icd-opencl-dev 
          sudo apt-get install -y clinfo
          python -m pip install --upgrade pip
          pip install numpy pytest pyopencl
```
Check that everything is looking good: 
```bash 
python setup_test.py
```
You should get something like this:
```Using device: Intel(R) Iris(R) Xe Graphics`` or any other Intel GPU device

### Try it yourself:

```bash
python tests.py
```
You can create your own tests tho!