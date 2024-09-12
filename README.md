# HandsOnPyCuda : Beginning with CUDA on GPUs

## PyCUDA Note1: `gpuarray`

GPU memory -> device memory

CPU memory -> host memory

In CUDA C: the transfer of data is carried out using `cudaMemcpyHostToDevice` and `cudaMemcpyDeviceToHost`. Memory allocation using `cudaMalloc` and deallocation using `cudaFree`

PyCUDA covers all overhead of memory allocation, deallocation and transfer using `gpuarray` class.

It performs automatic cleanup based on lifetime.

How to transfer data from the host to GPU?

Contain data in host memory using Numpy e.g. `host_data`

Transfer to GPU using `gpuarray.to_gpu(host_data)`

After computation, retrieve data from GPU using `gpuarray.get()`

#### Always set the datatypes of Numpy arrays transferred to GPU using `dtype`. 

#### Pointwise operations are intrinsically parallelizable.

In PyCUDA, GPU code is often compiled at runtime with the NVIDIA `nvcc` compiler and then subsequently called from
PyCUDA. This can lead to an unexpected slowdown, usually the first time a program or GPU operation is run in a given
Python session.

## `ElementWiseKernel` for performing pointwise computations

Implementing pointwise operation using inline code in CUDA C. Example below

`gpu_2x_ker = ElementwiseKernel(
	"float *in, float *out",
	"out[i] = 2*in[i];",
	"gpu_2x_ker")`

This is compiled externally by `nvcc` compiler and then launched at runtime via PyCUDA.

