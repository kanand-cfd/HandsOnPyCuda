# HandsOnPyCuda : Beginning with CUDA on GPUs

## PyCUDA Note1: gpuarray

GPU memory -> device memory

CPU memory -> host memory

In CUDA C: the transfer of data is carried out using 'cudaMemcpyHostToDevice' and 'cudaMemcpyDeviceToHost'. Memory allocation using ‘cudaMalloc’ and deallocation using ‘cudaFree’

PyCUDA covers all overhead of memory allocation, deallocation and transfer using ‘gpuarray’ class.

It performs automatic cleanup based on lifetime.

How to transfer data from the host to GPU?

Contain data in host memory using Numpy e.g. host_data
Transfer to GPU using gpuarray.to_gpu(host_data)

After computation, retrieve data from GPU using gpuarray.get()

### Always set the datatypes of Numpy arrays transferred to GPU using dtype. 

### Pointwise operations are intrinsically parallelizable. 
