# Torchdynamo test for Transforemrs

> No Success for Pegasus yet.

* [Enable torchdynamo with torch_tensorrt(fx path)](https://github.com/huggingface/transformers/pull/17765)
* [Inference benchmarks of Torchdynamo + FX2TRT(now in Torch-TensorRT)](https://github.com/huggingface/transformers/pull/17724)
* [TorchDynamo](https://github.com/pytorch/torchdynamo)
* [Sylvains tests](https://github.com/sgugger/torchdynamo-tests)

> We have moved TorchDynamo to pytorch/pytorch


## Installation [REF](https://github.com/pytorch/torchdynamo#requirements-and-setup)

1. create conda env

```bash
conda create --name dyn python=3.8
conda activate dyn
```

2. install dependencies 

_GPU:_

check which cuda version your local system has installed and make sure to install the corresponding pytorch package.


```bash
pip install numpy --pre torch[dynamo] --force-reinstall --extra-index-url https://download.pytorch.org/whl/nightly/cu116
```

_CPU_

```bash
pip install --pre torch --extra-index-url https://download.pytorch.org/whl/nightly/cpu
pip install transformers jupyter datasets accelerate
```

3. test it 

```python
import torch._dynamo as torchdynamo

print(torchdynamo.list_backends())
# ['ansor', 'aot_autograd', 'aot_cudagraphs', 'aot_eager', 'aot_inductor_debug', 'aot_print', 'aot_ts', 'aot_ts_nvfuser', 'aot_ts_nvfuser_nodecomps', 'cudagraphs', 'cudagraphs_ts', 'cudagraphs_ts_ofi', 'eager', 'fx2trt', 'inductor', 'ipex', 'nnc', 'nnc_ofi', 'nvprims_aten', 'nvprims_nvfuser', 'ofi', 'onednn', 'onnx2tensorrt', 'onnx2tensorrt_alt', 'onnx2tf', 'onnxrt', 'onnxrt_cpu', 'onnxrt_cpu_numpy', 'onnxrt_cuda', 'static_runtime', 'taso', 'tensorrt', 'torch2trt', 'torchxla_trace_once', 'torchxla_trivial', 'ts', 'ts_nvfuser', 'ts_nvfuser_ofi', 'tvm', 'tvm_meta_schedule']
```

4. Checkout example

* [pegasus](./Pegasus_torchdynamo.ipynb)


## Torchdynamo

TorchDynamo is a new tracer that uses Python’s frame evaluation API to automatically create FX traces from existing PyTorch programs. After capturing the FX graph, different backends can be deployed to lower the graph to an optimized engine. One solution is using the [TensorRT](https://developer.nvidia.com/tensorrt) or NVFuser as backend. You can choose one option below for performance boost.

