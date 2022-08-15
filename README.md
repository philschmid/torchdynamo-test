# Torchdynamo test for Transforemrs

> No Success for Pegasus yet.

* [Enable torchdynamo with torch_tensorrt(fx path)](https://github.com/huggingface/transformers/pull/17765)
* [Inference benchmarks of Torchdynamo + FX2TRT(now in Torch-TensorRT)](https://github.com/huggingface/transformers/pull/17724)
* [TorchDynamo](https://github.com/pytorch/torchdynamo)

## Installation [REF](https://github.com/huggingface/transformers/pull/17765)

1. create conda env

```bash
conda create --name dyn python=3.8 jupyter
conda activate dyn
```

2. install dependencies 

```bash
# install torch-nightly
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch-nightly
# pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu113 --upgrade
# install functorch (and reinstall after `git pull` later if need to sync up)
python -c "import torch; assert torch.__version__ > '1.12.0', 'Please install torch 1.13.0 or later'"

# clean is not working other
rm -rf TensorRT
rm -rf functorch
rm -rf torchdynamo

git clone https://github.com/pytorch/functorch
cd functorch
rm -rf build
pip install -e .[aot]
cd ..

git clone https://github.com/pytorch/torchdynamo
cd torchdynamo
pip install -r requirements.txt
python setup.py develop
python -c "import torchdynamo;"
cd ..

# install TensorRT
pip install nvidia-pyindex 
pip install nvidia-tensorrt=="8.2.4.2"
pip install torch-tensorrt -f https://github.com/NVIDIA/Torch-TensorRT/releases

# install torch_tensorrt (fx path)
git clone https://github.com/pytorch/TensorRT.git
cd TensorRT/py
python setup.py install --fx-only
cd ../..
python -c "import torch_tensorrt.fx"
python -c "import torchdynamo; assert 'fx2trt' in torchdynamo.list_backends(), 'Some error in your installation missing optimizer'"
python -c "from torchdynamo.optimizations import backends; x=backends.fx2trt_compiler"


```

3. install transformers

```bash
pip install transformers[sentencepiece] jupyter
```

4. Checkout example

* [pegasus](./Pegasus_torchdynamo.ipynb)


## Torchdynamo

TorchDynamo is a new tracer that uses Python’s frame evaluation API to automatically create FX traces from existing PyTorch programs. After capturing the FX graph, different backends can be deployed to lower the graph to an optimized engine. One solution is using the [TensorRT](https://developer.nvidia.com/tensorrt) or NVFuser as backend. You can choose one option below for performance boost.
```
TrainingArguments(torchdynamo="eager")      #enable eager model GPU. No performance boost
TrainingArguments(torchdynamo="nvfuser")    #enable nvfuser
TrainingArguments(torchdynamo="fx2trt")     #enable tensorRT fp32
TrainingArguments(torchdynamo="fx2trt-f16") #enable tensorRT fp16
```
This feature involves 3 different libraries. To install them, please follow the instructions below:  
- [Torchdynamo installation](https://github.com/pytorch/torchdynamo#requirements-and-setup)  
- [Functorch installation](https://github.com/pytorch/functorch#install)  
- [Torch-TensorRT(FX) installation](https://github.com/pytorch/TensorRT/blob/master/docsrc/tutorials/getting_started_with_fx_path.rst#installation)