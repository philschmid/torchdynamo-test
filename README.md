# Torchdynamo test for Transforemrs

* [Enable torchdynamo with torch_tensorrt(fx path)](https://github.com/huggingface/transformers/pull/17765)
* [TorchDynamo](https://github.com/pytorch/torchdynamo)

## Installation [REF](https://github.com/huggingface/transformers/pull/17765)

1. create conda env

```bash
conda create --name dyn python=3.8
conda activate dyn
```

2. install dependencies 

```bash
# install torch-nightly
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch-nightly

# install functorch (and reinstall after `git pull` later if need to sync up)
git clone https://github.com/pytorch/functorch
cd functorch
rm -rf build
pip install -e .[aot] -y

cd ..
git clone https://github.com/pytorch/torchdynamo
cd torchdynamo
pip install -r requirements.txt
python setup.py develop

# install TensorRT
pip install nvidia-pyindex 
pip install nvidia-tensorrt==8.2.4.2

# install torch_tensorrt (fx path)
cd ..
git clone https://github.com/pytorch/TensorRT.git
cd TensorRT/py
python setup.py install --fx-only
cd ..

# clean
rm -rf TensorRT
rm -rf functorch
rm -rf torchdynamo
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