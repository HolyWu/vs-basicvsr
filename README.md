# BasicVSR
BasicVSR: The Search for Essential Components in Video Super-Resolution and Beyond

Ported from https://github.com/xinntao/BasicSR


## Dependencies
- [NumPy](https://numpy.org/install)
- [PyTorch](https://pytorch.org/get-started), preferably with CUDA. Note that `torchvision` and `torchaudio` are not required and hence can be omitted from the command.
- [VapourSynth](http://www.vapoursynth.com/)


## Installation
```
pip install --upgrade vsbasicvsr
```


## Usage
```python
from vsbasicvsr import BasicVSR

ret = BasicVSR(clip)
```

See `__init__.py` for the description of the parameters.
