## Installation

### Requirements

- Linux or macOS (Windows is not currently officially supported)
- Python 3.6+
- PyTorch 1.3+
- CUDA 9.2+ (If you build PyTorch from source, CUDA 9.0 is also compatible)
- GCC 5+
- mmcv (tested with mmcv 0.6.2)

Recommended configuration: Python 3.7,PyTorch 1.4, CUDA 10.1 and mmcv 0.6.2.

### Install mmdetection with aLRPLoss

a. Create a conda virtual environment and activate it.

```shell
conda create -n aLRPLoss python=3.7 -y
conda activate aLRPLoss
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

```shell
conda install pytorch torchvision -c pytorch
```

Note: Make sure that your compilation CUDA version and runtime CUDA version match.
You can check the supported CUDA version for precompiled packages on the [PyTorch website](https://pytorch.org/).

`E.g.` If you have CUDA 10.1 installed under `/usr/local/cuda` and would like to install
PyTorch 1.4, you need to install the prebuilt PyTorch with CUDA 10.1.

```python
conda install pytorch=1.4 cudatoolkit=10.1 torchvision -c pytorch
```
c. Clone the repository.

```shell
git clone https://github.com/kemaloksuz/aLRPLoss.git
cd aLRPLoss
```

d. Install build requirements and then install mmdetection with aLRP Loss.
(We install our forked version of pycocotools via the github repo instead of pypi
for better compatibility with our repo.)

```shell
pip install -r requirements/build.txt
pip install "git+https://github.com/open-mmlab/cocoapi.git#subdirectory=pycocotools"
pip install -v -e .  # or "python setup.py develop"
```

If you build mmdetection on macOS, replace the last command with

```
CC=clang CXX=clang++ CFLAGS='-stdlib=libc++' pip install -e .
```

Note:

1. The git commit id will be written to the version number with step d, e.g. 0.6.0+2e7045c. The version will also be saved in trained models.
It is recommended that you run step d each time you pull some updates from github. If C++/CUDA codes are modified, then this step is compulsory.

    > Important: Be sure to remove the `./build` folder if you reinstall mmdet with a different CUDA/PyTorch version.

    ```
    pip uninstall mmdet
    rm -rf ./build
    find . -name "*.so" | xargs rm
    ```

2. Following the above instructions, mmdetection is installed on `dev` mode, any local modifications made to the code will take effect without the need to reinstall it (unless you submit some commits and want to update the version number).

3. If you would like to use `opencv-python-headless` instead of `opencv-python`,
you can install it before installing MMCV.

4. Some dependencies are optional. Simply running `pip install -v -e .` will only install the minimum runtime requirements. To use optional dependencies like `albumentations` and `imagecorruptions` either install them manually with `pip install -r requirements/optional.txt` or specify desired extras when calling `pip` (e.g. `pip install -v -e .[optional]`). Valid keys for the extras field are: `all`, `tests`, `build`, and `optional`.

### Install with CPU only
The code can be built for CPU only environment (where CUDA isn't available).

In CPU mode you can run the demo/webcam_demo.py for example.
However some functionality is gone in this mode:

- Deformable Convolution
- Deformable ROI pooling
- CARAFE: Content-Aware ReAssembly of FEatures
- nms_cuda
- sigmoid_focal_loss_cuda

So if you try to run inference with a model containing deformable convolution you will get an error.
Note: We set `use_torchvision=True` on-the-fly in CPU mode for `RoIPool` and `RoIAlign`

### A from-scratch setup script

Here is a full script for setting up this repository with conda.

```shell
conda create -n aLRPLoss python=3.7 -y
conda activate aLRPLoss

# install latest pytorch prebuilt with the default prebuilt CUDA version (usually the latest)
conda install -c pytorch pytorch torchvision -y
git clone https://github.com/kemaloksuz/aLRPLoss.git
cd aLRPLoss
pip install -r requirements/build.txt
pip install "git+https://github.com/open-mmlab/cocoapi.git#subdirectory=pycocotools"
pip install -v -e .
```
