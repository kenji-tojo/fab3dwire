# WireGrad

## What's This
This is the official open-souce implementation of the wire-shape optimization method presented by the paper **Fabricable 3D Wire Art (SIGGRAPH 2024 conference track)**.

Concretely, this directory includes the core ```wiregrad``` library that implements a differentiable renderer and useful regularizations for optimizaing 3D curves. We also proride examples ```./examples_*.py``` to create 3D wire arts from text and more, and a variety of *unit tests* in ```./tests/``` showing how to use our library in wider applications.

## Tested Environments
### Ubuntu 22.04.4 LTS
- Python 3.10.12
- gcc/g++ 11.4.0
- CUDA Toolkit 11.8
- cmake 3.22.1

### macOS Sonoma 14.3 & M3 Pro chip
- Python 3.12.2
- Apple clang 15.0.0
- cmake 3.28.3

## Resolving Depndencies
### When using CUDA 11.8 (recommended)
This code assumes CUDA Toolkit 11.8. Please modify the PyTorch-related lines in ```./requirements.txt``` if you want to use other CUDA versions.

The followings are the steps to install all the depndencies. You may want to use virtual environments like [```venv```](https://docs.python.org/3/library/venv.html) before pip-install.
1. Install the version 3.4 of [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) to let ```cmake``` run ```find_package(Eigen3 3.4 REQUIRED NO_MODULE)``` in step 4.
2. install PyTorch via ```pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118```
3. Run ```pip3 install -r ./requirements.txt``` to install Python dependencies.
4. Run ```pip3 install .``` to build and install our core ```wiregrad``` Python module.

### When using CPU
Although slow and depreciated, CPU-only execution is also supported EXCEPT ```./examples_text.py``` and ```./examples_twowires.py```. To use the CPU-only version, please replace the step 2 and 3 with:

2. Just skip this step.
3. ```pip3 install -r ./depndencies/cpu/requirements.txt```.

## Running the Code
To download the pre-trained models, please put your [Hugging Face](https://huggingface.co/) access token in ```./TOKEN```.

Then, all files in the form ```./example_*.py``` and the files under ```./tests/``` can be executed via
```
$ python3 [filename].py
```
Please see the beginning section of each file for a description. In most cases, the code produces self-explanatory outputs.

The code also provides options to customize execution. Please check them out if interested.

## Previewing the Results
Please use ```./show_bspline.py``` for previewing optimized B-spline curves. You can view a single curve by executing, for example,
```
$ python3 ./show_bspline.py --file ./data/results/mesh/bunny/controls.obj
```
Note that ```./data/results/``` contains the results of ```./example_*.py``` in our environment.

You can also specify a directory that contians multiple ```./controls_*.obj``` files:
```
$ python3 ./show_bspline.py --dir ./data/results/text/horse_bull
```

## Troubleshooting
Some errors you might encounter and how I solved them.
- ```urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: self-signed certificate in certificate chain (_ssl.c:1000)>``` -> see [this](https://stackoverflow.com/questions/68275857/urllib-error-urlerror-urlopen-error-ssl-certificate-verify-failed-certifica).

## Citing the Paper
If you find this project interesting and useful, consider citing:
```
@inproceedings{Tojo2024Wireart,
	title = {Fabricable 3D Wire Art},
	author = {Tojo, Kenji and Shamir, Ariel and Bickel, Bernd and Umetani, Nobuyuki},
	booktitle = {ACM SIGGRAPH 2024 Conference Proceedings},
	year = {2024},
	series = {SIGGRAPH '24}
}
```