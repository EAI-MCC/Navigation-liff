## Target-driven Visual Navigation Model using Deep Reinforcement Learning


## Introduction

This repocitory provides a Tensorflow implementation of the deep siamese actor-critic model for indoor scene navigation introduced in the following paper:

**[Target-driven Visual Navigation in Indoor Scenes using Deep Reinforcement Learning](http://web.stanford.edu/~yukez/papers/icra2017.pdf)**
<br>
[Yuke Zhu](http://web.stanford.edu/~yukez/), Roozbeh Mottaghi, Eric Kolve, Joseph J. Lim, Abhinav Gupta, Li Fei-Fei, and Ali Farhadi
<br>
[ICRA 2017, Singapore](http://www.icra2017.org/)
## Environment
It runs successfully on my machine.
![run](.asset/run.jpg)
![evaluation](.asset/evaluation.jpg)
- Architecture: `x86_64`
- Model Name: `Intel i7-8750H CPU @ 2.20GHz`
- OS: `Ubuntu 2004`
- software environment: `Conda 4.9.2`
- device :`single GPU - RTX 3090`

use conda or py-venv

Here is a conda package list 

`python 3.10` + `tensorflow 2.9.1`

```bash
# Name                    Version                   Build  Channel
_libgcc_mutex             0.1                 conda_forge    conda-forge
_openmp_mutex             4.5                       2_gnu    conda-forge
absl-py                   1.1.0                    pypi_0    pypi
astunparse                1.6.3                    pypi_0    pypi
bzip2                     1.0.8                h7b6447c_0  
ca-certificates           2022.4.26            h06a4308_0  
cachetools                5.2.0                    pypi_0    pypi
certifi                   2022.6.15                pypi_0    pypi
charset-normalizer        2.1.0                    pypi_0    pypi
cudatoolkit               11.3.1               h2bc3f7f_2  
cudnn                     8.4.1.50             hed8a83a_0    conda-forge
flatbuffers               1.12                     pypi_0    pypi
gast                      0.4.0                    pypi_0    pypi
google-auth               2.9.1                    pypi_0    pypi
google-auth-oauthlib      0.4.6                    pypi_0    pypi
google-pasta              0.2.0                    pypi_0    pypi
grpcio                    1.47.0                   pypi_0    pypi
h5py                      3.7.0                    pypi_0    pypi
idna                      3.3                      pypi_0    pypi
imageio                   2.19.5                   pypi_0    pypi
keras                     2.9.0                    pypi_0    pypi
keras-preprocessing       1.1.2                    pypi_0    pypi
ld_impl_linux-64          2.38                 h1181459_1  
libclang                  14.0.1                   pypi_0    pypi
libffi                    3.3                  he6710b0_2  
libgcc-ng                 12.1.0              h8d9b700_16    conda-forge
libgomp                   12.1.0              h8d9b700_16    conda-forge
libstdcxx-ng              12.1.0              ha89aaad_16    conda-forge
libuuid                   1.0.3                h7f8727e_2  
libzlib                   1.2.12               h166bdaf_2    conda-forge
markdown                  3.4.1                    pypi_0    pypi
ncurses                   6.3                  h5eee18b_3  
networkx                  2.8.4                    pypi_0    pypi
numpy                     1.23.1                   pypi_0    pypi
oauthlib                  3.2.0                    pypi_0    pypi
openssl                   1.1.1q               h7f8727e_0  
opt-einsum                3.3.0                    pypi_0    pypi
packaging                 21.3                     pypi_0    pypi
pillow                    9.2.0                    pypi_0    pypi
pip                       22.1.2                   pypi_0    pypi
protobuf                  3.19.4                   pypi_0    pypi
pyasn1                    0.4.8                    pypi_0    pypi
pyasn1-modules            0.2.8                    pypi_0    pypi
pyparsing                 3.0.9                    pypi_0    pypi
python                    3.10.4               h12debd9_0  
pywavelets                1.3.0                    pypi_0    pypi
readline                  8.1.2                h7f8727e_1  
requests                  2.28.1                   pypi_0    pypi
requests-oauthlib         1.3.1                    pypi_0    pypi
rsa                       4.8                      pypi_0    pypi
scikit-image              0.19.3                   pypi_0    pypi
scipy                     1.8.1                    pypi_0    pypi
setuptools                61.2.0                   pypi_0    pypi
six                       1.16.0                   pypi_0    pypi
sqlite                    3.38.5               hc218d9a_0  
tensorboard               2.9.1                    pypi_0    pypi
tensorboard-data-server   0.6.1                    pypi_0    pypi
tensorboard-plugin-wit    1.8.1                    pypi_0    pypi
tensorflow                2.9.1                    pypi_0    pypi
tensorflow-estimator      2.9.0                    pypi_0    pypi
tensorflow-io-gcs-filesystem 0.26.0                   pypi_0    pypi
termcolor                 1.1.0                    pypi_0    pypi
tifffile                  2022.5.4                 pypi_0    pypi
tk                        8.6.12               h1ccaba5_0  
typing-extensions         4.3.0                    pypi_0    pypi
tzdata                    2022a                hda174b7_0  
urllib3                   1.26.10                  pypi_0    pypi
werkzeug                  2.1.2                    pypi_0    pypi
wheel                     0.37.1             pyhd3eb1b0_0  
wrapt                     1.14.1                   pypi_0    pypi
xz                        5.2.5                h7f8727e_1  
zlib                      1.2.12               h7f8727e_2
```

`Note:`
**if you want to run it on your own machine,you should modify path/to/dataset on `train.py` line 37 and `evaluate.py` line 26.**

## Setup
This code is implemented in [Tensorflow API r1.0](https://www.tensorflow.org/api_docs/). You can follow the [online instructions](https://www.tensorflow.org/install/) to install Tensorflow 1.0. Other dependencies ([h5py](http://www.h5py.org/), [numpy](http://www.numpy.org/), [scikit-image](http://scikit-image.org/), [pyglet](https://bitbucket.org/pyglet/pyglet/wiki/Home)) can be install by [pip](https://pypi.python.org/pypi/pip): ```pip install -r requirements.txt```. This code has been tested with Python 2.7 and 3.5.

## Scenes
To facilitate training, we provide [hdf5](http://www.h5py.org/) dumps of the simulated scenes. Each dump contains the agent's first-person observations sampled from a discrete grid in four cardinal directions. To be more specific, each dump stores the following information row by row:

* **observation**: 300x400x3 RGB image (agent's first-person view)
* **resnet_feature**: 2048-d [ResNet-50](https://arxiv.org/abs/1512.03385) feature extracted from the observations
* **location**: (x, y) coordinates of the sampled scene locations on a discrete grid with 0.5-meter offset
* **rotation**: agent's rotation in one of the four cardinal directions, 0, 90, 180, and 270 degrees
* **graph**: a state-action transition graph, where ```graph[i][j]``` is the location id of the destination by taking action ```j``` in location ```i```, and ```-1``` indicates collision while the agent stays in the same place.
* **shortest_path_distance**: a square matrix of shortest path distance (in number of steps) between pairwise locations, where ```-1``` means two states are unreachable from each other.

Before running the code, please download the scene dumps using the following script:
```bash
./data/download_scene_dumps.sh
```
We are currently releasing one scene from each of the four scene categories, *bathroom*, *bedroom*, *kitchen*, and *living room*. Please contact me for information about additional scenes.
A ```keyboard_agent.py``` script is provided. This script allows you to load a scene dump and use the arrow keys to navigate a scene. To run the script, here is an example command:
```bash
# make sure the scene dump is in the data folder, e.g., ./data/bedroom_04.h5
python keyboard_agent.py --scene_dump ./data/bedroom_04.h5
```

These scene dumps enable us to train a (discrete) navigation agent without running the simulator during training or extracting ResNet features. Thus, it greatly improves training efficiency. The training code runs comfortably on CPUs (of my Macbook Pro). Due to legal concerns, our THOR simulator will be released later.

## Training and Evaluation
The parameters for training and evaluation are defined in ```constants.py```. The most important parameter is ```TASK_LIST```, which is a dictionary that defines the scenes and targets to be trained and evaluated on. The keys of the dictionary are scene names, and the values are a list of location ids in the scene dumps, i.e., navigation targets. We use a type of asynchronous advantage actor-critic model, similar to [A3C](https://arxiv.org/abs/1602.01783), where each thread trains for one target of one scene. Therefore, make sure the number of training threads ```PARALLEL_SIZE``` is *at least* the same as the total number of targets. You can use more threads to further parallelize training. For instance, when using 8 threads to train 4 targets, 2 threads will be allocated to train each target.

The model checkpoints are stored to ```CHECKPOINT_DIR```, and Tensorboard logs are written in ```LOG_FILE```. To train a target-driven navigation model, run the following script:
```bash
# train a model for targets defined in TASK_LIST
python train.py
```

For evaluation, we run 100 episodes for each target and report the mean/stddev length of the navigation trajectories. To evaluate a model checkpoint in ```CHECKPOINT_DIR```, run the following script:
```bash
# evaluate a checkpoint on targets defined in TASK_LIST
python evaluate.py
```

## Acknowledgements
I would like to acknowledge the following references that have offered great help for me to implement the model.
* ["Asynchronous Methods for Deep Reinforcement Learning", Mnih et al., 2016](https://arxiv.org/abs/1602.01783)
* [David Silver's Deep RL course](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)
* [muupan's async-rl repo](https://github.com/muupan/async-rl/wiki)
* [miyosuda's async_deep_reinforce repo](https://github.com/miyosuda/async_deep_reinforce)

## Citation
Please cite our ICRA'17 paper if you find this code useful for your research.
```
@InProceedings{zhu2017icra,
  title = {{Target-driven Visual Navigation in Indoor Scenes using Deep Reinforcement Learning}},
  author = {Yuke Zhu and Roozbeh Mottaghi and Eric Kolve and Joseph J. Lim and Abhinav Gupta and Li Fei-Fei and Ali Farhadi},
  booktitle = {{IEEE International Conference on Robotics and Automation}},
  year = 2017,
}
```

## License
MIT
