# csp-cpp
Evaluating the Convolutional Social Pooling (CSP) work using the PyTorch C++ frontend

# Installing  
You can create an Anaconda environment, install CMake and the C+++ distributions of PyTorch as follows:

1- Download the C++ distributions of PyTorch (LibTorch) ZIP archive:
```
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest.zip
```
Note that the above link has CPU-only LibTorch. 

2- Create and activate a new conda environment:
``` 
conda create --name cpp-pytorch
conda activate cpp-pytorch
```

3- Install CMake:
```
conda install -c anaconda cmake 
```
CMake is the recommended build system and will be well supported into the future.

# Evaluating CSP using C++
This repository includes a 'eval.cpp' script that loads a traced model and evaluates it on a given set of inputs. 
The model is already traced and saved at '/traced-models' and a 'CMakeLists.txt' file is provide to build the application. Follow these steps to test the code:

1- Clone the repository to your local machine.

2- Run the following commands to build the application from within the cloned repository folder:
```
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..
cmake --build . --config Release
``` 
where /absolute/path/to/libtorch should be the ABSOLUTE path to the unzipped LibTorch distribution.

3- Execute the resulting binary found in the build folder: 
```
./csp-cpp "../traced-models/traced_net_model.pt"
```