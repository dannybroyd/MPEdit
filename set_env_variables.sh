export LD_LIBRARY_PATH=$HOME/miniconda3/envs/mpd-splines-public/lib
export CPATH=$HOME/miniconda3/envs/mpd-splines-public/include

# OMPL Python bindings
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${THIS_DIR}/deps/pybullet_ompl/ompl/py-bindings:${THIS_DIR}/deps/pybullet_ompl/ompl/build/Release/lib:${PYTHONPATH}"

# CUDA 11.8
export PATH=/usr/local/cuda-11.8/bin:$PATH
export CUDA_HOME=/usr/local/cuda-11.8
