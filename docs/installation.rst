Installation
__________

Pom was originally developed as an extension to `Ais <https://ais-cryoet.readthedocs.io/en/latest/>`_, our software for streamlined tomogram segmentation. As such, it is practical to install both Ais and Pom in the same environment. If you already have an environment with Ais installed, you can just run 'pip install Pom-cryoET' in the same environment and skip the instructions below.

Installing Ais and Pom
^^^^^^^^^^^^
The easiest way to set up Ais and Pom is to create a new conda environment:

::

    conda create --name pom
    conda activate pom
    conda install python==3.9
    conda install pip
    pip install ais-cryoet, pom-cryoet

The required Pom commands are now available, and if you want you can launch Ais with either of these commands:

::

    ais
    ais-cryoet

In case you encounter issues with the installation, please ask for help on the `Ais <https://github.com/bionanopatterning/Ais>`_ or `Pom <https://github.com/bionanopatterning/Pom>`_ GitHub page.


CUDA & Tensorflow
^^^^^^^^^^^^
To enable processing on the GPU, tensorflow must be set up to use CUDA. This can be a bit of a pain, as only particular combination of versions of tensorflow, CUDA, cuDNN, and protobuf (a Python package) tend to work. When installing Ais via pip, the versioning should be handled, but CUDA must still be manually installed. For instructions, see:

Installing CUDA: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

Installing cuDNN: https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html

When running Ais from within an IDE some paths may need to be manually specified. In PyCharm, add the path to the zlib .dll to the run configuration environment variables as follows to enable tensorflow using the GPU:
LIBRARY_PATH=C:\Program Files\zlib123dllx64\dll_x64
