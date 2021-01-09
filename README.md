# PlasmaOpt

## Requirements

On a recent linux (e.g. Ubuntu > 18.04), most requirements should be met.
First install an MPI library of your choice via
    
    sudo apt install mpich

or
    
    brew install mpich

On mac, install gcc and python via

    brew install python gcc

and make sure to follow the instructions under _Caveats_ when installing python.

## Installation

To install run

    git clone --recursive git@github.com:florianwechsung/PyPlasmaOpt.git

or if you don't have SSH keys for GitHub set up

    git clone --recursive https://github.com/florianwechsung/PyPlasmaOpt.git

change into the directory

    cd PyPlasmaOpt/

and then depending on your platform run 

    make pip-mac
or 
    make pip-linux

Finally, you may need to run the following command to be able to "import pyplasmaopt" in scripts outside of this directory as the "make" command above does not seem to always function properly: 

    pip -e install . 

To check the installation

    pytest tests/

You can also run scripts in the "examples" folder to be certain that all the package dependencies are installed. 
