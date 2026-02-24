# SAT-Metropolis

Python implementation of the SAT-Metropolis algorithm presented in the accompanying paper: "SAT-Metropolis: Combining Markov Chain Monte Carlo with SAT/SMT sampling". SAT-Metropolis uses SAT/SMT samplers as proposal distributions to effectively sample in probabilistic inference problems with hard constraints.

**Code for experiments in the accompanying paper:** The notebook `experiments/experiments.ipynb` contains the code for all the experiments in the accompanying paper.

**Get started with SAT-Metropolis:** The notebooks in folders `experiments/sat` and `experiments/smt` contain multiple examples on using SAT-Metropolis with the different backends currently available, namely, SPUR, CMSGen and MegaSampler. Below we provide installation instructions for a conda environment for the library, and instructions to install each backend. 

After completing the installation steps below, any of the Jupyter notebooks can be executed by selecting the `sat_metropolis` kernel; the guide below includes making this kernel available in the system. The conda environment includes `jupyterlab` to execute Jupyter notebooks.

## Installation

1. Install [miniconda](https://docs.conda.io/projects/miniconda/en/latest/)
2. Create the `sat_metropolis` conda environment using the `environment.yml` file: `conda env create -n sat_metropolis -f environment.yml`
3. Activate environment: `conda activate sat_metropolis`
4. Add environemnt as a kernel: `python -m ipykernel install --user --name=sat_metropolis` 
5. Install the `sat_metropolis` library: `python -m pip install .`


6. Download and install [SPUR](https://github.com/ZaydH/spur). The following are the same steps as in [SPUR](https://github.com/ZaydH/spur) with some minor modifications.
   1. `git clone https://github.com/ZaydH/spur.git`
   2. `cd spur/`
   3. `sudo snap install cmake --classic`
   4. `sudo apt install libgmp-dev `
   5. Spur would not compile with newer versions of g++ because gcc does not include `stdint.h`. Thus, it is necessary to modify the flags in the `CMakeFile.txt` file as follows:
      ```
      set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -std=c++11 -Wall -include stdint.h")

      set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -std=c++11 -O3 -DNDEBUG -Wall -include stdint.h")

      set(CMAKE_CXX_FLAGS_PROFILING "${CMAKE_CXX_FLAGS_PROFILING} -std=c++11 -O3 -g
          -DNDEBUG -Wall -fno-omit-frame-pointer -include stdint.h")
      ```
   6. `./build.sh`
   7. `ln -s <path_to_spur_repo>/build/Release/spur ~/.local/bin/spur`


7. Download and install [MegaSampler](https://github.com/chaosite/MeGASampler). The following are the same steps as in [MegaSampler](https://github.com/chaosite/MeGASampler) with some minor modifications.
   1. `sudo apt install git build-essential python3-minimal python3-dev libjsoncpp-dev python3-venv`
   2. `python -m venv venv --upgrade`
   4. `source venv/bin/activate`
   5. `git clone https://github.com/chaosite/MeGASampler.git`
   6. `git clone https://github.com/chaosite/z3.git` (patched z3 for SMTSampler coverage)
   7. `pushd z3`
   8. `python scripts/mk_make.py --python`
   9. `cd build`
   10. `make install`
   11. `popd`
   12. `cd MeGASampler`
   13. `make`
   14. `conda env config vars set LD_LIBRARY_PATH=<path_to_dir_hosting_virtual_environment>/venv/lib` (this is an example to add the variable to a conda environment, it is also possible to simply add it to `.bashrc`)
   15. `ln -s <path_to_megasampler>/megasampler ~/.local/bin/megasampler`


8. Download and install [CMSGen](https://github.com/meelgroup/cmsgen). The following are the same steps as in [CMSGen](https://github.com/meelgroup/cmsgen).
    1. `sudo apt install zlib1g-dev help2man`
    2. `cd cmsgen`
    3. `mkdir build && cd build`
    4. `cmake ..`
    5. `make`
    6. `sudo make install`
    7. `sudo ldconfig`

9. Download and install [Unigen](https://github.com/meelgroup/unigen).
   1.  `mkdir unigen`
   2. `cd unigen`
   3. Install their static binary from their [Github releases](https://github.com/meelgroup/unigen/releases/tag/release%2F2.7.0).
   4. Place it inside your unigen folder
   5. Make the binary executable, in wsl run `chmod +x unigen`
   6. `ln -s <path_to_unigen/unigen ~/.local/bin/unigen`
