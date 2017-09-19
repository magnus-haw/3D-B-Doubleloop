# 3D-B-Doubleloop

3D B-vector measurements on the Caltech doubleloop experiment and associated plotting code.

## Getting Started

To get a local copy of this repository:
git clone "https://github.com/magnus-haw/3D-B-Doubleloop.git"

CSV format, calibrated data is located in "Bx-By-Bz.tar.gz". All values are in SI units. Time steps correspond to 10 ns each (e.g. t=1200 is equivalent to t=12 us). Each file is formated with a single column of data and cartesian positions for each column element are listed in the X.txt,Y.txt,Z.txt files.

The python plotting code is located in the plot_data folder. The only plotting script is PLOT_DATA.PY, the remaining scripts are all libraries.

## Prerequisites for Python plotting

Plotting software uses Mayavi2 and matplotlib libraries.

WINDOWS: recommend installing PythonXY which includes all packages (need to specify full install)

Unix: Use package manager to install. 
