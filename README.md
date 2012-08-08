IPComparison
============

Comparison of image processing algorithms using different technologies &
programming languages.

Requirements
============

 - OpenCV is used in the programs for comparing the different algorithms
implementations against it. OpenCV is a library for image processing and
computer vision algorithms that it is actively maintained for a open
source community.

 - CMake is used for configuring the project.

 - argtable2 library. This library is used in some programs for parsing the
 arguments passed to the programs.

Notes
=====

In this initial version of the repository, it will be available a program for
inverting images. The program contain a version of the algorithm for the
following technologies:

- Normal CPU instructions (C/C++).
- MMX instructions.
- SSE instructions.
- CUDA.
- OpenCL (Not yet available).
