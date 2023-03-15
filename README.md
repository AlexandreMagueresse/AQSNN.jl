# AQSNN.jl

This folder contains the codes that we used in our paper "Adaptive quadratures for nonlinear approximation of low-dimensional PDEs using smooth neural networks".

## CPWLisation

This module enables to approximate a smooth function that behaves linearly at infinity by a Continuous PieceWise Linear (CPWL) function. Two approximation spaces are made available:

* A free CPWL space where the degrees of freedom are the locations of the breakpoints and the values at the breakpoints (see `scripts/cpwl.jl`).
* A subspace consisting of functions that coincide with the tangents to the function to approximate at free points (see `scripts/tangent.jl`).

Several types of boundary conditions are available, from free end point (`PointFree`), prescribed value at the end point (`PointValue`), prescribed slope at the end point (`PointSlope`), or tangent at the end point (`PointTangent`). It is also possible to perform the approximation on unbounded domains (`Asymptote`). The optimisation is performed in $L^2$ norm thanks to ADAM or L-BFGS.

## AQSNN

Adaptive Quadrature for Smooth Neural Networks (AQSNN) is a library that provides a reliable numerical integration method for neural networks, applied to approximating the solution of elliptic partial differential equations.

The activation function is approximated by a CPWL function obtained from `CPWLisation`, and it is used to define a decomposition of the input space into regions where the network is almost linear. The loss function is evaluated at Gaussian points in each cell of the mesh.

We showcase functionalities of our library regarding the linearisation and the integration of a neural network in `scripts/0.tutorial.jl`. We ran all the numerical experiments of our article from the file `scripts/1.training.jl`. Helper files to visualise the learning curve of a network as well as the final pointwise error with respect to a manufactured solution can be follow the format `scripts/2.xxx.jl` are. Files of the shape `scripts/3.xxx.jl` create summaries of similar experiments with different hyperparameters or initialisations.