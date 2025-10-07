# AQSNN.jl

[![DOI](https://zenodo.org/badge/614422884.svg)](https://zenodo.org/badge/latestdoi/614422884)

This repository contains the codes that we used in our paper "Adaptive quadratures for nonlinear approximation of low-dimensional PDEs using smooth neural networks".

## CPWLisation

This module enables to approximate a smooth function that behaves linearly at infinity by a Continuous PieceWise Linear (CPWL) function. Two approximation spaces are made available:

* A free CPWL space where the degrees of freedom are the locations of the breakpoints and the values at the breakpoints (see `scripts/cpwl.jl`).
* A subspace consisting of functions that coincide with the tangents to the function to approximate at free points (see `scripts/tangent.jl`).

Several types of boundary conditions are available, from free end point (`PointFree`), prescribed value at the end point (`PointValue`), prescribed slope at the end point (`PointSlope`), or tangent at the end point (`PointTangent`). It is also possible to perform the approximation on unbounded domains (`Asymptote`). The optimisation is performed in $L^2$ norm thanks to ADAM or L-BFGS.

## AQSNN

Adaptive Quadrature for Smooth Neural Networks (AQSNN) is a library that provides a reliable numerical integration method for neural networks, applied to approximating the solution of elliptic partial differential equations.

The activation function is approximated by a CPWL function obtained from `CPWLisation`, and it is used to define a decomposition of the input space into regions where the network is almost linear. The loss function is evaluated at Gaussian points in each cell of the mesh.

We showcase functionalities of our library regarding the linearisation and the integration of a neural network in `scripts/0.tutorial.jl`.

# Reproducing the results of the article

**WARNING:** Our numerical experiments depend on random generators for the initialisation of the neural networks and the sampling of Monte-Carlo integration points. Random generators are not consistent across versions of Julia and of `Distributions.jl`. To obtain exactly the same numbers of the article, one must use version `1.8.0` of `julia` and version `0.25.79` of `Distributions.jl`.

### Instantiation
```
] activate CPWLisation/.
] instantiate
] activate AQSNN/.
] instantiate
```

### CPWLisation
To produce `fig2(a-b).pdf` in `plots/`, run the following.
```
include("CPWLisation/scripts/article.jl")
```
We used `Nmax = 25` in the article but this will take around one hour to converge, so we set `Nmax = 15` by default.

### AQSNN

#### Training
To train the models that we report in the article, run the following
```
include("AQSNN/scripts/1.run.jl")
```
**WARNING:** It will take around 11h to train all the networks. For convenience, we have attached the files we have used in `data/data.zip`. To use these files, simply unzip the archive in the folder `data/`.

#### Tables
To reproduce the tables of the article, run
```
include("AQSNN/scripts/2.table_exploration1D.jl")
include("AQSNN/scripts/2.table_exploration2D.jl")
include("AQSNN/scripts/2.table_initialisation1D.jl")
include("AQSNN/scripts/2.table_initialisation2D.jl")
include("AQSNN/scripts/2.table_reduction1D.jl")
include("AQSNN/scripts/2.table_reduction2D.jl")
```
This will create `table(2-5).txt` and `table(6-8)(a-b).txt` in `tables/`.

#### Figures
To reproduce the figures of the article, run
```
include("AQSNN/scripts/3.activations.jl")
include("AQSNN/scripts/3.comparison_1D.jl")
include("AQSNN/scripts/3.comparison_2D.jl")
include("AQSNN/scripts/3.mesh_2D.jl")
```
This will create `fig(1, 4-7)(a-c).pdf`, `fig(8-9, 11).vtu` and `fig10(a-b).pdf` in `plots/`. We have attached `fig(8-9, 11).pvsm` to obtain the same style in Paraview for figures 8, 9 and 11. The figures (4-7)c have not been included in our article. In Paraview, simply open `fig(8-9, 11).vtu` and load the corresponding `fig(8-9, 11).pvsm`.

#### Other tables and figures
Figure 3 and Table 1 were produced manually.