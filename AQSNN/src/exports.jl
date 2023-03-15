#############
# Polytopes #
#############
using AQSNN.Polytopes

export AbstractContext
export Context0D
export Context1D
export Context2D

export mandim
export embdim

export getBasis
export getNormal

export AbstractPolytope

export getContext
export getVertex
export getEdgeEnd

export isSimplex
export isReference
export isConvex

export GeneralPolytope

export ConvexPolytope

export SimplexPolytope
export ReferencePolytope
export Point
export Segment
export Triangle
export Quadrangle

export Domain
export CartesianDomain
export RegularDomain

export Boundary

export measure

export plotPolytope

##########
# Meshes #
##########
using AQSNN.Meshes

export AbstractMesh
export getPolytope
export getCells

export SimplexMesh
export simplexify

export ConvexMesh
export convexify

export ReferenceMesh
export referenceify

export CartesianMesh

export PolytopeSampler
export getMesh
export getSamples

export resample!

export plotMesh

#########
# Model #
#########
using AQSNN.Model

export AbstractInitialiser
export HeInitialiser
export UniformInitialiser
export ZeroInitialiser

export AbstractActivation
export AbstractReLU
export AbstractSigmoid
export Identity
export ReLU
export Absε
export Logε
export Erfε
export Spline²ε
export Spline³ε
export Tanh

export Dense

export Sequential

export weight
export bias
export paramsAll
export paramsWeights
export paramsMesh
export ∇⁰
export ∇¹
export ∇²
export ∇
export ∂ₙ
export Δ
export basisFunction

export FEMMin
export FEMMinFull
export FEMSum

export plotModel
export plotBasis

#################
# Linearisation #
#################
using AQSNN.Linearisation

export CPWLise

export AbstractLineariser
export Lineariser
export linearise!

export collapse!

export plotLineariser
export plotProjection

###############
# Integration #
###############
using AQSNN.Integration

export AbstractIntegrand
export DomainIntegrand
export BoundaryIntegrand
export ∫Ω
export ∫Γ
export norm2
export dot

export AbstractQuadrature
export initialise!
export integrate

export MonteCarloQuadrature
export DomainMonteCarloQuadrature
export BoundaryMonteCarloQuadrature

export ReferenceQuadrature
export getOrder
export getPoints
export getWeights
export transport

export MeshQuadrature

export AdaptiveQuadrature

################
# Optimisation #
################
using AQSNN.Optimisation

export Objective
export linear
export bilinear
export residual

export AbstractOptimiser
export ADAM
export SGD
export LBFGS
export LinearSolver

export initialise!
export step!

export Training
export train!

##################
# Postprocessing #
##################
using AQSNN.Postprocessing

export L1
export L2
export L∞
export H1
export H2

export plotTrain
export plotComp
export plotDiff

export save
export load
