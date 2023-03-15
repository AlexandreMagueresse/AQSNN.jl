module Model

using LinearAlgebra
using Distributions
using SpecialFunctions
using Knet
using AutoGrad
using PlotlyJS

using AQSNN.Helpers
using AQSNN.Polytopes
using AQSNN.Meshes

include("Initialisers.jl")
export AbstractInitialiser
export HeInitialiser
export UniformInitialiser
export ZeroInitialiser

include("Activations.jl")
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

include("Dense.jl")
export Dense

include("Sequential.jl")
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

include("FEMInit.jl")
export FEMMin
export FEMMinFull
export FEMSum

include("plots.jl")
export _plotInput
export _plotOutput
export _plotModels
export plotModel
export plotBasis

end # module Model
