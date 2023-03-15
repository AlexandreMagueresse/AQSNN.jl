module Optimisation

using LinearAlgebra
using IterativeSolvers
using Optim
using Knet

using AQSNN.Helpers
using AQSNN.Model
using AQSNN.Integration
import AQSNN.Integration: initialise!

include("Objective.jl")
export Objective
export linear
export bilinear
export residual

include("Optimisers.jl")
export AbstractOptimiser
export ADAM
export SGD
export LBFGS
export LinearSolver

export initialise!
export step!

include("Training.jl")
export Training
export train!

end # module Optimisation
