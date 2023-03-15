module Linearisation

using LinearAlgebra
using StaticArrays
using Statistics
using IterTools
using PlotlyJS

import Base: length, eachindex, getindex, firstindex, lastindex, iterate

using AQSNN.Helpers
using AQSNN.Polytopes
using AQSNN.Meshes
using AQSNN.Model

import AQSNN.Meshes: getMesh

include("CPWLiser.jl")
export CPWLise

include("Lineariser.jl")
export AbstractLineariser
export Lineariser
export elements
export linearise!

include("Lineariser0.jl")
include("Lineariser1.jl")
include("Lineariser2.jl")

include("Collapser.jl")
export collapse!

include("plots.jl")
export plotLineariser
export plotProjection

end # module Linearisation
