module Meshes

using LinearAlgebra
using StaticArrays
using PlotlyJS

using AQSNN.Helpers
using AQSNN.Polytopes
import Base: eltype, length, eachindex, getindex, firstindex, lastindex, iterate
import AQSNN.Polytopes: mandim, embdim

include("AbstractMesh.jl")
export AbstractMesh
export getPolytope
export getCells

include("SimplexMesh.jl")
export SimplexMesh
export simplexify

include("ConvexMesh.jl")
export ConvexMesh
export convexify

include("ReferenceMesh.jl")
export ReferenceMesh
export referenceify

include("CartesianMesh.jl")
export CartesianMesh

include("PolytopeSampler.jl")
export PolytopeSampler
export getMesh
export getSamples

export resample!

include("plots.jl")
export plotMesh

end # module Meshes
