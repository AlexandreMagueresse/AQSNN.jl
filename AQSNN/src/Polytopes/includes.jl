module Polytopes

import Base: eltype, length
using LinearAlgebra
using StaticArrays
using PlotlyJS

using AQSNN.Helpers

include("AbstractContext.jl")
export AbstractContext
export Context0D
export Context1D
export Context2D

export mandim
export embdim

export getBasis
export getNormal

include("AbstractPolytope.jl")
export AbstractPolytope

export getStore
export getIndices

export getVertex
export getEdgeEnd
export getContext

export isSimplex
export isReference
export isConvex

include("GeneralPolytope.jl")
export GeneralPolytope

include("ConvexPolytope.jl")
export ConvexPolytope

include("ReferencePolytope.jl")
export SimplexPolytope
export ReferencePolytope
export Point
export Segment
export Triangle
export Quadrangle

include("DomainFactory.jl")
export Domain
export CartesianDomain
export RegularDomain

include("BoundaryFactory.jl")
export Boundary

include("Measure.jl")
export measure

include("plots.jl")
export plotPolytope

end # module Polytopes
