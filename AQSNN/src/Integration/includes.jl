module Integration

import Base: +, -, *, ^
import Base: eltype, length
using LinearAlgebra
using Statistics
import LinearAlgebra: dot
using FastGaussQuadrature

using AQSNN.Helpers
using AQSNN.Polytopes
using AQSNN.Meshes
using AQSNN.Model
using AQSNN.Linearisation

include("Integrand.jl")
export AbstractIntegrand
export DomainIntegrand
export BoundaryIntegrand
export ∫Ω
export ∫Γ
export norm2
export dot

include("AbstractQuadrature.jl")
export AbstractQuadrature
export initialise!
export integrate

include("MonteCarloQuadrature.jl")
export DomainMonteCarloQuadrature
export BoundaryMonteCarloQuadrature
export MonteCarloQuadrature

include("ReferenceQuadrature.jl")
export ReferenceQuadrature
export getOrder
export getPoints
export getWeights
export transport

include("MeshQuadrature.jl")
export MeshQuadrature

include("AdaptiveQuadrature.jl")
export AdaptiveQuadrature

end # module Integration
