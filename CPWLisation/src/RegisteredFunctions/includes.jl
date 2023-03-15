module RegisteredFunctions

using CPWLisation.Helpers

include("RegisteredFunction.jl")
export RegisteredFunction
export ∇⁰
export ∇¹
export ϕ₋
export ϕ₊
export ∫ρ
export ∫ρx
export ∫ρ²
export ∫ρϕ₋²
export ∫ρϕ₊²
export ∫ρϕ²

include("ReLU.jl")
export ReLU

include("Tanh.jl")
export Tanh

end # module RegisteredFunctions
