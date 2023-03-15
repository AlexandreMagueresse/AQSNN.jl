#######################
# RegisteredFunctions #
#######################
using CPWLisation.RegisteredFunctions

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

export ReLU
export Tanh

################
# Optimisation #
################
using CPWLisation.Optimisation

export Asymptote
export PointFree
export PointValue
export PointSlope
export PointTangent

export cpwl_ADAM
export cpwl_BFGS
export tangent_ADAM
export tangent_BFGS
