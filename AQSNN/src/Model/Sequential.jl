##############
# Sequential #
##############
struct Sequential{T,E} <: Function
  architecture::Vector{Int}
  layers::Vector{Dense{T}}
end

function Sequential{T,E}(
  architecture::AbstractVector{Int}, activation::AbstractActivation;
  weightInitialiser::AbstractInitialiser{T}=HeInitialiser{T}(),
  biasInitialiser::AbstractInitialiser{T}=ZeroInitialiser{T}()
) where {T,E}
  L = length(architecture)
  layers = Vector{Dense{T}}()
  for i in 1:L-1
    nIn = architecture[i]
    nOut = architecture[i+1]
    activation = (i < L - 1) ? activation : Identity()
    push!(layers,
      Dense(
        nIn, nOut, activation,
        weightInitialiser, biasInitialiser
      )
    )
  end

  Sequential{T,E}(architecture, layers)
end

function Base.copy(model::Sequential{T,E}) where {T,E}
  architecture = model.architecture
  layers = Vector{Dense{T}}()
  for layer in model.layers
    push!(layers, Dense{T,typeof(layer.activation)}(
      Param(Knet.atype(copy(layer.weight))),
      Param(Knet.atype(copy(layer.bias))),
      layer.activation
    ))
  end

  Sequential{T,E}(architecture, layers)
end

##############
# Attributes #
##############
function weight(u::Sequential, layer::Int)
  u.layers[layer].weight
end

function bias(u::Sequential, layer::Int)
  u.layers[layer].bias
end

function paramsAll(u::Sequential)
  Knet.params(u)
end

function paramsWeights(u::Sequential)
  θs = Knet.params(u)
  θs[end-1:end]
end

function paramsMesh(u::Sequential)
  θs = Knet.params(u)
  θs[1:end-2]
end

###########
# Forward #
###########
function ∇⁰(u::Sequential, X)
  for layer in u.layers
    X = layer(X)
  end
  X
end

function ∇¹(u::Sequential, X)
  X̄ = Param(X)
  ∑Ū = @diff sum(∇⁰(u, X̄))
  AutoGrad.grad(∑Ū, X̄)
end

function ∇²(u::Sequential{T,E}, X) where {T,E}
  N = size(X, 2)
  H = zeros(T, E, E, N)

  X̄ = Param(X)
  for i in 1:E
    ∑∇ᵢŪ = @diff begin
      ∑Ū = @diff sum(∇⁰(u, X̄))
      sum(AutoGrad.full(AutoGrad.grad(∑Ū, X̄))[i, :])
    end
    ∇²ᵢU = AutoGrad.full(AutoGrad.grad(∑∇ᵢŪ, X̄))
    for j in 1:E
      H[i, j, :] .= ∇²ᵢU[j, :]
    end
  end
  H
end

function Δ(u::Sequential{T,E}, X) where {T,E}
  X̄ = Param(X)

  ∑∇Ū = @diff begin
    ∑Ū = @diff sum(∇⁰(u, X̄))
    sum(AutoGrad.grad(∑Ū, X̄)[1:1, :])
  end
  ΔU = AutoGrad.grad(∑∇Ū, X̄)[1:1, :]

  for i in 2:E
    ∑∇Ū = @diff begin
      ∑Ū = @diff sum(∇⁰(u, X̄))
      sum(AutoGrad.grad(∑Ū, X̄)[i:i, :])
    end
    ΔU += AutoGrad.grad(∑∇Ū, X̄)[i:i, :]
  end

  ΔU
end

(u::Sequential)(X) = ∇⁰(u, X)
∇(u::Sequential, X) = ∇¹(u, X)
∂ₙ(u::Sequential, normal, X) = normal' * ∇¹(u, X)

∇⁰(u::Sequential) = X -> ∇⁰(u, X)
∇¹(u::Sequential) = X -> ∇¹(u, X)
∇²(u::Sequential) = X -> ∇²(u, X)
∇(u::Sequential) = X -> ∇(u, X)
∂ₙ(u::Sequential, normal) = X -> normal' * ∇¹(u, X)
Δ(u::Sequential) = X -> Δ(u, X)

function basisFunction(u::Sequential, i::Int)
  v = copy(u)

  layer = v.layers[end]
  layer.weight .= 0
  layer.bias .= 0

  if i <= v.architecture[end-1]
    layer.weight[i] .= 1
  else
    layer.bias .= 1
  end

  v
end
