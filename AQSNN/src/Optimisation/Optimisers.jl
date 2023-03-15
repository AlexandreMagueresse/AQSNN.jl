abstract type AbstractOptimiser end

function initialise!(::AbstractOptimiser)
  @abstractmethod
end

function step!(::AbstractOptimiser, ::Function, args...)
  @abstractmethod
end

function makeType(∇L, ::Param{T}) where {T}
  convert(T, ∇L)
end

########
# ADAM #
########
mutable struct ADAM{T} <: AbstractOptimiser
  const params::Vector{Param}
  const lr::T
  const γ::T
  epoch::Int
  const normalisation::Real

  function ADAM(params::Vector{Param}, lr::T; γ::T=1, normalisation::Real=-1) where {T}
    new{T}(params, lr, γ, 0, normalisation)
  end
end

function initialise!(opt::ADAM)
  for param in opt.params
    if isa(param.opt, Knet.Adam)
      param.opt.lr = opt.lr
    else
      param.opt = Knet.Adam(lr=opt.lr)
    end
  end
end

function step!(opt::ADAM, u::Sequential, objective::Objective, β, dΩ, dΓ)
  opt.epoch += 1

  L = @diff residual(objective, u, β, dΩ, dΓ)
  for param in opt.params
    ∇L = AutoGrad.grad(L, param)
    if opt.normalisation > 0
      normalize!(∇L, opt.normalisation)
    end
    ∇L = makeType(∇L, param)
    Knet.update!(param, ∇L)

    if mod(opt.epoch, 100) == 0
      param.opt.lr *= opt.γ
    end
  end
end

#######
# SGD #
#######
struct SGD{T} <: AbstractOptimiser
  params::Vector{Param}
  lr::T
  normalisation::Real

  function SGD(params::Vector{Param}, lr::T, normalisation::Real=-1) where {T}
    new{T}(params, lr, normalisation)
  end
end

function initialise!(opt::SGD)
  for param in opt.params
    param.opt = Knet.SGD(lr=opt.lr)
  end
end

function step!(opt::SGD, u::Sequential, objective::Objective, β, dΩ, dΓ)
  L = @diff residual(objective, u, β, dΩ, dΓ)
  for param in opt.params
    ∇L = AutoGrad.grad(L, param)
    if opt.normalisation > 0
      normalize!(∇L, opt.normalisation)
    end
    ∇L = makeType(∇L, param)
    Knet.update!(param, ∇L)
  end
end

#########
# LBFGS #
#########
struct LBFGS <: AbstractOptimiser
  params::Vector{Param}
  normalisation::Real

  function LBFGS(params::Vector{Param}, normalisation::Real=-1)
    new(params, normalisation)
  end
end

function initialise!(::LBFGS)
end

function lossfun(ws, opt, u, objective, β, dΩ, dΓ)
  ps = opt.params
  s = 1
  for p in ps
    l = length(p)
    copyto!(p.value, ws[s:s+l-1])
    s += l
  end

  residual(objective, u, β, dΩ, dΓ)
end

function gradfun(ws, gs, opt, u, objective, β, dΩ, dΓ)
  ps = opt.params
  s = 1
  for p in ps
    l = length(p)
    copyto!(p.value, ws[s:s+l-1])
    s += l
  end

  s = 1
  L = @diff residual(objective, u, β, dΩ, dΓ)
  for p in ps
    l = length(p.value)
    ∇L = AutoGrad.grad(L, p)
    if opt.normalisation > 0
      normalize!(∇L, opt.normalisation)
    end
    gs[s:s+l-1] .= reshape(∇L, l)
    s += l
  end
end

function step!(opt::LBFGS, u::Sequential, objective::Objective, β, dΩ, dΓ)
  ps = opt.params
  p0 = zeros(eltype(ps[1]), sum(length, ps))
  s = 1
  for p in ps
    l = length(p.value)
    p0[s:s+l-1] .= reshape(p.value, l)
    s += l
  end

  Optim.optimize(
    (ws) -> lossfun(ws, opt, u, objective, β, dΩ, dΓ),
    (gs, ws) -> gradfun(ws, gs, opt, u, objective, β, dΩ, dΓ),
    p0,
    Optim.Options(
      iterations=2,
      f_tol=1e-8,
      g_tol=1e-32,
      allow_f_increases=false,
    )
  )
  nothing
end

################
# LinearSolver #
################
struct LinearSolver <: AbstractOptimiser
end

function initialise!(::LinearSolver)
end

function step!(::LinearSolver, u::Sequential{T,D}, objective::Objective, β, dΩ, dΓ) where {T,D}
  N = u.architecture[end-1] + 1

  A = zeros(T, N, N)
  L = zeros(T, N)

  for i in 1:N
    ui = basisFunction(u, i)
    L[i] = linear(objective, ui, β, dΩ, dΓ)
    for j in i:N
      uj = basisFunction(u, j)
      A[i, j] = bilinear(objective, ui, uj, β, dΩ, dΓ)
    end
  end
  S = Symmetric(A)

  wb = pinv(S) * L
  d = norm(S * wb .- L)
  try
    wb_c = S \ L
    d_c = norm(S * wb_c .- L)
    if d_c < d
      wb = wb_c
      d = d_c
    end
  catch
  end
  wb_c = cg(S, L)
  d_c = norm(S * wb_c .- L)
  if d_c < d
    wb = wb_c
    d = d_c
  end

  u.layers[end].weight .= wb[1:end-1]'
  u.layers[end].bias .= wb[end]
  nothing
end
