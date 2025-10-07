using Plots
using LaTeXStrings
using CPWLisation

#################
# Documentation #
#################
# This file finds the optimal abscissa for the approximation of SoftReLU and Tanh by a
# CPWL function that coincides with its tangents at the abscissa.

# For both functions, we perform the optimisation on half the plane (R- for ReLU so that
# it takes bounded values, and R+ for Tanh). The objective function `L` is the squared L2
# norm of the difference between the function and its approximation. As a result,
# the L2 norm on R as a whole is `sqrt(2 * L)`.

###################
# Hyperparameters #
###################
# Type for the abscissa
T = Float64

# Maximum number of free abscissa
# Warning: the execution time dramatically increases with N (explanation below)
# Nmax =  5       2 s
# Nmax = 10      15 s
# Nmax = 15     100 s
# Nmax = 20   1 500 s
Nmax = 5

# Learning rate
lr = 1.0e-2

# The optimisation will stop when the norm of the gradient of the objective function
# w.r.t. the free abscissa drops below ε
ε = 1.0e-12

############
# SoftReLU #
############
# The slant asymptotes of SoftReLU intersect at (0, 0), defining a 2-piece approximation
# (it is the ReLU function itself). The 3-piece approximation is obtained by placing a
# tangent at zero, and there is still no free abscissa here. We compute the corresponding
# L2 norms manually.

# The search for the optimal abscissa is extremely slow for large values of N because the
# decay rate of the difference between SoftReLU and its slant asymptotes (i.e. the decay
# rate of its second derivative) is only polynomial (2 * ε^2 / x^3). Even after the
# number of epochs provided below for N > 18, the norm of the gradient is still > ε.

# Using the notations of the article, we take ε = 0.1, meaning that ρ(0) = 0.1.
ρ = CPWLisation.ReLU(T(0.1))
bc₋ = Asymptote()
bc₊ = PointValue(T(0), T(ρ(0)))
errs_relu = Vector{Float64}()
ξs_relu = Vector{Vector{Float64}}()

# For the 2-piece approximation, the L2 norm is sqrt(8 * ε^3 / 3), where ε is
push!(errs_relu, sqrt(8 * 0.1^3 / 3))
push!(ξs_relu, T[])

# For the 3-piece approximation, we can still rely the distance function provided in this
# library, with no free abscissa.
ξs = T[]
err = CPWLisation.Optimisation.objective_tangent(ρ, bc₋, bc₊, ξs, verbose=false)
push!(errs_relu, sqrt(2 * err))
push!(ξs_relu, ξs)

# Main loop for N > 0, corresponding to 2*N+2 pieces
epochs = [
  20, 30, 60, 80, 120, 150, 150, 200, 200, 250,
  300, 500, 800, 2000, 2500, 3500, 8000, 12000, 16000, 20000,
  24000, 28000, 35000, 50000, 80000
]

for N in 1:Nmax
  epochs_ADAM = epochs[N]
  epochs_BFGS = div(epochs_ADAM, 2)
  ξs, L² = tangent_ADAM(ρ, bc₋, bc₊, N; lr, epochs=epochs_ADAM, ε)
  ξs, L² = tangent_BFGS(ρ, bc₋, bc₊, N; epochs=epochs_BFGS, ε, ξs)
  err = CPWLisation.Optimisation.objective_tangent(ρ, bc₋, bc₊, ξs, verbose=false)
  push!(errs_relu, sqrt(2 * err))
  push!(ξs_relu, ξs)
end

ns_relu = collect(3:2:(2*length(errs_relu)-1))
pushfirst!(ns_relu, 2)

########
# Tanh #
########
# The slant asymptotes of Tanh do not intersect, so we need to add a tangent at 0.
# The corresponding 3-piece approximation has no free abscissa so we separate this case.

# Even for large values of N, the search for the optimal abscissa converges reasonably
# fast because the decay rate of the difference between tanh and its slant asymptotes
# is exponential (-8 / exp(2x)).

ρ = CPWLisation.Tanh{T}()
bc₋ = PointValue(T(0), T(ρ(0)))
bc₊ = Asymptote()
errs_tanh = Vector{Float64}()
ξs_tanh = Vector{Vector{Float64}}()

ξs = T[]
err = CPWLisation.Optimisation.objective_tangent(ρ, bc₋, bc₊, ξs, verbose=false)
push!(errs_tanh, sqrt(2 * err))
push!(ξs_tanh, ξs)

# Main loop for N > 0, corresponding to 2*N+3 pieces
epochs = [
  10, 30, 50, 70, 80, 100, 120, 150, 200, 200,
  200, 250, 300, 300, 400, 400, 500, 600, 750, 800,
  1000, 1000, 1100, 1500, 1500
]

for N in 1:Nmax
  epochs_ADAM = epochs[N]
  epochs_BFGS = div(epochs_ADAM, 2)
  ξs, L² = tangent_ADAM(ρ, bc₋, bc₊, N; lr, epochs=epochs_ADAM, ε)
  ξs, L² = tangent_BFGS(ρ, bc₋, bc₊, N; epochs=epochs_BFGS, ε, ξs)
  err = CPWLisation.Optimisation.objective_tangent(ρ, bc₋, bc₊, ξs, verbose=false)
  push!(errs_tanh, sqrt(2 * err))
  push!(ξs_tanh, ξs)
end

ns_tanh = 3:2:(2*length(errs_tanh)+1)

##############
# Figure 2.a #
##############
N = 1
ρ = CPWLisation.Tanh{T}()
# ρ = ReLU(T(0.1))
xmax = 5

if Nmax >= N
  free = ξs_tanh[N]
  all_free = [-reverse(free)..., free...]
  xmax = !isempty(all_free) ? max(all_free[end] + 1, xmax) : xmax
  xmin = -xmax
  abscissa = [-reverse(free)..., free...]
  if ρ isa CPWLisation.Tanh
    name = L"\operatorname{\tanh}"
    fixed = T[xmin, 0, xmax]
    pushfirst!(abscissa, -T(Inf))
    insert!(abscissa, searchsortedfirst(abscissa, T(0)), T(0))
    push!(abscissa, T(Inf))
  else
    name = L"\operatorname{ReLU}_\varepsilon"
    fixed = T[xmin, xmax]
    pushfirst!(abscissa, T(Inf))
    push!(abscissa, T(Inf))
  end

  # ρ
  xs = xmin:0.01:xmax
  plot(xs, ρ.(xs), lw=2, ls=:dot, label=name)

  # π[ρ]
  a₀, b₀ = ϕ₋(ρ)
  ξ₊ = abscissa[2]
  a₊, b₊ = CPWLisation.∇¹(ρ, ξ₊), CPWLisation.∇⁰(ρ, ξ₊) - ξ₊ * CPWLisation.∇¹(ρ, ξ₊)
  x₋ = xmin
  x₊ = (b₊ - b₀) / (a₀ - a₊)
  xs = x₋:0.01:x₊
  plot!(xs, a₀ .* xs .+ b₀, color=2, lw=2, label="")

  for n in 2:length(abscissa)-1
    ξ₋ = abscissa[n-1]
    if !isfinite(ξ₋)
      a₋, b₋ = ϕ₋(ρ)
    else
      a₋, b₋ = CPWLisation.∇¹(ρ, ξ₋), CPWLisation.∇⁰(ρ, ξ₋) - ξ₋ * CPWLisation.∇¹(ρ, ξ₋)
    end

    ξ₀ = abscissa[n]
    a₀, b₀ = CPWLisation.∇¹(ρ, ξ₀), CPWLisation.∇⁰(ρ, ξ₀) - ξ₀ * CPWLisation.∇¹(ρ, ξ₀)

    ξ₊ = abscissa[n+1]
    if !isfinite(ξ₊)
      a₊, b₊ = ϕ₊(ρ)
    else
      a₊, b₊ = CPWLisation.∇¹(ρ, ξ₊), CPWLisation.∇⁰(ρ, ξ₊) - ξ₊ * CPWLisation.∇¹(ρ, ξ₊)
    end

    x₋ = max((b₀ - b₋) / (a₋ - a₀), xmin)
    x₊ = min((b₊ - b₀) / (a₀ - a₊), xmax)
    xs = x₋:0.01:x₊
    plot!(xs, a₀ .* xs .+ b₀, color=2, lw=2, label="")
  end

  ξ₋ = abscissa[end-1]
  a₋, b₋ = CPWLisation.∇¹(ρ, ξ₋), CPWLisation.∇⁰(ρ, ξ₋) - ξ₋ * CPWLisation.∇¹(ρ, ξ₋)
  a₀, b₀ = ϕ₊(ρ)
  x₋ = (b₀ - b₋) / (a₋ - a₀)
  x₊ = xmax
  xs = x₋:0.01:x₊
  plot!(xs, a₀ .* xs .+ b₀, color=2, lw=2, label="")

  if ρ isa CPWLisation.Tanh
    n = 2 * (N - 1) + 3
  else
    n = 2 * (N - 1) + 2
  end
  plot!(T[], T[], color=2, lw=2, label=L"\pi_{%$(n)}(" * name * L")")

  # abscissa
  scatter!(all_free, ρ.(all_free), markercolor=3, markershape=:circle, label="free")
  scatter!(fixed, ρ.(fixed), markercolor=4, markershape=:rect, label="fixed")
  plot!(legend=:topleft, legendfontsize=10)

  plot_path = joinpath("results", "figures", "fig2a_$(N).pdf")
  mkpath(dirname(plot_path))
  savefig(plot_path)
end

##############
# Figure 2.b #
##############
plot()
plot!(log10.(ns_tanh), log10.(errs_tanh), label=L"\tanh")
plot!(log10.(ns_relu), log10.(errs_relu), label=L"\operatorname{ReLU}_\varepsilon")
plot!(log10.(ns_relu), -2 .* log10.(ns_relu), label="Slope -2")
plot!(legend=:topright, legendfontsize=10)
plot!(xlabel=L"\log\ n")
plot!(ylabel=L"\log\ \|f - \pi_n[f]\|_{L^2(\mathbb{R})}")

plot_path = joinpath("results", "figures", "fig2b.pdf")
mkpath(dirname(plot_path))
savefig(plot_path)
