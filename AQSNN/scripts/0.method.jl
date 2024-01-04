using AQSNN
using Plots
using LaTeXStrings
using Random

###############
# CPWLisation #
###############
# * When the number of parts is low, use best approximation in L^2 (from CPWLisation.jl)
# * When the number of parts is high, use the iterative tangents approach

ρ = Absε(0.1f0)
parts = 3:100
ys = Vector{Float32}(undef, length(parts))
for (i, part) in enumerate(parts)
  _, _, _, ε = CPWLise(ρ, part - 2)
  ys[i] = ε
  println(i)
end

plot()
plot!(collect(parts), log10.(ys), label=L"\operatorname{ReLU}_\varepsilon")

ρ = Tanh()
parts = 3:100
ys = Vector{Float32}(undef, length(parts))
for (i, part) in enumerate(parts)
  _, _, _, ε = CPWLise(ρ, part - 3)
  ys[i] = ε
  println(i)
end

plot!(collect(parts), log10.(ys), label=L"\tanh")

plot!(xlabel="Number of pieces", ylabel=L"\log\ \|\!\!\|\rho - \pi_n[\rho]\,\|\!\!\|_\infty")
savefig("plots/cpwlisation_L∞.pdf")

###############
# Integration #
###############
Ω = CartesianDomain(-1.0, +1.0)
Ωτ = simplexify(Ω)
Ωκ = convexify(Ω)
T = Float32
E = embdim(Ω)

Random.seed!(1)
ρ = Tanh()
u = Sequential{T,E}([1, 2, 1], ρ)
u.layers[1].weight[1] = 3
u.layers[1].bias[1] = 1
u.layers[1].weight[2] = 2
u.layers[1].bias[2] = -1

u.layers[2].weight[1] = +1
u.layers[2].weight[2] = -1
u.layers[2].bias[1] = 0

∫ex = log(cosh(4) / cosh(2)) / 3 - log(cosh(1) / cosh(3)) / 2

NΩMCs = 1:35
∫MCs = Vector{Float32}(undef, length(NΩMCs))
for (i, NΩ) in enumerate(NΩMCs)
  dΩmc = MonteCarloQuadrature(Ωτ, NΩ)
  ∫MCs[i] = ∫Ω(u) * dΩmc
end

plot()
plot!(NΩMCs, log10.(abs.(∫MCs .- ∫ex)), label="MC")

Ps = [0, 2, 4, 6]
Os = [2, 5, 10]

for O in Os
  NΩAQs = Vector{Int64}(undef, length(Ps))
  ∫AQs = Vector{Float32}(undef, length(Ps))

  for (i, P) in enumerate(Ps)
    Ωlin = Lineariser(Ωκ, ρ, P)
    dΩaq = AdaptiveQuadrature(u, Ωlin, O, 0.0, false)
    initialise!(dΩaq)

    NΩAQs[i] = length(Ωlin)
    ∫AQs[i] = ∫Ω(u) * dΩaq
  end
  plot!(NΩAQs, log10.(abs.(∫AQs .- ∫ex)), label="AQ, order $O")
end
plot!(xlabel="Number of points", ylabel="Integration error (log)", legend=:bottomright)

savefig("plots/integration.pdf")

########################
# Activation functions #
########################
x = -1:0.001:+1
ε = 0.1
ρ = ReLU()
ρ₁ = Absε(ε)
ρ₂ = Logε(ε)
ρ₃ = Erfε(ε)

plot(x, ∇⁰.(ρ, x), label="ReLU", c="black", ls=:dash, lw=2)
plot!(x, ∇⁰.(ρ₁, x), label="Absolute", c="red", lw=2)
plot!(x, ∇⁰.(ρ₂, x), label="Heaviside", c="blue", lw=2)
plot!(x, ∇⁰.(ρ₃, x), label="Mollifier", c="green", lw=2)
Plots.plot!(legend=:topleft, legendfontsize=11)
savefig("plots/fig1a.pdf")

plot(x, ∇¹.(ρ, x), label="ReLU", c="black", ls=:dash, lw=2)
plot!(x, ∇¹.(ρ₁, x), label="Absolute", c="red", lw=2)
plot!(x, ∇¹.(ρ₂, x), label="Heaviside", c="blue", lw=2)
plot!(x, ∇¹.(ρ₃, x), label="Mollifier", c="green", lw=2)
Plots.plot!(legend=:topleft, legendfontsize=11)
savefig("plots/fig1b.pdf")

plot(x, ∇².(ρ, x), label="ReLU", c="black", ls=:dash, lw=2)
plot!(x, ∇².(ρ₁, x), label="Absolute", c="red", lw=2)
plot!(x, ∇².(ρ₂, x), label="Heaviside", c="blue", lw=2)
plot!(x, ∇².(ρ₃, x), label="Mollifier", c="green", lw=2)
Plots.plot!(legend=:topleft, legendfontsize=11)
savefig("plots/fig1c.pdf")
