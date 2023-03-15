using AQSNN
using Plots
using LaTeXStrings
using Random

##################
# CPWLisation L∞ #
##################
# When the number of parts is low, use best approximation in L^2
# When the number of parts is high, use the iterative tangents approach

ρ = Absε(0.1f0)
parts = 3:100
ys = Vector{Float32}(undef, length(parts))
for (i, part) in enumerate(parts)
  _, _, _, ε = CPWLise(ρ, part - 2)
  ys[i] = ε
  println(i)
end

plot(collect(parts), log10.(ys), label=L"\mathrm{ReLU}_\varepsilon")

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

##################
# CPWLisation L² #
##################
function ρn(x, ξs, αs, βs)
  i = searchsortedfirst(ξs, x)
  α, β = αs[i], βs[i]
  α * x + β
end

ρ = Tanh()
ρ = Absε(0.1f0)
parts = 3:2:15

x = -1:0.01:+1
integrals = Vector{Float64}()
r = 100
N = 100000

for part in parts
  ξs, αs, βs, ε = CPWLise(ρ, part)

  integral = 0
  for _ in 1:N
    x = r .* (2 .* rand() .- 1)
    integral += abs2(ρn(x, ξs, αs, βs) - ρ(x))
  end
  integral = sqrt(integral / N)
  push!(integrals, ε) #integral)
end
plot(log10.(parts), log10.(integrals))
(log10(integrals[end]) - log10(integrals[1])) / (log10(parts[end]) - log10(parts[1]))

N = 7
ρ = Tanh()
x = -5:0.01:5

xs, as, bs, ε = CPWLise(ρ, N)
if N == 5
  ξsfree = [-1.162452945276609, 1.162452945276609]
elseif N == 7
  ξsfree = [-1.6315255319296034, -0.8337120236555233, 0.8337120236555233, 1.6315255319296034]
end
ξsfixed = [x[1], 0.0, x[end]]

y = tanh.(x)
ŷ = [ρn(xi, xs, as, bs) for xi in x]

υsfree = tanh.(ξsfree)
υsfixed = tanh.(ξsfixed)

plot(x, y, label=L"\tanh")
plot!(x, ŷ, label=L"\pi_7(\tanh)")
scatter!(ξsfree, υsfree, label="free", m=:circle)
scatter!(ξsfixed, υsfixed, label="fixed", m=:rect)
plot!(legend=:topleft, legendfontpointsize=10,)

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
plotModel(u, Ωτ)

∫ex = log(cosh(4) / cosh(2)) / 3 - log(cosh(1) / cosh(3)) / 2

NΩMCs = 1:35
∫MCs = Vector{Float32}(undef, length(NΩMCs))
for (i, NΩ) in enumerate(NΩMCs)
  dΩmc = MonteCarloQuadrature(Ωτ, NΩ)
  ∫MCs[i] = ∫Ω(u) * dΩmc
end
plot(NΩMCs, log10.(abs.(∫MCs .- ∫ex)), label="MC")

collapse = 0.0
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
  plot!(NΩAQs, log10.(abs.(∫AQs .- ∫ex)), label="AQ($O)")
end
display(plot!(xlabel="Number of points", ylabel="Integration error (log)", legend=:bottomright))
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

plot(x, ∇⁰.(ρ, x), label="ReLU")
plot!(x, ∇⁰.(ρ₁, x), label="Absolute")
plot!(x, ∇⁰.(ρ₂, x), label="Heaviside")
plot!(x, ∇⁰.(ρ₃, x), label="Mollifier")
display(Plots.plot!(legend=:topleft))
savefig("plots/regularisation_zero.pdf")

plot(x, ∇¹.(ρ, x), label="ReLU")
plot!(x, ∇¹.(ρ₁, x), label="Absolute")
plot!(x, ∇¹.(ρ₂, x), label="Heaviside")
plot!(x, ∇¹.(ρ₃, x), label="Mollifier")
display(Plots.plot!(legend=:topleft))
savefig("plots/regularisation_one.pdf")

plot(x, ∇².(ρ, x), label="ReLU")
plot!(x, ∇².(ρ₁, x), label="Absolute")
plot!(x, ∇².(ρ₂, x), label="Heaviside")
plot!(x, ∇².(ρ₃, x), label="Mollifier")
display(Plots.plot!(legend=:topleft))
savefig("plots/regularisation_two.pdf")
