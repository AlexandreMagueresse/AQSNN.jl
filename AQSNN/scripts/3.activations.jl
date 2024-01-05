using AQSNN
using Plots
using LaTeXStrings
using Random

x = -1:0.001:+1
ε = 0.1
ρ = AQSNN.ReLU()
ρ₁ = AQSNN.Absε(ε)
ρ₂ = AQSNN.Logε(ε)
ρ₃ = AQSNN.Erfε(ε)

plot(x, AQSNN.∇⁰.(ρ, x), label="ReLU", c="black", ls=:dash, lw=2)
plot!(x, AQSNN.∇⁰.(ρ₁, x), label="Absolute", c="red", lw=2)
plot!(x, AQSNN.∇⁰.(ρ₂, x), label="Heaviside", c="blue", lw=2)
plot!(x, AQSNN.∇⁰.(ρ₃, x), label="Mollifier", c="green", lw=2)
Plots.plot!(legend=:topleft, legendfontsize=11)

plot_path = joinpath("results", "figures", "fig1a.pdf")
mkpath(dirname(plot_path))
savefig(plot_path)

plot(x, AQSNN.∇¹.(ρ, x), label="ReLU", c="black", ls=:dash, lw=2)
plot!(x, AQSNN.∇¹.(ρ₁, x), label="Absolute", c="red", lw=2)
plot!(x, AQSNN.∇¹.(ρ₂, x), label="Heaviside", c="blue", lw=2)
plot!(x, AQSNN.∇¹.(ρ₃, x), label="Mollifier", c="green", lw=2)
Plots.plot!(legend=:topleft, legendfontsize=11)

plot_path = joinpath("results", "figures", "fig1b.pdf")
mkpath(dirname(plot_path))
savefig(plot_path)

plot(x, AQSNN.∇².(ρ, x), label="ReLU", c="black", ls=:dash, lw=2)
plot!(x, AQSNN.∇².(ρ₁, x), label="Absolute", c="red", lw=2)
plot!(x, AQSNN.∇².(ρ₂, x), label="Heaviside", c="blue", lw=2)
plot!(x, AQSNN.∇².(ρ₃, x), label="Mollifier", c="green", lw=2)
Plots.plot!(legend=:topleft, legendfontsize=11)

plot_path = joinpath("results", "figures", "fig1c.pdf")
mkpath(dirname(plot_path))
savefig(plot_path)
