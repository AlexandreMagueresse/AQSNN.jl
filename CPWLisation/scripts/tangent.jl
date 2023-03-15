using CPWLisation

T = Float64
N = 2

# Function to approximate and boundary conditions
ρ = Tanh{T}()
bc₋ = PointFree(T(0))
bc₊ = Asymptote()

ρ = ReLU(T(0.1))
bc₋ = Asymptote()
bc₊ = PointFree(T(0))

# Learning rate, number of epochs and tolerance
lr = T(1.0e-2)
epochs = 3000
ε = T(1.0e-12)

# Main loop
ξs, L² = tangent_ADAM(ρ, bc₋, bc₊, N, lr=lr, epochs=epochs, ε=ε)#, ξs=ξs)
display(CPWLisation.Optimisation.objective_tangent(ρ, bc₋, bc₊, ξs, verbose=true))
ξs, L² = tangent_BFGS(ρ, bc₋, bc₊, N, epochs=epochs, ε=ε, ξs=ξs)
display(CPWLisation.Optimisation.objective_tangent(ρ, bc₋, bc₊, ξs, verbose=true))
