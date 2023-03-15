using CPWLisation

T = Float64
N = 3

# Function to approximate and boundary conditions
ρ = Tanh{T}()
bc₋ = PointValue(T(0), T(ρ(0)))
bc₋ = PointTangent(T(0), T(0), T(1))
bc₊ = Asymptote()

ρ = ReLU(T(0.1))
bc₋ = Asymptote()
bc₊ = PointSlope(T(0), T(1 / 2))
bc₊ = PointFree(T(0))

# Learning rate, number of epochs and tolerance
lr = T(1.0e-3)
epochs = 3000
ε = T(1.0e-12)

# Main loop
xs, ys, L² = cpwl_ADAM(ρ, bc₋, bc₊, N, lr=lr, epochs=epochs, ε=ε)#, xs=xs, ys=ys)
println(sqrt(2 * L²))
xs, ys, L² = cpwl_BFGS(ρ, bc₋, bc₊, N, epochs=epochs, ε=ε, xs=xs, ys=ys)
println(sqrt(2 * L²))
