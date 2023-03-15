function FEMMin(
  target::Function, Ω::AbstractPolytope{T,1,1},
  partition::Tuple{Int}=(10,), activation::Function=ReLU()
) where {T}
  N = partition[1]
  model = Sequential{T,1}(
    [1, 1 * N, 1],
    activation
  )

  x₋, x₊ = getVertex(Ω, 1)[1], getVertex(Ω, 2)[1]
  x = reshape(convert(Array{T}, range(x₋, x₊, N)), (1, N))
  h = (x₊ - x₋) / (N - 1)
  h⁻¹ = inv(h)

  layer = model.layers[1]
  i = 1
  for xi in x
    # h⁻¹ (x - xᵢ) + 1
    layer.weight[i] = h⁻¹
    layer.bias[i] = 1 - h⁻¹ * xi
    i += 1
  end

  A = zeros(Int16, N, N)
  for i in 1:N
    for j in 1:i
      # reluⱼ(xᵢ) = max(h⁻¹ (xᵢ - xⱼ) + 1, 0)
      A[i, j] = 1 + i - j
    end
  end
  b = target(x)[1, :]

  layer = model.layers[2]
  layer.weight .= (A \ b)'
  layer.bias .= 0

  model
end

function FEMMin(
  target::Function, Ω::AbstractPolytope{T,2,2},
  partition::Tuple{Int,Int}=(10, 10), activation::Function=ReLU()
) where {T}
  Nx, Ny = partition
  model = Sequential{T,2}(
    [
      2,
      2 * Nx * Ny,
      4 * Nx * Ny,
      1 * Nx * Ny,
      1
    ],
    activation
  )

  x₋, x₊ = extrema(i -> getVertex(Ω, i)[1], 1:length(Ω))
  y₋, y₊ = extrema(i -> getVertex(Ω, i)[2], 1:length(Ω))
  hx = (x₊ - x₋) / (Nx - 1)
  hy = (y₊ - y₋) / (Ny - 1)
  hx⁻¹ = inv(hx)
  hy⁻¹ = inv(hy)

  x = range(x₋, x₊, Nx)
  y = range(y₋, y₊, Ny)
  xy = zeros(T, (2, Nx * Ny))
  r = 1
  for xi in x
    for yj in y
      xy[1, r] = xi
      xy[2, r] = yj
      r += 1
    end
  end

  layer = model.layers[1]
  layer.weight .= 0
  layer.bias .= 0
  r = 1
  for i in 1:Nx
    for j in 1:Ny
      # hy⁻¹ (y - yⱼ) + 1
      layer.weight[r, 2] = hy⁻¹
      layer.bias[r] = 1 - hy⁻¹ * y[j]
      # hx⁻¹ (x - xᵢ) + 1
      layer.weight[r+1, 1] = hx⁻¹
      layer.bias[r+1] = 1 - hx⁻¹ * x[i]
      r += 2
    end
  end

  minMat = T[
    +1 +1
    -1 -1
    +1 -1
    -1 +1
  ]
  minVec = T[+1 -1 -1 -1]

  # Take combinations two by two
  layer = model.layers[2]
  layer.weight .= 0
  layer.bias .= 0
  for i in 0:Nx*Ny-1
    layer.weight[4*i+1:4*i+4, 2*i+1:2*i+2] = minMat
  end

  layer = model.layers[3]
  layer.weight .= 0
  layer.bias .= 0
  for i in 1:Nx*Ny
    layer.weight[i, 4*i-3:4*i] = minVec
  end
  layer.weight ./= 2

  A = zeros(Int16, Nx * Ny, Nx * Ny)
  for i in 1:Nx
    for j in 1:Ny
      for k in 1:i
        for l in 1:j
          # reluₖₗ(xᵢ, yⱼ) = min(max(1 + hx⁻¹ (xᵢ - xₖ), 0), max(1 + hy⁻¹ (yⱼ - yₗ)))
          A[(i-1)*Nx+j, (k-1)*Nx+l] = 1 + min(i - k, j - l)
        end
      end
    end
  end
  b = target(xy)'

  layer = model.layers[4]
  layer.weight .= (A \ b)'
  layer.bias .= 0
  model
end

function FEMMinFull(
  target::Function, Ω::AbstractPolytope{T,1,1},
  partition::Tuple{Int}=(10,), activation::Function=ReLU()
) where {T}
  N = partition[1]
  model = Sequential{T,1}(
    [1, 2 * N, 4 * N, 1 * N, 1],
    activation
  )

  x₋, x₊ = getVertex(Ω, 1)[1], getVertex(Ω, 2)[1]
  x = reshape(convert(Array{T}, range(x₋, x₊, N)), (1, N))
  h = (x₊ - x₋) / (N - 1)
  h⁻¹ = inv(h)

  layer = model.layers[1]
  i = 1
  for xi in x
    # + h⁻¹ (x - xᵢ) + 1
    layer.weight[i] = h⁻¹
    layer.bias[i] = 1 - h⁻¹ * xi
    # - h⁻¹ (x - xᵢ) + 1
    layer.weight[i+1] = -h⁻¹
    layer.bias[i+1] = 1 + h⁻¹ * xi
    i += 2
  end

  minMat = T[
    +1 +1
    -1 -1
    +1 -1
    -1 +1
  ]
  minVec = T[+1 -1 -1 -1]

  layer = model.layers[2]
  layer.weight .= 0
  layer.bias .= 0
  for i in 0:N-1
    layer.weight[4*i+1:4*i+4, 2*i+1:2*i+2] = minMat
  end

  layer = model.layers[3]
  layer.weight .= 0
  layer.bias .= 0
  for i in 1:N
    layer.weight[i, 4*i-3:4*i] = minVec
  end
  layer.weight ./= 2

  layer = model.layers[4]
  layer.weight .= target(x)
  layer.bias .= 0

  model
end

function FEMMinFull(
  target::Function, Ω::AbstractPolytope{T,2,2},
  partition::Tuple{Int,Int}=(10, 10), activation::Function=ReLU()
) where {T}
  Nx, Ny = partition
  model = Sequential{T,2}(
    [
      2,
      2 * (Nx + Ny + Nx + Ny - 1),
      4 * (Nx + Ny + Nx + Ny - 1),
      1 * (Nx + Ny + Nx + Ny - 1),
      4 * Nx * Ny + Nx + Ny - 1,
      Nx * Ny + Nx + Ny - 1,
      4 * Nx * Ny,
      Nx * Ny,
      1
    ],
    activation
  )

  x₋, x₊ = extrema(i -> getVertex(Ω, i)[1], 1:length(Ω))
  y₋, y₊ = extrema(i -> getVertex(Ω, i)[2], 1:length(Ω))
  hx = (x₊ - x₋) / (Nx - 1)
  hy = (y₊ - y₋) / (Ny - 1)
  hx⁻¹ = inv(hx)
  hy⁻¹ = inv(hy)

  x = range(x₋, x₊, Nx)
  y = range(y₋, y₊, Ny)
  xy = zeros(T, (2, Nx * Ny))
  r = 1
  for xi in x
    for yj in y
      xy[1, r] = xi
      xy[2, r] = yj
      r += 1
    end
  end

  layer = model.layers[1]
  layer.weight .= 0
  layer.bias .= 0
  i = 1
  for xi in x
    # + hx⁻¹ (x - xᵢ) + 1
    layer.weight[i, 1] = +hx⁻¹
    layer.bias[i] = 1 - hx⁻¹ * xi
    # - hx⁻¹ (x - xᵢ) + 1
    layer.weight[i+1, 1] = -hx⁻¹
    layer.bias[i+1] = 1 + hx⁻¹ * xi
    i += 2
  end
  for yi in y
    # + hy⁻¹ (y - yᵢ) + 1
    layer.weight[i, 2] = +hy⁻¹
    layer.bias[i] = 1 - hy⁻¹ * yi
    # - hy⁻¹ (y - yᵢ) + 1
    layer.weight[i+1, 2] = -hy⁻¹
    layer.bias[i+1] = 1 + hy⁻¹ * yi
    i += 2
  end
  for xi in x
    # + hx⁻¹ (x - xᵢ) + hy⁻¹ (y - yᵢ) + 1
    layer.weight[i, :] = [+hx⁻¹, +hy⁻¹]
    layer.bias[i] = 1 - hx⁻¹ * xi - hy⁻¹ * y₋
    # - hx⁻¹ (x - xᵢ) - hy⁻¹ (y - yᵢ) + 1
    layer.weight[i+1, :] = [-hx⁻¹, -hy⁻¹]
    layer.bias[i+1] = 1 + hx⁻¹ * xi + hy⁻¹ * y₋
    i += 2
  end
  for yi in y[2:end]
    # + hx⁻¹ (x - xᵢ) + hy⁻¹ (y - yᵢ) + 1
    layer.weight[i, :] = [+hx⁻¹, +hy⁻¹]
    layer.bias[i] = 1 - hx⁻¹ * x₊ - hy⁻¹ * yi
    # - hx⁻¹ (x - xᵢ) - hy⁻¹ (y - yᵢ) + 1
    layer.weight[i+1, :] = [-hx⁻¹, -hy⁻¹]
    layer.bias[i+1] = 1 + hx⁻¹ * x₊ + hy⁻¹ * yi
    i += 2
  end

  minMat = T[
    +1 +1
    -1 -1
    +1 -1
    -1 +1
  ]
  minMat1 = T[+1 -1 +1 -1]
  minMat2 = T[+1 -1 -1 +1]
  minVec = T[+1 -1 -1 -1]

  # Create hats two by two
  layer = model.layers[2]
  layer.weight .= 0
  layer.bias .= 0
  for i in 0:(Nx+Ny+Nx+Ny-1)-1
    layer.weight[4*i+1:4*i+4, 2*i+1:2*i+2] = minMat
  end

  layer = model.layers[3]
  layer.weight .= 0
  layer.bias .= 0
  for i in 1:(Nx+Ny+Nx+Ny-1)
    layer.weight[i, 4*i-3:4*i] = minVec
  end
  layer.weight ./= 2

  # Take combinations vertical and horizontal
  layer = model.layers[4]
  layer.weight .= 0
  layer.bias .= 0
  k = 1
  for i in 1:Nx
    for j in 1:Ny
      layer.weight[k:k+3, i] = minMat1
      layer.weight[k:k+3, Nx+j] = minMat2
      k += 4
    end
  end
  # Keep diagonals
  for i in 1:(Nx+Ny-1)
    layer.weight[4*Nx*Ny+i, Nx+Ny+i] = 1
  end

  layer = model.layers[5]
  layer.weight .= 0
  layer.bias .= 0
  for i in 1:Nx*Ny
    layer.weight[i, 4*i-3:4*i] = minVec
  end
  layer.weight ./= 2
  # Keep diagonals
  for i in 1:(Nx+Ny-1)
    layer.weight[Nx*Ny+i, 4*Nx*Ny+i] = 1
  end

  # Take combinations with diagonals
  layer = model.layers[6]
  layer.weight .= 0
  layer.bias .= 0
  k = 1
  for i in 1:Nx
    for j in 1:Ny
      layer.weight[k:k+3, Nx*(i-1)+j] = minMat1
      layer.weight[k:k+3, Nx*Ny+i+j-1] = minMat2
      k += 4
    end
  end

  layer = model.layers[7]
  layer.weight .= 0
  layer.bias .= 0
  for i in 1:Nx*Ny
    layer.weight[i, 4*i-3:4*i] = minVec
  end
  layer.weight ./= 2

  # Weights
  layer = model.layers[8]
  layer.weight .= target(xy)
  layer.bias .= 0

  model
end

function FEMSum(
  target::Function, Ω::AbstractPolytope{T,1,1},
  partition::Tuple{Int}=(10,), activation::Function=ReLU()
) where {T}
  N = partition[1]
  model = Sequential{T,1}(
    [1, N + 2, N, 1],
    activation
  )

  x₋, x₊ = getVertex(Ω, 1)[1], getVertex(Ω, 2)[1]
  x = reshape(convert(Array{T}, range(x₋, x₊, N)), (1, N))
  h = (x₊ - x₋) / (N - 1)
  h⁻¹ = inv(h)

  layer = model.layers[1]
  for i in 1:N+2
    # h⁻¹ (x - xᵢ₋₁) + 1
    layer.weight[i] = h⁻¹
    layer.bias[i] = 1 - h⁻¹ * x₋ - (i - 1)
  end

  layer = model.layers[2]
  layer.weight .= 0
  layer.bias .= 0
  for i in 2:N+1
    layer.weight[i-1, i-1] = 1
    layer.weight[i-1, i] = -2
    # layer.weight[i-1, i+1] = 1
  end

  A = model.layers[2](model.layers[1](x))
  b = target(x)[1, :]

  layer = model.layers[3]
  layer.weight .= (A \ b)'
  layer.bias .= 0

  model
end

function FEMSum(
  target::Function, Ω::AbstractPolytope{T,2,2},
  partition::Tuple{Int,Int}=(10, 10), activation::Function=ReLU()
) where {T}
  Nx, Ny = partition
  model = Sequential{T,2}(
    [
      2,
      Nx + 4 + Ny + 4 + Nx + Ny + 1,
      Nx * Ny,
      1
    ],
    activation
  )

  x₋, x₊ = extrema(i -> getVertex(Ω, i)[1], 1:length(Ω))
  y₋, y₊ = extrema(i -> getVertex(Ω, i)[2], 1:length(Ω))
  hx = (x₊ - x₋) / (Nx - 1)
  hy = (y₊ - y₋) / (Ny - 1)
  hx⁻¹ = inv(hx)
  hy⁻¹ = inv(hy)

  x = range(x₋, x₊, Nx)
  y = range(y₋, y₊, Ny)
  xy = zeros(T, (2, Nx * Ny))
  r = 1
  for xi in x
    for yj in y
      xy[1, r] = xi
      xy[2, r] = yj
      r += 1
    end
  end

  layer = model.layers[1]
  layer.weight .= 0
  layer.bias .= 0
  r = 1
  for i in 1:Nx+4
    # hx⁻¹ (x - xᵢ₋₂) + 1
    layer.weight[r, 1] = hx⁻¹
    layer.bias[r] = 1 - hx⁻¹ * x₋ - (i - 2)
    r += 1
  end
  for j in 1:Ny+4
    # hy⁻¹ (y - yⱼ₋₂) + 1
    layer.weight[r, 2] = hy⁻¹
    layer.bias[r] = 1 - hy⁻¹ * y₋ - (j - 2)
    r += 1
  end
  for k in Ny-1:-1:-(Nx + 1)
    # hx⁻¹ (x - x₋ + k) - hy⁻¹ (y - y₋ - k)
    layer.weight[r, :] = [hx⁻¹, -hy⁻¹]
    layer.bias[r] = 1 - hx⁻¹ * x₋ + hy⁻¹ * y₋ + k
    r += 1
  end

  layer = model.layers[2]
  layer.weight .= 0
  layer.bias .= -3
  r = 1
  ox = 0
  oy = Nx + 4
  oz = Nx + 4 + Ny + 4
  for j in 1:Ny
    for i in 1:Nx
      ix = i + 2
      iy = j + 2
      iz = Ny - j + i + 1
      layer.weight[r, ox+ix-2] = 1
      layer.weight[r, ox+ix] = -2
      # layer.weight[r, ox+ix+2] = 1

      layer.weight[r, oy+iy-2] = 1
      layer.weight[r, oy+iy] = -2
      # layer.weight[r, oy+iy+2] = 1

      layer.weight[r, oz+iz-1] = 1
      layer.weight[r, oz+iz] = -2
      # layer.weight[r, oz+iz+1] = 1
      r += 1
    end
  end

  A = model.layers[2](model.layers[1](xy))
  b = target(xy)'

  layer = model.layers[3]
  layer.weight .= (A \ b)'
  layer.bias .= 0

  model
end
