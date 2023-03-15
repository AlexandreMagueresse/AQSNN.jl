function save(filename::String, Ω::AbstractPolytope, u::Sequential; kwargs...)
  while isfile(filename)
    file, ext = splitext(filename)
    filename = file * "!" * ext
  end

  ρ = u.layers[1].activation
  dict = Dict(
    "Ω" => Ω,
    "A" => u.architecture,
    "ρ" => typeof(ρ).name.name,
    "ε" => hasfield(typeof(ρ), :εp) ? ρ.εp : -1,
    "weights" => [l.weight for l in u.layers],
    "biases" => [l.bias for l in u.layers],
    "extra" => kwargs
  )

  FileIO.save(filename, dict)
end

function load(filename::String, Ω, ρ)
  dict = FileIO.load(filename)

  T = eltype(Ω)
  E = embdim(Ω)
  u = Sequential{T,E}(dict["A"], ρ)

  for (l, w, b) in zip(u.layers, dict["weights"], dict["biases"])
    l.weight .= w
    l.bias .= b
  end

  u, dict["extra"]
end
