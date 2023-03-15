module AQSNNRunTests

using Test

@time @testset "Polytopes" begin
  include("Polytopes/runtests.jl")
end

@time @testset "Integration" begin
  include("Integration/runtests.jl")
end

end # module AQSNNRunTests
