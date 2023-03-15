module IntegrationTests

using Test

@testset "ReferenceQuadrature" begin
  include("ReferenceQuadratureTests.jl")
end

end # module IntegrationTests
