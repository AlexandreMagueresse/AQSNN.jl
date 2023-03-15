module Postprocessing

using FileIO
using PlotlyJS

using AQSNN.Polytopes
using AQSNN.Meshes
using AQSNN.Model
using AQSNN.Integration

include("metrics.jl")
export L1
export L2
export L∞
export H1
export H2

include("plots.jl")
export plotTrain
export plotComp
export plotDiff

include("io.jl")
export save
export load

end # module
