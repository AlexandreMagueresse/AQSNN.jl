module Helpers

import Base: first, last, push!, delete!, isdone, iterate

include("Macros.jl")
export @abstractmethod
export @notimplemented
export @notreachable

include("LinearAlgebra.jl")
export dot₋
export dot₊
export dot₋₋
export dot₋₊
export norm2₋
export norm2₊
export norm₋
export det2
export sol2

include("Geometry.jl")
export middif
export inTriangle
export areaTriangle
export convexAngle
export cosAngle

include("CircularVector.jl")
export CircularNode
export CircularVector
export prevnext

end # module Helpers
