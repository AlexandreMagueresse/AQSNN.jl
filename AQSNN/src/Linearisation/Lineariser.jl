abstract type AbstractLineariser{T,M,E,C<:ConvexMesh} end

function getMesh(::AbstractLineariser)
  @abstractmethod
end

########
# Base #
########
function length(::AbstractLineariser)
  @abstractmethod
end

function eachindex(lin::AbstractLineariser)
  eachindex(1:length(lin))
end

function getindex(::AbstractLineariser, ::Int)
  @abstractmethod
end

function firstindex(::AbstractLineariser)
  @abstractmethod
end

function lastindex(::AbstractLineariser)
  @abstractmethod
end

function iterate(lin::AbstractLineariser, i::Int=1)
  if i > length(lin)
    nothing
  else
    (getindex(lin, i), i + 1)
  end
end

#################
# Linearisation #
#################
function linearise!(
  u::Sequential, lins::AbstractVector{<:AbstractLineariser};
  layer::Int=1
)
  for lin in lins
    linearise!(u, lin; layer=layer)
  end
end

function linearise!(
  ::Sequential, ::AbstractLineariser;
  layer::Int
)
  @notimplemented
end

#################
# Sorted arrays #
#################
function searchSorted(
  v::AbstractVector{T}, x::T, start::I=1, stop::I=length(v)
) where {T,I<:Integer}
  searchsorted(v, x, start, stop, Base.Order.Forward)
end

function intersectSorted(v::AbstractVector{T}, x::T, y::T) where {T}
  x, y = minmax(x, y)
  L = length(v)
  rx = searchSorted(v, x)
  ry = searchSorted(v, y, max(1, rx.stop), L)
  rx.start:ry.stop
end

########
# Line #
########
struct Line{T}
  number::Int
  p̄::Vector{T}
  Δp::Vector{T}
  abscissae::Vector{T}
  points::Vector{Int}
end

#########
# Graph #
#########
struct Graph{T}
  neighbours::Dict{Int,Vector{Int}}
  marks::Dict{Tuple{Int,Int},Int}
  lines::Dict{Tuple{Int,Int},Int}

  function Graph{T}() where {T}
    new(
      Dict{Int,Tuple{Int,Int,Int,Int}}(),
      Dict{Tuple{Int,Int},Int}(),
      Dict{Tuple{Int,Int},Int}()
    )
  end
end

function addNode!(graph::Graph{T}, n::Int) where {T}
  graph.neighbours[n] = Vector{Int}()
  nothing
end

function getEdges(graph::Graph)
  keys(graph.marks)
end

function addEdge!(
  graph::Graph{T}, p::Int, q::Int, line::Int, isBoundary::Bool
) where {T}
  (p == q) && return nothing
  k = minmax(p, q)
  haskey(graph.marks, k) && return nothing

  idx = searchsortedfirst(graph.neighbours[p], q)
  insert!(graph.neighbours[p], idx, q)

  idx = searchsortedfirst(graph.neighbours[q], p)
  insert!(graph.neighbours[q], idx, p)

  graph.marks[k] = isBoundary ? 1 : 0
  graph.lines[k] = line
  nothing
end

function addMark!(graph::Graph, p::Int, q::Int)
  k = minmax(p, q)
  m = get(graph.marks, k, 0) + 1
  graph.marks[k] = m

  if m == 2
    delete!(graph.marks, k)
    # delete!(graph.lines, k)

    idx = searchsortedfirst(graph.neighbours[p], q)
    deleteat!(graph.neighbours[p], idx)

    idx = searchsortedfirst(graph.neighbours[q], p)
    deleteat!(graph.neighbours[q], idx)
  end
end

function getNeighbours(graph::Graph, p::Int)
  get(graph.neighbours, p, Vector{Int}())
end

function getLine(graph::Graph, p::Int, q::Int)
  get(graph.lines, minmax(p, q), -1)
end

####################
# CircularIterator #
####################
struct CircularIterator{L}
  list::L
end

function Base.iterate(iter::CircularIterator, state=1)
  if state == length(iter.list) + 1
    nothing
  elseif state == length(iter.list)
    ((iter.list[end], iter.list[begin]), state + 1)
  else
    ((iter.list[state], iter.list[state+1]), state + 1)
  end
end
