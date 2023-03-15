mutable struct CircularNode{T}
  data::T
  prev::CircularNode{T}
  next::CircularNode{T}

  function CircularNode{T}() where {T}
    node = new()
    node.prev = node
    node.next = node
    node
  end

  function CircularNode{T}(data) where {T}
    node = new(data)
    node.prev = node
    node.next = node
    node
  end

  function CircularNode{T}(data, prev, next) where {T}
    new(data, prev, next)
  end
end

function CircularNode(data::T, prev, next) where {T}
  CircularNode{T}(data, prev, next)
end

mutable struct CircularVector{T}
  handle::CircularNode{T}

  function CircularVector{T}() where {T}
    handle = CircularNode{T}()
    new(handle)
  end
end

function prevnext(c::CircularVector, n::CircularNode)
  prev = n.prev
  next = n.next
  if prev == c.handle
    last(c), next
  elseif next == c.handle
    prev, first(c)
  else
    prev, next
  end
end

#####################
# Extension of base #
#####################
Base.first(c::CircularVector) = c.handle.next
Base.last(c::CircularVector) = c.handle.prev

function Base.push!(c::CircularVector, element)
  next = CircularNode(element, c.handle.prev, c.handle)
  c.handle.prev.next = next
  c.handle.prev = next
  next
end

function Base.delete!(c::CircularVector, n::CircularNode)
  n.prev.next = n.next
  n.next.prev = n.prev
  c
end

function Base.iterate(c::CircularVector, n::CircularNode=c.handle.next)
  (n == c.handle) ? nothing : (n, n.next)
end
