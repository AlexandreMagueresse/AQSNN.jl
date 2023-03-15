function collapse!(::AbstractLineariser)
  @notimplemented
end

function collapse!(lins::AbstractVector{<:AbstractLineariser})
  for lin in lins
    collapse!(lin)
  end
end

###############
# Dimension 0 #
###############
function collapse!(::AbstractLineariser{T,0}, rate::T=T(0.2)) where {T}
end

###############
# Dimension 1 #
###############
function collapse!(lin::AbstractLineariser{T,1}, rate::T=T(0.2)) where {T}
  ms = map(measure, lin)
  m̄ = rate * median(ms)

  found = true
  while found
    found = false

    for i in 2:length(lin.abscissae)-1
      (ms[i-1] >= m̄) && (ms[i] >= m̄) && continue
      deleteat!(lin.abscissae, i)
      deleteat!(lin.points, i)
      deleteat!(lin.images, i)
      deleteat!(ms, i)
      ms[i-1] = measure(lin[i-1])
      found = true
      break
    end
  end
end

###############
# Dimension 2 #
###############
function commonEdges(src, dst)
  srcL, dstL = length(src), length(dst)
  swapped = false

  if srcL > dstL
    src, dst = dst, src
    srcL, dstL = dstL, srcL
    swapped = true
  end
  reverse!(src)

  this = zeros(Int, 2 * srcL - 1)
  prev = zeros(Int, 2 * srcL - 1)

  maxL, srcEnd, dstEnd = 0, 0, 0

  for (dstJ, dstI) in enumerate(1:2*dstL-1)
    (dstJ > dstL) && (dstI -= dstL)

    for (srcJ, srcI) in enumerate(1:2*srcL-1)
      (srcJ > srcL) && (srcI -= srcL)
      (src[srcI] != dst[dstI]) && continue

      if (srcJ == 1) || (dstJ == 1)
        this[srcJ] = 1
      else
        this[srcJ] = prev[srcJ-1] + 1
      end

      if this[srcJ] > maxL
        maxL = this[srcJ]
        srcEnd, dstEnd = srcJ, dstJ
      end
    end
    copy!(prev, this)
    this .= 0
  end

  srcBeg = srcEnd - (maxL - 1)
  dstBeg = dstEnd - (maxL - 1)

  (srcEnd > srcL) && (srcEnd -= srcL)
  (dstEnd > dstL) && (dstEnd -= dstL)

  reverse!(src)
  srcEnd = srcL + 1 - srcEnd
  srcBeg = srcL + 1 - srcBeg

  if swapped
    src, dst = dst, src
    srcBeg, dstBeg = dstBeg, srcBeg
    srcEnd, dstEnd = dstEnd, srcEnd
  end

  maxL, (srcBeg, srcEnd), (dstBeg, dstEnd)
end

function collapse!(lin::AbstractLineariser{T,2}, rate::T=T(0.2)) where {T}
  ms = map(measure, lin)
  m̄ = rate * median(ms)

  ε = 10 * eps(T)
  u, v = getBasis(getPolytope(getMesh(lin)))

  active = [true for _ in 1:length(lin)]
  found = true
  while found
    found = false

    for (srcIdx, srcCycle) in enumerate(lin.cycles)
      # Skip large and inactive cells
      !active[srcIdx] && continue
      (ms[srcIdx] >= m̄) && continue
      srcL = length(srcCycle)

      # Find destination to merge with
      # * Most convex angles
      # * Largest area
      # Find common edges on both at the same time
      maxMsr, maxCvx = zero(T), 0
      dstIdx, dstCycle, dstL = -1, srcCycle, -1
      maxEdge, srcBeg, srcEnd, dstBeg, dstEnd = 0, -1, -1, -1, -1

      for (k₋, k₊) in CircularIterator(srcCycle)
        # Find dst on the other side of edge
        _, idx₁, idx₂ = lin.adjacency[minmax(k₋, k₊)]
        dstidx = (idx₁ == srcIdx) ? idx₂ : idx₁

        # Skip if boundary or inactve
        (dstidx == -1) && continue
        !active[dstidx] && continue

        # Initialise as candidate
        dstcycle = lin.cycles[dstidx]
        dstl = length(dstcycle)
        maxmsr, maxcvx = ms[dstidx], 0

        # Find location of common edges in dst
        maxedge, (srcbeg, srcend), (dstbeg, dstend) = commonEdges(srcCycle, dstcycle)

        # Extract indices of neighbour points in src and dst
        src₋ = srcCycle[(srcbeg - 1 >= 1) ? (srcbeg - 1) : srcL - 1]
        src₊ = srcCycle[(srcend + 1 <= srcL) ? (srcend + 1) : 1]
        dst₋ = dstcycle[(dstbeg + 1 <= dstl) ? (dstbeg + 1) : 1]
        dst₊ = dstcycle[(dstend - 1 >= 1) ? (dstend - 1) : dstl - 1]

        # Check whether angles are convex
        for (i₋, i₀, i₊) in ((src₋, k₋, dst₋), (dst₊, k₊, src₊))
          p₋, p₀, p₊ = lin.points[i₋], lin.points[i₀], lin.points[i₊]
          Δs₋, Δt₋ = dot₋(p₋, p₀, u), dot₋(p₋, p₀, v)
          Δs₊, Δt₊ = dot₋(p₊, p₀, u), dot₋(p₊, p₀, v)
          if det2(Δs₊, Δt₊, Δs₋, Δt₋) < ε
            maxcvx += 1
          end
        end

        if maxcvx > maxCvx
          maxMsr, maxCvx = maxmsr, maxcvx
          dstIdx, dstCycle, dstL = dstidx, dstcycle, dstl
          maxEdge, srcBeg, srcEnd, dstBeg, dstEnd = maxedge, srcbeg, srcend, dstbeg, dstend
        elseif maxmsr > maxMsr
          maxMsr = maxmsr
          dstIdx, dstCycle, dstL = dstidx, dstcycle, dstl
          maxEdge, srcBeg, srcEnd, dstBeg, dstEnd = maxedge, srcbeg, srcend, dstbeg, dstend
        end
      end

      # Make dst longer
      if srcL > dstL
        srcIdx, dstIdx = dstIdx, srcIdx
        srcCycle, dstCycle = dstCycle, srcCycle
        srcL, dstL = dstL, srcL
        srcBeg, dstBeg = dstBeg, srcBeg
        srcEnd, dstEnd = dstEnd, srcEnd
      end

      # Remove in between
      dstBeg += 1
      (dstBeg > dstL) && (dstBeg = 1)
      for _ in 1:maxEdge-2
        deleteat!(dstCycle, dstBeg)
        dstEnd -= 1
      end

      # Concatenate cycles
      while true
        srcBeg += 1
        (srcBeg > srcL) && (srcBeg = 1)
        (srcBeg == srcEnd) && break
        insert!(dstCycle, dstBeg, srcCycle[srcBeg])
        dstBeg += 1
      end

      # Deactivate src cycle
      active[srcIdx] = false

      # Update measures and adjacency
      ms[dstIdx] = measure(lin[dstIdx])

      for (k₋, k₊) in CircularIterator(srcCycle)
        k = minmax(k₋, k₊)
        a, b, c = lin.adjacency[k]

        # Replace src by dst
        # If (src, dst), do not write (dst, dst) but (dst, -1)
        other = (b == srcIdx) ? c : b
        if other == dstIdx
          lin.adjacency[k] = (a, dstIdx, -1)
        else
          lin.adjacency[k] = (a, dstIdx, other)
        end
      end

      found = true
      break
    end
  end

  i = length(lin.cycles)
  while i > 0
    if !active[i]
      deleteat!(lin.cycles, i)
    end
    i -= 1
  end

  error
end
