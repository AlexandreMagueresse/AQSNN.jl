macro abstractmethod(message="This function belongs to an abstract type and must be specialised.")
  quote
    error($(esc(message)))
  end
end

macro notimplemented(message="This function is not implemented yet.")
  quote
    error($(esc(message)))
  end
end

macro notreachable(message="This line cannot be reached.")
  quote
    error($(esc(message)))
  end
end
