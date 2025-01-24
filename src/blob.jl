"""
A pointer to a `T` stored inside a Blob.
"""
struct Blob{T}
    base::Ptr{Nothing}
    offset::Int64
    limit::Int64

    function Blob{T}(base::Ptr{Nothing}, offset::Int64, limit::Int64) where {T}
        @assert isbitstype(T)
        new(base, offset, limit)
    end
end

function Blob(ref::Base.RefValue{T}) where T
    Blob{T}(pointer_from_objref(ref), 0, sizeof(T))
end

function Blob(base::Ptr{T}, offset::Int64 = 0, limit::Int64 = sizeof(T)) where {T}
    Blob{T}(Ptr{Nothing}(base), offset, limit)
end

function Blob{T}(blob::Blob) where T
    Blob{T}(getfield(blob, :base), getfield(blob, :offset), getfield(blob, :limit))
end

function Blob{T}(blob::Blob, rel_offset::Int64) where {T}
    Blob{T}(
        getfield(blob, :base),
        getfield(blob, :offset) + rel_offset,
        getfield(blob, :limit)
    )
end

function assert_same_allocation(blob1::Blob, blob2::Blob)
    @noinline _throw(blob1, blob2) =
        throw(AssertionError("These blobs do not share the same allocation: $blob1 - $blob2"))
    if getfield(blob1, :base) != getfield(blob2, :base)
        _throw(blob1, blob2)
    end
end

function Base.pointer(blob::Blob{T}) where T
    convert(Ptr{T}, getfield(blob, :base) + getfield(blob, :offset))
end

function Base.:+(blob::Blob{T}, offset::Integer) where T
    Blob{T}(getfield(blob, :base), getfield(blob, :offset) + offset, getfield(blob, :limit))
end

function Base.:-(blob1::Blob, blob2::Blob)
    assert_same_allocation(blob1, blob2)
    getfield(blob1, :offset) - getfield(blob2, :offset)
end

@noinline function _throw_assert_not_null_error(typename::Symbol)
    throw(AssertionError("Null pointer dereference in Blob{$(typename)}"))
end

@noinline function boundscheck(blob::Blob{T}) where T
    base = getfield(blob, :base)
    offset = getfield(blob, :offset)
    limit = getfield(blob, :limit)
    element_size = self_size(T)
    if (offset < 0) || (offset + element_size > limit)
        throw(BoundsError())
    end
    if base == Ptr{Nothing}(0)
        _throw_assert_not_null_error(T.name.name)
    end
end

Base.@propagate_inbounds function Base.getindex(blob::Blob{T}) where T
    @boundscheck boundscheck(blob)
    unsafe_load(blob)
end

# TODO(jamii) do we need to align data?
"""
    self_size(::Type{T}, args...) where {T}

The number of bytes needed to allocate `T` itself.

Defaults to `sizeof(T)`.
"""
Base.@assume_effects :foldable function self_size(::Type{T}) where T
    # This function is marked :foldable to encourage constant folding for this types-only
    # static computation.
    if isempty(fieldnames(T))
        sizeof(T)
    else
        # Recursion is the fastest way to compile this, confirmed with benchmarks.
        # Alternatives considered:
        # - +(Iterators.map(self_size, fieldtypes(T))...)
        # - _iterative_sum_field_sizes for-loop (below).
        # Splatting is always slower, and breaks after ~30 fields.
        # The for-loop is faster after around 15-30 fields, so we pick an
        # arbitrary cutoff of 20:
        if fieldcount(T) > 20
            _iterative_sum_field_sizes(T)
        else
            _recursive_sum_field_sizes(T)
        end
    end
end
function _iterative_sum_field_sizes(::Type{T}) where T
    out = 0
    for f in fieldtypes(T)
        out += Blobs.self_size(f)
    end
    out
end
Base.@assume_effects :foldable _recursive_sum_field_sizes(::Type{T}) where {T} =
    _recursive_sum_field_sizes(T, Val(fieldcount(T)))
Base.@assume_effects :foldable _recursive_sum_field_sizes(::Type, ::Val{0}) = 0
Base.@assume_effects :foldable function _recursive_sum_field_sizes(::Type{T}, ::Val{i}) where {T,i}
    return self_size(fieldtype(T, i)) + _recursive_sum_field_sizes(T, Val(i-1))
end

# Recursion scales better than splatting for large numbers of fields.
Base.@assume_effects :foldable @inline function blob_offset(::Type{T}, i::Int) where {T}
    # Beyond this size, the tuple-construction in blob_offsets(T) refuses to const-fold,
    # in the *dynamic* `i` case, so we would end up with runtime tuple
    # construction and many many allocations.
    # For larger structs, doing dynamic field access, we elect to have a single
    # dynamic dispatch here with friendlier performance.
    if fieldcount(T) <= 32
        blob_offsets(T)[i]
    else
        _recursive_sum_field_sizes(T, Val(i - 1))
    end
end

Base.@assume_effects :foldable @inline function blob_offsets(::Type{T}) where {T}
    _recursive_field_offsets(T)
end
Base.@assume_effects :foldable _recursive_field_offsets(::Type{T}) where {T} =
    _recursive_field_offsets(T, Val(fieldcount(T)))
Base.@assume_effects :foldable _recursive_field_offsets(::Type, ::Val{0}) = ()
Base.@assume_effects :foldable _recursive_field_offsets(::Type, ::Val{1}) = (0,)
Base.@assume_effects :foldable function _recursive_field_offsets(::Type{T}, ::Val{i}) where {T,i}
    tup = _recursive_field_offsets(T, Val(i-1))
    return (tup..., tup[end] + self_size(fieldtype(T, i-1)))
end


# Manually write a compile-time loop in the type domain, to enforce constant-folding the
# fieldindexes even for large structs (with e.g. 100 fields). This might make compiling a
# touch slower, but it allows this to work for even large structs, like the manually-written
# `@generated` functions did before.
@inline function fieldindexes(::Type{T}) where {T}
    return _recursive_fieldindexes(T, Val(fieldcount(T)))
end
_recursive_fieldindexes(::Type{T}, ::Val{0}) where {T} = ()
function _recursive_fieldindexes(::Type{T}, ::Val{i}) where {T,i}
    next = _recursive_fieldindexes(T, Val(i-1))
    names = (fieldnames(typeof(next))..., fieldname(T, i))
    return NamedTuple{names}((next..., i))
end

# NOTE: An important optimization here is that the static operations that can be performed
# only on the type do not depend on the possibly runtime value `field`. We precompute the
# fieldname => fieldidx lookup table at compile time (as a NamedTuple), then use it at
# runtime. If the field is a known compiler constant (as in the `x.y` case), all the better.
@inline function Base.getindex(blob::Blob{T}, field::Symbol) where {T}
    fieldidx_lookup = fieldindexes(T)
    if !haskey(fieldidx_lookup, field)
        _throw_missing_field_error(T, field)
    end
    i = fieldidx_lookup[field]
    FT = fieldtype(T, i)
    Blob{FT}(blob + blob_offset(T, i))
end
@noinline _throw_missing_field_error(T, field) = error("$T has no field $field")

@noinline function _throw_getindex_boundserror(blob::Blob, i::Int)
    throw(BoundsError(blob, i))
end
@inline function Base.getindex(blob::Blob{T}, i::Int) where {T}
    @boundscheck if i < 1 || i > fieldcount(T)
        _throw_getindex_boundserror(blob, i)
    end
    return Blob{fieldtype(T, i)}(blob + Blobs.blob_offset(T, i))
end

Base.@propagate_inbounds function Base.setindex!(blob::Blob{T}, value::T) where T
    @boundscheck boundscheck(blob)
    unsafe_store!(blob, value)
end

# if the value is the wrong type, try to convert it (just like setting a field normally)
Base.@propagate_inbounds function Base.setindex!(blob::Blob{T}, value) where T
    setindex!(blob, convert(T, value))
end

macro _make_new(type, args)
    # :splatnew lets you directly invoke the type's inner constructor with a Tuple,
    # bypassing any effects from any custom constructors.
    return Expr(:splatnew, esc(type), esc(args))
end
@inline function Base.unsafe_load(blob::Blob{T}) where {T}
    if isempty(fieldnames(T))
        unsafe_load(pointer(blob))
    else
        # This recursive definition is *almost* as fast as the `@generated` code. On julia
        # 1.10, it has a single invoke function call here, which adds a few ns overhead.
        # But on julia 1.11, this generates the expected code and is just as fast.
        # We are sticking with this version though, to save the `@generated` compilation time.
        @_make_new(T, _unsafe_load_fields(blob, Val(fieldcount(T))))
    end
end
@inline _unsafe_load_fields(::Blob, ::Val{0}) = ()
function _unsafe_load_fields(blob::Blob{T}, ::Val{I}) where {T, I}
    @inline
    types = fieldnames(T)
    return (_unsafe_load_fields(blob, Val(I-1))..., unsafe_load(getindex(blob, types[I])))
end

@inline function Base.unsafe_store!(blob::Blob{T}, value::T) where {T}
    if isempty(fieldnames(T))
        unsafe_store!(pointer(blob), value)
        value
    else
        _unsafe_store!(blob, value, Val(fieldcount(T)))
        value
    end
end
# On julia 1.11, this is equivalantly fast to the `@generated` version.
# On julia 1.10, this is about 2x slower than generated for medium structs: ~10 ns vs ~5 ns.
# We will go with the recursive version, to avoid the compilation cost.
@inline _unsafe_store!(::Blob{T}, ::T, ::Val{0}) where {T} = nothing
function _unsafe_store!(blob::Blob{T}, value::T, ::Val{I}) where {T, I}
    @inline
    types = fieldnames(T)
    _unsafe_store!(blob, value, Val(I-1))
    unsafe_store!(getindex(blob, types[I]), getproperty(value, types[I]))
    nothing
end

# if the value is the wrong type, try to convert it (just like setting a field normally)
function Base.unsafe_store!(blob::Blob{T}, value) where {T}
    unsafe_store!(blob, convert(T, value))
end

# syntax sugar

function Base.propertynames(::Blob{T}, private::Bool=false) where T
    fieldnames(T)
end

function Base.getproperty(blob::Blob{T}, field::Symbol) where T
    getindex(blob, field)
end

function Base.setproperty!(blob::Blob{T}, field::Symbol, value) where T
    setindex!(blob, Val(field), value)
end

function rewrite_address(expr)
    if !(expr isa Expr)
        esc(expr)
    elseif expr.head == :.
        (object, field) = expr.args
        if field isa QuoteNode
            fieldname = field.value
        elseif field isa Expr && field.head == :quote
            fieldname = field.args[1]
        else
            error("Impossible?")
        end
        :(getindex($(rewrite_address(object)), $(QuoteNode(fieldname))))
    elseif expr.head == :ref
        object = expr.args[1]
        :(getindex($(rewrite_address(object)), $(map(esc, expr.args[2:end])...)))
    elseif expr.head == :macrocall
        rewrite_address(macroexpand(expr))
    else
        error("Don't know how to compute address for $expr")
    end
end

"""
    @a blob.x[2].y

Get a `Blob` pointing at the *address* of `blob.x[2].y`.
"""
macro a(expr)
    rewrite_address(expr)
end

function rewrite_value(expr)
    if (expr isa Expr) && (expr.head == :(=))
        if length(expr.args) == 2
            :(unsafe_store!($(rewrite_address(expr.args[1])), $(esc(expr.args[2]))))
        else
            error("Don't know how to compute assignment $expr")
        end
    else
        :(unsafe_load($(rewrite_address(expr))))
    end
end

"""
    @v blob.x[2].y

Get the *value* at `blob.x[2].y`.

    @v blob.x[2].y = 42

Set the *value* at `blob.x[2].y`.

NOTE macros bind tightly, so:

    # invalid syntax
    @v blob.x[2].y < 42

    # valid syntax
    (@v blob.x[2].y) < 42
"""
macro v(expr)
    rewrite_value(expr)
end

# patch pointers on the fly during load/store!

function self_size(::Type{Blob{T}}) where T
    sizeof(Int64)
end

@inline function Base.unsafe_load(blob::Blob{Blob{T}}) where {T}
    offset = unsafe_load(Blob{Int64}(blob))
    Blob{T}(getfield(blob, :base), getfield(blob, :offset) + offset, getfield(blob, :limit))
end

@inline function Base.unsafe_store!(blob::Blob{Blob{T}}, value::Blob{T}) where {T}
    assert_same_allocation(blob, value)
    offset = getfield(value, :offset) - getfield(blob, :offset)
    unsafe_store!(Blob{Int64}(blob), offset)
end
