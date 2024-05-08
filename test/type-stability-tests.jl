@testitem "type-stability" begin
    struct Quux
        x::BlobVector{Int}
        y::Float64
    end
    struct Bar
        a::Int
        b::BlobBitVector
        c::Bool
        d::BlobVector{Float64}
        e::Blob{Quux}
    end

    @test Blobs.self_size(Bar) == 8 + 16 + 1 + 16 + 8 # Blob{Quux} is smaller in the blob

    function Blobs.child_size(::Type{Quux}, x_len::Int64, y::Float64)
        T = Quux
        +(Blobs.child_size(fieldtype(T, :x), x_len))
    end

    function Blobs.child_size(::Type{Bar}, b_len::Int64, c::Bool, d_len::Int64, x_len::Int64, y::Float64)
        T = Bar
        +(Blobs.child_size(fieldtype(T, :b), b_len),
        Blobs.child_size(fieldtype(T, :d), d_len),
        Blobs.child_size(fieldtype(T, :e), x_len, y))
    end

    function Blobs.init(quux::Blob{Quux}, free::Blob{Nothing}, x_len::Int64, y::Float64)
        free = Blobs.init(quux.x, free, x_len)
        quux.y[] = y
        free
    end

    function Blobs.init(bar::Blob{Bar}, free::Blob{Nothing}, b_len::Int64, c::Bool, d_len::Int64, x_len::Int64, y::Float64)
        free = Blobs.init(bar.b, free, b_len)
        free = Blobs.init(bar.d, free, d_len)
        free = Blobs.init(bar.e, free, x_len, y)
        bar.c[] = c
        free
    end

    bar = Blobs.malloc_and_init(Bar, 10, false, 20, 15, 1.5)

    # Test type stability
    @testset "getindex" begin
        test_getproperty1(b) = b.e
        @test @inferred(test_getproperty1(bar)) === bar.e
        test_getproperty2(b) = b.d
        @test @inferred(test_getproperty2(bar)) === bar.d
    end

    @testset "unsafe_load" begin
        @test @inferred(unsafe_load(bar)) isa Bar
    end

    @testset "self_size" begin
        @test @inferred(Blobs.self_size(Bar)) === 49
    end
end
