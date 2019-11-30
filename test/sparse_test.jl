# test simple solving
for i in 1:10
    A = rand(20, 20)
    sA = sparse(A)
    b = rand(20)

    @test fnnls(sA, b) ≈ fnnls(A, b)
    @test nnls(sA, b) ≈ nnls(A, b)
    @test pivot(sA, b) ≈ pivot(A, b)
    @test nonneg_lsq(sA,b;alg=:pivot, variant=:cache) ≈ nonneg_lsq(A,b;alg=:pivot, variant=:cache)
end
