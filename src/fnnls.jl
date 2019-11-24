"""
x = fnnls(AtA, Atb; ...)

Returns x that solves A*x = b in the least-squares sense, subject to x >=0. The
inputs are the cross-products AtA = A'*A and Atb = A'*b. Uses the modified
active set method of Bro and De Jong (1997).

Optional arguments:
    tol: tolerance for nonnegativity constraints
    max_iter: maximum number of iterations (counts inner loop iterations)

References:
    Bro R, De Jong S. A fast non-negativitity-constrained least squares
    algorithm. Journal of Chemometrics. 11, 393–401 (1997)
"""
function fnnls(AtA, Atb::AbstractVector{T}; kwargs...) where {T}
    work = fnnls_work(AtA, Atb)
    fnnls!(work, AtA, Atb; kwargs...)
end

function fnnls_work(AtA, Atb::AbstractVector{T}) where {T}
    n    = size(AtA,1)
    x    = zeros(T, n)
    s    = zeros(T, n)
    w    = similar(x)
    P    = BitArray(undef, n)
    nP   = BitArray(undef, n)
    iP   = BitArray(undef, n)
    AtAP = similar(AtA)
    AtbP = similar(Atb)
    return (x=x, s=s, w=w, P=P, nP=nP, iP=iP, AtAP=AtAP, AtbP=AtbP)
end

function fnnls!(
        work,
        AtA,
        Atb::AbstractVector{T};
        tol::T = sqrt(eps(T)),
        max_iter::Int = 30 * size(AtA, 2)
    ) where {T}

    x, s, w, P, nP, iP, AtAP, AtbP = work
    n = size(AtA,1)
    x .= 0
    s .= 0

    # P is a bool array storing positive elements of x
    # i.e., x[P] > 0 and x[~P] == 0
    P  .= x .> tol
    nP .= .!(P)

    mul!(w, AtA, x)
    w .= Atb .- w; #w = Atb - AtA*x

    function anyP(f, x, P)
        @inbounds for i in eachindex(x,P)
            if P[i] && f(x[i])
                return true
            end
        end
        return false
    end

    function argmaxP(x, P)
        α = T(-Inf)
        out = 1
        @inbounds for i in eachindex(x, P)
            if P[i] && (x[i] > α)
                α = max(α, x[i])
                out = i # i = argmax(w .* nP)
            end
        end
        return out
    end

    function assigntoP!(dst, srcP, P)
        j = 0
        @inbounds for i in eachindex(dst, P)
            if P[i]
                j += 1
                dst[i] = srcP[j]
            end
        end
        return dst
    end

    function copytocorner!(dst, src, P)
        j1, j2 = 0, 0
        @inbounds for i2 in 1:size(src, 2)
            if P[i2]
                j2 += 1
                j1  = 0
                for i1 in 1:size(src, 1)
                    if P[i1]
                        j1 += 1
                        dst[j1,j2] = src[i1,i2]
                    end
                end
            end
        end
        return j1, j2
    end

    function copytotop!(dst, src, P)
        j1 = 0
        @inbounds for i1 in 1:size(src, 1)
            if P[i1]
                j1 += 1
                dst[j1] = src[i1]
            end
        end
        return j1
    end

    function alpha(x, s, P)
        α = oftype(Inf, one(eltype(x))/one(eltype(s)))
        @inbounds for i in eachindex(x, s, P)
            if P[i]
                α = min(α, x[i]/(x[i] - s[i]))
            end
        end
        return α
    end

    function solveP!(y, Atmp, ytmp, A, x, P) # s[P] = AtA[P,P] \ Atb[P]
        # (1)
        # y[P] .= A[P,P] \ x[P]
        # (2)
        # assigntoP!(y, Symmetric(A[P,P]) \ x[P], P) # 670us
        # assigntoP!(y, A[P,P] \ x[P], P) # 547 us
        # (3)
        invA = cholesky!(Symmetric(A[P,P]); check = false) # 482 us
        # invA = cholesky!(A[P,P]; check = false) # 510 ua
        assigntoP!(y, ldiv!(invA, x[P]), P)
        # (4)
        # m_, n_ = copytocorner!(Atmp, A, P)
        # p_ = copytotop!(ytmp, x, P)
        # invA = cholesky!(Symmetric(view(Atmp, 1:m_, 1:n_)); check = false)
        # outP = ldiv!(invA, view(ytmp, 1:p_)) # 517 us
        # assigntoP!(y, outP, P)
    end        

    # We have reached an optimum when either:
    #   (a) all elements of x are positive (no nonneg constraints activated)
    #   (b) ∂f/∂x = A' * (b - A*x) > 0 for all nonpositive elements of x
    iter = 0
    while sum(P) < n && anyP(>(tol), w, nP) && iter < max_iter

        # find i that maximizes w, restricting i to indices not in P
        # Note: the while loop condition guarantees at least one w[~P]>0
        i = argmaxP(w, nP) # i = argmax(w .* nP)

        # Move i to P
        P[i]  = true
        nP[i] = false

        # Solve least-squares problem, with zeros for columns/elements not in P
        solveP!(s, AtAP, AtbP, AtA, Atb, P) # s[P] = AtA[P,P] \ Atb[P]
        @inbounds for i in eachindex(s, nP)
            nP[i] && (s[i] = 0) # s[nP] .= zero(eltype(s)) # zero out elements not in P
        end        

        # Inner loop: deal with negative elements of s
        while anyP(<=(tol), s, P)
            iter += 1

            # find indices in P where s is negative
            @inbounds iP .= (s .<= tol) .& P

            # calculate step size, α, to prevent any xᵢ from going negative
            α = alpha(x, s, iP) # minimum(x[iP] ./ (x[iP] .- s[iP]))

            # update solution (pushes some xᵢ to zero)
            @inbounds x .+= α .* (s .- x)

            # Remove all i in P where x[i] == 0
            @inbounds for i = 1:n
                if P[i] && abs(x[i]) < tol
                    P[i]  = false # remove i from P
                    nP[i] = true
                end
            end

            # Solve least-squares problem again, zeroing nonpositive columns
            solveP!(s, AtAP, AtbP, AtA, Atb, P) # s[P] = AtA[P,P] \ Atb[P]
            @inbounds for i in eachindex(s, nP)
                nP[i] && (s[i] = 0) # s[nP] .= zero(eltype(s)) # zero out elements not in P
            end
        end

        # update solution
        x .= s
        mul!(w, AtA, x)
        w .= Atb .- w; #w = Atb - AtA*x
    end

    return x
end

function fnnls(A,
               B::AbstractMatrix;
               gram::Bool = false,
               use_parallel::Bool = true,
               kwargs...)

    n = size(A,2)
    k = size(B,2)

    if gram
        # A,B are actually Gram matrices
        AtA = A
        AtB = B
    else
        # cache matrix computations
        AtA = A'*A
        AtB = A'*B
    end

    if use_parallel && nprocs()>1
        X = @distributed (hcat) for i=1:k
            fnnls(AtA, AtB[:,i]; kwargs...)
        end
    else
        X = Array{eltype(B)}(undef,n,k)
        for i = 1:k
            X[:,i] = fnnls(AtA, AtB[:,i]; kwargs...)
        end
    end

    return X
end
