using LinearAlgebra

error_eps = 1e-6

function generate_random_matrix(m, n, a, b)
    return [a+(b-a)*rand() for _ in 1:1:m, _ in 1:1:n]
end

function truncate(U, D, V, eps)
    gamma = size(D,1)
    for i in size(D,1):-1:2
        if D[i] < eps
            gamma = i-1
        end
    end
    V_tr = V'
    new_U = U[:, 1:gamma]
    new_D = diagm(D)[1:gamma , 1:gamma]
    new_V_tr = V_tr[1:gamma, :]
    return new_U, new_D, new_V_tr
end

function main()
    A = generate_random_matrix(1000, 400, 0, 1)
    U, D, V = svd(A)
    println(size(U), size(D), size(V))
    # println(D)
    built_A = U*diagm(D)*V'
    println(size(built_A))
    println(all(abs.(built_A-A) .< error_eps))
    U_truncated, D_truncated, V_tr_truncated = truncate(U, D, V, D[end]+0.0005)
    println(size(U_truncated), size(D_truncated), size(V_tr_truncated))
    tr_built_A = U_truncated*D_truncated*V_tr_truncated
    # println(abs.(tr_built_A-A) .< error_eps)
    println(all(abs.(tr_built_A-A) .< error_eps))

    println(A[1:4 , 1:4])
    println(tr_built_A[1:4 , 1:4])
end

main()