include("create_compressed_matrix.jl")
using .CompressedMatrixCreationModule


error_eps = 1e-2


function generate_random_matrix(m, n, a, b)
    A = zeros(m, n)
    for i in 1:1:m 
        for j in 1:1:n 
            A[i,j] = a + (b-a)*rand()
        end
    end
    return A
end


function are_equal(A, B)
    if !all(size(A) .== size(B))
        return false
    end

    return all(abs.(A-B) .< error_eps)

end


function test_matrix_compression(low_m, high_m, low_n, high_n, a, b, tests, gamma, eps)
    correct_outputs = 0
    incorrect_ouputs = 0
    for _ in 1:1:tests 
        A = generate_random_matrix(rand(low_m:1:high_m), rand(low_n:1:high_n), a, b)
        cmn = perform_matrix_compression(A, gamma=gamma, eps=eps)
        built_A = build_matrix_from_compressed(size(A), cmn)
        if are_equal(A, built_A)
            correct_outputs += 1
        else 
            # for i in 1:1:size(A,1)
            #     for j in 1:1:size(A,2)
            #         if abs(A[i,j] - built_A[i,j]) > error_eps
            #             println(A[i,j], " ", built_A[i,j], "\n")
            #         end
            #     end
            # end
            # println("\n\n\n")
            incorrect_ouputs += 1    
        end
    end

    return correct_outputs, incorrect_ouputs
end


function run_compression_tests()
    low_m_for_test, high_m_for_test, low_n_for_test, high_n_for_test = 15, 50, 15, 50
    # low_m_for_test, high_m_for_test, low_n_for_test, high_n_for_test = 1, 10, 1, 10
    a_for_test, b_for_test = 0, 50
    correct, incorrect = test_matrix_compression(low_m_for_test, high_m_for_test, low_n_for_test, high_n_for_test, a_for_test, b_for_test, 15, 1, 0.1)
    println(correct, " ", incorrect)
end    


function test_compressed_matrix_addition(low_m, high_m, low_n, high_n, a, b, tests, gamma, eps)
    correct_outputs = 0
    incorrect_outputs = 0
    for _ in 1:1:tests 
        m, n = rand(low_m:1:high_m), rand(low_n:1:high_n)
        A = generate_random_matrix(m, n, a, b)
        B = generate_random_matrix(m, n, a, b)
        cm_A = perform_matrix_compression(A, gamma=gamma, eps=eps)
        cm_B = perform_matrix_compression(B, gamma=gamma, eps=eps)
        cm_added = cm_A + cm_B
        AB = A+B
        built_addded = build_matrix_from_compressed(size(cm_added), cm_added)
        if are_equal(A+B, built_addded)
            correct_outputs += 1
        else 
            for i in 1:1:size(A,1)
                for j in 1:1:size(A,2)
                    if abs(AB[i,j] - built_addded[i,j]) > error_eps
                        println(AB[i,j], " ", built_addded[i,j], " ", A[i,j], " ", B[i,j], " ", i, " ", j, "\n")
                    end
                end
            end
            println("\n\n\n")
            incorrect_outputs += 1    
        end
    end

    return correct_outputs, incorrect_outputs
end


function run_compressed_addition_tests()
    low_m_for_test, high_m_for_test, low_n_for_test, high_n_for_test = 15, 50, 15, 50
    # low_m_for_test, high_m_for_test, low_n_for_test, high_n_for_test = 1, 10, 1, 10
    a_for_test, b_for_test = 0, 50
    correct, incorrect = test_compressed_matrix_addition(low_m_for_test, high_m_for_test, low_n_for_test, high_n_for_test, a_for_test, b_for_test, 15, 1, 0.1)
    println(correct, " ", incorrect)
end


function test_compressed_matrix_multiplication(low_m, high_m, low_n, high_n, a, b, tests, gamma, eps)
    correct_outputs = 0
    incorrect_outputs = 0
    for _ in 1:1:tests 
        m, n, p = rand(low_m:1:high_m), rand(low_n:1:high_n), rand(low_m:1:high_m)
        A = generate_random_matrix(m, n, a, b)
        B = generate_random_matrix(n, p, a, b)
        cm_A = perform_matrix_compression(A, gamma=gamma, eps=eps)
        cm_B = perform_matrix_compression(B, gamma=gamma, eps=eps)
        cm_mult = cm_A * cm_B
        AB = A*B
        built_mult = build_matrix_from_compressed(size(cm_mult), cm_mult)
        if are_equal(A*B, built_mult)
            correct_outputs += 1
        else 
            for i in 1:1:size(AB,1)
                for j in 1:1:size(AB,2)
                    if abs(AB[i,j] - built_mult[i,j]) > error_eps
                        println(AB[i,j], " ", built_mult[i,j], " ", i, " ", j, "\n")
                    end
                end
            end
            println("\n\n\n")
            incorrect_outputs += 1    
        end
    end

    return correct_outputs, incorrect_outputs
end


function run_compressed_multiplication_tests()
    # low_m_for_test, high_m_for_test, low_n_for_test, high_n_for_test = 15, 50, 15, 50
    low_m_for_test, high_m_for_test, low_n_for_test, high_n_for_test = 1, 10, 1, 10
    a_for_test, b_for_test = 0, 50
    correct, incorrect = test_compressed_matrix_multiplication(low_m_for_test, high_m_for_test, low_n_for_test, high_n_for_test, a_for_test, b_for_test, 15, 1, 0.1)
    println(correct, " ", incorrect)
end


run_compression_tests()
run_compressed_addition_tests()
# run_compressed_multiplication_tests()