module CompressedMatrixCreationModule

import Base: +, *, size, iterate

error_eps = 1e-4

using Pkg
include("compressed_matrix.jl")
using .CompressedMatrixModule
using LinearAlgebra

export perform_matrix_compression, build_matrix_from_compressed

function trunc_svd(A::Matrix{Float64}; gamma = min(size(A,1), size(A,2)))
    gamma = min(gamma, size(A,1), size(A,2))
    U, D, V = svd(A)
    D = diagm(D)
    U = U[: , 1:gamma]
    D = D[1:gamma , 1:gamma]
    V = V[: , 1:gamma]
    return U, D, V'
end


function compress_submatrix(A, r_min, r_max, c_min, c_max, U, D, V_tr, gamma)
    if all(abs.(A[r_min:r_max , c_min:c_max]) .< error_eps)
        v = create_new_compressed_matrix_node()
        v.rank = 0
        v.addr = (r_min, r_max, c_min, c_max)
        return v
    else
        sigmas = diag(D)
        v = create_new_compressed_matrix_node()
        v.rank = gamma
        v.addr = (r_min, r_max, c_min, c_max)
        v.singular_values = sigmas[1:gamma]
        v.U_matrix = U[: , 1:gamma]
        v.V_tr_matrix = D[1:gamma , 1:gamma] * V_tr[1:gamma , :]
        return v
    end
end


function compress_matrix(A, r_min=1, r_max=size(A,1), c_min=1, c_max=size(A,2); gamma=1, eps=1)
    if r_min == r_max && c_min == c_max
        v = create_new_compressed_matrix_node()
        v.rank = 1
        v.addr = (r_min, r_max, c_min, c_max)
        v.U_matrix = reshape([A[r_min, c_min]], 1, 1)
        return v
    end   
    
    U, D, V_tr = trunc_svd(A[r_min:r_max , c_min:c_max], gamma=gamma+1)
    if gamma+1 > size(D,1)
        gamma = size(D,1)
        if D[gamma, gamma] - eps < error_eps
            v = compress_submatrix(A, r_min, r_max, c_min, c_max, U, D, V_tr, gamma)
            return v 
        end
    elseif D[gamma+1, gamma+1] - eps < error_eps
        v = compress_submatrix(A, r_min, r_max, c_min, c_max, U, D, V_tr, gamma)
        return v
    end 

    r_mid = div(r_min+r_max,2)
    c_mid = div(c_min+c_max,2)
    right_upper_child, left_lower_child, right_lower_child = nothing, nothing, nothing
    left_upper_child = compress_matrix(A, r_min, r_mid, c_min, c_mid, gamma=gamma, eps=eps)
    if c_mid < c_max
        right_upper_child = compress_matrix(A, r_min, r_mid, c_mid+1, c_max, gamma=gamma, eps=eps)
    end
    if r_mid < r_max
        left_lower_child = compress_matrix(A, r_mid+1, r_max, c_min, c_mid, gamma=gamma, eps=eps)
        if c_mid < c_max
            right_lower_child = compress_matrix(A, r_mid+1, r_max, c_mid+1, c_max, gamma=gamma, eps=eps)
        end            
    end

    v = create_new_compressed_matrix_node()
    v.addr = (r_min, r_max, c_min, c_max)
    v.left_upper_child = left_upper_child
    v.right_upper_child = right_upper_child
    v.left_lower_child = left_lower_child
    v.right_lower_child = right_lower_child
    return v

end


function perform_matrix_compression(A, r_min=1, r_max=size(A,1), c_min=1, c_max=size(A,2); gamma=1, eps=1)
    return CompressedMatrix(compress_matrix(A, r_min, r_max, c_min, c_max; gamma=gamma, eps=eps), size(A))
end


function build_matrix_based_on_tree(target_matrix, node)
    r_min, r_max, c_min, c_max = node.addr

    if !isnothing(node.U_matrix)
        if !isnothing(node.V_tr_matrix)
            target_matrix[r_min:r_max , c_min:c_max] += node.U_matrix*node.V_tr_matrix
        else
            target_matrix[r_min:r_max , c_min:c_max] += node.U_matrix;
        end
    end

    if !isnothing(node.left_upper_child)
        build_matrix_based_on_tree(target_matrix, node.left_upper_child)
    end

    if !isnothing(node.right_upper_child)
        build_matrix_based_on_tree(target_matrix, node.right_upper_child)
    end
    
    if !isnothing(node.left_lower_child)
        build_matrix_based_on_tree(target_matrix, node.left_lower_child)
    end
    
    if !isnothing(node.right_lower_child)
        build_matrix_based_on_tree(target_matrix, node.right_lower_child)
    end
    
end


function build_matrix_from_compressed(matrix_size, compressed_matrix)
    built_matrix = zeros(matrix_size[1], matrix_size[2])
    build_matrix_based_on_tree(built_matrix, compressed_matrix.head)
    return built_matrix
end


function +(cmn1::Union{CompressedMatrixNode, Nothing}, cmn2::Union{CompressedMatrixNode, Nothing})::Union{CompressedMatrixNode, Nothing}
    if isnothing(cmn1) && isnothing(cmn2)
        return nothing   
    elseif isnothing(cmn1) 
        return deepcopy(cmn2)
    elseif isnothing(cmn2)
        return deepcopy(cmn1)          
    elseif !all(size(cmn1) .== size(cmn2))
        throw(ArgumentError("Cannot add matrix nodes of different sizes")) 
    elseif !all(cmn1.addr .== cmn2.addr)
        throw(ArgumentError("Incompatible matrix nodes in terms of addr"))
    end

    add_cmn::Union{CompressedMatrixNode, Nothing} = nothing
    character = "NA"

    if cmn1.addr[1] == cmn1.addr[2] && cmn1.addr[3] == cmn1.addr[4]
        character = "A"
        add_cmn = create_new_compressed_matrix_node()
        if !isnothing(cmn1.V_tr_matrix)
            if !isnothing(cmn2.V_tr_matrix)
                add_cmn.U_matrix = cmn1.U_matrix * cmn1.V_tr_matrix + cmn2.U_matrix * cmn2.V_tr_matrix
            else 
                add_cmn.U_matrix = cmn1.U_matrix * cmn1.V_tr_matrix + cmn2.U_matrix
            end
        elseif !isnothing(cmn2.V_tr_matrix)
            add_cmn.U_matrix = cmn1.U_matrix + cmn2.U_matrix * cmn2.V_tr_matrix
        else
            add_cmn.U_matrix = cmn1.U_matrix + cmn2.U_matrix
        end
        # if !isnothing(cmn1.V_tr_matrix) || !isnothing(cmn2.V_tr_matrix)
        #     println(cmn1.U_matrix, " ", cmn1.V_tr_matrix)
        #     println(cmn2.U_matrix, " ", cmn2.V_tr_matrix)
        #     println("!")
        # end
        add_cmn.rank = cmn1.rank
        add_cmn.addr = cmn1.addr
    
    elseif cmn1.rank == 0 && cmn2.rank == 0
        character = "B"
        add_cmn = create_new_compressed_matrix_node()
        add_cmn.rank = 0
        add_cmn.addr = cmn1.addr

    elseif cmn1.rank == 0
        character = "C"
        add_cmn = deepcopy(cmn2)
       
    elseif cmn2.rank == 0
        character = "D"
        add_cmn = deepcopy(cmn1)    
    
    elseif is_compressed(cmn1) && is_compressed(cmn2) 
        character = "E"
        u_wave = [cmn1.U_matrix cmn2.U_matrix]
        v_wave = [cmn1.V_tr_matrix ; cmn2.V_tr_matrix]
        gamma = max(cmn1.rank, cmn2.rank)
        u_dash, d_dash, v_tr_dash = trunc_svd(u_wave*v_wave, gamma=gamma)
        sigmas = diag(d_dash)
        add_cmn = create_new_compressed_matrix_node()
        add_cmn.rank = gamma
        add_cmn.addr = cmn1.addr
        add_cmn.singular_values = sigmas[1:gamma]
        add_cmn.U_matrix = u_dash
        add_cmn.V_tr_matrix = d_dash*v_tr_dash
    
    elseif !is_compressed(cmn1) && !is_compressed(cmn2)
        character = "F"
        add_cmn = create_new_compressed_matrix_node()
        add_cmn.rank = cmn1.rank
        add_cmn.addr = cmn1.addr
        add_cmn.left_upper_child = cmn1.left_upper_child + cmn2.left_upper_child
        add_cmn.left_lower_child = cmn1.left_lower_child + cmn2.left_lower_child
        add_cmn.right_upper_child = cmn1.right_upper_child + cmn2.right_upper_child
        add_cmn.right_lower_child = cmn1.right_lower_child + cmn2.right_lower_child

    elseif is_compressed(cmn1) && !is_compressed(cmn2)
        character = "G"
        cmn1_left_upper, cmn1_left_lower, cmn1_right_upper, cmn1_right_lower = break_up_compressed_cmn(cmn1)
        add_cmn = create_new_compressed_matrix_node()
        add_cmn.rank = cmn1.rank 
        add_cmn.addr = cmn1.addr 
        add_cmn.left_upper_child = cmn1_left_upper + cmn2.left_upper_child
        add_cmn.left_lower_child = cmn1_left_lower + cmn2.left_lower_child
        add_cmn.right_upper_child = cmn1_right_upper + cmn2.right_upper_child
        add_cmn.right_lower_child = cmn1_right_lower + cmn2.right_lower_child

    elseif !is_compressed(cmn1) && is_compressed(cmn2)
        character = "H"
        cmn2_left_upper, cmn2_left_lower, cmn2_right_upper, cmn2_right_lower = break_up_compressed_cmn(cmn2)
        add_cmn = create_new_compressed_matrix_node()
        add_cmn.rank = cmn2.rank 
        add_cmn.addr = cmn2.addr 
        add_cmn.left_upper_child = cmn1.left_upper_child + cmn2_left_upper
        add_cmn.left_lower_child = cmn1.left_lower_child + cmn2_left_lower
        add_cmn.right_upper_child = cmn1.right_upper_child + cmn2_right_upper
        add_cmn.right_lower_child = cmn1.right_lower_child + cmn2_right_lower
    end

    if all(size(add_cmn) .== (1,1)) && !isnothing(add_cmn.V_tr_matrix)
        println(character)
    end
    return add_cmn

end


function remove_unneccessary_children(cmn::Union{CompressedMatrixNode, Nothing})::Union{CompressedMatrixNode, Nothing}
    if isnothing(cmn)
        return nothing
    end

    new_cmn = cmn 
    while all(size(new_cmn) .== (1,1)) && isnothing(new_cmn.U_matrix) && !isnothing(new_cmn.left_upper_child)
        new_cmn = cmn.left_upper_child
    end

    return new_cmn
end


function *(cmn1::Union{CompressedMatrixNode, Nothing}, cmn2::Union{CompressedMatrixNode, Nothing})::Union{CompressedMatrixNode, Nothing}
    if isnothing(cmn1) || isnothing(cmn2)
        return nothing
    elseif size(cmn1, 2) != size(cmn2, 1)
        throw(ArgumentError("Cannot multiply compressed matrix nodes of incompatible shapes"))
    end

    mult_cmn::Union{CompressedMatrixNode, Nothing} = nothing
    
    if all(size(cmn1) .== (1,1)) && all(size(cmn2) .== (1,1))
        mult_cmn = create_new_compressed_matrix_node()
        mult_cmn.addr = (1, 1, 1, 1)
        mult_cmn.rank = 1
        mult_cmn.U_matrix = cmn1.U_matrix * cmn2.U_matrix

    elseif cmn1.rank == 0 || cmn2.rank == 0
        mult_cmn = create_new_compressed_matrix_node()
        mult_cmn.rank = 0
        mult_cmn.addr = (1, size(cmn1, 1), 1, size(cmn2, 2))

    elseif is_compressed(cmn1) && is_compressed(cmn2)
        mult_cmn = create_new_compressed_matrix_node()
        V1_tr_matrix = cmn1.V_tr_matrix
        U_dash = nothing
        if !isnothing(V1_tr_matrix)
            U_dash = cmn1.U_matrix * V1_tr_matrix * cmn2.U_matrix 
        else
            U_dash = cmn1.U_matrix * cmn2.U_matrix
        end
        mult_cmn.addr = (1, size(cmn1, 1), 1, size(cmn2, 2))
        mult_cmn.U_matrix = U_dash
        mult_cmn.V_tr_matrix = cmn2.V_tr_matrix   
        mult_cmn.rank = size(mult_cmn.U_matrix, 2)

    elseif !is_compressed(cmn1) && !is_compressed(cmn2)
        mult_cmn = create_new_compressed_matrix_node()
        mult_cmn.left_upper_child = cmn1.left_upper_child * cmn2.left_upper_child + cmn1.right_upper_child * cmn2.left_lower_child
        mult_cmn.right_upper_child = cmn1.left_upper_child * cmn2.right_upper_child + cmn1.right_upper_child * cmn2.right_lower_child
        mult_cmn.left_lower_child = cmn1.left_lower_child * cmn2.left_upper_child + cmn1.right_lower_child * cmn2.left_lower_child
        mult_cmn.right_lower_child = cmn1.left_lower_child * cmn2.right_upper_child + cmn1.right_lower_child * cmn2.right_lower_child
        mult_cmn.addr = (1, size(cmn1, 1), 1, size(cmn2, 2))
        
    elseif is_compressed(cmn1) && !is_compressed(cmn2)
        mult_cmn = create_new_compressed_matrix_node()
        cmn1_left_upper, cmn1_left_lower, cmn1_right_upper, cmn1_right_lower = break_up_compressed_cmn(cmn1)
        mult_cmn.left_upper_child = cmn1_left_upper * cmn2.left_upper_child + cmn1_right_upper * cmn2.left_lower_child
        mult_cmn.right_upper_child = cmn1_left_upper * cmn2.right_upper_child + cmn1_right_upper * cmn2.right_lower_child
        mult_cmn.left_lower_child = cmn1_left_lower * cmn2.left_upper_child + cmn1_right_lower * cmn2.left_lower_child
        mult_cmn.right_lower_child = cmn1_left_lower * cmn2.right_upper_child + cmn1_right_lower * cmn2.right_lower_child
        mult_cmn.addr = (1, size(cmn1, 1), 1, size(cmn2, 2))

    elseif !is_compressed(cmn1) && is_compressed(cmn2)
        mult_cmn = create_new_compressed_matrix_node()
        cmn2_left_upper, cmn2_left_lower, cmn2_right_upper, cmn2_right_lower = break_up_compressed_cmn(cmn2)
        mult_cmn.left_upper_child = cmn1.left_upper_child * cmn2_left_upper + cmn1.right_upper_child * cmn2_left_lower
        mult_cmn.right_upper_child = cmn1.left_upper_child * cmn2_right_upper + cmn1.right_upper_child * cmn2_right_lower
        mult_cmn.left_lower_child = cmn1.left_lower_child * cmn2_left_upper + cmn1.right_lower_child * cmn2_left_lower
        mult_cmn.right_lower_child = cmn1.left_lower_child * cmn2_right_upper + cmn1.right_lower_child * cmn2_right_lower
        mult_cmn.addr = (1, size(cmn1, 1), 1, size(cmn2, 2))

    end

    mult_cmn = remove_unneccessary_children(mult_cmn)

    return mult_cmn

end


function +(cm1::CompressedMatrix, cm2::CompressedMatrix)
    if !all(cm1.size .== cm2.size) 
        throw(ArgumentError("Cannot add matrixes of different sizes"))
    end
    return CompressedMatrix(cm1.head + cm2.head, cm1.size)
end


function *(cm1::CompressedMatrix, cm2::CompressedMatrix)
    if cm1.size[2] != cm2.size[1]
        throw(ArgumentError("Cannot multiply matrixes of incompatible shapes"))
    end
    mult_cm = CompressedMatrix(cm1.head * cm2.head,  (cm1.size[1], cm2.size[2]))
    fix_addrs(mult_cm)
    return mult_cm
end


end