module CompressedMatrixModule

export CompressedMatrixNode, create_new_compressed_matrix_node, CompressedMatrix, is_compressed, break_up_compressed_cmn, change_addr_to_mock, fix_addrs

import Base: +, *, size, iterate

mutable struct CompressedMatrixNode
    rank::Union{Int64, Nothing}
    addr::Union{Tuple{Int64, Int64, Int64, Int64}, Nothing} 

    left_upper_child::Union{CompressedMatrixNode, Nothing}
    left_lower_child::Union{CompressedMatrixNode, Nothing}
    right_upper_child::Union{CompressedMatrixNode, Nothing}
    right_lower_child::Union{CompressedMatrixNode, Nothing}

    singular_values
    U_matrix::Union{Matrix{Float64}, Nothing}
    V_tr_matrix::Union{Matrix{Float64}, Nothing}

end


function size(cmn::CompressedMatrixNode)
    return (cmn.addr[2]-cmn.addr[1]+1, cmn.addr[4]-cmn.addr[3]+1)
end


function size(cmn::CompressedMatrixNode, dim::Int64)
    if dim == 1
        return cmn.addr[2]-cmn.addr[1]+1
    elseif dim == 2
        return cmn.addr[4]-cmn.addr[3]+1
    else 
        throw(ArgumentError("Compressed matrix only has two dimansions")) 
    end 
end

function is_compressed(cmn::CompressedMatrixNode)
    # if isnothing(cmn.left_upper_child) && isnothing(cmn.left_lower_child) && isnothing(cmn.right_upper_child) && isnothing(cmn.right_lower_child) && isnothing(cmn.U_matrix) && cmn.rank != 0
    #     println("!")
    # end
    return isnothing(cmn.left_upper_child) && isnothing(cmn.left_lower_child) && isnothing(cmn.right_upper_child) && isnothing(cmn.right_lower_child) && !isnothing(cmn.U_matrix)
end


function change_addr_to_mock(cmn::Union{CompressedMatrixNode, Nothing})
    if isnothing(cmn1) 
        return
    end

    new_r_max, new_c_max = size(cmn)
    cmn.addr = (1, new_r_max, 1, new_c_max)
end


function create_new_compressed_matrix_node()::CompressedMatrixNode
    return CompressedMatrixNode(
        nothing, nothing,
        nothing, nothing, nothing, nothing,
        nothing, nothing, nothing
    )
end


function break_up_compressed_cmn(cmn::CompressedMatrixNode)
    r_min, r_max, c_min, c_max = cmn.addr
    r_mid = div(r_min+r_max,2)
    c_mid = div(c_min+c_max,2)
    right_upper_child, left_lower_child, right_lower_child = nothing, nothing, nothing

    gamma = cmn.rank
    U_border = r_mid - r_min + 1
    V_border = c_mid - c_min + 1
    U_end = r_max - r_min + 1
    V_end = c_max - c_min + 1

    left_upper_child = create_new_compressed_matrix_node()
    left_upper_child.addr = (r_min, r_mid, c_min, c_mid)
    left_upper_child.rank = gamma
    left_upper_child.U_matrix = cmn.U_matrix[1:U_border , 1:end]
    if !isnothing(cmn.V_tr_matrix)
        if r_min == r_mid && c_min == c_mid 
            left_upper_child.U_matrix *= cmn.V_tr_matrix[1:end , 1:V_border]
        else
            left_upper_child.V_tr_matrix = cmn.V_tr_matrix[1:end , 1:V_border]
        end
    end

    if U_border < U_end 
        left_lower_child = create_new_compressed_matrix_node()
        left_lower_child.addr = (r_mid+1, r_max, c_min, c_mid)
        left_lower_child.rank = gamma 
        left_lower_child.U_matrix = cmn.U_matrix[U_border+1:end , 1:end]
        if !isnothing(cmn.V_tr_matrix)
            if r_mid+1 == r_max && c_min == c_max
                left_lower_child.U_matrix *= cmn.V_tr_matrix[1:end , 1:V_border]
            else
                left_lower_child.V_tr_matrix = cmn.V_tr_matrix[1:end , 1:V_border]
            end
        end

        if V_border < V_end
            right_lower_child = create_new_compressed_matrix_node()
            right_lower_child.addr = (r_mid+1, r_max, c_mid+1, c_max)
            right_lower_child.rank = gamma 
            right_lower_child.U_matrix = cmn.U_matrix[U_border+1:end , 1:end]
            if !isnothing(cmn.V_tr_matrix)
                if r_mid+1 == r_max && c_mid+1 == c_max
                    right_lower_child.U_matrix *= cmn.V_tr_matrix[1:end , V_border+1:end]
                else
                    right_lower_child.V_tr_matrix = cmn.V_tr_matrix[1:end , V_border+1:end]
                end
            end
        end
    end
    
    if V_border < V_end
        right_upper_child = create_new_compressed_matrix_node()
        right_upper_child.addr = (r_min, r_mid, c_mid+1, c_max)
        right_upper_child.rank = gamma 
        right_upper_child.U_matrix = cmn.U_matrix[1:U_border , 1:end]
        if !isnothing(cmn.V_tr_matrix)
            if r_min == r_mid && c_mid+1 == c_max
                right_upper_child.U_matrix *= cmn.V_tr_matrix[1:end , V_border+1:end]
            else
                right_upper_child.V_tr_matrix = cmn.V_tr_matrix[1:end , V_border+1:end]
            end
        end
    end

    return left_upper_child, left_lower_child, right_upper_child, right_lower_child
end


function fix_addrs(cmn::Union{CompressedMatrixNode, Nothing}, r_min::Int64, r_max::Int64, c_min::Int64, c_max::Int64)
    if isnothing(cmn)
        return 
    end

    cmn.addr = (r_min, r_max, c_min, c_max)

    r_mid = div(r_min+r_max, 2)
    c_mid = div(c_min+c_max, 2)
    fix_addrs(cmn.left_upper_child, r_min, r_mid, c_min, c_mid)
    fix_addrs(cmn.left_lower_child, r_mid+1, r_max, c_min, c_mid)
    fix_addrs(cmn.right_upper_child, r_min, r_mid, c_mid+1, c_max)
    fix_addrs(cmn.right_lower_child, r_mid+1, r_max, c_mid+1, c_max)

end


mutable struct CompressedMatrix
    head::CompressedMatrixNode
    size::Tuple{Int64, Int64}

end


function size(cmn::CompressedMatrix)
    return cmn.size
end


function size(cmn::CompressedMatrix, dim::Int64)
    return cmn.size[dim]
end


function fix_addrs(cm::CompressedMatrix)
    fix_addrs(cm.head, 1, size(cm, 1), 1, size(cm, 2))
end


end