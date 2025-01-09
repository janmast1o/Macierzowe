module CompressedMatrixModule

export CompressedMatrixNode, create_new_compressed_matrix_node, CompressedMatrix, is_compressed, break_up_compressed_cmn

import Base: +, size, iterate

mutable struct CompressedMatrixNode
    rank
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


function is_compressed(cmn::CompressedMatrixNode)
    return isnothing(cmn.left_upper_child) && isnothing(cmn.left_lower_child) && isnothing(cmn.right_upper_child) && isnothing(cmn.right_lower_child)
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
    left_upper_child.V_tr_matrix = cmn.V_tr_matrix[1:end , 1:V_border]

    if U_border+1 < U_end 
        left_lower_child = create_new_compressed_matrix_node()
        left_lower_child.addr = (r_mid+1, r_max, c_min, c_mid)
        left_lower_child.rank = gamma 
        left_lower_child.U_matrix = cmn.U_matrix[U_border+1:end , 1:end]
        left_lower_child.V_matrix = cmn.V_tr_matrix[1:end , 1:V_border]

        if V_border+1 < V_end
            right_lower_child = create_new_compressed_matrix_node()
            right_lower_child.addr = (r_mid+1, r_max, c_mid+1, c_max)
            right_lower_child.rank = gamma 
            right_lower_child.U_matrix = cmn.U_matrix[U_border+1:end , 1:end]
            right_lower_child.V_tr_matrix = cmn.V_tr_matrix[1:end , V_border+1:end]
        end
    end
    
    if V_border+1 < V_end
        right_upper_child = create_new_compressed_matrix_node()
        right_upper_child.addr = (r_min, r_mid, c_mid+1, c_max)
        right_upper_child.rank = gamma 
        right_upper_child.U_matrix = cmn.U_matrix[1:U_border , 1:end]
        right_upper_child.V_tr_matrix = cmn.V_tr_matrix[1:end , V_border+1:end]
    end

    return left_upper_child, left_lower_child, right_upper_child, right_lower_child
end


mutable struct CompressedMatrix
    head::CompressedMatrixNode
    size::Tuple{Int64, Int64}

end

function size(cmn::CompressedMatrix)
    return cmn.size
end

end