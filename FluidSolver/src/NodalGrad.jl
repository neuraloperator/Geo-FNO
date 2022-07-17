export NodalGrad, Compute_Weights_LeastSquares!, Compute_Nodal_Gradients!

mutable struct NodalGrad
    node_eto_nodes
    dxdxᵀ⁻¹
end

function NodalGrad(X::Array{Float64,2}, edges)

    
        nnodes = size(X, 2)
        nedges = size(edges, 2)
        node_eto_nodes_set = [Set(Int64[]) for i =1:nnodes]
        node_eto_nodes = [Int64[] for i =1:nnodes]
    
        for e = 1:nedges
            i, j = edges[:, e]   
                 
            push!(node_eto_nodes_set[i], j)
            push!(node_eto_nodes_set[j], i)
        end

        for n = 1:nnodes
            node_eto_nodes[n] = collect(node_eto_nodes_set[n])
        end

    
        dxdxᵀ⁻¹ = zeros(Float64, 2, 2, nnodes)
    
        Compute_Weights_LeastSquares!(X, node_eto_nodes, dxdxᵀ⁻¹)
    
        return NodalGrad(node_eto_nodes, dxdxᵀ⁻¹)
    end

function Compute_Weights_LeastSquares!(X::Array{Float64,2}, node_eto_nodes, dxdxᵀ⁻¹)

    dxdxᵀ⁻¹[:,:,:] .= 0.0

    nnodes = length(node_eto_nodes)

    #TODO parallel
    for i = 1:nnodes
        for j in node_eto_nodes[i]
            
            dxdxᵀ⁻¹[:, :, i] .+= (X[:, i] - X[:, j])*(X[:, i] - X[:, j])'
        end
        dxdxᵀ⁻¹[:, :, i] .= inv(dxdxᵀ⁻¹[:, :, i])
    end
end

"""
Compute nodal gradient at node j with least square
(V_jk - V_j)ᵀ = (x_jk - x_j)ᵀ ⋅ dV_j, jk = 1,2,...,N_j are neigboring points, dV_j is a (dim, nV) array
∑_k^N_j  (x_jk - x_j)(V_jk - V_j)ᵀ = ∑_k^N_j(x_jk - x_j)(x_jk - x_j)ᵀ ⋅ dV_j 
dV_j = dxdxᵀ⁻¹ ∑_k^N_j  (x_jk - x_j)(V_jk - V_j)ᵀ
"""
function Compute_Nodal_Gradients!(X::Array{Float64,2}, V::Array{Float64,2},  ngrad::NodalGrad,  dV::Array{Float64,3})
    
    node_eto_nodes = ngrad.node_eto_nodes
    dxdxᵀ⁻¹ = ngrad.dxdxᵀ⁻¹

    nnodes = length(node_eto_nodes)

    dV  .= 0

    #TODO parallel
    for i = 1:nnodes
        for j in node_eto_nodes[i]
            
            dV[:,:,i] .+= (V[:, j] - V[:, i]) * ( (X[:, j] - X[:, i])' * dxdxᵀ⁻¹[:,:,i] )
        end
    end
end



# function Compute_Weighted_Nodal_Gradients!(mesh::QuadMesh, V::Array{Float64,2}, dV::Array{Float64,3})
#     nnodes, nV = size(V)
#     node_eto_nodes = mesh.node_eto_nodes
#     wdxdxᵀ = mesh.wdxdxᵀ 
#     X = mesh.X
#     #
#     dVdxᵀ = zeros(Float64, 2)
#     for j = 1:nV
#         wdxdxᵀ .= 0.0
#         for i = 1:nnodes
#             dVdxᵀ .= 0.0
#             for ni in node_eto_nodes[i]
#                 w = 1.0/(1.0 + ((V[ni, j] - V[i, j])
#                 /sqrt((X[ni, 1] - X[i, 1])^2 + (X[ni, 2] - X[i, 2])^2))^2)
                
#                 dVdxᵀ .+= w*(V[ni, j] - V[i, j])*(X[ni, :] - X[i, :])
#                 wdxdxᵀ[i, :, :] .+= w*(X[ni,:] - X[i,:])*(X[ni,:] - X[i,:])'
#             end

#             dV[i, j, :] .= wdxdxᵀ[i,:,:]\dVdxᵀ
#         end
#     end
# end

