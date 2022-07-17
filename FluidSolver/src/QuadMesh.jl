export QuadMesh, Compute_Nodal_Gradients!, Visual_Tripcolor

function _pair_sort(a, b)
    return (a > b ? (b, a) : (a, b))
end

"""
Geometric States
Julia is column major, to improve cache hit ratio
"""
mutable struct QuadMesh
    
    # Mesh coordiniates 
    nnodes::Int64
    X::Array{Float64,2}        # (2, nnodes) array, locations of all grid nodes at n step 
    X_nm1::Array{Float64,2}    # (2, nnodes) array, locations of all grid nodes at n-1 step
    X_nm2::Array{Float64,2}    # (2, nnodes) array, locations of all grid nodes at n-2 step
    

    # contral volume areas
    ctrlVol::Array{Float64,1}        # (2, nnodes) array, dual cell areas for all grid nodes at n step 
    ctrlVol_nm1::Array{Float64,1}    # (2, nnodes) array, dual cell areas for all grid nodes at n-1 step 
    ctrlVol_nm2::Array{Float64,1}    # (2, nnodes) array, dual cell areas for all grid nodes at n-2 step 

    # quad elements
    nelems::Int64
    elems::Array{Int64,2}       # (4, nelems) array, the element list
    tri_elems::Array{Int64,2}   # (3, nelems) array, the element list of triangular elements, for visulization purpose

    # edges for the flux
    nedges::Int64
    edges::Array{Int64,2}                 # (2, nnodes) array, edge list (e1, e2) with e1 < e2
    edgeNorm::Array{Float64,2}            # (4, nnodes) array, area weighted nomrmal ν_ij associated with each edge (i, j) at n step
    edgeNormVel::Array{Float64,1}         # (nnodes) array, normal direction velocity of the mesh ẋ⋅ν_ij/|ν_ij| ,associated with each edge (i, j)
    edgeNorm_nm1::Array{Float64,2}        # (4, nnodes) array, area weighted nomrmal ν_ij associated with each edge (i, j) at n - 1 step
    edgeNormVel_nm1::Array{Float64,1}     # (nnodes) array, normal direction velocity of the mesh ẋ⋅ν_ij/|ν_ij| ,associated with each edge (i, j) at n - 1 step
    edgeNorm_nm2::Array{Float64,2}        # (4, nnodes) array, area weighted nomrmal ν_ij associated with each edge (i, j) at n - 2 step
    edgeNormVel_nm2::Array{Float64,1}     # (nnodes) array, normal direction velocity of the mesh ẋ⋅ν_ij/|ν_ij| ,associated with each edge (i, j) at n - 2 step

    minEdgeLens::Array{Float64,1}     # (nnodes) array, minimum edge length at each node
    # nodal gradient structure
    ngrad::NodalGrad

    
    bcType::Array{Int64,2}           # (4, nbcEdges) array, i, j, bcType, bcData_Id (i.e. different farfield conditions), i,j is counterclockwise
    bcData::Array{Array{Float64,2}}  # bcData[bcType][:,bcData_Id] is the data
    bcNorm::Array{Float64,2}         # (2, nbcEdges) array, area weighted nomrmal ν_ij/2 associated with the bc edge at n step
    bcNormVel::Array{Float64,1}      # (nbcEdges) array, normal direction velocity associated with the bc edge  at n step
    
    bcNorm_nm1::Array{Float64,2}         # (2, nbcEdges) array, area weighted nomrmal ν_ij/2 associated with the bc edge at n - 1 step
    bcNormVel_nm1::Array{Float64,1}      # (nbcEdges) array, normal direction velocity associated with the bc edge  at n - 1 step
    bcNorm_nm2::Array{Float64,2}         # (2, nbcEdges) array, area weighted nomrmal ν_ij/2 associated with the bc edge at n - 1 step
    bcNormVel_nm2::Array{Float64,1}      # (nbcEdges) array, normal direction velocity associated with the bc edge  at n - 1 step
    
end


@doc """
xy : Float64[nnodes, 2], coordinate
elem: Int64[nelems, 4],  element node ids in counterclockwise or clockwise
bcMap: Int64[nbc_edge, 4],  boundary edge i, j, is boundary type k, id l  [i,j,k,l] 
bc_data: Array of Array,  boundary type k, data id l 
"""
function QuadMesh(X::Array{Float64,2}, elems::Array{Int64,2}, bcMap::Array{Int64,2}, bcData::Array{Array{Float64,2},1})

    nnodes = size(X, 2)
    nelems = size(elems, 2)
    
    tri_elems = zeros(Int64, 3, 2nelems)
    for i = 1:nelems
        tri_elems[:, 2(i-1)+1] .= elems[1, i], elems[2, i], elems[3, i]
        tri_elems[:, 2(i-1)+2] .= elems[3, i], elems[4, i], elems[1, i]
    end

    ctrlVol, edges, edgeNorm,  minEdgeLens, bcType, bcNorm = Init_QuadMesh(X, elems, bcMap)
    
    ngrad = NodalGrad(X, edges)

    nedges = size(edges, 2)

    # mesh history
    X_nm1 = copy(X)
    X_nm2 = copy(X)

    # control volume history
    ctrlVol_nm1 = copy(ctrlVol)
    ctrlVol_nm2 = copy(ctrlVol)

    # edge history
    edgeNorm_nm1 = copy(edgeNorm)
    edgeNorm_nm2 = copy(edgeNorm)

    edgeNormVel = zeros(Float64, nedges)
    edgeNormVel_nm1 = copy(edgeNormVel)
    edgeNormVel_nm2 = copy(edgeNormVel)


    nbcEdges = size(bcMap,1)
    bcNorm_nm1 = copy(bcNorm)
    bcNorm_nm2 = copy(bcNorm)

    bcNormVel = zeros(Float64, nbcEdges)
    bcNormVel_nm1 = copy(bcNormVel)
    bcNormVel_nm2 = copy(bcNormVel)


    QuadMesh(nnodes, X, X_nm1, X_nm2, 
             ctrlVol, ctrlVol_nm1, ctrlVol_nm2, 
             nelems, elems, tri_elems, 
             nedges, edges,  edgeNorm, edgeNormVel, edgeNorm_nm1, edgeNormVel_nm1, edgeNorm_nm2, edgeNormVel_nm2,
             minEdgeLens,
             ngrad,
             bcType, bcData, bcNorm, bcNormVel,
             bcNorm_nm1, bcNormVel_nm1, bcNorm_nm2, bcNormVel_nm2)
end




function Init_QuadMesh(X::Array{Float64,2}, elems::Array{Int64,2}, bcMap::Array{Int64,2})
    nelems = size(elems,2)
        
    nnodes = size(X, 2)

    ctrlVol = zeros(Float64, nnodes)
    
    edge_to_elems = Dict{Tuple{Int64,Int64},Array{Int64,1}}()

    elem_centroids = zeros(Float64, 2, nelems)

    for e = 1:nelems

        elem = elems[:, e]
        
        elem_centroids[:, e] = mean(X[:, elem], dims=2)

        edge_pairs = [_pair_sort(elem[1], elem[2]), _pair_sort(elem[2], elem[3]), _pair_sort(elem[3], elem[4]), _pair_sort(elem[4], elem[1])]

        # update area
        for edge_pair in edge_pairs
            n1, n2 = edge_pair

            x1, y1 = X[:, n1]
            x2, y2 = X[:, n2]
            x3, y3 = elem_centroids[:, e]

            area = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)) / 2.0
            ctrlVol[n1] +=  area / 2.0
            ctrlVol[n2] +=  area / 2.0
        end

        # update edge_to_elems map
        
        for edge_pair in edge_pairs
                # n is the third node number
            if haskey(edge_to_elems, edge_pair)
                edge_to_elems[edge_pair][2]  = e
            else
                edge_to_elems[edge_pair] = [e, -1]
            end
        end
    end

        # build edge related quantities
    nedges = length(edge_to_elems)
    edges = zeros(Int64, 2, nedges)
    edgeNorm = zeros(Float64, 2, nedges)

    i = 1
    for (edge, elems) in edge_to_elems

        edges[:, i] .= edge
        n1, n2 = edge
        x1, y1 = X[:, n1]
        x2, y2 = X[:, n2]

        xm, ym = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        e1, e2 = elems

        if (e1 != -1)
            xc, yc = elem_centroids[:, e1]
                # xc-xm, yc-ym  ->  ym-yc, xc-xm
            direction = sign((x2 - x1) * (ym - yc) + (y2 - y1) * (xc - xm))
            edgeNorm[:, i] .+= direction * (ym - yc), direction * (xc - xm)
        end
        if (e2 != -1)
            xc, yc = elem_centroids[:, e2]
                # xc-xm, yc-ym  ->  ym-yc, xc-xm
            direction = sign((x2 - x1) * (ym - yc) + (y2 - y1) * (xc - xm))
            edgeNorm[:, i] .+= direction * (ym - yc), direction * (xc - xm)
        end

        i += 1

    end


    minEdgeLens = zeros(Float64, nnodes) .+ Inf
    for e = 1:nedges
        i, j = edges[:, e]
        x1, y1 = X[:, i]
        x2, y2 = X[:, j]
        len = sqrt((x2 - x1)^2 + (y2 - y1)^2)
            
        if len < minEdgeLens[i]
            minEdgeLens[i] = len
        end
        if len < minEdgeLens[j]
            minEdgeLens[j] = len
        end
    end


    # boundary conditions
    nbcEdge = size(bcMap, 2)

    bcType = zeros(Int64, 4, nbcEdge)        # n1, n2, bc_type, bc_data_type (i.e. different farfield conditions)
    bcNorm = zeros(Float64, 2, nbcEdge) 

    for i = 1:nbcEdge

        n1, n2, bc_type, bc_data_type = bcMap[:, i]
    
        x1, y1 = X[:, n1]
    
        x2, y2 = X[:, n2]

        xm, ym = (x1 + x2) / 2.0, (y1 + y2) / 2.0


        e1, e2 = edge_to_elems[_pair_sort(n1, n2)] 

        @assert(e2 == -1)

        # determine any other node in the same element
        n3 = elems[1, e1]
        for n in elems[:, e1]
            if n != n1 && n != n2
                n3 = n
                break
            end
        end
        
        x3, y3 = X[:, n3]

        norm = [-(y2 - y1), x2 - x1]

        direction = sign(norm[1] * (xm - x3) + norm[2] * (ym - y3))
            
        norm = direction * norm
            
            
        bcType[:, i] .= n1, n2,  bc_type, bc_data_type
        bcNorm[:, i] .= norm

    end

    return ctrlVol, edges, edgeNorm,  minEdgeLens, bcType, bcNorm
end




# Visualize the nodal quantity on the mesh

function Visual_Tripcolor(mesh::QuadMesh, V::Array{Float64,1}, filename::String="None"; equal_axis = true)
    figure()
    tripcolor(mesh.X[1, :], mesh.X[2, :], mesh.tri_elems' .- 1, V, cmap = "viridis")
    colorbar()

    if equal_axis
        axis("equal")
    end
    if filename != "None"
        savefig(filename)
        close("all")
    end

    
    
end