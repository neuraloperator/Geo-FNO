export NSSolver, Solve!, Input


function MeshGrid(i, j)

    return  [4i-3, 4i-2, 4i-1, 4i,   4i-3, 4i-2, 4i-1, 4i,  4i-3, 4i-2, 4i-1, 4i,   4i-3, 4i-2, 4i-1, 4i],
            [4j-3, 4j-3, 4j-3, 4j-3, 4j-2, 4j-2, 4j-2, 4j-2,  4j-1, 4j-1, 4j-1, 4j-1,   4j, 4j, 4j, 4j]
end

mutable struct Input
    # Problem 
    problem_type::String

    # Time discretization
    time_integrator::String
    time_step_computation::String
    cfl_init_number::Float64
    cfl_max_number::Float64
    ## Steady simulation
    max_iteration::Int64
    converge_eps::Float64
    ## Unsteady simulation
    t_end::Float64
    dt::Float64
    # Spatial discretization
    limiter_type::String
    limiter_beta::Float64
    limiter_eps::Float64

    # Physical parameters
    gamma::Float64
end

function Input(; problem_type::String, time_integrator::String, time_step_computation::String = "Global_CFL", 
    cfl_init_number::Float64 = 0.8, cfl_max_number::Float64 = 0.8, max_iteration::Int64 = 1000, converge_eps::Float64=1e-3, 
    t_end::Float64=0.0, dt::Float64 = -1.0,limiter_type::String = "Minmod", limiter_beta::Float64 = 1.0/3.0, limiter_eps::Float64 = 1.0e-15, gamma=1.4)
    
    return Input(problem_type, 
        time_integrator, time_step_computation, cfl_init_number, cfl_max_number, max_iteration, converge_eps, t_end, dt,
        limiter_type, limiter_beta, limiter_eps, 
        gamma)
end


mutable struct NSSolver
    mesh::QuadMesh            # QuadMesh
    W::Array{Float64,2}   # conservative variables
    V::Array{Float64,2}   # primitive variables
    dV::Array{Float64,3}  # primitive variables

    limiter::Limiter

    W0::Array{Float64,2}  # conservative variables container
    V0::Array{Float64,2}  # primitive variables container

    k1::Array{Float64,2}  # residual  container
    k2::Array{Float64,2}  # residual  container
    k3::Array{Float64,2}  # residual  container
    R::Array{Float64,2}   # residual 

    # Physics parameters
    gamma::Float64
    # mu::Float64
    # Pr::Float64

end


# W0, QuadMesh structure 
# params: dictionary for different parameters
function NSSolver(mesh::QuadMesh, input::Input)
        
    nnodes = mesh.nnodes
    W = zeros(Float64, 4, nnodes) #conservative state variables at the present step
    V = zeros(Float64, 4, nnodes) #primitive state variables at the present step
    dV = zeros(Float64, 4, 2, nnodes) #derivatives of primitive state variables at the present step


    limiter = Limiter(input.limiter_type, input.limiter_beta, input.limiter_eps)

    W0 = zeros(Float64, 4, nnodes)  #container
    V0 = zeros(Float64, 4, nnodes)  #container

    k1 = zeros(Float64, 4, nnodes)  #container
    k2 = zeros(Float64, 4, nnodes)  #container
    k3 = zeros(Float64, 4, nnodes)  #container
    R  = zeros(Float64, 4, nnodes)   #container

    gamma = input.gamma
    NSSolver(mesh, W, V, dV, limiter, W0, V0, k1, k2, k3, R, gamma)
end


# W0, initial condition (conservative state variables)
# methods: dictionary for method parameters
# T: end time
function Solve!(self::NSSolver, W0::Array{Float64,2}, input::Input)
    
    W, V = self.W, self.V
    W[:,:] .= W0
    gamma = self.gamma

    if input.problem_type == "Unsteady"
        SAVE_PRIM = false
        t, dt, istep = 0.0, 0.0, 0
        t_end = input.t_end
        while t < t_end

            # compute time step size
            dt = ComputeTimeStep(self, W, input)
        
            if dt + t > t_end
                dt = T - t_end
            end

            # update solution in W for the next time step 
            Time_Advance!(self, W, dt, input)

            t += dt
            # save primitive variables
            istep += 1
            if SAVE_PRIM
                Conser_To_Prim!(W, V, gamma)
                @save "Data/V-" * string(istep) * ".jld2" V
            end

        end
    elseif input.problem_type == "Steady"

        MAXITE, EPS1 = input.max_iteration, input.converge_eps 
        istep = 0
        
        res_all = Float64[]

        Conser_To_Prim!(W, V, gamma)
        self.R[:,:] .= 0.0
        Spatial_Residual!(self, V, self.R)
        res0 =  norm(self.R, 2)
        res = res0

        while res > res0*EPS1 && istep < MAXITE

            cfl = input.cfl_init_number +  (input.cfl_max_number - input.cfl_init_number)*min(istep/100, 1.0)
            # compute time step size
            dt = ComputeTimeStep(self, W, input, cfl)
  
            # update solution in W for the next time step 
            Time_Advance!(self, W, dt, input)
            
            # fail safe
            Conser_To_Prim!(W, V, gamma)
            if minimum(V[1, :]) < 0.0 || minimum(V[4, :]) < 0.0
                @info "Negative density/pressure!"
                return W
            end

            # R is updated in Time_Advance!
            # self.R[:,:] .= 0.0
            # Spatial_Residual!(self, V, self.R)
            res = norm(self.R, 2)

            istep += 1
            
            if istep % 200 == 0
            # if istep % 1000 == 0
                @show ("Residual = ", res, " Relative resiual = ", res/res0, "; ite/MAXITE = " , istep, "/", MAXITE)
            end
            
            push!(res_all, res/res0)

        end

    else
        error("methods[ProblemType] ", methods["ProblemType"], " has not implemented")
    end
    return W
end



function ComputeTimeStep(self::NSSolver, W::Array{Float64,2}, input::Input, cfl::Float64)

    if input.time_step_computation == "Constant"
        return  input.dt

    elseif input.time_step_computation == "Global_CFL" || input.time_step_computation == "Local_CFL"
    
        gamma = self.gamma

        V = self.V

        Conser_To_Prim!(W, V, gamma)

        c = sqrt.(gamma * V[4, :] ./ V[1, :])

        wave_speed = sqrt.(V[2, :].^2 + V[3, :].^2) + c  
        
        dt = cfl * self.mesh.minEdgeLens ./ wave_speed

        if input.time_step_computation == "Global_CFL" 
            dt = minimum(dt)
        end

        return dt

    else
        error("TimeStep method ", input.time_step_computation, " has not implemented yet")
    end
end


# advance to next time step
# W: current conservative state
# update W to the state at the next time state
# update: 
# R: the residual 
# W: current state
function Time_Advance!(self::NSSolver, W::Array{Float64,2}, dt::Union{Float64,Array{Float64,1}}, input::Input)

    mesh = self.mesh
    V0, V, W0, R = self.V0, self.V,  self.W0, self.R
    k1, k2, k3 = self.k1, self.k2, self.k3

    gamma = self.gamma


    Conser_To_Prim!(W, V, gamma)
    

    # @info "u ∈ [", minimum(V[2, :]), " , ", maximum(V[2, :]), "] v ∈ [", minimum(V[3, :]), " , ", maximum(V[3, :]), "]" 
    # @info "ρ ∈ [", minimum(V[1, :]), " , ", maximum(V[1, :]), "] p ∈ [", minimum(V[4, :]), " , ", maximum(V[4, :]), "]" 
    
    ctrlVol = mesh.ctrlVol
    

    if input.time_integrator == "ForwardEuler"
        # W^{n+1} - W^{n} = dt * R^{n}/ Ω
        R[:,:] .= 0
        Spatial_Residual!(self, V, R)
        
        W .+=  R ./ ctrlVol' .* dt'
    elseif input.time_integrator == "BackwardEuler"
        # TODO only for steady simulation
        # @assert(input.problem_type == "Steady")
        # R.=0.0
        # Spatial_Residual!(self, V, R)
        # J = Spatial_Residual_Jacob(self, V, W)
        # α = input.cfl_number
        # Ω = repeat(ctrlVol', 4)[:]

        # W .-=  α *reshape( (J \ R[:]) , size(R) ) * 0.1
        # W0 .= W
        # Conser_To_Prim!(W0, V0, gamma)
        # R .= 0.0
        # Spatial_Residual!(self, V0, R)
        # J = Spatial_Residual_Jacob(self, V0, W0)
        # W .-= 0.01*reshape(J\R[:] , size(R)) 
      


        # Newton with inexact Jacobian
        # W^{n+1} = W^{n} + dt Residual(W^{n+1}) / Ω
        # 
        # Solve ΔW - dt * Residual(W^{n} + α ΔW ) / Ω = 0
        #
        # J = Ω⁻¹(Ω - αdt * ∂Residual/∂W(W^{n} + α ΔW )
        
        k1 .= 0.0                              # ΔW
        α = 1.0
        Ω_dt = repeat(ctrlVol' ./ dt', 4)[:]   
        for i = 1:2
            W0 .= W + α*k1
            Conser_To_Prim!(W0, V0, gamma)
            # fail safe
            if minimum(V0[1, :]) < 0.0 || minimum(V0[4, :]) < 0.0
                break
            end
            
            R .= 0.0
            Spatial_Residual!(self, V0, R)
            J = Spatial_Residual_Jacob(self, V0, W0)
            k2 .= k1 - R ./ ctrlVol' .* dt'
            k1 .-= reshape((Diagonal(Ω_dt) - α*J) \  (Ω_dt .* k2[:]) , size(R)) 
        end
        W .+= k1



        # Jacobian Free Newton Krylov
        # solve F(Q) = 0
        # Qⁿ⁺¹ = Qⁿ - dF/dQ(Qⁿ)^{-1} F(Qⁿ)
        # 
        # To solve 
        # [dF/dQ(Qⁿ) P^{-1}] [P ΔQ] = F(Qⁿ)
        # [dF/dQ(Qⁿ) P^{-1}] v =   F(Qⁿ + ϵ P^{-1}v) - F(Qⁿ) / ϵ
        # Solve Ω ΔW - dt * Residual(W^{n} + α ΔW )  = 0

        
        # α = 1.0
        # Ω = repeat(ctrlVol', 4)[:]
        # b = similar(Ω); 
        # ΔΔW = similar(Ω);  ΔΔW.= 0.0
        # ΔW = similar(Ω);  ΔW.= 0.0 #R ./ ctrlVol' .* dt'
        # function F(ΔW)
        #     R .= 0.0
        #     W0 .= W + α * reshape(ΔW, size(W))
        #     Conser_To_Prim!(W0, V0, gamma)
        #     Spatial_Residual!(self, V0, R)

        #     # @show "F: ", norm(Ω .* ΔW - dt * R[:])
        #     return Ω .* ΔW - dt * R[:]
        # end

        # function Jac(ΔW)
        #     R .= 0.0
        #     W0 .= W + α * reshape(ΔW, size(W))
        #     Conser_To_Prim!(W0, V0, gamma)
        #     J = Spatial_Residual_Jacob(self, V0, W0)
        #     return lu(Diagonal(Ω) - α*dt*J)
        # end

        # # solve for dF(ΔW)^{-1} F(ΔW)
        # for i = 1:2
        #     b.= F(ΔW)
        #     Pr = Jac(ΔW)

        #     # F = Ω ΔW - dt * Residual(W^{n} + α ΔW )
            
        #     n = length(Ω)
        #     Qp = ΔW
        #     Fp = F(ΔW)
        #     ϵ = 1e-3
        #     opA = MatrixOperator(F, Qp, Fp, ϵ, n)
            
        #     # @info "Test gres! with MatrixOperator"
        #     _, ch = gmres!(ΔΔW, opA, b; Pr = Pr,  log=true)
        #     ΔW .-= ΔΔW

        #     res = norm(F(ΔW))
        #     @info ch
        #     @info "norm(F(ΔW)) = ", res
        #     if res < 1e-4
        #         break
        #     end
            


        #     # @info "norm(ΔΔW) = ", norm(ΔΔW)

        # end
        # W .+= reshape(ΔW, size(W))

        
        
        
    elseif input.time_integrator == "RK2"
        k1[:,:] .= 0

        Spatial_Residual!(self, V, k1)

        
        W0[:,:] .= W + k1 ./ ctrlVol' .* dt'

        k2[:,:] .= 0
        Conser_To_Prim!(W0, V, gamma)

        Spatial_Residual!(self, V, k2);

        R[:,:] = 1.0 / 2.0 * (k1 + k2)

        W .+=  R ./ ctrlVol' .* dt'
    
    elseif input.time_integrator == "RK3"
        k1[:,:] .= 0
        Spatial_Residual!(self, V, k1)

        W0[:,:] .= W + k1 ./ ctrlVol' .* dt'
        k2[:,:] .= 0
        Conser_To_Prim!(W0, V, gamma)
        Spatial_Residual!(self, V, k2);

        W0[:,:] .= 3/4*W + 1/4*W0 + 1/4*k2 ./ ctrlVol' .* dt'
        k3[:,:] .= 0
        Conser_To_Prim!(W0, V, gamma)
        Spatial_Residual!(self, V, k3);
        


        R[:,:] = 1.0 / 6.0 * (k1 + k2 + 4*k3)
        W .+=  R ./ ctrlVol' .* dt'
    else
        error("Time integrator ", method, "has not implemented")
    end

    
    

end


# Compute spatial residual vector on each node 
# V: primitive state vector
# dW/dt = R
function Spatial_Residual!(self::NSSolver, V::Array{Float64,2}, R::Array{Float64,2})

    dV = self.dV

    Compute_Nodal_Gradients!(self.mesh.X, V, self.mesh.ngrad, dV)

    Euler_Flux_Residual!(self, V, dV, R)

    Euler_Boundary_Residual!(self, V, R)

end





# Compute spatial residual vector due to internal fluxes and add them to R
# V: primitive state vector
function Euler_Flux_Residual!(self::NSSolver, V::Array{Float64,2}, dV::Array{Float64,3}, R::Array{Float64,2})
    
    mesh = self.mesh
    nedges = mesh.nedges
    gamma = self.gamma

    X = self.mesh.X
    limiter = self.limiter
    Δi, Δj = zeros(Float64, 4), zeros(Float64, 4)
    dx = zeros(Float64, 2)

    
    for e = 1:nedges

        i, j = mesh.edges[:, e]

        V_i, V_j = V[:, i], V[:, j]

        nu_ij = mesh.edgeNorm[:, e]

        
        # linear reconstruction
        dx .= X[:, j] - X[:, i]
        Δi .=  dV[:, :, i] * dx
        Δj .= -dV[:, :, j] * dx

        v_L, v_R = limiter.Recon(V_i, V_j, Δi, Δj)

        flux = Roe_Flux(v_L, v_R, nu_ij, gamma)

        R[:, i] -= flux
        R[:, j] += flux
  
    end

end

# Compute spatial residual vector due to boundary fluxes and add them to R
# V: primitive state vector
function Euler_Boundary_Residual!(self::NSSolver, V::Array{Float64,2}, R::Array{Float64,2})

    mesh = self.mesh

    bcType = mesh.bcType     
    bcNorm = mesh.bcNorm
    bcData = mesh.bcData
    gamma = self.gamma

    nbc = size(bcType, 2)

    for e = 1:nbc

        i, j, bc_type, bc_data_type = bcType[:, e]

        V_i = V[:, i]
        V_j = V[:, j]

        nu_e = mesh.bcNorm[:, e]
        if (bc_type == 1) # wall
                # weakly impose slip_wall boundary condition
             
            R[:, i] -= Wall_Flux(V_i,  nu_e/2, gamma)
            R[:, j] -= Wall_Flux(V_j,  nu_e/2, gamma)
            

        elseif (bc_type == 2) # free_stream

            W_oo = bcData[bc_type][:, bc_data_type]


            R[:, i] -= Steger_Warming_Flux(V_i, W_oo, nu_e/2, gamma)
            R[:, j] -= Steger_Warming_Flux(V_j, W_oo, nu_e/2, gamma)
  
        end
    end
end



# Compute spatial residual vector on each node 
# V: primitive state vector
# dW/dt = R
function Spatial_Residual_Jacob(self::NSSolver, V::Array{Float64,2}, W::Array{Float64, 2})

  
    nnodes = size(V)[2]
    gamma = self.gamma
    dV_dW = zeros(nnodes, 4, 4)
    Conser_To_Prim_Jacob!(W, dV_dW, gamma)

    rowI, rowJ, rowV = Euler_Flux_Residual_Jacob(self, V, dV_dW)
    rowI_b , rowJ_b, rowV_b = Euler_Boundary_Residual_Jacob(self, V, dV_dW)

    return sparse([rowI; rowI_b], [rowJ; rowJ_b], [rowV; rowV_b], 4nnodes, 4nnodes, +)
    # return sparse(rowI_b, rowJ_b, rowV_b)

end



# Compute spatial residual vector due to internal fluxes and add them to R
# W
# V: primitive state vector
function Euler_Flux_Residual_Jacob(self::NSSolver, V::Array{Float64,2}, dV_dW::Array{Float64, 3})
    
    mesh = self.mesh
    nedges = mesh.nedges
    gamma = self.gamma

    J_i, J_j = zeros(4, 4), zeros(4, 4)

    rowI, rowJ, rowV = Int64[], Int64[], Float64[]
    for e = 1:nedges

        i, j = mesh.edges[:, e]

        V_i, V_j = V[:, i], V[:, j]

        nu_ij = mesh.edgeNorm[:, e]

        Roe_Flux_Jacb!(V_i, V_j, nu_ij, gamma, J_i, J_j)

        dflux_dWi = J_i * dV_dW[i, :, :]
        dflux_dWj = J_j * dV_dW[j, :, :]
        # R[:, i] -= flux
        # ∂R_i/∂W_i = -dflux_dWi  
        # ∂R_i/∂W_j = -dflux_dWj
        row_i, col_i =  MeshGrid(i, i) 
      
        append!(rowI, row_i)
        append!(rowJ, col_i)
        append!(rowV, -dflux_dWi[:])

        row_i, col_j =  MeshGrid(i, j) 
        append!(rowI, row_i)
        append!(rowJ, col_j)
        append!(rowV, -dflux_dWj[:])

        # R[:, j] += flux
        # ∂R_j/∂V_i = dflux_dWi 
        # ∂R_j/∂V_j = dflux_dWj

        row_j, col_i =  MeshGrid(j, i) 
        append!(rowI, row_j)
        append!(rowJ, col_i)
        append!(rowV, dflux_dWi[:])

        row_j, col_j =  MeshGrid(j, j) 
        append!(rowI, row_j)
        append!(rowJ, col_j)
        append!(rowV, dflux_dWj[:])
  
    end

    return rowI, rowJ, rowV

end

# Compute spatial residual vector due to boundary fluxes and add them to R
# V: primitive state vector
function Euler_Boundary_Residual_Jacob(self::NSSolver, V::Array{Float64,2}, dV_dW::Array{Float64,3})

    mesh = self.mesh

    bcType = mesh.bcType     
    bcNorm = mesh.bcNorm
    bcData = mesh.bcData
    gamma = self.gamma

    J_i, J_j = zeros(4, 4), zeros(4, 4)

    nbc = size(bcType, 2)

    rowI, rowJ, rowV = Int64[], Int64[], Float64[]
    for e = 1:nbc

        i, j, bc_type, bc_data_type = bcType[:, e]

        V_i = V[:, i]
        V_j = V[:, j]

        nu_e = mesh.bcNorm[:, e]
        if (bc_type == 1) # wall
            # weakly impose slip_wall boundary condition
             
            # R[:, i] -= Wall_Flux(V_i,  nu_e/2, gamma)
            # R[:, j] -= Wall_Flux(V_j,  nu_e/2, gamma)


            Wall_Flux_Jacb!(V_i, nu_e/2, gamma, J_i)
            dflux_dWi = J_i * dV_dW[i, :, :]
            # R[:, i] -= flux
            # ∂R_i/∂W_i = -dflux_dWi  
            row_i, col_i =  MeshGrid(i, i) 
            append!(rowI, row_i)
            append!(rowJ, col_i)
            append!(rowV, -dflux_dWi[:])


            Wall_Flux_Jacb!(V_j, nu_e/2, gamma, J_j)
            dflux_dWj = J_j * dV_dW[j, :, :]
            # R[:, i] -= flux
            # ∂R_i/∂W_i = -dflux_dWi  
            row_j, col_j =  MeshGrid(j, j) 
            append!(rowI, row_j)
            append!(rowJ, col_j)
            append!(rowV, -dflux_dWj[:])

            

        elseif (bc_type == 2) # free_stream

            W_oo = bcData[bc_type][:, bc_data_type]

            # R[:, i] -= Steger_Warming_Flux(V_i, W_oo, nu_e/2, gamma)
            # R[:, j] -= Steger_Warming_Flux(V_j, W_oo, nu_e/2, gamma)


            Steger_Warming_Flux_Jacb!(V_i, W_oo, nu_e/2, gamma, J_i)
            dflux_dWi = J_i * dV_dW[i, :, :]
            # R[:, i] -= flux
            # ∂R_i/∂W_i = -dflux_dWi  
            row_i, col_i =  MeshGrid(i, i) 
            append!(rowI, row_i)
            append!(rowJ, col_i)
            append!(rowV, -dflux_dWi[:])


            Steger_Warming_Flux_Jacb!(V_j, W_oo, nu_e/2, gamma, J_j)
            dflux_dWj = J_j * dV_dW[j, :, :]
            # R[:, i] -= flux
            # ∂R_i/∂W_i = -dflux_dWi  
            row_j, col_j =  MeshGrid(j, j) 
            append!(rowI, row_j)
            append!(rowJ, col_j)
            append!(rowV, -dflux_dWj[:])
  
        end
    end

    return rowI, rowJ, rowV
end
