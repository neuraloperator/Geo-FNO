using Revise
using PyPlot
using JLD2
using LinearAlgebra
using Statistics
using Distributions
using Random
using Roots
using FluidSolver

# symmetrical 4-digit NASA airfoil
# the airfoil is in [0,1]
function NACA_shape(x; digit=12) 
    5*(digit/100.0)*(0.2969sqrt(x) - 0.1260x - 0.3516x^2 + 0.2843x^3 - 0.1036x^4)
end


function NACA_shape_arc_len(x; digit=12) 
    5*(digit/100.0)*(0.2969sqrt(x)*x/1.5 - 0.1260x^2/2 - 0.3516x^3/3 + 0.2843x^4/4 - 0.1036x^5/5)
end


# Nx point on top skin
function NACA_shape_mesh(Nx; method="stretching", ratio = 1.0)
    
    if method == "uniform"
        xx = Array(LinRange(0, 1, Nx))
    elseif method == "stretching"
        xx = zeros(Nx)
        xx[2:end] = GeoSpace(0, 1, Nx-1; r =ratio^(1/(Nx-3)))
        xx[2] = xx[3]/4.0 
    else
        error("method : ", method, " is not recognized")
    end

    xx .= xx[end:-1:1]
    yy = [NACA_shape.(xx); -NACA_shape.(xx[Nx-1:-1:1])] 
    xx = [xx; xx[Nx-1:-1:1]]

    return xx, yy
end

# The undeformed box is 
# 0.5 - Lx/2  (8)        0.5 - Lx/6  (7)         0.5 + Lx/6  (6)          0.5 + Lx/2  (5)         (y = Ly/2)
#
# 0.5 - Lx/2  (1)        0.5 - Lx/6  (2)         0.5 + Lx/6  (3)          0.5 + Lx/2  (4)         (y = -Ly/2)
#
# basis function at node (i)   is Bᵢ   = Φᵢ(x) Ψ₁(y)    (1 ≤ i ≤ 4)
# basis function at node (i+4) is Bᵢ₊₄ = Φᵢ(x) Ψ₂(y)    (1 ≤ i ≤ 4)
#
# The map is 
# (x, y) -> (x, y) + dᵢ Bᵢ(x,  y)
#
function NACA_sdesign(d::Array{FT,1}, x, y; Lx=1.5, Ly=0.2) where {FT<:AbstractFloat}
    x₁, x₂, x₃, x₄ = 0.5 - Lx/2,  0.5 - Lx/6,   0.5 + Lx/6,    0.5 + Lx/2
    y₁, y₂  = - Ly/2,  Ly/2

    
    Φ₁(x) = (x - x₂)*(x - x₃)*(x - x₄)  / ((x₁ - x₂)*(x₁ - x₃)*(x₁ - x₄))
    Φ₂(x) = (x - x₁)*(x - x₃)*(x - x₄)  / ((x₂ - x₁)*(x₂ - x₃)*(x₂ - x₄))
    Φ₃(x) = (x - x₁)*(x - x₂)*(x - x₄)  / ((x₃ - x₁)*(x₃ - x₂)*(x₃ - x₄))
    Φ₄(x) = (x - x₁)*(x - x₂)*(x - x₃)  / ((x₄ - x₁)*(x₄ - x₂)*(x₄ - x₃))

    Ψ₁(y) = (y - y₂)/(y₁ - y₂)
    Ψ₂(y) = (y - y₁)/(y₂ - y₁)

    B = [Φ₁(x)*Ψ₁(y), Φ₂(x)*Ψ₁(y),  Φ₃(x)*Ψ₁(y),  Φ₄(x)*Ψ₁(y),  Φ₄(x)*Ψ₂(y), Φ₃(x)*Ψ₂(y), Φ₂(x)*Ψ₂(y),  Φ₁(x)*Ψ₂(y)]
    return x, y + d'*B

end




function NACA_sdesign_demo()
    Lx, Ly = 1.5, 0.2
    bbox = [0.5 - Lx/2, 0.5 - Lx/6, 0.5 + Lx/6, 0.5 + Lx/2,  0.5 + Lx/2, 0.5 + Lx/6, 0.5 - Lx/6, 0.5 - Lx/2, 0.5 - Lx/2]
    bboy = [-Ly/2, -Ly/2, -Ly/2, -Ly/2,  Ly/2, Ly/2, Ly/2, Ly/2, -Ly/2]

    Nx = 50
    xx, yy = NACA_shape_mesh(Nx; method="stretching")
    xx_new, yy_new = similar(xx), similar(yy)

    fig, ax = subplots(4, 2, sharex=true, sharey=true, figsize=(12, 6))
    vdisp = Ly/2

    for ip = 1:8
        d = zeros(8)
        d[ip] = vdisp
        for i = 1:length(xx)
            xx_new[i], yy_new[i] = NACA_sdesign(d, xx[i], yy[i]; Lx = Lx, Ly  = Ly)
        end
        
        
        ax[ip].plot(xx, yy, "-", color="blue")
        ax[ip].plot(xx_new, yy_new, color="black")
        ax[ip].plot(bbox, bboy, "--o", color="blue")

        ax[ip].scatter(bbox[ip] , bboy[ip]+ vdisp, color="black")

        Nb = 20
        bbox_deform  = Array(LinRange(0.5 - Lx/2, 0.5 + Lx/2, Nb))
        bboy_deform  = similar(bbox_deform)
        bboy_deform .= ( ip <= 4 ? -Ly/2.0 : Ly/2.0)
        for i = 1:Nb
            bbox_deform[i], bboy_deform[i] = NACA_sdesign(d, bbox_deform[i], bboy_deform[i]; Lx = Lx, Ly  = Ly)
        end
        ax[ip].plot(bbox_deform, bboy_deform, "--", color="black")
    end
    
    fig.savefig("NACA-demo.pdf")
end


# generate mesh between a and b 
# dx0, dx0*r, dx0*r^2 ... dx0*r^{N-2}
# b - a = dx0*(r^{N-1} - 1)/(r - 1)
function GeoSpace(a, b, N; r = -1.0, dx0= -1.0)

    xx = Array(LinRange(a, b, N))

    if r > 1 || dx0 > 0
        if r > 1 
            dx0 = (b - a)/((r^(N - 1) - 1)/(r - 1))

            dx = dx0
            for i = 2:N-1
                xx[i] = xx[i-1] + dx
                dx *= r
            end

        else
            # first use r=1.05 to generate half of the grids
            # then compute r and generate another half of the grids
            f(r)  = (r-1)*(b-a) - dx0*(r^(N-1) - 1)
            Df(r) = (b-a) - dx0*(N-1)*r^(N-2)  
            r = find_zero(f, (1+1e-4, 1.5), Roots.Bisection())
            if r > 1.02
                r = min(r, 1.02)
                dx = dx0
                Nf = div(3N, 4)
               

                for i = 2:Nf
                    xx[i] = xx[i-1] + dx
                    dx *= r
                end
                a = xx[Nf]
                dx0 = dx
                
                f = (r) -> (r-1)*(b-a)-dx0*(r^(N - Nf) - 1)
                Df = (r) -> (b-a)-dx0*(N-Nf)*r^(N - Nf - 1)
                r0= 1.0 + 2*((b-a)/(dx0) - (N-Nf))/((N-Nf)*(N-Nf-1))
                
                r = find_zero(f, (1+1e-4, 2.0), Roots.Bisection())
                for i = Nf+1:N-1
                    xx[i] = xx[i-1] + dx
                    dx *= r
                end

            else
                dx = dx0
                for i = 2:N-1
                    xx[i] = xx[i-1] + dx
                    dx *= r
                end
            end
            
        end 
    
        
    end

    return xx
end



function Cgrid2Cylinder!(cnx1, cnx2, cny, Cgrid, Cylinder)
    nx1, nx2, ny = cnx1+1, cnx2+1, cny+1
    @assert(size(Cylinder) == (2*nx1+cnx2-1, cny+1))
    @assert(size(Cgrid) == ((2*nx1+cnx2-1)*cny + (nx1+cnx2-1),))

    for j = 1:cny+1
        if j == 1
            Cylinder[1:cnx1+cnx2, j] .= Cgrid[1:cnx1+cnx2]
            Cylinder[cnx1+cnx2+1:2*nx1+cnx2-1, j] .= Cylinder[cnx1+1:-1:1, j]
        else
            Cylinder[:, j] .= Cgrid[(j-2)*(2cnx1+cnx2+1) + cnx1+cnx2+1: (j-2)*(2cnx1+cnx2+1) + cnx1+cnx2+1 + 2cnx1+cnx2]
        end
    end
end

function Cylinder2Cgrid!(cnx1, cnx2, cny, Cylinder, Cgrid)
    nx1, nx2, ny = cnx1+1, cnx2+1, cny+1

    @assert(size(Cylinder) == (2*nx1+cnx2-1, cny+1))
    @assert(size(Cgrid) == ((2*nx1+cnx2-1)*cny + (nx1+cnx2-1),))

    for j = 1:cny+1
        if j == 1
            Cgrid[1:cnx1+cnx2] .= Cylinder[1:cnx1+cnx2, j]
        else
            Cgrid[(j-2)*(2cnx1+cnx2+1) + cnx1+cnx2+1: (j-2)*(2cnx1+cnx2+1) + cnx1+cnx2+1 + 2cnx1+cnx2] .= Cylinder[:, j]
        end
    end
end

function PlotCylinderGrid(xx_Cylinder, yy_Cylinder, filename::String="None"; equal_axis = true)
    plot(xx_Cylinder, yy_Cylinder, color = "black", linewidth=0.1)
    plot(xx_Cylinder', yy_Cylinder', color = "black", linewidth=0.1)

    if equal_axis
        axis("equal")
    end
    if filename != "None"
        savefig(filename)
        close("all")
    end
end




function Theta2Mesh(θ_field; cnx1 = 50, cnx2 = 120, cny = 50, R = 40, Rc = 1.0, L = 40, dy0 = 2.0/120.0)
    @assert(cnx2%2 == 0)
    cnx = 2cnx1 + cnx2
    nx1, nx2, ny = cnx1+1, cnx2+1, cny+1 #points
    nnodes = (2*nx1+cnx2-1)*cny + (nx1+cnx2-1)

    xx_airfoil, yy_airfoil = NACA_shape_mesh(div(cnx2,2)+1; method="stretching")



    xx_inner = GeoSpace(0, 1, nx1; dx0 = sqrt((xx_airfoil[1] - xx_airfoil[2])^2 + (yy_airfoil[1] - yy_airfoil[2])^2)/(L-1))
    xx_outer = GeoSpace(Rc, L, nx1)
    wy = GeoSpace(0, 1, ny; dx0 = dy0/R)
    
    
    for i = 1:cnx2+1
        xx_airfoil[i], yy_airfoil[i] = NACA_sdesign(θ_field, xx_airfoil[i], yy_airfoil[i])
    end
    
    
    xy_inner, xy_outer = zeros(2*nx1+cnx2-1,2), zeros(2*nx1+cnx2-1,2)
    # top flat 
    xy_inner[nx1:-1:1, 1] .= xx_airfoil[1]*(1 .- xx_inner)  +  L * xx_inner
    xy_inner[nx1:-1:1, 2] .= yy_airfoil[1]*(1 .- xx_inner)
    xy_outer[1:nx1, 1] .= xx_outer[end:-1:1] 
    xy_outer[1:nx1, 2] .= R
    # airfoil
    xy_inner[nx1:nx1+cnx2, 1] .= xx_airfoil
    xy_inner[nx1:nx1+cnx2, 2] .= yy_airfoil

    θθ = LinRange(π/2, 3*π/2, nx2)
    xy_outer[nx1:nx1+cnx2, 1] .= R*cos.(θθ) .+ Rc
    xy_outer[nx1:nx1+cnx2, 2] .= R*sin.(θθ)
    # bottom flat
    xy_inner[nx1+cnx2:2*nx1+cnx2-1, 1] .= xy_inner[nx1:-1:1, 1]
    xy_inner[nx1+cnx2:2*nx1+cnx2-1, 2] .= xy_inner[nx1:-1:1, 2]
    xy_outer[nx1+cnx2:2*nx1+cnx2-1, 1] .= xx_outer #LinRange(1.0, L+1.0, nx1)
    xy_outer[nx1+cnx2:2*nx1+cnx2-1, 2] .= -R

    
    
    # Construct Cylinder grid

    xx_Cylinder = zeros(Float64, 2*nx1+cnx2-1, cny+1)
    yy_Cylinder = zeros(Float64, 2*nx1+cnx2-1, cny+1)
    
    
    for j=1:ny
        xx_Cylinder[:,j] = xy_inner[:,1]*(1 - wy[j]) + xy_outer[:,1]*wy[j];
        yy_Cylinder[:,j] = xy_inner[:,2]*(1 - wy[j]) + xy_outer[:,2]*wy[j];
    end
    
    return xx_airfoil, yy_airfoil, xx_Cylinder, yy_Cylinder
end

# c: number of cells
# cnx1 C mesh behind trailing edge 
# cnx2 C mesh around airfoil
# cny radial direction
# 
# The airfoil is in [0,1]
# R: radius of C mesh 
# L: the right end of the mesh
# the bounding box of the mesh is [Rc-R, L], [-R, R]
#
# dy0, vertical mesh size
function NACA(θ_field; cnx1 = 2, cnx2 = 6, cny = 3, 
                        R = 10, Rc = 0.0, L = 10, dy0 = 0.05,
                        cfl_init_number = 0.5, cfl_max_number = 10.0,
                        time_integrator = "BackwardEuler",
                        eps1 = 1.0e-6, maxite = 2000)
    @assert(cnx2%2 == 0)
    cnx = 2cnx1 + cnx2
    nx1, nx2, ny = cnx1+1, cnx2+1, cny+1 #points
    nnodes = (2*nx1+cnx2-1)*cny + (nx1+cnx2-1)
    
    
    xx_airfoil, yy_airfoil, xx_Cylinder, yy_Cylinder = Theta2Mesh(θ_field; cnx1 = cnx1, cnx2 = cnx2, cny = cny, R = R, Rc = Rc, L = L, dy0 = dy0)
    

    
    # Generate mesh node coordinates
    xys = zeros(Float64, 2 , nnodes)
    Cylinder2Cgrid!(cnx1, cnx2, cny, xx_Cylinder, @views xys[1, :])
    Cylinder2Cgrid!(cnx1, cnx2, cny, yy_Cylinder, @views xys[2, :])

    # Generate element list
    elems = zeros(Int64, 4, cnx*cny)
    for j = 1:cny
        for i = 1:cnx
            if j == 1
                eid2 = i + (nx1+cnx2-1)
                
                eid = (i ≤ cnx1+cnx2 ? i : nx1 + 1 - (i - (cnx1+cnx2)) )
                eid4 = (i < cnx1+cnx2 ? i+1 : nx1 - (i - (cnx1+cnx2)) )
                elems[:, (i - 1)+(j - 1)*cnx + 1] .= eid,  eid2 , eid2+1,  eid4
            else
                eid = (nx1+cnx2-1) + i + (j - 2)*(2*nx1+cnx2-1)
                elems[:, (i - 1)+(j - 1)*cnx + 1] .= eid, eid + (cnx+1),  eid + (cnx+2),  eid + 1
            end
        end
    end

    # Generate boundary map, e1, e2, bc_type, bc_date_typ
    bc_map = zeros(Int64, 4,  2cnx1 + cnx2 + 2cny + cnx2)
    ibc = 1


    start = (2*nx1+cnx2-1)*cny + (nx1+cnx2-1)
    for i = 1:cny
        stride = (i == cny ? (nx1+cnx2-1  + 2*nx1+cnx2-2) : (2*nx1+cnx2-1))
        bc_map[:, ibc] .= start, start - stride, 2, 1 # farfield outflow (bottom half)
        start = start - stride
        ibc += 1
    end
    start = 1
    for i = 1:cny
        stride = (i == 1 ? (nx1+cnx2-1) : (2*nx1+cnx2-1))
        bc_map[:, ibc] .= start, start + stride, 2, 1 # farfield outflow (top half)
        start = start + stride
        ibc += 1
    end
    for i = 1:2cnx1+cnx2
        eid = i + (nx1+cnx2-1) + (cny - 1)*(2*nx1+cnx2-1)
        bc_map[:, ibc] .= eid, eid + 1, 2, 2 # farfield inflow
        ibc += 1
    end
    # 
    for i = 1:cnx2
        eid = i + cnx1
        eid2 = (i == cnx2 ? cnx1+1 : eid + 1)
        bc_map[:, ibc] .= eid, eid2, 1, 1 # wall boundary
        ibc += 1
    end


    
    # Generate far-field boundary conditions
    gamma = 1.4
    p_oo = 1.0; rho_oo = 1.0; M_oo = 0.8; Aot = 0.0 * pi/180
    u_oo, v_oo = [cos(Aot*π/180) , sin(Aot*π/180)] *  M_oo*sqrt(gamma*p_oo/rho_oo)


    W_oo = zeros(Float64, 4, 2)
    W_oo[:, 1] .= Prim_To_Conser([rho_oo, u_oo, v_oo, p_oo], gamma)
    W_oo[:, 2] .= Prim_To_Conser([rho_oo, u_oo, v_oo, p_oo], gamma)


    bc_data = [Array{Float64}(undef, 0, 0), W_oo]

    # Generate mesh object
    mesh = QuadMesh(xys, elems, bc_map, bc_data)



    # Generate solver
    

    # TimeIntegrator: RK2, ForwardEuler
    # ProblemType: Steady, Unsteady,  UnsteadyAeroelastic
    input = Input(problem_type = "Steady", max_iteration = maxite, converge_eps = eps1,
                time_integrator = time_integrator,   time_step_computation = "Local_CFL", 
                cfl_init_number = cfl_init_number, cfl_max_number = cfl_max_number,
                limiter_type = "Minmod", limiter_beta = 1.0/3.0,
                gamma = gamma)

    solver = NSSolver(mesh, input)
    # Generate initial conditions
    W0 = zeros(Float64, 4, nnodes)


    @. W0[1, :] = W_oo[1, 1]
    @. W0[2, :] = W_oo[2, 1]
    @. W0[3, :] = W_oo[3, 1]
    @. W0[4, :] = W_oo[4, 1]

    W = Solve!(solver,  W0, input)

    # Postprocess
    V = zeros(Float64, 4, nnodes)
    Conser_To_Prim!(W, V, gamma)
    M = similar(V[2, :])
    # failsafe
    if minimum(V[1, :]) > 0.0 && minimum(V[4, :]) > 0.0  
        M .= sqrt.(V[2, :].^2 + V[3, :].^2)./sqrt.(gamma*V[4, :]./V[1, :])
    end
    
    Q = zeros(5, nnodes)
    for i = 1:4
        Q[i, :] .= V[i, :]
    end
    Q[5, :] .= M
    
    
    Q_Cylinder = zeros(5, size(xx_Cylinder,1), size(xx_Cylinder, 2))
    for i = 1:5
        Cgrid2Cylinder!(cnx1, cnx2, cny, Q[i,:], @view Q_Cylinder[i, :, :])
    end
    
    
    return solver, Q, norm(solver.R, 2), xx_Cylinder, yy_Cylinder, Q_Cylinder
    
end

# NACA_sdesign_demo()

# Random.seed!(3)
# N_data = 2
# θ_field = rand(Uniform(-0.1, 0.1), N_data, 8);
# θ_field[:, 1] .= 0.0
# @time xx, yy, Q, res, xx_Cylinder, yy_Cylinder, Q_Cylinder = NACA(θ_field[1, :]; cnx1 = 40, dy0 = 1/40, cnx2 = 80, cny = 40, R = 40, Rc = 1.0, L = 40, cfl = 0.1, maxite = 2000, plot_or_not = true)