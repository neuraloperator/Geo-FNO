export Prim_To_Conser, Prim_To_Conser!, Conser_To_Prim, Conser_To_Prim!, Roe_Flux, Steger_Warming_Flux, Wall_Flux
export Roe_Flux_Jacb!, Steger_Warming_Flux_Jacb!, Wall_Flux_Jacb!

function Prim_To_Conser(V::Array{Float64,1}, gamma::Float64)
    rho, vx, vy, p = V
    return [rho,rho * vx, rho * vy,rho * (vx * vx + vy * vy) / 2 + p / (gamma - 1)]
end



function Prim_To_Conser!(V::Array{Float64,2}, W::Array{Float64,2}, gamma::Float64)

    W[1, :] = V[1, :]
    W[2, :] = V[2, :] .* V[1, :]
    W[3, :] = V[3, :] .* V[1, :]
    W[4, :] = 0.5 * V[1, :] .* (V[2, :].^2 + V[3, :].^2) + V[4, :] / (gamma - 1.0)
end

function Prim_To_Conser_Jacob!(V::Array{Float64,1}, dW_dV::Array{Float64,2}, gamma::Float64)
    rho, vx, vy, p = V
    dW_dV[1, :] .= 1.0, 0.0, 0.0, 0.0
    dW_dV[2, :] .= vx, rho, 0.0, 0.0
    dW_dV[3, :] .= vy, 0, rho, 0.0
    dW_dV[4, :] .= (vx * vx + vy * vy) / 2, rho*vx, rho*vy, 1/(gamma - 1)

    # return [rho,rho * vx, rho * vy, rho * (vx * vx + vy * vy) / 2 + p / (gamma - 1)]
end

function Prim_To_Conser_Jacob!(V::Array{Float64,2}, dW_dV::Array{Float64,3}, gamma::Float64)
    nnodes = size(V)[2]
    for i = 1:nnodes
        rho, vx, vy, p = V[:, i]
        dW_dV[i, 1, :] .= 1.0, 0.0, 0.0, 0.0
        dW_dV[i, 2, :] .= vx, rho, 0.0, 0.0
        dW_dV[i, 3, :] .= vy, 0, rho, 0.0
        dW_dV[i, 4, :] .= (vx * vx + vy * vy) / 2, rho*vx, rho*vy, 1/(gamma - 1)
        
    end

end

function Conser_To_Prim(W::Array{Float64,1}, gamma::Float64)
    w1, w2, w3, w4 = W
    rho = w1;
    vx = w2 / w1;
    vy = w3 / w1
    p = (w4 - w2 * vx / 2 - w3 * vy / 2) * (gamma - 1)

    return [rho,vx,vy,p]
end

function Conser_To_Prim_Jacob!(W::Array{Float64,1}, dV_dW::Array{Float64,2}, gamma::Float64)
    w1, w2, w3, w4 = W
    dV_dW[1, :] .= 1,        0,       0,   0
    dV_dW[2, :] .= -w2/w1^2, 1/w1,    0,   0
    dV_dW[3, :] .= -w3/w1^2, 0,     1/w1,  0
    dV_dW[4, :] .= (w2^2+w3^2)/(2w1^2)*(gamma - 1),    -w2/w1* (gamma - 1), -w3/w1* (gamma - 1),   (gamma - 1)

end


function Conser_To_Prim_Jacob!(W::Array{Float64,2}, dV_dW::Array{Float64,3}, gamma::Float64)
    nnodes = size(W)[2]
    for i = 1:nnodes
        w1, w2, w3, w4 = W[:, i]
        dV_dW[i, 1, :] .= 1,        0,       0,   0
        dV_dW[i, 2, :] .= -w2/w1^2, 1/w1,    0,   0
        dV_dW[i, 3, :] .= -w3/w1^2, 0,     1/w1,  0
        dV_dW[i, 4, :] .= (w2^2+w3^2)/(2w1^2)*(gamma - 1),    -w2/w1* (gamma - 1), -w3/w1* (gamma - 1),   (gamma - 1)
        
    end
end

function Conser_To_Prim!(W::Array{Float64,2}, V::Array{Float64,2}, gamma::Float64)
    V[1, :] = W[1, :]
    V[2, :] = W[2, :] ./ W[1, :]
    V[3, :] = W[3, :] ./ W[1, :]
    V[4, :] = (W[4, :] - 0.5 * W[2, :] .* V[2, :] - 0.5 * W[3, :] .* V[3, :]) * (gamma - 1.0)
end


# Exact flux
# Primitive state variable vector V_ij
# normal n_ij
function Euler_Flux(V_ij::Array{Float64,1}, n_ij::Array{Float64,1}, gamma::Float64)

    rho, vx, vy, p = V_ij

    E = 0.5 * rho * (vx^2 + vy^2) + p / (gamma - 1.0)

    v_n = vx * n_ij[1] + vy * n_ij[2]

    return [rho * v_n, rho * vx * v_n + p * n_ij[1] , rho * vy * v_n + p * n_ij[2], (E + p) * v_n]

end

# Roe's Flux method 1
# Primitive state variable vector V_i, Vj
# normal n_ij, from node i to node j
function Roe_Flux(V_i::Vector, V_j::Vector, n_ij::Array{Float64,1}, gamma::Float64)
    n_len = sqrt(n_ij[1]^2 + n_ij[2]^2)

    # construct unit normal/tagent vector
    n_ij = n_ij / n_len
    t_ij = [-n_ij[2], n_ij[1]]

    # left state
    rho_l, u_l, v_l, p_l = V_i;
    un_l = u_l * n_ij[1] + v_l * n_ij[2]
    ut_l = u_l * t_ij[1] + v_l * t_ij[2]
    a_l = sqrt(gamma * p_l / rho_l)
    h_l = 0.5 * (v_l * v_l + u_l * u_l) + gamma * p_l / (rho_l * (gamma - 1.0));

    # right state
    rho_r, u_r, v_r, p_r = V_j;
    un_r = u_r * n_ij[1] + v_r * n_ij[2]
    ut_r = u_r * t_ij[1] + v_r * t_ij[2]
    a_r = sqrt(gamma * p_r / rho_r)
    h_r = 0.5 * (v_r * v_r + u_r * u_r) + gamma * p_r / (rho_r * (gamma - 1.0));

    # compute the Roe-averaged quatities
    RT = sqrt(rho_r / rho_l)
    rho_rl = RT * rho_l

    u_rl = (u_l + RT * u_r) / (1.0 + RT)

    v_rl = (v_l + RT * v_r) / (1.0 + RT)
    h_rl = (h_l + RT * h_r) / (1.0 + RT)
    a_rl = sqrt((gamma - 1) * (h_rl - 0.5 * (u_rl * u_rl + v_rl * v_rl)))
    un_rl = u_rl * n_ij[1] + v_rl * n_ij[2]
    ut_rl = u_rl * t_ij[1] + v_rl * t_ij[2]


    # wave strengths
    dp = p_r - p_l
    drho = rho_r - rho_l
    dun = un_r - un_l
    dut = ut_r - ut_l
    du = [(dp - rho_rl * a_rl * dun) / (2.0 * a_rl * a_rl),  rho_rl * dut,  drho - dp / (a_rl * a_rl),  (dp + rho_rl * a_rl * dun) / (2.0 * a_rl * a_rl)]

    # compute the Roe-average wave speeds
    ws = [abs(un_rl - a_rl), abs(un_rl), abs(un_rl), abs(un_rl + a_rl)]


    # compute the right characteristic eigenvectors
    P_inv = [[1.0                    0.0          1.0                            1.0];
             [u_rl - a_rl * n_ij[1]    t_ij[1]        u_rl                           u_rl + a_rl * n_ij[1]];
             [v_rl - a_rl * n_ij[2]    t_ij[2]        v_rl                           v_rl + a_rl * n_ij[2]];
             [h_rl - un_rl * a_rl       ut_rl         0.5 * (u_rl * u_rl + v_rl * v_rl)      h_rl + un_rl * a_rl]]


    f_l = [rho_l * un_l, rho_l * un_l * u_l + p_l * n_ij[1],  rho_l * un_l * v_l + p_l * n_ij[2], rho_l * h_l * un_l ]
    f_r = [rho_r * un_r, rho_r * un_r * u_r + p_r * n_ij[1],  rho_r * un_r * v_r + p_r * n_ij[2], rho_r * h_r * un_r ]

  
    flux = 0.5 * (f_r + f_l  - P_inv * (du .* ws))

    return n_len * flux

end

function  Roe_Flux_Jacb!(V_i::Vector, V_j::Vector, n_ij::Array{Float64,1}, gamma::Float64, J_i::Array{Float64,2}, J_j::Array{Float64,2})  
    Roe_Flux_Jacb_Helper = (V) -> Roe_Flux(V[1:4], V[5:8], n_ij, gamma)
    J = ForwardDiff.jacobian(Roe_Flux_Jacb_Helper, [V_i; V_j])
    J_i .= J[:, 1:4]
    J_j .= J[:, 5:8]
end

# Steger Warming Flux
# Primitive state variable vector V_i
# Conservative far-field state variable vector W_oo
# outward normal n_ioo
function Steger_Warming_Flux(V_i::Vector, W_oo::Array{Float64,1}, n_ioo::Array{Float64,1}, gamma::Float64)
    n_len = sqrt(n_ioo[1]^2 + n_ioo[2]^2)

    # construct unit normal vector

    tilde_nx, tilde_ny = n_ioo / n_len

    rho, vx, vy, p = V_i


    # normal velocity
    v = vx * n_ioo[1] + vy * n_ioo[2]

    c = sqrt(gamma * p / rho)


    Dp = [max(0, v), max(0, v), max(0, v + c * n_len),max(0, v - c * n_len)]
    Dm = [min(0, v), min(0, v), min(0, v + c * n_len),min(0, v - c * n_len)]

    theta = tilde_nx * vx + tilde_ny * vy

    phi = sqrt((gamma - 1) / 2 * (vx * vx + vy * vy))

    beta = 1 / (2 * c * c)

    Q = [[1.0       0.0       1.0    1.0];
          [vx                             tilde_ny           vx + tilde_nx * c                        vx - tilde_nx * c];
          [vy                            -tilde_nx           vy + tilde_ny * c                        vy - tilde_ny * c];
          [phi * phi / (gamma - 1)     tilde_ny * vx - tilde_nx * vy     (phi * phi + c * c) / (gamma - 1) + c * theta    (phi * phi + c * c) / (gamma - 1) - c * theta]]

    Qinv = [[1 - phi * phi / (c * c)                 (gamma - 1) * vx / c^2                      (gamma - 1) * vy / c^2                         -(gamma - 1) / c^2];
            [-(tilde_ny * vx - tilde_nx * vy)       tilde_ny                             -tilde_nx                                0.0];
            [beta * (phi^2 - c * theta)           beta * (tilde_nx * c - (gamma - 1) * vx)   beta * (tilde_ny * c - (gamma - 1) * vy)       beta * (gamma - 1)];
            [beta * (phi^2 + c * theta)          -beta * (tilde_nx * c + (gamma - 1) * vx)   -beta * (tilde_ny * c + (gamma - 1) * vy)      beta * (gamma - 1)]]

 
    fp = 0.5 * rho / gamma * [2.0 * (gamma - 1) * Dp[1] + Dp[3] + Dp[4],
                          2.0 * (gamma - 1) * Dp[1] * vx + Dp[3] * (vx + c * tilde_nx) + Dp[4] * (vx - c * tilde_nx),
                          2.0 * (gamma - 1) * Dp[1] * vy + Dp[3] * (vy + c * tilde_ny) + Dp[4] * (vy - c * tilde_ny),
                          (gamma - 1) * Dp[1] * (vx * vx + vy * vy) + 0.5 * Dp[3] * ((vx + c * tilde_nx)^2 + (vy + c * tilde_ny)^2) + 0.5 * Dp[4] * ((vx - c * tilde_nx)^2 + (vy - c * tilde_ny)^2) + (3.0 - gamma) * (Dp[3] + Dp[4]) * c * c / (2 * (gamma - 1))]

    fm = Q * (Dm .* (Qinv * W_oo))

 
    return fp + fm

end

function  Steger_Warming_Flux_Jacb!(V_i::Vector, W_oo::Array{Float64,1}, n_ioo::Array{Float64,1}, gamma::Float64, J_i::Array{Float64,2})  
    Steger_Warming_Flux_Jacb_Helper = (V) -> Steger_Warming_Flux(V, W_oo, n_ioo, gamma)
    J_i .= ForwardDiff.jacobian(Steger_Warming_Flux_Jacb_Helper, V_i)
end


# Wall Flux
# Primitive state variable vector V_i
# outward wall normal n_i
function Wall_Flux(V_i::Vector, n_i::Array{Float64,1}, gamma::Float64)

    rho, vx, vy, p = V_i

    return [0.0, p * n_i[1] , p * n_i[2], 0.0]

end


function  Wall_Flux_Jacb!(V_i::Array{Float64,1}, n_i::Array{Float64,1}, gamma::Float64, J_i::Array{Float64,2})  
    J_i .= 0.0
    J_i[2, 4], J_i[3, 4] = n_i[1], n_i[2]
end


# function  Wall_Flux_Jacb!(V_i::Array{Float64,1}, n_i::Array{Float64,1}, gamma::Float64, J_i::Array{Float64,2}) 
#     Wall_Flux_Jacb_Helper = (V) -> Wall_Flux(V, n_i, gamma)
#     J_i .= ForwardDiff.jacobian(Wall_Flux_Jacb_Helper, V_i)
# end