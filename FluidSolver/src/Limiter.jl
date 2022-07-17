export Limiter

mutable struct Limiter
    limiter_type::String
    β::Float64
    ε::Float64

    Recon::Function
        
end

function Limiter(limiter_type::String, β::Float64 = 1.0/3.0, ε::Float64 = 1.0e-15)

    if limiter_type == "Constant"
        Recon = (V_i::Array{Float64, 1}, V_j::Array{Float64, 1}, dV_i::Array{Float64, 1}, dV_j::Array{Float64, 1}) -> Constant_Recon(V_i, V_j)
    elseif limiter_type == "VanAlbala"
        Recon = (V_i::Array{Float64, 1}, V_j::Array{Float64, 1}, dV_i::Array{Float64, 1}, dV_j::Array{Float64, 1}) -> VanAlbala_Recon(V_i, V_j, dV_i, dV_j, β, ε)
    elseif limiter_type == "Venkatakrishnan"
        Recon = (V_i::Array{Float64, 1}, V_j::Array{Float64, 1}, dV_i::Array{Float64, 1}, dV_j::Array{Float64, 1}) -> Venkatakrishnan_Recon(V_i, V_j, dV_i, dV_j, β, ε)
    elseif limiter_type == "Minmod"
        Recon = (V_i::Array{Float64, 1}, V_j::Array{Float64, 1}, dV_i::Array{Float64, 1}, dV_j::Array{Float64, 1}) -> Minmod_Recon(V_i, V_j, dV_i, dV_j, β, ε)
    else
        error("Limiter type ", limiter_type, " has not implemented")
    end
    
    Limiter(limiter_type, β, ε, Recon)
end


#  vi *------* vj
#  v_i v_j are two primitive state variables
#  dv_i  = ∇v_i *(xj - xi)
#  dv_j  = ∇v_j *(xi - xj)
function Constant_Recon(V_i::Array{Float64, 1}, V_j::Array{Float64, 1})
    return V_i, V_j
end

function Central_Recon(V_i::Array{Float64, 1}, V_j::Array{Float64, 1})
    return (V_i + V_j)/2.0, (V_i + V_j)/2.0
end


function VA_Slope_Limiter(a::Array{Float64, 1},b::Array{Float64, 1},ε::Float64)
    slope = zeros(Float64, 4)
    for i = 1:4
        if (a[i]*b[i] >= 0.0)
            slope[i] =  (a[i]*(b[i]^2 + ε) + b[i]*(a[i]^2 + ε))/(a[i]^2 + b[i]^2 + 2*ε)
        end
    end
    return slope
end
function VanAlbala_Recon(V_i::Array{Float64, 1}, V_j::Array{Float64, 1}, dV_i::Array{Float64, 1}, dV_j::Array{Float64, 1}, β::Float64, ε::Float64)
    Δij = V_j-V_i
    v_L = V_i + 0.5*VA_Slope_Limiter( Δij,   (1-4β)*Δij + 4β*dV_i, ε) 
    v_R = V_j + 0.5*VA_Slope_Limiter(-Δij,  -(1-4β)*Δij + 4β*dV_j, ε)
    return v_L, v_R
end


function Venkat_Slope_Limiter(a::Array{Float64, 1},b::Array{Float64, 1},ε::Float64)
    slope = zeros(Float64, 4)
    for i = 1:4
        if (a[i]*b[i] >= 0.0)
            slope[i] =  sign(a[i])*min(abs(a[i]), abs(b[i])) #(a[i]*(b[i]^2 + ε) + 2*b[i]*a[i]^2 )/(a[i]^2 + 2*b[i]^2 + a[i]*b[i] + ε)
        end
    end
    return slope
end
function Venkatakrishnan_Recon(V_i::Array{Float64, 1}, V_j::Array{Float64, 1}, dV_i::Array{Float64, 1}, dV_j::Array{Float64, 1}, β::Float64, ε::Float64)
    Δij = V_j-V_i
    v_L = V_i + 0.5*Venkat_Slope_Limiter( Δij,   (1-4β)*Δij + 4β*dV_i, ε) 
    v_R = V_j + 0.5*Venkat_Slope_Limiter(-Δij,  -(1-4β)*Δij + 4β*dV_j, ε)
    return v_L, v_R
end


function Minmod_Slope_Limiter(a::Array{Float64, 1},b::Array{Float64, 1},ε::Float64)
    slope = zeros(Float64, 4)
    for i = 1:4
        if (a[i]*b[i] >= 0.0)
            slope[i] =  sign(a[i])*min(abs(a[i]), abs(b[i])) 
        end
    end
    return slope
end

function Minmod_Recon(V_i::Array{Float64, 1}, V_j::Array{Float64, 1}, dV_i::Array{Float64, 1}, dV_j::Array{Float64, 1}, β::Float64, ε::Float64)
    Δij = V_j-V_i
    v_L = V_i + 0.5*Minmod_Slope_Limiter( Δij,   (1-4β)*Δij + 4β*dV_i, ε) 
    v_R = V_j + 0.5*Minmod_Slope_Limiter(-Δij,  -(1-4β)*Δij + 4β*dV_j, ε)
    return v_L, v_R
end