# MOVING or even number means compute load
const BC_SLIP_WALL_FIXED = 1
const BC_SLIP_WALL_MOVING = 2
const BC_ISOTHERMAL_WALL_FIXED = 3
const BC_ISOTHERMAL_WALL_MOVING = 4
const BC_ADIABATIC_WALL_FIXED = 5
const BC_ADIABATIC_WALL_MOVING = 6

const BC_INLET_FIXED = 7
const BC_INLET_MOVING = 8
const BC_OUTLET_FIXED = 9
const BC_OUTLET_MOVING = 10

const BC_SYMMETRY = 11

function Compute_First_Order_BC_Flux!(bc_type::Int64, bc_data::Array{Float64,1}, bc_norm, bc_norm_vel, V, flux)
    if (bc_type == BC_ISOTHERMAL_WALL_FIXED || 
        bc_type == BC_ISOTHERMAL_WALL_MOVING ||
        bc_type == BC_SLIP_WALL_FIXED ||
        bc_type == BC_SLIP_WALL_MOVING || 
        bc_type == BC_ADIABATIC_WALL_FIXED ||
        bc_type == BC_ADIABATIC_WALL_MOVING ||
        bc_type == BC_SYMMETRY)

        # normal velocity is zero
        p = V[4]
        
        flux[1] = 0 
        flux[2] = p*bc_norm_vel[1]
        flux[3] = p*bc_norm_vel[2]
        flux[4] = 0
        
    else if (bc_type == BC_INLET_FIXED ||
        bc_type ==  BC_INLET_MOVING ||
        bc_type ==  BC_OUTLET_FIXED || 
        bc_type ==  BC_OUTLET_MOVING) 

        Steger_Warming_Flux!(V::Array{Float64,1}, bc_data::Array{Float64,1}, bc_norm::Array{Float64,1}, 
                            bc_norm_vel::Array{Float64,1}, gamma::Float64, flux::Array{Float64,1})
        

        
        
    else
    end
end