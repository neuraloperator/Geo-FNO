__precompile__(true)
module FluidSolver

using Revise
using LinearAlgebra
using Statistics 
using JLD2
using MAT
using PyPlot
using ForwardDiff
using SparseArrays
using IterativeSolvers

include("NewtonSolver.jl")

include("Flux.jl")
include("NodalGrad.jl")
include("Limiter.jl")

include("QuadMesh.jl")

include("NSSolver.jl")

end