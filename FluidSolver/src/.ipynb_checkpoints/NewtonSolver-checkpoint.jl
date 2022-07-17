# Jacobian free Newton Krylov solver
# solve F(Q) = 0
# Qⁿ⁺¹ = Qⁿ - dF/dQ(Qⁿ)^{-1} F(Qⁿ)
# 
# To solve 
# [dF/dQ(Qⁿ) P^{-1}] [P ΔQ] = F(Qⁿ)
# [dF/dQ(Qⁿ) P^{-1}] v =   F(Qⁿ + ϵ P^{-1}v) - F(Qⁿ) / ϵ

using IterativeSolvers
using LinearAlgebra
using SparseArrays

struct MatrixOperator
    F
    Qp
    Fp
    ϵ
    size::Int64
end
Base.size(op::MatrixOperator, i...) = op.size
Base.eltype(op::MatrixOperator) = Float64
function LinearAlgebra.mul!(Ax, self::MatrixOperator, x) 
    F, Fp, Qp, ϵ = self.F, self.Fp, self.Qp, self.ϵ
    Ax .= (F(Qp + ϵ*x) - Fp)/ϵ

    # @info "norm(Ax) = ", norm(Ax)
end
LinearAlgebra.:*(A::MatrixOperator,B::AbstractVector) = (C = similar(B); LinearAlgebra.mul!(C,A,B))



# A = [2.0 3.0; 1.0 2.0]
# b= [1.0 ; 2.0]
# Pr = [2.0 0.0; 0.0 2.0]
# ΔW = [0.0; 0.0]
# @info "Test gres!"
# _, history = gmres!(ΔW, A, b)
# @assert(norm(A * ΔW - b ) < 1.0e-8)

# @info "Test gres! with right preconditioner"
# gmres!(ΔW, A, b; Pr = Pr)
# @assert(norm(A * ΔW - b ) < 1.0e-8)



# F = (x)-> A*x
# n = length(b)
# Qp = zeros(n)
# Fp = F(Qp)
# ϵ = 1e-3
# opA = MatrixOperator(F, Qp, Fp, ϵ, n)
# @info "Test gres! with MatrixOperator"
# gmres!(ΔW, opA, b;)
# @assert(norm(A * ΔW - b ) < 1.0e-8)

# @info "Test gres! with MatrixOperator and right preconditioner"
# Pr = sparse(Pr)
# gmres!(ΔW, opA, b; Pr = (Pr))
# @assert(norm(A * ΔW - b ) < 1.0e-8)
