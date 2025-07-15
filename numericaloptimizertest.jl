using LinearAlgebra

include("numericaloptimizers.jl")

# Simple test problem: minimize f(a, p) = ||a - p||^2 (negative to turn gradient ascent into descent)
function f(a, p, batch=nothing)
    return -sum((a .- p[1]).^2)  # Use p[1] because p is a 1-tuple
end


A = 100
dim = 100                      
b_init = [randn(dim) for _ in 1:A]
p_init = [randn(dim) for _ in 1:A]

batch_fn = () -> rand(1:A, 1) #implements batch size 1.
b_final, p_final = outer_parallel_sgd_hpc!(f, b_init, p_init; η=0.1, max_iter=500, batch_fn=batch_fn)


for a in 1:A
    δ = norm(b_final[a] - p_init[a])
    println("Agent $a: ‖b - p‖ = ", round(δ, digits=10))
end