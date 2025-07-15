using ForwardDiff
using Base.Threads

# ??? Future: ASK what platform everyone is using. Julia has Zygote which is GPU accelerated.
# No need to overkill for now.
# grad = Zygote.gradient(a -> f(a, p), a)[1]  # Gradient w.r.t. a

# Inner SGD optimizer as described in the latex (see the SGD-like algo.tex)
# Leave batch_fn off to do non-stochastic GD, and instead use something like:
# batch_fn= () -> rand(..., 1) for full SGD, or change 1 to be minibatch.
# a_init are the variables, p are the PARAMETERS. We do not take partials of parameters.
# Eg consider f(x,y) = cx+dy. a = {x,y}, p={c,d}, ergo gradient is 2D.
# η is the learning rate.
# !NOTE: Here we are making copies of a, instead we may want to update a in place for performance.
function inner_sgd!(f, a_init::Vector, p::Tuple; η=0.01, max_iter=100, batch_fn=nothing)
    a = copy(a_init)
    for _ in 1:max_iter
        minibatch = batch_fn === nothing ? nothing : batch_fn()
        gradient = ForwardDiff.gradient(a -> f(a, p, minibatch), a)
        a .+= η .* gradient
    end

    return a
end

# Outer optimization using parallel best-response dynamics (see the SGD-like algo.tex)
# The arguments are the same as above, except for the one exception of b and p now being vectors of vectors, where each vector 
# inside the vector of vectors belongs to a single agent/element of A.
# !Important, this is one iteration only. You'd wrap this in a for loop to do many iterations.
# !I made this like so in order for you to be able to tune hyperparameters, you probably want to smooth out the η
# !parameter as the algo progresses.
function outer_parallel_sgd!(f, b_init::Vector{Vector{Float64}}, p_init::Vector{Vector{Float64}}; η=0.01, max_iter=100, batch_fn=nothing)
    #!Again, lets maybe update in place
    #!Leaving as a disclaimer for myself, I need to think about race conditions if we are doing some parallelism.
    A = length(b_init)
    b = deepcopy(b_init)
    p = deepcopy(p_init)

    # Update all b_a independently
    for a in 1:A
        b[a] = inner_sgd!(f, b[a], (p[a],); η=η, max_iter=max_iter, batch_fn=batch_fn)
    end

    # Update all p_a independently, conditioned on updated b
    for a in 1:A
        p[a] = inner_sgd!(f, p[a], (b[a],); η=η, max_iter=max_iter, batch_fn=batch_fn)
    end

    return b, p
end



# Same fn as above, more performant as it uses parallelism
# !!! VERY VERY IMPORTANT - this modifies b and p in place unlike the other two.
function outer_parallel_sgd_hpc!(f, b::Vector{Vector{Float64}}, p::Vector{Vector{Float64}}; η=0.01, max_iter=100, batch_fn=nothing)
    A = length(b)

    #!Note to self - I don't think there are races here as the individual vector writes are non-overlapping.
    Threads.@threads for a in 1:A
        b[a] = inner_sgd!(f, b[a], (p[a],); η=η, max_iter=max_iter, batch_fn=batch_fn)
    end

    Threads.@threads for a in 1:A
        p[a] = inner_sgd!(f, p[a], (b[a],); η=η, max_iter=max_iter, batch_fn=batch_fn)
    end

    return b, p
end
