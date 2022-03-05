#------------------------------
# Convenience function for benchmarking
#------------------------------

"""
    generate_all(n, p, q, q_info; seed::Int=12)

Convenience function to generate all simulation data necessary to run the model, 
    with baseline variables simulated based on the group membership (`groupsonly`).

    > ! Use is really for convenience only !
        -- Assumes that `true_u0`, `sol_group1` and `sol_group2`, the initial condition for the 
        ODE and the two true ODE solutions are defined in the global scope !

Inputs: 

    `n`: number of individuals to be simulated 

    `p`: number of time-dependent variables to be simulated 

    `q`: number of baseline variables to be simulated 

    `q_info`: number of informative baseline variables to be simulated 
        (the other `q` - `q_info` variables represent pure noise)

Returns: 

    `xs`: vector of length `n` = n_individuals, where the `i`th element is a (n_vars=p x n_timepoints) matrix 
        containing the time-dependent variables of the `i`th individual in the dataset

    `x_baseline`: vector of length `n` = n_individuals, where the `i`th  element is a vector of length (n_baselinevars=q)
        containing the baseline information for the `i`th individual in the dataset, simulated based on group membership information 

    `tvals`: vector of length `n` = n_individuals, where the `i`th element is a vector of length 1 (or more generally n_timepoints_i)
        containing the time point of the `i`th individual's second measurement (or all the timepoints after the baseline visit)

    `group1`: vector of integers giving the indices of the individuals in group 1

    `group2`: vector of integers giving the indices of the individuals in group 2
"""
function generate_all(n, p, q, q_info; seed::Int=12)
    # set seed for reproducibility
    Random.seed!(seed)
    # choose simulation setting for baseline variables 
    baseline_simulation = "groupsonly" # alternatives: "trueparams" or "groupsonly"
    # generate time dependent variables
    xs, tvals, group1, group2 = generate_xs(n, p, true_u0, sol_group1, sol_group2, maxntps=1, σ_var=0.1, σ_ind=0.1); # default vals: t_start=1.5, t_end=10, maxntps=10, dt=0.1, σ_var=0.1, σ_ind=0.5
    # generate baseline variables
    if baseline_simulation == "trueparams"
        x_baseline = generate_baseline(n, q, q_info, group1, true_odeparams_group1, true_odeparams_group2); # defaul vals: σ_info=0.1, σ_noise=0.1
    elseif baseline_simulation == "groupsonly"
        x_baseline = generate_baseline(n, q, q_info, group1,  σ_info=0.5, σ_noise=0.5); # default vals: σ_info=1, σ_noise=1
    else
        error("Please select either 'trueparams' or 'groupsonly' as mode for simulation of baseline variables")
    end
    return xs, x_baseline, tvals, group1, group2
end

"""
    run_benchmark(trainingdata, m)

Convenience function to wrap the training of the ODE-VAE model `m` on the input data `trainingdata`. 
    Number of epochs is hard-coded to 35, learning rate is set to 0.0005. 

Inputs: 

    `trainingdata`: zipped training data to use as input for the model, obtained in the main script as `zip(xs, x_baseline, tvals,)`
"""
function run_benchmark(trainingdata, m)
    L = loss_wrapper(m)
    ps = getparams(m)
    opt = ADAM(0.0005)
    for epoch in 1:35
        Flux.train!(L, ps, trainingdata, opt)
    end
end

using Printf
"""
    prettymemory(b)

Adapted from source code of some print / summary method in `BenchmarkTools.jl`: 
    converts the memory bytes used `b` into human understandable format: KiB, MiB, GiB. 

Input: 

    `b`: Integer giving the bytes of memory consumed 

Returns:

    a string giving the memory defined by `b` in KiBs, MiBs, or GiB, according to what is the most sensible unit. 
"""
function prettymemory(b)
    if b < 1024
        return string(b, " bytes")
    elseif b < 1024^2
        value, units = b / 1024, "KiB"
    elseif b < 1024^3
        value, units = b / 1024^2, "MiB"
    else
        value, units = b / 1024^3, "GiB"
    end
    return string(@sprintf("%.2f", value), " ", units)
end
