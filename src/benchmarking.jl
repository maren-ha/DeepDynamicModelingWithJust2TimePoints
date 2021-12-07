#------------------------------
# Convenience function for in benchmarking
#------------------------------

# define convenience function to generate data for simpler benchmark loop
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

function run_benchmark(trainingdata, m)
    L = loss_wrapper(m)
    ps = getparams(m)
    opt = ADAM(0.0005)
    for epoch in 1:35
        Flux.train!(L, ps, trainingdata, opt)
    end
end

using Printf
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
