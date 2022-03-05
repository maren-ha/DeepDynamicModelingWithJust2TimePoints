#------------------------------
# functions to simulate data 
#------------------------------

"""
    generate_xs(n, p, true_u0, sol_group1, sol_group2; t_start=1.5, t_end=10, maxntps = 10, dt=0.1, σ_var=0.1, σ_ind=0.5)

Generates simulated data by sampling `n` observations of `p` variables at between 1 and `maxntps` timepoints for each individual 
    by randomly selecting one of the true underlying ODE solutions given by `sol_group1` and `sol_group2`, taking its values 
    at a randomly sampled number between 1 and `maxntps` of randomly sampled time points and adding variable-specific 
    and individual-specific errors to the values of the true trajectories, where the variance of the error terms is controlled by 
    `σ_var` and `σ_ind`. 

Inputs: 

    `n`: number of individuals to simulate 

    `p`: number of time-dependent variables to simulate - should be divisible by the number of the true underlying 
        trajectory dimensions, so the first (p/n_true_dimensions) variables can be noisy versions of the first dimension of 
        the true dynamics, and so on. 

    `true_u0`: vector stating the initial condition of the ground-truth underlying ODE systems from which to simulate the data 

    `sol_group1`: true ODE solution of the first group 

    `sol_group2`: true ODE solution of the second group 

Optional Keyword arguments: 

    `t_start`: Earliest time point possible for follow-up measurements, start of the interval from which to sample the 
        subsequent measurement time point(s). Default = 1.5

    `t_end`: Latest time point possible for follow-up measurements, end of the interval from which to sample the 
        subsequent measurement time point(s). Default = 10

    `maxntps`: maximum number of time points per individual after the baseline timepoint. Default = 1

    `dt`: time steps at which to solve the ODE. Needed to ensure correct array sizes. Default = 0.1

    `σ_var`: variance with which to sample the variable-specific error terms. Default = 0.1

    `σ_ind`: variance with which to sample the individual-specific error terms. Default = 0.5

Returns:

    `xs`: vector of length `n` = n_individuals, where the `i`th element is a (n_vars=p x n_timepoints) matrix 
        containing the simulated values of the time-dependent variables of the `i`th individual in the dataset

    `tvals`: vector of length `n` = n_individuals, where the `i`th element is a vector of length 1 (or more generally n_timepoints_i)
        containing the simulated time point of the `i`th individual's second measurement (or all the timepoints after the baseline visit)

    `group1`: indices of all individuals in group1 

    `group2`: indices of all individuals in group1 
"""
function generate_xs(n, p, true_u0, sol_group1, sol_group2; t_start=1.5, t_end=10, maxntps = 10, dt=0.1, σ_var=0.1, σ_ind=0.5)

    # (1) generate artifical group labels
    groups = zeros(n)
    groups[randperm(n)[1:Int(floor(n/2))]] .= 1 # for groups of the same size
    group1 = findall(x -> x==1, groups)
    group2 = findall(x -> x==0, groups)

    # (2) generate artificial time stamps
    ntps = rand(1:maxntps, n)
    tvals = [sort(rand(t_start:dt:t_end,ntps[i])) for i in 1:n]

    # (3) obtain true values as solutions of the ODEs at the initial time point and the drawn second time point 
    # check for equal number of variables:
    if p%2 != 0
        error("Please select an even number of variables")
    end
    # true starting point    
    z_t0_p1 = true_u0[1] # for variables 1-(p/2)
    z_t0_p2 = true_u0[2] # for variables (p/2+1)-p
    z_t0 = repeat([z_t0_p1, z_t0_p2], inner=Int(p/2))

    # now use ODE solutions to obtain true temporal development value
    # for all individuals in both variables u1 and u2
    z_later_ts = collect((i ∈ group1) ? (Array(sol_group1)[:,Int.(round.(tvals[i].*(1 ./dt)).+1)]) : (Array(sol_group2)[:,Int.(round.(tvals[i].*(1 ./dt)).+1)]) for i in 1:n)

    # (4) sample variable- specific and individual-specific errors at both time points
    # variable specific random effect (general difficulty measuring that specific variable)
    us = rand(Normal(0,σ_var),p) 

    xs = []
    for i in 1:n 
        # make time series structure, should have shape (p x ntps[i])
        cur_timeseries = zeros(p, ntps[i]+1)
        for j in 1:p 
            cur_timeseries[j,1] = z_t0[j] + us[j] + randn() .* σ_ind
            for tp in 1:ntps[i]
                if j <= Int(p/2)
                    cur_timeseries[j,tp+1] = z_later_ts[i][1,tp] + us[j] + randn() .* σ_ind
                else
                    cur_timeseries[j,tp+1] = z_later_ts[i][2,tp] + us[j] + randn() .* σ_ind
                end
            end
        end
        push!(xs, cur_timeseries)
    end

    return xs, tvals, group1, group2
end

"""
    generate_baseline(n, q, q_info, group1; σ_info=1, σ_noise=1)

Generates simulated baseline data by sampling `n` observations of `q` baseline variables, of which only the first `q_info`
    are informative, and the other ones are just pure noise variables, based on the group membership information. 
    This information is given by `group1`, the indices of all individuals in group1, based on which the other indices in 
    group 2 can be inferred, since union(group1, group2) = {1,...,n}. 
    Baseline measurements are simulated by encoding group membership as 1 or -1 and drawing from N(0,σ_info) or N(1, σ_info),
    repectively. For the noise variables, data are simulated by drawing from N(0, σ_noise). 

Inputs: 

    `n`: number of individuals to simulate 

    `q`: number of baseline variables to simulate 

    `q_info`: number of informative baseline variables. 

    `group1`: indices of all individuals in group1 - since [group1, group2] = {1,...,n}, the `group2` indices can be inferred from that

Optional keyword arguments: 

    `σ_info`: variance with which to sample from the group membership information in the informative baseline variables terms. Default = 1

    `σ_noise`: variance with which to sample the noise baseline variables terms. Default = 1

Returns:

    `x_params`: vector of length `n` = n_individuals, where the `i`th  element is a vector of length (n_baselinevars=q)
        containing the baseline information for the `i`th individual in the dataset 
"""
function generate_baseline(n, q, q_info, group1; σ_info=1, σ_noise=1)
    zs = fill(1,(n,1))
    zs[group1].= -1
    means = fill(0,n,q)
    means[:,1:q_info] .= zs

    vars=fill(σ_noise,q)
    vars[1:q_info] .= σ_info
    x_params = [cat([rand(Normal(means[i,j],vars[j])) for j in 1:q]..., dims=1) for i in 1:n]
    return x_params
end

"""
    generate_baseline(n, q, q_info, group1, true_odeparams_group1, true_odeparams_group2; σ_info=0.1, σ_noise=0.1)

Generates simulated baseline data by sampling `n` observations of `q` baseline variables, of which only the first `q_info`
    are informative, and the other ones are just pure noise variables, based on the true ODE parameters passed as 
    `true_odeparams_group1` and `true_odeparams_group2`. 
    Baseline measurements are simulated by sampling from the true parameters with a standard deviation of σ_info. 
    For the noise variables, data are simulated by drawing from N(0, σ_noise). 

Inputs: 

    `n`: number of individuals to simulate 

    `q`: number of baseline variables to simulate 

    `q_info`: number of informative baseline variables. 

    `group1`: indices of all individuals in group1 - since [group1, group2] = {1,...,n}, the `group2` indices can be inferred from that

Optional keyword arguments: 

    `σ_info`: variance with which to sample from the group membership information in the informative baseline variables terms. Default = 0.1. 

    `σ_noise`: variance with which to sample the noise baseline variables terms. Default = 0.1. 

Returns:

    `x_params`: vector of length `n` = n_individuals, where the `i`th  element is a vector of length (n_baselinevars=q)
        containing the baseline information for the `i`th individual in the dataset 
"""
function generate_baseline(n, q, q_info, group1, true_odeparams_group1, true_odeparams_group2; σ_info=0.1, σ_noise=0.1)
    signs = fill(1,(n,1))
    signs[group1] .= -1
    z1s = zeros(n,1)
    z1s[signs .== -1] .= true_odeparams_group1[1]
    z1s[signs .== 1] .= true_odeparams_group2[1]
    z2s = zeros(n,1)
    z2s[signs .== -1] .= true_odeparams_group1[4]
    z2s[signs .== 1] .= true_odeparams_group2[4]

    means = zeros(n,q)
    means[:,1:Int(floor(q_info/2))] .= z1s
    means[:,Int(floor(q_info/2))+1:q_info] .=z2s

    vars=fill(σ_noise,q)
    vars[1:q_info] .= σ_info
    x_params = [cat([rand(Normal(means[i,j],vars[j])) for j in 1:q]..., dims=1) for i in 1:n]
    return x_params
end

"""
    generate_baseline_4params(n, q, q_info, group1, true_odeparams_group1, true_odeparams_group2; σ_info=0.1, σ_noise=0.1)

Version of `generate_baseline(...)` for 4 ODE parameters to be learned and encoded in the baseline variables, not just 2.
    Generates simulated baseline data by sampling `n` observations of `q` baseline variables, of which only the first `q_info`
    are informative, and the other ones are just pure noise variables, based on the true ODE parameters passed as 
    `true_odeparams_group1` and `true_odeparams_group2`. 
    Baseline measurements are simulated by sampling from the true parameters with a standard deviation of σ_info. 
    For the noise variables, data are simulated by drawing from N(0, σ_noise). 

Inputs: 

    `n`: number of individuals to simulate 

    `q`: number of baseline variables to simulate 

    `q_info`: number of informative baseline variables. 

    `group1`: indices of all individuals in group1 - since [group1, group2] = {1,...,n}, the `group2` indices can be inferred from that

Optional keyword arguments: 

    `σ_info`: variance with which to sample from the group membership information in the informative baseline variables terms. Default = 0.1

    `σ_noise`: variance with which to sample the noise baseline variables terms. Default = 0.1

Returns:

    `x_params`: vector of length `n` = n_individuals, where the `i`th  element is a vector of length (n_baselinevars=q)
        containing the baseline information for the `i`th individual in the dataset 
"""
function generate_baseline_4params(n, q, q_info, group1, true_odeparams_group1, true_odeparams_group2; σ_info=0.1, σ_noise=0.1)
    signs = fill(1,n)
    signs[group1] .= -1
    zs = cat(repeat([true_odeparams_group2], n)..., dims=2)
    zs[:,findall(signs .==-1)] .= true_odeparams_group1
    zs = zs'

    means = zeros(n,q)
    means[:,1:Int(floor(q_info/4))] .= zs[:,1]
    means[:,Int(floor(q_info/4))+1:Int(floor(q_info/2))] .= zs[:,2]
    means[:,Int(floor(q_info/2))+1:Int(floor(q_info*3/4))] .= zs[:,3]
    means[:,Int(floor(q_info*3/4))+1:q_info] .= zs[:,4]

    vars=fill(σ_noise, q)
    vars[1:q_info] .= σ_info
    x_params = [[rand(Normal(means[i,j],vars[j])) for j in 1:q] for i in 1:n]
    return x_params
end