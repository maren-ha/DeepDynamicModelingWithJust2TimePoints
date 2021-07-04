"""
functions to simulate data 
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

#= # not yet adapted to multiple time points! 
function generate_xs_nonlinear(n, p, true_u0, sol_group1, sol_group2; t_start=0, t_end=10, dt=0.1, σ_var=0.1, σ_ind=0.1)

    # (1) generate artifical group labels
    groups = zeros(n)
    groups[randperm(n)[1:Int(floor(n/2))]] .= 1 # for groups of the same size
    group1 = findall(x -> x==1, groups)
    group2 = findall(x -> x==0, groups)

    # (2) generate artificial time stamps
    tvals = [rand(t_start:dt:t_end) for i in 1:n]

    # (3) obtain true values as solutions of the ODEs at the initial time point and the drawn second time point 
    # check for equal number of variables:
    if p%2 != 0
        error("Please select an even number of variables")
    end
    # true starting point
    z_t0_p1 = true_u0[1]     # for variables 1-p1
    z_t0_p2 = true_u0[2]     # for variables (p1+1)-p 
    z_t0 = repeat([z_t0_p1, z_t0_p2], inner=Int(p/2))

    # now solve ODE system to obtain true temporal development value
    # for all individuals in both variables u1 and u2
    z_t1 = collect((i ∈ group1) ? (sol_group1[Int(round(tvals[i]*(1/dt))+1)]) : (sol_group2[Int(round(tvals[i]*(1/dt))+1)]) for i in 1:n)
    z_t1_p1 = collect(z_t1[i][1] for i in 1:n)
    z_t1_p2 = collect(z_t1[i][2] for i in 1:n)
    z_t1_mat = cat(repeat(z_t1_p1,1,Int(p/2)), repeat(z_t1_p2,1,Int(p/2)), dims=2)

    # (4) sample variable- specific and individual-specific errors at both time points
    # variable specific random effect (general difficulty measuring that specific variable)
    us_t0 = [rand(Normal(0,σ_var)) for i in 1:p]
    us_t1 = [rand(Normal(0,σ_var)) for i in 1:p]

    # individual specific measurement error
    eps_t0 = randn(n,p) .* σ_ind#0.5#0.1
    eps_t1 = randn(n,p) .* σ_ind#0.5#0.1

    # add those to the true values and combine everything 
    # combine all to xs: x_ij(t) = z(t) + u_j + eps_ij; i=1,...,n, j=1,...,p
    x_t0 = cat([cat([z_t0[j] .+ us_t0[j] .+ eps_t0[i,j] for i in 1:n]..., dims=1) for j in 1:p]..., dims=2)
    x_t1 = cat([cat([z_t1_mat[i,j] .+ us_t1[j] .+ eps_t1[i,j] for i in 1:n]..., dims=1) for j in 1:p]..., dims=2)
    # (5) combine everything into one array 
    xs = [hcat(x_t0[i,:], x_t1[i,:]) for i in 1:size(x_t0,1)]
    return xs, tvals, group1, group2
end
=# 