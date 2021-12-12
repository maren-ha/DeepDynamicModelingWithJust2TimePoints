#------------------------------
# Linear ODE system with four unkown parameters
#------------------------------

  
#------------------------------
# setup 
#------------------------------

using Pkg;

# all paths are relative to the repository main folder

Pkg.activate(".")
Pkg.instantiate()
Pkg.status()

using Distributions
using Random
using Flux
using DiffEqFlux
using OrdinaryDiffEq
using Plots 
using LaTeXStrings
gr()

include("../src/simulation.jl")
include("../src/model.jl")
include("../src/batches.jl")
include("../src/plotting.jl")

#------------------------------
# define and obtain ground-truth developments
#------------------------------
  
# define initial condition
true_u0 = Float32[4, 2]
# define time span on which to solve the ODE
tspan = (0.0f0, 10.0f0)
# define parameters for the two distinct groups
true_odeparams_group1 = Float32[-0.2, 0.1, 0.1, -0.2]
true_odeparams_group2 = Float32[-0.2, 0.1, -0.1, 0.25]

# define corresponding ODE problems for the two groups
prob1 = ODEProblem(linear_2d_system,true_u0,tspan,true_odeparams_group1)
prob2 = ODEProblem(linear_2d_system,true_u0,tspan,true_odeparams_group2)
  
# solve ODE systems to obtain "true" underlying trajectory in each group
dt=0.1
sol_group1 = solve(prob1, Tsit5(), saveat = dt);
sol_group2 = solve(prob2, Tsit5(), saveat = dt);

#------------------------------
# simulate data based on true trajectories
#------------------------------

# generate data
n = 100
p = 10
q, q_info = 50, 20
baseline_simulation="trueparams"

Random.seed!(12)#Random.seed!(1234)
xs, tvals, group1, group2 = generate_xs(n, p, true_u0, sol_group1, sol_group2; maxntps=1, σ_ind=0.1); # default vals: t_start=1.5, t_end=10, maxntps=10, dt=0.1, σ_var=0.1, σ_ind=0.5
if baseline_simulation == "trueparams"
    x_baseline = generate_baseline_4params(n, q, q_info, group1, true_odeparams_group1, true_odeparams_group2); # defaul vals: σ_info=0.1, σ_noise=0.1
elseif baseline_simulation == "groupsonly"
    x_baseline = generate_baseline(n, q, q_info, group1); # default vals: σ_info=1, σ_noise=1
else
    error("Please select either 'trueparams' or 'groupsonly' as mode for simulation of baseline variables")
end

#------------------------------
# define model
#------------------------------

zdim = 2
nODEparams = 4
m = init_vae(p, q, zdim, nODEparams, prob1, seed=550) # 543
opt = ADAM(0.0005) # 0.001
nepochs = 10

batchsize = 10
kernel=tricube  

L = batchloss_wrapper(m)
ps = getparams(m)

evalcb_zs() = eval_z_trajectories(xs, x_baseline, tvals, group1, sol_group1, sol_group2, m, dt)

#------------------------------
# train model
#------------------------------

numbers_of_correctly_classified = []

for epoch in 1:10
    distmat = getdistmat_odesols_mean(m, xs, x_baseline; centralise=true);
    minibatches, batch_weights = findminibatches_distmat(distmat, batchsize, kernel);
    #minibatches, batch_weights = randomminibatches(x_baseline,batchsize)
    corr_class_g1, corr_class_g2, corr_class_total  = evaluate_minibatches_pergroup(minibatches,group1,group2);
    @info "epoch " epoch
    @info "group1 : $(round(corr_class_g1, digits=3)) correctly classified"
    @info "group2 : $(round(corr_class_g2, digits=3)) correctly classified"
    @info "in total $(round(corr_class_total, digits=3)) correctly classified"
    batch_xs = collect(xs[minibatches[i]] for i in 1:length(xs));
    batch_x_baseline = collect(x_baseline[minibatches[i]] for i in 1:length(x_baseline));
    batch_tvals = collect(tvals[minibatches[i]] for i in 1:length(tvals));
    batchdata = zip(batch_xs, batch_x_baseline, batch_tvals, batch_weights);
    push!(numbers_of_correctly_classified,[corr_class_g1, corr_class_g2, corr_class_total])
    Flux.train!(L, ps, batchdata, opt)
    @show(mean(L.(batch_xs, batch_x_baseline, batch_tvals, batch_weights)))
    evalcb_zs()
end

#------------------------------
# visualize results
#------------------------------

data = simdata(xs, x_baseline, tvals, group1, group2);

plot(plot_truesolution(2, data, sol_group1, sol_group2, showdata=false), 
    plot_truesolution(1,  data, sol_group1, sol_group2, showdata=false), 
    plot_truesolution(2, data, sol_group1, sol_group2, showdata=true), 
    plot_truesolution(1,  data, sol_group1, sol_group2, showdata=true),
    layout = (2,2),
    legend = false
)

p1 = plot_batch_solution(66, xs, x_baseline, tvals, group1, sol_group1, sol_group2, m, dt)
p2 = plot_batch_solution(87, xs, x_baseline, tvals, group1, sol_group1, sol_group2, m, dt, showlegend=false)

p3 = allindsplot(2, data, m, sol_group1, sol_group2)
p4 = allindsplot(1, data, m, sol_group1, sol_group2, showlegend=false)

plot(p3, p4, layout = (1,2))