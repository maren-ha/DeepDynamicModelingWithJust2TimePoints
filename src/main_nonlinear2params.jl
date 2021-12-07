#------------------------------
# Non-linear ODE system with two unkown parameters
#------------------------------


#------------------------------
# Setup 
#------------------------------

using Pkg;

# all paths are relative to the repository main folder

Pkg.activate("src/.")
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

include("simulation.jl")
include("model.jl")
include("plotting.jl")

#------------------------------
# Define and obtain ground-truth developments
#------------------------------

true_u0 = Float32[2, 2] 
tspan = (0.0f0, 10.0f0)
true_odeparams_group1 = Float32[1.0, 1.0, 1.0, 0.5] 
true_odeparams_group2 = Float32[0.5, 1.0, 1.0, 2.0]

lvprob1 = ODEProblem(lotka_volterra, true_u0, tspan, true_odeparams_group1)
lvprob2 = ODEProblem(lotka_volterra, true_u0, tspan, true_odeparams_group2)

# true trajectory of each group
dt=0.1
sol_group1 = solve(lvprob1, Tsit5(), saveat = dt);
sol_group2 = solve(lvprob2, Tsit5(), saveat = dt);
#plot(sol_group1)
#plot(sol_group2)

#------------------------------
# Simulate data based on true trajectories
#------------------------------

n = 200
p = 10
q, q_info = 50, 20 
baseline_simulation = "groupsonly"

Random.seed!(1234)
xs, tvals, group1, group2 = generate_xs(n, p, true_u0, sol_group1, sol_group2, t_start=0.0, maxntps=1, σ_var=0.1, σ_ind=0.1); # default vals: t_start=1.5, t_end=10, maxntps=10, dt=0.1, σ_var=0.1, σ_ind=0.5
if baseline_simulation == "trueparams"
    x_baseline = generate_baseline(n, q, q_info, group1, true_odeparams_group1, true_odeparams_group2); # defaul vals: σ_info=0.1, σ_noise=0.1
elseif baseline_simulation == "groupsonly"
    x_baseline = generate_baseline(n, q, q_info, group1,  σ_info=0.5, σ_noise=0.5); # default vals: σ_info=1, σ_noise=1
else
    error("Please select either 'trueparams' or 'groupsonly' as mode for simulation of baseline variables")
end

# look at data: 
data = simdata(xs, x_baseline, tvals, group1, group2);

plot(plot_truesolution(2, data, sol_group1, sol_group2, showdata=false), 
    plot_truesolution(1,  data, sol_group1, sol_group2, showdata=false), 
    plot_truesolution(2, data, sol_group1, sol_group2, showdata=true), 
    plot_truesolution(1,  data, sol_group1, sol_group2, showdata=true),
    layout = (2,2),
    legend = false
)

#------------------------------
# Define and train model
#------------------------------

zdim = nODEparams = 2
m = init_vae(p, q, zdim, nODEparams, lvprob2, seed=84)
L = loss_wrapper(m)
ps = getparams(m)
opt = ADAM(0.0001)
trainingdata = zip(xs, x_baseline, tvals);
evalcb() = @show(mean(L.(xs, x_baseline, tvals)))
evalcb_zs() = eval_z_trajectories(xs, x_baseline, tvals, group1, sol_group1, sol_group2, m, dt)

for epoch in 1:40
    Flux.train!(L, ps, trainingdata, opt)
    evalcb()
    evalcb_zs()
end

#------------------------------
# Look at results
#------------------------------

plot(allindsplot(2, data, m, sol_group1, sol_group2),
    allindsplot(1, data, m, sol_group1, sol_group2, showlegend=false),
    layout = (1,2)
)  

# individual plots 
p1 = plot_individual_solutions(46, xs, x_baseline, tvals, group1, sol_group1, sol_group2, m, dt)
p2 = plot_individual_solutions(182, xs, x_baseline, tvals, group1, sol_group1, sol_group2, m, dt, showlegend=false)

# solutions across the entire dataset
p3 = allindsplot(2, data, m, sol_group1, sol_group2)
p4 = allindsplot(1, data, m, sol_group1, sol_group2, showlegend=false)
plot(p3,p4, layout = (1,2))