"""
Setup 
"""

using Pkg;

Pkg.activate(".")
Pkg.instantiate()
Pkg.status()

using Distributions
using Random
using Flux
using DiffEqFlux
using Optim
using DifferentialEquations
using Plots 
using LaTeXStrings
gr()

include("simulation.jl")
include("model.jl")
include("plotting.jl")

"""
Define and obtain ground-truth developments
"""

# define true underlying ODE system
function linear_2d_system(du,u,p,t)
    a11, a12, a21, a22 = p
    z1,z2 = u
    du[1] = dz1 = a11 * z1 + a12 * z2
    du[2] = dz2 = a21 * z1 + a22 * z2
end
  
# define initial condition
true_u0 = Float32[2, 1]
# define time span on which to solve the ODE
tspan = (0.0f0, 10.0f0)
# define parameters for the two distinct groups
true_odeparams_group1 = Float32[-0.2, 0.00, 0.00, -0.2]
true_odeparams_group2 = Float32[-0.2, 0.00, 0.00, 0.2]
  
# define corresponding ODE problems for the two groups
prob1 = ODEProblem(linear_2d_system,true_u0,tspan,true_odeparams_group1)
prob2 = ODEProblem(linear_2d_system,true_u0,tspan,true_odeparams_group2)
  
# solve ODE systems to obtain "true" underlying trajectory in each group
dt=0.1
sol_group1 = solve(prob1, Tsit5(), saveat = dt);
sol_group2 = solve(prob2, Tsit5(), saveat = dt);

"""
Simulate data based on true trajectories
"""

# define numbers of individuals and variables 
n = 100 
p = 10
q, q_info = 50, 10

# choose simulation setting for baseline variables 
baseline_simulation = "trueparams" # alternative: "groupsonly"

# set seed for reproducibility
Random.seed!(12);

# generate time dependent variables
xs, tvals, group1, group2 = generate_xs(n, p, true_u0, sol_group1, sol_group2); # default vals: t_start=1.5, t_end=10, maxntps=10, dt=0.1, σ_var=0.1, σ_ind=0.5
# generate baseline variables
if baseline_simulation == "trueparams"
    x_baseline = generate_baseline(n, q, q_info, group1, true_odeparams_group1, true_odeparams_group2); # defaul vals: σ_info=0.1, σ_noise=0.1
elseif baseline_simulation == "groupsonly"
    x_baseline = generate_baseline(n, q, q_info, group1); # default vals: σ_info=1, σ_noise=1
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

"""
Define and train model
"""

zdim = nODEparams = 2
m = init_vae(p, q, zdim, nODEparams, prob1)
L = loss_wrapper(m)
ps = getparams(m)
opt = ADAM(0.0005)
trainingdata = zip(xs, x_baseline, tvals);
evalcb() = @show(mean(L.(xs, x_baseline, tvals)))
evalcb_zs() = eval_z_trajectories(xs, x_baseline, tvals, group1, sol_group1, sol_group2, m, dt)

for i in 1:40
    Flux.train!(L, ps, trainingdata, opt)
    evalcb()
    evalcb_zs()
end

"""
Look at results
"""

plot(allindsplot(2, data, m, sol_group1, sol_group2),
    allindsplot(1, data, m, sol_group1, sol_group2),
    layout = (1,2)
)  