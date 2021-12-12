#------------------------------
# Runtime and complexity analysis  
# for linear ODE system with two unkown parameters
#------------------------------


#------------------------------
# Setup 
#------------------------------

using Distributed

addprocs(5)

@everywhere using Pkg; 
#@everywhere cd("/Users/imbi-mac-102/Desktop/ManuscriptMasterThesis/code/") # omit or change accordingly on server 
@everywhere Pkg.activate(".")
Pkg.instantiate()

@everywhere using BenchmarkTools
@everywhere using DataFrames
@everywhere using Distributed
@everywhere using Distributions
@everywhere using Random
@everywhere using Flux
@everywhere using DiffEqFlux
@everywhere using OrdinaryDiffEq
@everywhere using SharedArrays

@everywhere include("../src/simulation.jl")
@everywhere include("../src/model.jl")
@everywhere include("../src/benchmarking.jl")

#------------------------------
# Define and obtain ground-truth developments
#------------------------------

# define initial condition
@everywhere true_u0 = Float32[2, 1]
# define time span on which to solve the ODE
@everywhere tspan = (0.0f0, 10.0f0)
# define parameters for the two distinct groups
@everywhere true_odeparams_group1 = Float32[-0.2, 0.00, 0.00, -0.2]
@everywhere true_odeparams_group2 = Float32[-0.2, 0.00, 0.00, 0.2]
  
# define corresponding ODE problems for the two groups
@everywhere prob1 = ODEProblem(linear_2d_system,true_u0,tspan,true_odeparams_group1)
@everywhere prob2 = ODEProblem(linear_2d_system,true_u0,tspan,true_odeparams_group2)
  
# solve ODE systems to obtain "true" underlying trajectory in each group
@everywhere dt=0.1
@everywhere sol_group1 = solve(prob1, Tsit5(), saveat = dt);
@everywhere sol_group2 = solve(prob2, Tsit5(), saveat = dt);


#------------------------------
# Train model using benchmarking 
#------------------------------

# define number of observations, variables and baseline variables to try 
@everywhere n_obs = [50, 100, 250, 500, 1000, 2000, 5000]
@everywhere n_vars = [10, 20, 50, 100, 200]
@everywhere n_baselinevars = [10, 20, 50, 100, 200]
@everywhere lenobs, lenvars, lenbvars = length(n_obs), length(n_vars), length(n_baselinevars)

# construct dataframe: n, p, q, time, memory, allocations
benchmarkdf = DataFrame(n_obs = cat(n_obs, fill(100, lenvars + lenbvars), dims=1),
        n_vars = cat(fill(10, lenobs), n_vars, fill(10, lenbvars), dims=1),
        n_baselinevars = cat(fill(50, lenobs + lenvars), n_baselinevars, dims=1),
        time = fill(0.0, lenobs+lenvars+lenbvars),
        gctime = fill(0.0, lenobs+lenvars+lenbvars),
        memory = fill(0, lenobs+lenvars+lenbvars),
        allocs = fill(0, lenobs+lenvars+lenbvars)
)

# get it to a shared array for distributed computing
benchmarkarray = SharedArrays.SharedMatrix{Float64}(size(Matrix(benchmarkdf)));
benchmarkarray[:,1:3] = Matrix(benchmarkdf)[:,1:3];

@everywhere eval($benchmarkarray);

# scenario 1: fixed number of time-dep and baseline variables, varying number of observations 
@sync @distributed for n_ind in 1:lenobs
    # warmup (first run takes longer because of precompilation times and shouldnt be included)
    n_warmup, p_warmup, q_warmup, q_info_warmup = 100, 10, 10, 10
    xs, x_baseline, tvals, group1, group2 = generate_all(n_warmup, p_warmup, q_warmup, q_info_warmup);
    trainingdata = zip(xs, x_baseline, tvals);
    zdim = nODEparams = 2
    m = init_vae(p_warmup, q_warmup, zdim, nODEparams, prob1)
    L = loss_wrapper(m)
    ps = getparams(m)
    opt = ADAM(0.0005)
    for epoch in 1:35
        Flux.train!(L, ps, trainingdata, opt)
    end
    println("warmup done")
    # now start for real
    n, p, q = n_obs[n_ind], 10, 50 
    println("n=$n, p=$p, q=$q")
    q_info = Int(q/5)
    xs, x_baseline, tvals, group1, group2 = generate_all(n, p, q, q_info);
    trainingdata = zip(xs, x_baseline, tvals);
    zdim = nODEparams = 2
    m = init_vae(p, q, zdim, nODEparams, prob1)
    b = @benchmark run_benchmark($trainingdata, $m) samples=1 evals=1
    println("training done")
    row = n_ind
    benchmarkarray[row,4] = b.times[1] # times
    benchmarkarray[row,5] = b.gctimes[1] # gctimes 
    benchmarkarray[row,6] = b.memory # memory 
    benchmarkarray[row,7] = b.allocs # allocations
end

# scenario 2: fixed number of observations and baseline variables, varying number of time-dependent variables
@sync @distributed for p_ind in 1:lenvars
    n, p, q = 100, n_vars[p_ind], 50 
    println("n=$n, p=$p, q=$q")
    q_info = Int(q/5)
    xs, x_baseline, tvals, group1, group2 = generate_all(n, p, q, q_info);
    trainingdata = zip(xs, x_baseline, tvals);
    zdim = nODEparams = 2
    m = init_vae(p, q, zdim, nODEparams, prob1)
    b = @benchmark run_benchmark($trainingdata, $m) samples=1 evals=1
    println("training done")
    row = lenobs + p_ind
    benchmarkarray[row,4] = b.times[1] # times
    benchmarkarray[row,5] = b.gctimes[1] # gctimes 
    benchmarkarray[row,6] = b.memory # memory 
    benchmarkarray[row,7] = b.allocs # allocations
end

# scenario 3: fixed number of observations and time-dependent variables, varying number of baseline variables
@sync @distributed for q_ind in 1:lenbvars
    n, p, q = 100, 10, n_baselinevars[q_ind]
    println("n=$n, p=$p, q=$q")
    q_info = Int(q/5)
    xs, x_baseline, tvals, group1, group2 = generate_all(n, p, q, q_info);
    trainingdata = zip(xs, x_baseline, tvals);
    zdim = nODEparams = 2
    m = init_vae(p, q, zdim, nODEparams, prob1)
    b = @benchmark run_benchmark($trainingdata, $m) samples=1 evals=1
    println("training done")
    row = lenobs + lenvars + q_ind
    benchmarkarray[row,4] = b.times[1] # times
    benchmarkarray[row,5] = b.gctimes[1] # gctimes 
    benchmarkarray[row,6] = b.memory # memory 
    benchmarkarray[row,7] = b.allocs # allocations
end

# if desired: save as JLD2 file 
using JLD2 
JLD2.@save "results/benchmarkresults.jld2" benchmarkarray
# and re-load from saved
JLD2.@load "results/benchmarkresults.jld2" 
benchmarkarray = eval(:benchmarkarray)

# copy back to dataframe, to be saved later as CSV
benchmarkdf[:,:time] = benchmarkarray[:,4]
benchmarkdf[:,:gctime] = benchmarkarray[:,5]
benchmarkdf[:,:memory] = benchmarkarray[:,6]
benchmarkdf[:,:allocs] = benchmarkarray[:,7]
benchmarkdf[:,:time_in_seconds] = round.(benchmarkarray[:,4] .* 1e-9, digits=3)

# turn memory into human-readable format (taken from BenchmarkTools.jl source code)
benchmarkdf[:,:prettymemory] = prettymemory.(benchmarkarray[:,6])

# save entire dataframe as CSV
using CSV 
CSV.write("results/benchmarkresults.csv", benchmarkdf)

# extract tables as in manuscript appendix and save as CSV files

# different number of observations for fixed p (10) and q (50)
rows_obs = findall(x -> x.n_vars == 10 && x.n_baselinevars == 50, eachrow(benchmarkdf))
times_obs = benchmarkdf[rows_obs, [:n_obs, :time_in_seconds, :prettymemory]]
CSV.write("results/benchmarkresults_obs.csv", times_obs)

rows_vars = findall(x -> x.n_obs == 100 && x.n_baselinevars == 50, eachrow(benchmarkdf))
times_vars = benchmarkdf[rows_vars, [:n_vars, :time_in_seconds, :prettymemory]]
CSV.write("results/benchmarkresults_vars.csv", times_vars)

rows_bvars = findall(x -> x.n_obs == 100 && x.n_vars == 10, eachrow(benchmarkdf))
times_bvars = benchmarkdf[rows_bvars, [:n_baselinevars, :time_in_seconds, :prettymemory]]
CSV.write("results/benchmarkresults_baselinevars.csv", times_bvars)
