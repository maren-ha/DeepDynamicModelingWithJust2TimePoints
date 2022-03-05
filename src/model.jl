#------------------------------
# functions to define and manipulate ODE-VAE model 
#------------------------------

#------------------------------
# ODE systems
#------------------------------

"""
    linear_2d_system(du,u,p,t)

Defines a 2D system of linear ODEs with parameters `p` in-place: 
    u'(t) = A * u(t), u(t0)=u0, where the system matrix A is constructed from the parameter vector `p`.
    Written to align with the interface of `DifferentialEquations.jl` to be used to construct and solve `ODEProblems` later on. 

Inputs: 

    `du`: left-hand side of the ODE u'(t) = A * u(t), where A is a 2x2 matrix, and u is a 2-dimensional vector 

    `u`: function for which to solve the ODE 

    `p`: parameters of the ODE system, in this case the system matrix, has to be passed as vector 

    `t`: has to be defined as argument, time to solve the ODE for, is needed later in the `DifferentialEquations.ODEProblem` interface

Returns: 

    `du`: left-hand side of the ODE u'(t) = A * u(t), where A is a 2x2 matrix, and u is a 2-dimensional vector 
"""
function linear_2d_system(du,u,p,t)
    a11, a12, a21, a22 = p
    z1,z2 = u
    du[1] = dz1 = a11 * z1 + a12 * z2
    du[2] = dz2 = a21 * z1 + a22 * z2
end

"""
    lotka_volterra(du,u,p,t)

Defines a 2D Lotka-Volterra ODE system with parameters `p` in-place: 
    u1'(t) = α * u1(t) - β*u1(t)u2(t), u2'(t) = γ*u1(t)*u2(t) - δ*u2(t), u(t0)=u0. 
    Written to align with the interface of `DifferentialEquations.jl` to be used  to construct and solve `ODEProblems` later on. 

Inputs: 

    `du`: left-hand side of the ODE defined above 

    `u`: function for which to solve the ODE 

    `p`: parameters of the ODE system, inthis case `p` = α, β, γ, δ, has to be passed as vector 

    `t`: has to be defined as argument, time to solve the ODE for, is needed later in the `DifferentialEquations.ODEProblem` interface

Returns: 

    `du`: left-hand side of the ODE defined above 
"""
function lotka_volterra(du,u,p,t)
    α, β, γ, δ = p
    z1,z2 = u
    du[1] = dz1 = α*z1 - β*z1*z2 #prey; α: growth, β: meeting rate of prey&predator=predation rate
    du[2] = dz2 = γ*z1*z2 - δ*z2 #predator; γ: growth of predator, δ: loss rate of predator
end


#------------------------------
# define and initialize model 
#------------------------------

mutable struct odevae
    p 
    q
    zdim 
    paramNN
    encoder
    encodedμ 
    encodedlogσ 
    decoder
    decodedμ 
    decodedlogσ 
    ODEprob
end

"""
    downscaled_glorot_uniform(dims...)

Creates an array of size (`dims`) of small random numbers to be used as initialization for neural network weights. 
    Adapted from the `Flux.glorot_uniform()` function used as default for weight initialization of neural networks, 
    but with a rescaling factor to make the initial weights smaller. 
    - this helps to prevent solver instabilities when training ODE-VAEs with Lotka-Volterra ODE systems 
"""
downscaled_glorot_uniform(dims...) = (rand(Float32, dims...) .- 0.5f0) .* sqrt(0.01f0/sum(dims)) # smaller weights initialisation

"""
    init_vae(p, q, zdim, nODEparams, ODEprob; seed::Int=1234)

Initializes an ODE-VAE model. 

Inputs: 

    `p`: number of time-dependent variables 

    `q`: number of baseline variables 

    `zdim`: dimension of the VAE latent space 

    `nODEparams:` number of ODE parameters to be learned (2 or 4 depending on scenario)

    `ODEprob`: a `DifferentialEquations.ODEProblem`: the ODE problem that should be solved in the latent space of the ODE-VAE

    `seed`: optional keyword argument: which seed to use for the random weight initialization, to ensure reproducibility. 
        Default = 1234

Returns: 

    `model` - an ODE-VAE model as instance of the `odevae` struct 
"""
function init_vae(p, q, zdim, nODEparams, ODEprob; seed::Int=1234)
    myinit = ODEprob.f.f == lotka_volterra ? downscaled_glorot_uniform : Flux.glorot_uniform
    shift(arg) = ODEprob.f.f == lotka_volterra ? tanh.(arg) .+ 1 : sigmoid(arg).-0.5
    # seed
    Random.seed!(seed)
    # parameter network
    paramNN = Chain(Dense(q,q,tanh, init=myinit), 
                Dense(q, nODEparams, arg ->(shift(arg)), init=myinit), 
                #Dense(q, nODEparams, arg -> (2 ./(1 .+ exp.(-arg))) .- 1.0), 
                Flux.Diagonal(nODEparams))
    #   VAE encoder
    Dz, Dh = zdim, p
    encoder, encodedμ, encodedlogσ = Dense(p, Dh, arg ->(tanh.(arg) .+ 1)), Dense(Dh, Dz), Chain(Dense(Dh, Dz, arg -> -relu(arg)), Flux.Diagonal(Dz))
    # VAE decoder
    decoder, decodedμ, decodedlogσ = Dense(Dz, Dh, tanh), Dense(Dh, p), Dense(Dh, p)

    model = odevae(p, q, zdim, paramNN, encoder, encodedμ, encodedlogσ, decoder, decodedμ, decodedlogσ, ODEprob)
    return model
end

#------------------------------
# define model loss 
#------------------------------

"""
    latentz(μ, logσ)

Samples from the posterior of the latent variable Z, an isotropic Gaussian parameterized
    by the encoder outputs μ and logσ using the reparametrization trick.

Inputs: 

    `μ`: scalar or array of means as outputted by the encoder, 
        parameterizing the mean of the variational Gaussian posterior 

    `logσ`: scalar or array outputted by the encoder, 
        parameterizing the log of the standard deviation of the variational Gaussian posterior

Returns: 

    a sample from N(μ, σ) drawn according to the reparametrization trick (Kingma and Welling, 2014)
"""
latentz(μ, logσ) = μ .+ sqrt.(exp.(logσ)) .* randn(Float32,size(μ)...) 

"""
    kl_q_p(μ, logσ)

Calculates the KL-divergence between the standard normal distribution N(0,1) used as a prior for the 
    latent Z, and the variational posterior q_{Z∣x}(z) given by N(μ, σ), with μ and logσ parameterized by the encoder.

Inputs: 

    `μ`: array of means of size (`p` x n_timepoints) as outputted by the encoder, 
        parameterizing the mean of the variational Gaussian posterior 

    `logσ`: array of size (`p` x n_timepoints) outputted by the encoder, 
        parameterizing the log of the standard deviation of the variational Gaussian posterior

Returns: 

    the sum of the analytical values of the KL divergence between N(0,1) and N(μ, σ) across timepoints, 
        i.e., a vector of length (n_timedepvars)
"""
kl_q_p(μ, logσ) = 0.5 .* sum(exp.(logσ) + μ.^2 .- 1 .- (logσ),dims=1)

"""
    logp_x_z(m::odevae, x, z)

Evaluates the log-likelihood of the sample `x` under the distribution of p_{X∣z}(x) parameterized 
    by the decoder of the ODE-VAE model `m` based on a sample `z` of the latent variable Z. 

Inputs: 

    `m`: ODE-VAE model 

    `x`: individual data sample of size (`p` x n_timepoints) 

    `z`: sample from the latent representation of `m` of size (`p` x n_timepoints) 

Returns: 

    the sum of the log-likehood of `x` across time points, i.e., a vector of length (n_timedepvars)
"""
logp_x_z(m::odevae, x, z) = sum(logpdf.(Normal.(m.decodedμ(m.decoder(z)), sqrt.(exp.(m.decodedlogσ(m.decoder(z))))), x),dims=1) 

"""
    sqnorm(x) 

Calculates the squared euclidean norm of an input vector `x`. 
"""
sqnorm(x) = sum(abs2, x)

"""
    reg(m::odevae)

Calculates a regularization term penalizing large weights and biases in the decoder neural networks 
    of the ODE-VAE model `m` to be used in the loss function. 
    Essentially a L2-penalty, i.e., penalizes large L2 norm of the decoder parameters. 
"""
reg(m::odevae) = sum(sqnorm, Flux.params(m.decoder,m.decodedμ,m.decodedlogσ))

"""
    getparams(m::odevae)

Collects the parameters, i.e., the weights and biases of all neural network layers of the ODE-VAE model `m`, 
    using the `Flux.params()` function. 
"""
getparams(m::odevae) = Flux.params(m.encoder, m.encodedμ, m.encodedlogσ, m.decoder, m.decodedμ, m.decodedlogσ, m.paramNN)

"""
    loss(X, Y, t, m::odevae)

Calculates the loss of a single individual's observations given by time-dependent measurements `X`, 
    baseline information `Y`, time points of the second or subsequent measurements `t`, and the ODE-VAE model `m`, 
    based on an adapted version of the ELBO that uses the value of the latent representation as obtained from 
    solving a linear or Lotka-Volterra `ODEProblem` for the mean of the latent space of `m`, where individual-specific ODE-
    parameters are inferred from the baseline variables with an additional neural network, `m.paramNN`. 

Inputs: 

    `X`: array of size (`p` x n_timepoints) containing the time-dependent measurements of one individual

    `Y`: vector of length `p` containing the baseline measurements of one individual

    `t`: vector of length n_timepoints_i-1, containing the time point(s) of the individual's subsequent measurement(s) after baseline

    `m`: `odevae` struct - ODE-VAE model based on which the loss is calculated 

Returns: 

    scalar value of the loss function: negative adapted ELBO + decoder weight regularization + regularization to 
        enforce consistency between the latent representation means before and after solving the ODE.
"""
function loss(X, Y, t, m::odevae)
    latentμ, latentlogσ = m.encodedμ(m.encoder(X)), m.encodedlogσ(m.encoder(X))
    learnedparams = m.paramNN(Y)
    curparams = [learnedparams[1], m.ODEprob.p[2], m.ODEprob.p[3], learnedparams[2]]
    curts = Int.(floor.(t .*(1 ./dt) .+1))
    smoothμ = Array(solve(m.ODEprob, Tsit5(), u0 = [latentμ[1,1], latentμ[2,1]], p=curparams, saveat=dt))[:,curts]
    combinedμ = hcat(latentμ[:,1],smoothμ)
    combinedz = latentz.(combinedμ, latentlogσ)
    ELBO = 1.0 .* logp_x_z(m, X, combinedz) .- 0.5 .* kl_q_p(combinedμ, latentlogσ)
    lossval = sum(-ELBO) + 0.01*reg(m) + 1.0*sum((smoothμ .- latentμ[:,2:end]).^2)
    return lossval
end

"""
    loss_wrapper(m::odevae)

Convenience wrapper function: maps a ODE-VAE model `m` to the `loss` function, so that the resulting 
    output function is a function of the training data (`X`, `Y`, `t`) only.

Input: 

    `m::odevae`: ODE-VAE model based on which the loss shall be calculated 

Returns: 

    a function `(X, Y, t) -> loss(X, Y, t, m)`
"""
function loss_wrapper(m::odevae) 
    return function (X, Y, t) loss(X, Y, t, m) end 
end

# to experiment: 
# xs, tvals, group1, group2 = generate_xs(n, p, true_u0, sol_group1, sol_group2);
# trainingdata = zip(xs, x_baseline, tvals);
# X, Y, t = first(trainingdata)