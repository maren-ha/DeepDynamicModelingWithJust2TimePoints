#------------------------------
# functions to define and manipulate ODE-VAE model 
#------------------------------

#------------------------------
# ODE systems
#------------------------------

# Linear 
function linear_2d_system(du,u,p,t)
    a11, a12, a21, a22 = p
    z1,z2 = u
    du[1] = dz1 = a11 * z1 + a12 * z2
    du[2] = dz2 = a21 * z1 + a22 * z2
end

# Lotka-Volterra
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

downscaled_glorot_uniform(dims...) = (rand(Float32, dims...) .- 0.5f0) .* sqrt(0.01f0/sum(dims)) # smaller weights initialisation

# initialise model
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

latentz(μ, logσ) = μ .+ sqrt.(exp.(logσ)) .* randn(Float32,size(μ)...) # sample latent z,

kl_q_p(μ, logσ) = 0.5 .* sum(exp.(logσ) + μ.^2 .- 1 .- (logσ),dims=1)

logp_x_z(m::odevae, x, z) = sum(logpdf.(Normal.(m.decodedμ(m.decoder(z)), sqrt.(exp.(m.decodedlogσ(m.decoder(z))))), x),dims=1) # get reconstruction error

sqnorm(x) = sum(abs2, x)
reg(m::odevae) = sum(sqnorm, Flux.params(m.decoder,m.decodedμ,m.decodedlogσ)) # regularisation term in loss

getparams(m::odevae) = Flux.params(m.encoder, m.encodedμ, m.encodedlogσ, m.decoder, m.decodedμ, m.decodedlogσ, m.paramNN) # get parameters of VAE model

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

function loss_wrapper(m::odevae) 
    return function (X, Y, t) loss(X, Y, t, m) end 
end

# to experiment: 
# xs, tvals, group1, group2 = generate_xs(n, p, true_u0, sol_group1, sol_group2);
# trainingdata = zip(xs, x_baseline, tvals);
# X, Y, t = first(trainingdata)