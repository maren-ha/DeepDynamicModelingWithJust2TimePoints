"""
functions to define and manipulate VAE model 
"""
# model
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

# initialise model
function init_vae(p, q, zdim, nODEparams, ODEprob; seed::Int=1234)
    # seed
    Random.seed!(seed)
    # parameter network
    paramNN = Chain(Dense(q,q,tanh), Dense(q,nODEparams, arg ->(sigmoid(arg).-0.5)), Flux.Diagonal(nODEparams)) #.*10 vorher
    #   VAE encoder
    Dz, Dh = zdim, p
    encoder, encodedμ, encodedlogσ = Dense(p, Dh, arg ->(tanh.(arg) .+ 1)), Dense(Dh, Dz), Chain(Dense(Dh, Dz, arg -> -relu(arg)), Flux.Diagonal(Dz))
    # VAE decoder
    decoder, decodedμ, decodedlogσ = Dense(Dz, Dh, tanh), Dense(Dh, p), Dense(Dh, p)

    model = odevae(p, q, zdim, paramNN, encoder, encodedμ, encodedlogσ, decoder, decodedμ, decodedlogσ, ODEprob)
    return model
end

# VAE functions
latentz(μ, logσ) = μ .+ sqrt.(exp.(logσ)) .* randn(Float32,size(μ)...) # sample latent z,

kl_q_p(μ, logσ) = 0.5 .* sum(exp.(logσ) + μ.^2 .- 1 .- (logσ),dims=1)

logp_x_z(m::odevae, x, z) = sum(logpdf.(Normal.(m.decodedμ(m.decoder(z)), sqrt.(exp.(m.decodedlogσ(m.decoder(z))))), x),dims=1) # get reconstruction error

# enode(mu1_t0, mu2_t0,curparams,curt) = diffeq_rd(Float32.(curparams),prob1,Tsit5(),u0=[mu1_t0, mu2_t0],saveat=dt)[Int(curt)] # ODE solving step, deprecated

sqnorm(x) = sum(abs2, x)
reg(m::odevae) = sum(sqnorm, Flux.params(m.decoder,m.decodedμ,m.decodedlogσ)) # regularisation term in loss
# reg_old(m::odevae) = sum(x->sum(x.^2), Flux.params(m.decoder,m.decodedμ,m.decodedlogσ))

getparams(m::odevae) = Flux.params(m.encoder, m.encodedμ, m.encodedlogσ, m.decoder, m.decodedμ, m.decodedlogσ, m.paramNN) # get parameters of VAE model

function loss(X, Y, t, m::odevae)
    latentμ, latentlogσ = m.encodedμ(m.encoder(X)), m.encodedlogσ(m.encoder(X))
    learnedparams = m.paramNN(Y)
    curparams = [learnedparams[1], 0.00, 0.00, learnedparams[2]]
    curts = Int.(floor.(t .*(1 ./dt) .+1))
    smoothμ = Array(solve(m.ODEprob, Tsit5(), u0 = [latentμ[1,1], latentμ[2,1]], p=curparams, saveat=dt))[:,curts]
    #smoothμ = Tracker.collect(enode(latentμ[1,1], latentμ[2,1], curparams, curt))
    combinedμ = hcat(latentμ[:,1],smoothμ)
    combinedz = latentz.(combinedμ, latentlogσ)
    ELBO = 1.0 .* logp_x_z(m, X, combinedz) .- 0.5 .* kl_q_p(combinedμ, latentlogσ)
    lossval = sum(-ELBO) + 0.01*reg(m) + 0.5*sum((smoothμ .- latentμ[:,2:end]).^2)
    #lossval = sum(-ELBO) + 0.5*sum((smoothμ .- latentμ[:,2]).^2)
    return lossval
end

function loss_wrapper(m::odevae) 
    return function (X, Y, t) loss(X, Y, t, m) end 
end

"""
# to experiment: 
xs, tvals, group1, group2 = generate_xs(n, p, true_u0, sol_group1, sol_group2);
trainingdata = zip(xs, x_baseline, tvals);
X, Y, t = first(trainingdata)
"""

"""
extra functions for minibatch approach 
"""

function batchloss_individual_parameters(bX, bY, bts, bweights, m::odevae)
    batchloss = 0.0
    for ind in 1:batchsize
        latentμ, latentlogσ = m.encodedμ(m.encoder(bX[ind])), m.encodedlogσ(m.encoder(bX[ind]))
        curts = Int.(bts[ind] .*(1 ./dt) .+1)
        curparams = m.paramNN(bY[ind])
        smoothμ = Array(solve(m.ODEprob, Tsit5(), u0 = [latentμ[1,1], latentμ[2,1]], p=curparams, saveat=dt))[:,curts]
        combinedμ = hcat(latentμ[:,1],smoothμ)
        combinedz = latentz.(combinedμ, latentlogσ)
        ELBO = 1.0 .* logp_x_z(m, bX[ind], combinedz) .- 0.5 .* kl_q_p(combinedμ, latentlogσ)
        lossval = sum(-ELBO) .+ 0.01*reg(m) .+ 0.5 .* sum((smoothμ .- latentμ[:,2:end]).^2)
        batchloss += lossval * bweights[ind]
    end
    return batchloss
end

function batchloss_wrapper(m::odevae)
    return function(bX, bY, bts, bweights) batchloss_individual_parameters(bX, bY, bts, bweights, m) end
end

function tricube(x)
    if abs(x) <= 1
        val = (1 .- abs.(x).^3).^3
    else
        val = 0
    end
    return val
end

function maxk(a, indrange) #inspired and adapted from https://discourse.julialang.org/t/what-is-julias-maxk-matlab-that-returns-the-indice-of-top-k-largest-values/14100/7
    maxkinds = partialsortperm(a, indrange)
    return [maxkinds, a[maxkinds]]
end

euclidean_dist(x,y) = length(x) == length(y) ? sqrt(sum((x .- y).^2)) : error("lengths of the two vectors don't match")

mean_dist(x,y) = length(x) == length(y) ? (1/length(x))*sum(abs.(x.-y)) : error("lengths of the two vectors don't match")

function randomminibatches(Y, batchsize)
    minibatches = []
    n=length(Y)
    for ind in 1:n
        distrange = collect(1:n)[1:end .!= ind]
        randombatch = cat(ind, shuffle(distrange)[1:(batchsize-1)], dims=1)
        push!(minibatches, randombatch)
    end
    randomweights = collect(fill(1/batchsize, batchsize) for i=1:n)
    minibatches, randomweights
end

function getdistmat_odesols_mean(m::odevae, xs, x_params; centralise=true)
    n = length(xs)
    odeparams = m.paramNN.(x_params)
    odeμ0s = collect(m.encodedμ.(m.encoder.(xs))[ind][:,1] for ind in 1:n)
    tgrid = collect(11:10:101)
    dims = length(odeμ0s[1])
    odesols = collect(Array(solve(m.ODEprob, Tsit5(), u0 = odeμ0s[ind], p=odeparams[ind], saveat=dt))[:,tgrid] for ind in 1:n)
    if centralise # additional centralising step: subtract mean of trajectory for all inds and dims
        for ind in 1:n
            for d in 1:dims
                mean_per_dim = mean(odesols[ind][d,j] for j in 1:length(tgrid))
                odesols[ind][d,:] .-= mean_per_dim
            end
        end
    end
    mean_distmat = fill(0.0,(n,n));     # für jeden Zeitpunkt eine Distanzmatrix, dann mitteln
    for i in 1:n
        for j in 1:i
            mean_distmat[i,j] = (1.0/sqrt(length(tgrid))) .* sum(euclidean_dist(odesols[i][:,tp], odesols[j][:,tp]) for tp in 1:length(tgrid))
            #mean_distmat[i,j] = (1.0/sqrt(length(tgrid))) .* mean(sqrt(sum((odesols[i][d,tp]-odesols[j][d,tp]).^2 for tp in 1:length(tgrid))) for d in 1:dims)
            mean_distmat[j,i] = mean_distmat[i,j]
        end
    end
    return mean_distmat
end

function findminibatches_distmat(distmat, batchsize, kernel; bandwidth=1.0)
    n = size(distmat,2)
    minibatches =  []
    allweights = []
    for ind in 1:n
        distrange = collect(1:n)[1:end .!= ind]
        distvec = collect(distmat[ind,j] for j in distrange)
        batch_dist = maxk(distvec,1:(batchsize-1))
        map!(x -> (x >= ind ? x+1 : x), batch_dist[1], batch_dist[1])
        new_batch_dist = cat(0, batch_dist[2]..., dims=1)
        minibatch = cat(ind, Int.(batch_dist[1]), dims=1)
        kernelsum = sum(kernel.(new_batch_dist ./ bandwidth))
        weights = kernel.(new_batch_dist ./ bandwidth) ./kernelsum
        push!(allweights, weights)
        push!(minibatches, minibatch)
    end
    minibatches, allweights
end

function findminibatches_distmat(distmat, batchsize)
    n = size(distmat,2)
    minibatches =  []
    for ind in 1:n
        distrange = collect(1:m)[1:end .!= ind]
        distvec = collect(distmat[ind,j] for j in distrange)
        batch_dist = maxk(distvec,1:(batchsize-1))
        map!(x -> (x >= ind ? x+1 : x), batch_dist[1], batch_dist[1])
        new_batch_dist = cat(0, batch_dist[2]..., dims=1)
        minibatch = cat(ind, Int.(batch_dist[1]), dims=1)
        push!(minibatches, minibatch)
    end
    allweights = fill(fill(1/batchsize, batchsize), n)
    minibatches, allweights
end

# count the number of individuals of the same group as the reference individual in each minibatch
function evaluate_minibatches(minibatches, group1, group2)
    correctly_classified = 0
    for ind in 1:length(minibatches)
        cur_minibatch = minibatches[ind]
        correct_group = cur_minibatch[1] ∈ group1 ? group1 : group2
        curnum = length(intersect(cur_minibatch[2:end],correct_group))
        correctly_classified += curnum
    end
    return correctly_classified
end

function evaluate_minibatches_pergroup(minibatches, group1, group2)
    correctly_classified_group1 = 0
    correctly_classified_group2 = 0
    for ind in 1:length(minibatches)
        cur_minibatch = minibatches[ind]
        correct_group = cur_minibatch[1]∈group1 ? group1 : group2
        curnum = length(intersect(cur_minibatch[2:end],correct_group))
        if correct_group == group1
            correctly_classified_group1 += curnum
        else
            correctly_classified_group2 += curnum
        end
    end
    total_gr1 = length(group1) .* (batchsize-1)
    total_gr2 = length(group2) .* (batchsize-1)
    total = length(minibatches) .* (batchsize-1)
    return (correctly_classified_group1 ./total_gr1 .*100),
    (correctly_classified_group2 ./total_gr2 .* 100),
    ((correctly_classified_group1 + correctly_classified_group2)./total .*100)
end
