#------------------------------
# functions for minibatch approach 
#------------------------------

"""
    batchloss_individual_parameters(bX, bY, bts, bweights, m::odevae) 

Calculates the ELBO-based ODE-VAE loss function for a minibatch of individuals, based on the 
    batch individuals' time-dependent measurements `bX`, their baseline information `bY`, 
    time points of the second or subsequent measurements `bts`, and the ODE-VAE model `m`. 
    The loss is based on an adapted version of the ELBO that uses the value of the latent 
    representation as obtained from solving a linear `ODEProblem` for the mean of the latent 
    space of `m`, where individual-specific ODE-parameters are inferred from the baseline 
    variables with an additional neural network, `m.paramNN`.

Inputs: 

    `bX`: vector of length `batchsize`, where the `i`th element is a (n_vars x n_timepoints) matrix 
        containing the time-dependent variables of the `i`th individual from the batch 

    `bY`: vector of length `batchsize`, where the `i`th  element is a vector of length (n_baselinevars)
        containing the baseline information for the `i`th individual in the batch  

    `bts`: vector of length `batchsize`, where the `i`th  element is a vector of length (n_timepoints_i-1)
        containing the time points of the measurements of the `i`th individual in the batch after the baseline timepoint

    `bweights`: vector of length `batchsize`, where the `i`th  element is a value between 0 and 1 
        indicating th weight assigned to the `i`th individual based on the similarity its latent dynamics
        to those of the reference individual 

    `m`: `odevae` struct - ODE-VAE model based on which the loss is calculated 

Returns: 

    a scalar giving the loss of the batch, obtained as weighted average of the individual's loss values. 
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

"""
    batchloss_wrapper(m::odevae)

Convenience wrapper function: maps a ODE-VAE model `m` to the `batchloss_individual_parameters` function, 
    so that the resulting output function is a function of the training data (`bX`, `bY`, `bts`, `bweights`) only.

Input: 

    `m::odevae`: ODE-VAE model based on which the loss shall be calculated 

Returns: 

    a function `(bX, bY, bts, bweights) -> batchloss_individual_parameters(bX, bY, bts, bweights, m)`
"""
function batchloss_wrapper(m::odevae)
    return function(bX, bY, bts, bweights) batchloss_individual_parameters(bX, bY, bts, bweights, m) end
end

"""
    tricube(x)

Implements the tricube kernel function, returns the tricube value of an input scalar `x`.
    (see definition here: https://en.wikipedia.org/wiki/Kernel_(statistics)#Kernel_functions_in_common_use)
"""
function tricube(x)
    if abs(x) <= 1
        val = (1 .- abs.(x).^3).^3
    else
        val = 0
    end
    return val
end

"""
    mink(a, indrange)

Returns the smallest values of the input vector `a` according to the indices in `indrange` and their indices in `a`. 
    I.e., if `indrange` = 1:k, it returns the indices of the `k` smallest values in `a` together with the values themselves.

Inputs: 

    `a`: vector of scalars, to be queried for the smallest elements

    `indrange`: range of indices specifying the range of the smallest values desired:
        for `indrange` = j:k, the `j`th- to `k`th-smallest element of `a` will be selected

Returns: 

    `minkinds`: indices of the `indrange` smallest values in `a`

    `a[minkinds]`: values of `a` at those `indrange` smallest values 
"""
function mink(a, indrange) #inspired and adapted from https://discourse.julialang.org/t/what-is-julias-maxk-matlab-that-returns-the-indice-of-top-k-largest-values/14100/7
    minkinds = partialsortperm(a, indrange)
    return [minkinds, a[minkinds]]
end

"""
    euclidean_dist(x,y)

Calculates the Euclidean distance between input vectors `x` and `y` that have to be of the same length.
"""
euclidean_dist(x,y) = length(x) == length(y) ? sqrt(sum((x .- y).^2)) : error("lengths of the two vectors don't match")

"""
    mean_dist(x,y)

Calculates the mean absolute (=mean L1) distance between input vectors `x` and `y` that have to be of the same length.
"""
mean_dist(x,y) = length(x) == length(y) ? (1/length(x))*sum(abs.(x.-y)) : error("lengths of the two vectors don't match")

"""
    randomminibatches(Y, batchsize)

Randomly assigns a minibatch of size `batchsize` to each element in `Y` corresponding to one individual in the dataset, 
    consisting of indices giving the position of the respective batch individuals in the dataset.

Inputs: 
    `Y`: vector of length (n_individuals) containing arbitrary information at index `i` 
        about the `i`th individual - for example, the individual's baseline measurements 

    `batchsize`: integer specifying the desired batch size 
    
Returns: 
    `minibatches`: a vector of length (n_individuals), where the `i`th element is a vector 
        of length `batchsize` of integers between 1 and `length(Y)` = n_individuals 
        giving the indices of the individuals in the `i`th minibatch around the reference individual `i`

    `randomweights`: a vector of `length(Y)` = n_individuals, where the `i`th element is a vector 
        of length `batchsize` containing the weights for all the individual in the `i`th minibatch 
        around the reference individual `i`, which are calculated as `1/length(Y)`
"""
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

"""
    getdistmat_odesols_mean(m::odevae, xs, x_params; centralise=true)

Calculates the distance matrix of all individuals based on the L2 distance of the ODE solutions 
   in the latent space of `m` with individual-specific ODE parameters obtained from `m` with `x_params`.

Inputs: 
    `m::odevae`: current ODE-VAE model 

    `xs`: vector of length `n` = n_individuals, where the `i`th element is a (n_vars=p x n_timepoints) matrix 
        containing the time-dependent variables of the `i`th individual in the dataset

    `x_params`: vector of length `batchsize`, where the `i`th  element is a vector of length (n_baselinevars=q)
        containing the baseline information for the `i`th individual in the dataset 

    `centralise`: optional keyword argument, whether or not to centralise the ODE solutions before calculating 
        the (approximation to the) L2 distances by subtracting the mean of the trajectory in each dimension
        (to ensure that solutions are considered close because they share the same trend). Default=`true`

Returns: 

    a distance matrix of shape (n x n), where the `(i,j)`the element is the L2 distance between the 
        ODE solutioms of individuals `i` and `j`.
"""
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

"""
    findminibatches_distmat(distmat, batchsize, kernel; bandwidth=1.0)

For each individual, identify a minibatch of the `batchsize` closest individuals 
    according to the distance matrix `distmat` and calculate weights by transforming 
    the distances with a kernel function `kernel`.

Inputs: 

    `distmat`: (`n`=n_individuals x `n`) symmetric matrix of distances between all individuals

    `batchsize`: integer giving the desired size of the minibatch 

    `kernel`: kernel function to use to transform the distances to weights 

    `bandwidth`: optional keyword argument specifying the bandwidth of the kernel. Default = 1

Returns: 

    `minibatches`: a vector of length `n` = `n_individuals`, where the `i`th element is a vector of 
        length `batchsize`, containing the indices of the minibatch of the `batchsize` closest individuals 
        around the reference individual `i`

    `allweights`: a vector of length `n` = `n_individuals`, where the `i`th element is a vector of 
        length `batchsize`, containing the weights of the batchsize` closest individuals to the reference individual `i`
"""
function findminibatches_distmat(distmat, batchsize, kernel; bandwidth=1.0)
    n = size(distmat,2)
    minibatches =  []
    allweights = []
    for ind in 1:n
        distrange = collect(1:n)[1:end .!= ind]
        distvec = collect(distmat[ind,j] for j in distrange)
        batch_dist = mink(distvec,1:(batchsize-1))
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

"""
    findminibatches_distmat(distmat, batchsize)

For each individual, identify a minibatch of the `batchsize` closest individuals 
    according to the distance matrix `distmat`.
    Weights are trivially calculated as 1/`batchsize` for each individual. 

Inputs: 

    `distmat`: (n=n_individuals x n) symmetric matrix of distances between all individuals

    `batchsize`: integer giving the desired size of the minibatch 

Returns: 

    `minibatches`: a vector of length `n` = n_individuals, where the `i`th element is a vector of 
        length `batchsize`, containing the indices of the minibatch of the `batchsize` closest individuals 
        around the reference individual `i`
        
    `allweights`: a vector of length `n` = n_individuals, where the `i`th element is a vector of 
        length `batchsize`, containing the trivial 1/`n_batchsize` weights of the `batchsize` closest 
        individuals to the reference individual `i`
"""
function findminibatches_distmat(distmat, batchsize)
    n = size(distmat,2)
    minibatches =  []
    for ind in 1:n
        distrange = collect(1:m)[1:end .!= ind]
        distvec = collect(distmat[ind,j] for j in distrange)
        batch_dist = mink(distvec,1:(batchsize-1))
        map!(x -> (x >= ind ? x+1 : x), batch_dist[1], batch_dist[1])
        new_batch_dist = cat(0, batch_dist[2]..., dims=1)
        minibatch = cat(ind, Int.(batch_dist[1]), dims=1)
        push!(minibatches, minibatch)
    end
    allweights = fill(fill(1/batchsize, batchsize), n)
    minibatches, allweights
end

"""
    evaluate_minibatches(minibatches, group1, group2)

Assumes the dataset can be classified into two groups. Based on the group labels `group1` and `group2`, 
    for each batch, the number of individuals that are in the same group as the batch's reference individual 
    is counted and the sum of these counts across all minibatches is returned.

Inputs: 

    `minibatches`: a vector of length `n` = n_individuals, where the `i`th element is a vector of 
        length `batchsize`, containing the indices of the minibatch of the `batchsize` closest individuals 
        around the reference individual `i`

    `group1`: vector of integers giving the indices of the individuals in group 1

    `group2`: vector of integers giving the indices of the individuals in group 2, 
        such that `length(group1)` + `length(group2)` = `n` = n_individuals 

Returns: 

    `correctly_classified`: sum over the number of correctly classified individuals in each batch, 
        where "correctly classified" is defined as being from the same group as the batch's reference individual 
"""
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

"""
    evaluate_minibatches_pergroup(minibatches, group1, group2)

Assumes the dataset can be classified into two groups. Based on the group labels `group1` and `group2`, 
    for each group, in each batch, the number of individuals that are in the same group as the batch's reference individual 
    is counted. For each group, the corresponding percentage of the sum of these counts across all minibatches 
    divided by the size of the group of these counts across all minibatches is returned.

Inputs: 

    `minibatches`: a vector of length `n` = n_individuals, where the `i`th element is a vector of 
        length `batchsize`, containing the indices of the minibatch of the `batchsize` closest individuals 
        around the reference individual `i`

    `group1`: vector of integers giving the indices of the individuals in group 1

    `group2`: vector of integers giving the indices of the individuals in group 2, 
        such that `length(group1)` + `length(group2)` = `n` = n_individuals 

Returns: 

    for each group, the percentages of the correctly classified individuals from one group among all individuals in the group.
"""
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
