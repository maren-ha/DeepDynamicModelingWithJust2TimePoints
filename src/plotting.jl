#------------------------------
# some plotting functions for simulated data
#------------------------------

struct simdata
    xs
    x_baseline
    tvals
    group1
    group2
end

function plot_truesolution(group, data::simdata, sol_group1, sol_group2; showdata=true)
    if group == 1
        sol = sol_group1
        groupinds = data.group1
        legendposition = :topright
    else
        sol = sol_group2
        groupinds = data.group2
        legendposition = :topleft
    end
    curplot = plot(sol.t, sol',
                label = [L"\mathrm{true~solution~}u_1" L"\mathrm{true~solution~}u_2"],
                legend = legendposition,
                legendfontsize = 12,
                line=(3, ["#ff7f0e" "#1f77b4"])
                )
    if !showdata
        return curplot
    else
        for ind in 1:length(data.xs[groupinds])
            for var in 1:size(data.xs[groupinds][1],1)
                color = "#ffbb78" 
                if var > 5
                    color = "#aec7e8"
                end
                Plots.scatter!(cat(0,data.tvals[groupinds][ind], dims=1), data.xs[groupinds][ind][var,:], label="", marker=(:c,6,color))
            end
        end
    end
    return curplot
end

function eval_z_trajectories(xs, x_params,tvals, group1, sol_group1, sol_group2, m, dt; swapcolorcoding::Bool=false) # look at trajectories of first 9 individuals during training
    plotarray=[]
    for ind in 2:5#9
        curgroup = ind ∈ group1 ? 1 : 2
        colors_truesol = swapcolorcoding ? ["#ff7f0e" "#1f77b4"] : ["#1f77b4" "#ff7f0e"]
        curxs = xs[ind]
        curmu, cursi = m.encodedμ(m.encoder(curxs)), m.encodedlogσ(m.encoder(curxs))
        curz = latentz.(curmu, cursi)
        learnedparams = m.paramNN(x_params[ind])
        if length(m.paramNN.layers[end].α) == 2
            curparams = Float32[learnedparams[1], m.ODEprob.p[2], m.ODEprob.p[3], learnedparams[2]]
        else
            curparams = learnedparams
        end    
        curts = tvals[ind] .* (1 ./dt) .+ 1
        origt1s = repeat([tvals[ind]], length(curxs[:,1]))
        origxst0 = curxs[:,1]
        origxsotherts = curxs[:,2:end]
        #curprob = ODEProblem(linear_2d_system,curmu[:,1],tspan,curparams)
        cursol = solve(m.ODEprob, Tsit5(), u0 = curmu[:,1], p=curparams, saveat = dt)
        #cursol = solve(curprob, Tsit5(), saveat = dt)
        curplot = curgroup == 1 ? plot(sol_group1.t, sol_group1'; legend=false, line=([:dot :dot], 3, colors_truesol)) : plot(sol_group2.t, sol_group2'; legend=false, line=([:dot :dot], 3, colors_truesol))
        plot!(cursol.t, cursol'; line=(2, ["#1f77b4" "#ff7f0e"]))
        Plots.scatter!(cat(0,tvals[ind], dims=1), curmu[1,:], marker = (:c, 4, "#1f77b4")) 
        Plots.scatter!(cat(0,tvals[ind], dims=1), curmu[2,:], marker = (:c, 4, "#ff7f0e"))
        Plots.scatter!(zeros(10,1), origxst0; marker=(:c, 3, "#bab0ac"), alpha=0.5)
        for tp in 1:length(tvals[ind])
            Plots.scatter!(repeat([tvals[ind][tp]], length(curxs[:,1])), origxsotherts[:,tp]; marker=(:c, 3, "#bab0ac"), alpha=0.5)
        end
        push!(plotarray, curplot)
    end
    myplot = plot(plotarray[:]..., layout=(2,2))#layout=(3,3))
    display(myplot)
end

function plot_individual_solutions(ind, xs, x_baseline,tvals, group1, sol_group1, sol_group2, m, dt; swapcolorcoding::Bool=false, showlegend::Bool=true) # look at trajectories of one individual after training 
    curgroup = ind ∈ group1 ? 1 : 2
    colors_truesol = swapcolorcoding ? ["#ff7f0e" "#1f77b4"] : ["#1f77b4" "#ff7f0e"]
    if curgroup == 1
        sol = sol_group1
        legend = showlegend ? :topright : false
        truesollabels = ""
        smoothlabels = ""
        label1 = ""
        label2 = ""
    else
        sol = sol_group2
        legend = showlegend ? :topleft : false
        truesollabels = [L"\mathrm{true~}u_1" L"\mathrm{true~}u_2"]
        smoothlabels = [L"\mathrm{smooth~}\mu_1" L"\mathrm{smooth~}\mu_2"]
        label1 = L"\mu_1 \mathrm{~from~encoder}"
        label2 = L"\mu_2 \mathrm{~from~encoder}"
    end
    curxs = xs[ind]
    curmu, cursi = m.encodedμ(m.encoder(curxs)), m.encodedlogσ(m.encoder(curxs))
    learnedparams = m.paramNN(x_baseline[ind])
    if length(m.paramNN.layers[end].α) == 2
        curparams = Float32[learnedparams[1], m.ODEprob.p[2], m.ODEprob.p[3], learnedparams[2]]
    else
        curparams = learnedparams
    end    
    cursol = solve(m.ODEprob, Tsit5(), u0 = curmu[:,1], p=curparams, saveat = dt)
    curplot = plot(sol.t, sol'; labels=truesollabels, legend=legend, line=([:dot :dot], 3, colors_truesol))
    plot!(cursol.t, cursol'; labels=smoothlabels, line=(2, ["#1f77b4" "#ff7f0e"]))
    Plots.scatter!(cat(0,tvals[ind], dims=1), curmu[1,:], label = label1, marker = (:c, 6, "#1f77b4")) 
    Plots.scatter!(cat(0,tvals[ind], dims=1), curmu[2,:], label = label2, marker = (:c, 6, "#ff7f0e"))
    plot!(xlab="time", ylab="value of latent representation")
    #display(curplot)

    return curplot

end

function allindsplot(group, data::simdata, m, sol_group1, sol_group2; swapcolorcoding::Bool=false, showlegend::Bool=true)
    # get data
    xs, x_params, tvals = data.xs, data.x_baseline, data.tvals
    # set parameters
    if group == 1
        sol = sol_group1
        groupinds = group1
        legendposition = showlegend ? :topright : false
        ylims = (-0.3,4.2)
    else
        sol = sol_group2
        groupinds = group2
        legendposition = showlegend ? :topleft : false
        ylims = (-0.5,12)
    end

    # plot true solution
    colors_truesol = swapcolorcoding ? ["#ff7f0e" "#1f77b4"] : ["#1f77b4" "#ff7f0e"]
    allindsplot1 = plot(sol.t, sol';
                    #ylims = ylims,
                    labels=[L"\mathrm{true~}u_1" L"\mathrm{true~}u_2"],
                    legend=legendposition,
                    line=([:dot :dot], 4, colors_truesol)#["#e6550d" "#3182bd"])
    )
    # get data from group currently looked at
    groupxs = xs[groupinds]
    groupx_params = x_params[groupinds]
    grouptvals = tvals[groupinds]
    # plot ODE solutions (= smooth latent μs as function of t)    
    for ind in 1:length(xs[groupinds])
        curxs = groupxs[ind]
        curmu, cursi = m.encodedμ(m.encoder(curxs)), m.encodedlogσ(m.encoder(curxs))
        learnedparams = m.paramNN(groupx_params[ind])
        if length(m.paramNN.layers[end].α) == 2
            curparams = Float32[learnedparams[1], m.ODEprob.p[2], m.ODEprob.p[3], learnedparams[2]]
        else
            curparams = learnedparams
        end
        cursol = solve(m.ODEprob, Tsit5(), u0 = curmu[:,1], p=curparams, saveat = dt)
        if ind == 2
            labels = [L"\mathrm{smooth~}\mu_1" L"\mathrm{smooth~}\mu_2"]
            label1 = L"\mu_1 \mathrm{~from~encoder}"
            label2 = L"\mu_2 \mathrm{~from~encoder}"
        else
            labels = ""
            label1 = ""
            label2 = ""
        end
        plot!(cursol.t, cursol'; label=labels, line=(0.5, ["#3182bd" "#e6550d"])) #["#6baed6" "#fd8d3c"]
    end
    # plot latent μs obtained directly from the encoder before the ODE solving step
    for ind in 1:length(xs[groupinds])
        curxs = groupxs[ind]
        curmu, cursi = m.encodedμ(m.encoder(curxs)), m.encodedlogσ(m.encoder(curxs))
        if ind == 2
            labels = [L"\mathrm{smooth~}\mu_1" L"\mathrm{smooth~}\mu_2"]
            label1 = L"\mu_1 \mathrm{~from~encoder}"
            label2 = L"\mu_2 \mathrm{~from~encoder}"
        else
            labels = ""
            label1 = ""
            label2 = ""
        end
        Plots.scatter!(cat(0,grouptvals[ind], dims=1),curmu[1,:]; label=label1, marker=(:c,4,"#9ecae1"))
        Plots.scatter!(cat(0,grouptvals[ind], dims=1),curmu[2,:]; label=label2, marker=(:c,4,"#fdae6b"))
    end
    plot!(xlab="time", ylab="value of latent representation")

    return allindsplot1

end

function plot_batch_solution(ind, xs, x_baseline, tvals, group1, sol_group1, sol_group2, m, dt; swapcolorcoding::Bool=false, showlegend::Bool=true)
    distmat = getdistmat_odesols_mean(m, xs, x_baseline; centralise=true);
    minibatches, batch_weights = findminibatches_distmat(distmat, batchsize, kernel);
    batch_xs = collect(xs[minibatches[i]] for i in 1:length(xs));
    batch_x_baseline = collect(x_baseline[minibatches[i]] for i in 1:length(x_baseline));
    batch_tvals = collect(tvals[minibatches[i]] for i in 1:length(tvals));

    curgroup = ind ∈ group1 ? 1 : 2
    colors_truesol = swapcolorcoding ? ["#fd8d3c" "#6baed6"] : ["#6baed6" "#fd8d3c"]
    curxs = batch_xs[ind][1] # reference individual 
    curmu, cursi = m.encodedμ(m.encoder(curxs)), m.encodedlogσ(m.encoder(curxs))
    curparams = m.paramNN(batch_x_baseline[ind][1])
    cursol = solve(m.ODEprob, Tsit5(), u0 = curmu[:,1], p = curparams, saveat = dt)
    if curgroup == 1
        curplot = plot(sol_group1.t, sol_group1';
                        labels=[L"\mathrm{true~}u_1" L"\mathrm{true~}u_2"],
                        legend=showlegend ? :topright : false,
                        line=([:dot, :dot], 3, colors_truesol)#orange, blue
                        #size=(1000,500)
                        )
    else
        curplot = plot(sol_group2.t, sol_group2';
                        labels=[L"\mathrm{true~}u_1" L"\mathrm{true~}u_2"],
                        legend=showlegend ? :topleft : false,
                        line=([:dot, :dot], 3, colors_truesol),
                        #size=(1000,500)
                        )
    end
    for i in 2:batchsize
        bcurxs = batch_xs[ind][i]
        bcurmu, bcursi = m.encodedμ(m.encoder(bcurxs)), m.encodedlogσ(m.encoder(bcurxs))
        bcurparams = m.paramNN(batch_x_baseline[ind][i])
        bcursol = solve(m.ODEprob, Tsit5(), u0 = bcurmu[:,1], p = bcurparams, saveat = dt)
        if i==batchsize
            labels = [L"\mathrm{smooth~}\mu_1\mathrm{~batch~individual}" L"\mathrm{smooth~}\mu_2\mathrm{~batch~individual}"]
            label1 = L"\mu_1\mathrm{~from~encoder,~batch~individual}"
            label2 = L"\mu_2\mathrm{~from~encoder,~batch~individual}"
        else
            labels, label1, label2="", "", ""
        end
        plot!(bcursol.t, bcursol'; labels=labels, line=(0.5, ["#9ecae1" "#fdae6b"])) # blue, orange
        scatter!(cat(0,batch_tvals[ind][i], dims=1), bcurmu[1,:]; label=label1, marker=(:c,4,"#9ecae1"))
        scatter!(cat(0,batch_tvals[ind][i], dims=1), bcurmu[2,:]; label=label2, marker=(:c,4,"#fdae6b"))
    end
    plot!(cursol.t, cursol';
        labels=[L"\mathrm{smooth~}\mu_1\mathrm{~reference~individual}" L"\mathrm{smooth~}\mu_2\mathrm{~reference~individual}"],
        line=(2.5, ["#1f77b4" "#e6550d"])
        )
    scatter!(cat(0,batch_tvals[ind][1], dims=1),curmu[1,:];
        label = L"\mu_1\mathrm{~from~encoder,~reference~individual}", marker=(:c,7,"#1f77b4"))
    scatter!(cat(0,batch_tvals[ind][1], dims=1),curmu[2,:];
        label = L"\mu_2\mathrm{~from~encoder,~reference~individual}", marker=(:c,7,"#e6550d"))
    plot!(xlab="time", ylab="value of latent representation")
    display(curplot)
    return curplot 
end
