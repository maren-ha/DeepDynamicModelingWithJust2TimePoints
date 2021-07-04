"""
some plotting functions
"""

struct simdata
    xs
    x_baseline
    tvals
    group1
    group2
end

struct CohortData
    cohort 
    cohort_xs 
    cohort_xs_baseline
    cohort_tvals 
    cohort_ids 
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

function eval_z_trajectories(xs, x_params,tvals, group1, sol_group1, sol_group2, m, dt) # look at trajectories of first 9 individuals during training
    plotarray=[]
    for ind in 1:4#9
        curgroup = ind ∈ group1 ? 1 : 2
        curxs = xs[ind]
        curmu, cursi = m.encodedμ(m.encoder(curxs)), m.encodedlogσ(m.encoder(curxs))
        curz = latentz.(curmu, cursi)
        learnedparams = m.paramNN(x_params[ind])
        if length(m.paramNN.layers[end].α) == 2
            curparams = Float32[learnedparams[1], 0.00, 0.00, learnedparams[2]]
        else
            curparams = learnedparams
        end    
        curts = tvals[ind] .* (1 ./dt) .+ 1
        origt1s = repeat([tvals[ind]], length(curxs[:,1]))
        origxst0 = curxs[:,1]
        origxsotherts = curxs[:,2:end]
        curprob = ODEProblem(linear_2d_system,curmu[:,1],tspan,curparams)
        cursol = solve(curprob, Tsit5(), saveat = dt)
        curplot = curgroup == 1 ? plot(sol_group1.t, sol_group1'; legend=false, line=([:dot :dot], 3, ["#1f77b4" "#ff7f0e"])) : plot(sol_group2.t, sol_group2'; legend=false, line=([:dot :dot], 3, ["#1f77b4" "#ff7f0e"]))
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

function allindsplot(group, data::simdata, m::odevae, sol_group1, sol_group2)
    # get data
    xs, x_params, tvals = data.xs, data.x_baseline, data.tvals
    # set parameters
    if group == 1
        sol = sol_group1
        groupinds = group1
        legendposition = :topright
        ylims = (-0.3,4.2)
    else
        sol = sol_group2
        groupinds = group2
        legendposition = :topleft
        ylims = (-0.5,12)
    end
    # plot true solution
    allindsplot1 = plot(sol.t, sol';
                    #ylims = ylims,
                    labels=[L"\mathrm{true~}u_1" L"\mathrm{true~}u_2"],
                    legend=legendposition,
                    line=([:dot :dot], 4, ["#ff7f0e" "#1f77b4"])#["#e6550d" "#3182bd"])
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
            curparams = Float32[learnedparams[1], 0.00, 0.00, learnedparams[2]]
        else
            curparams = learnedparams
        end
        curprob = ODEProblem(linear_2d_system,curmu[:,1],tspan,curparams)
        cursol = solve(curprob, Tsit5(), saveat = dt)
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

    return allindsplot1

end

function create_cohortplot(m::odevae, cohort_data, tspan, dt)
    cohortplot = plot()
    for i in 1:length(cohort_data.cohort_xs)
        curxs, curxs_baseline, curtvals = cohort_data.cohort_xs[i], cohort_data.cohort_xs_baseline[i], cohort_data.cohort_tvals[i]
        curmu, cursi = m.encodedμ(m.encoder(curxs)), m.encodedlogσ(m.encoder(curxs))
        learnedparams = m.paramNN(curxs_baseline)
        if length(m.paramNN.layers[end].α) == 2
            curparams = Float32[learnedparams[1], 0.00, 0.00, learnedparams[2]]
        else
            curparams = learnedparams
        end
        curprob = ODEProblem(linear_2d_system,curmu[:,1],tspan,curparams)
        cursol = solve(curprob, Tsit5(), saveat = dt)
        plot!(cursol.t, cursol'; line=(0.5, ["#3182bd" "#e6550d"]), legend=false)
        Plots.scatter!(cat(0,curtvals, dims=1),curmu[1,:]; marker=(:c,4,"#9ecae1"), legend=false)
        Plots.scatter!(cat(0,curtvals, dims=1),curmu[2,:]; marker=(:c,4,"#fdae6b"), legend=false, title="cohort $(cohort_data.cohort)")
    end
    return cohortplot
end



