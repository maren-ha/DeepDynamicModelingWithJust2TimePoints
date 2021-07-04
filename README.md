# Deep dynamic modeling with just two time points: Can we still allow for individual trajectories?

In this repository, code to reproduce the results in the [arXiv preprint](https://arxiv.org/abs/2012.00634) on learning individual trajectories by integrating ODE systems into a VAE latent space based on data from only two time points is hosted. 

> Maren Hackenberg, Philipp Harms, Thorsten Schmidt and Harald Binder (2020): Deep dynamic modeling with just two time points: Can we still allow for individual trajectories?
> arXiv preprint: https://arxiv.org/abs/2012.00634

![](figures/example_nonlinear.png)

The project addresses a questions often coming up in the context of newly set-up epidemiological cohort studies and clinical registries: What can be learned from the data in an early phase of the study, when only a baseline characterization and one (or very few) follow-up measurement are available? 
Since such longitudinal biomedical data are often characterized by a sparse time grid and individual-specific development patterns and are thus challenging to model, we investigate whether combining deep learning with dynamic modeling can be useful for uncovering complex structure in such small data settings. 

Our approach is based on a variational autoencoder (VAE) to obtain a latent representation, where we integrate a system of ordinary differential equations into the VAE latent space for dynamic modeling. We fit individual-specific trajectories by using additional baseline variables to infer individual-specific ODE parameters, thus using implicit regularity assumptions on individuals’ similarity. 

![](figures/modelarchitecture.png)

Similarity is a crucial point here more generally: While such an extreme setting with only two time points might seem hopeless for dynamic modeling, irregular spacing of measurements might nevertheless enable modeling of trajectories: If the second measurements of similar individuals occur at different time points, then these aggregated measurements are informative about each individual trajectory. Thus, we explicitly leverage individuals’ similarity in an extension of the approach, where we enrich each individual’s information by assigning it to a group of similar ones and weight individuals’ contribution to the overall loss function according to their similarity.

![](figures/trainingonbatches.png)

Using simulated data, we show to what extent the approach can recover individual trajectories from ODE systems with two and four unknown parameters and infer groups of individuals with similar trajectories, and where it breaks down.

An illustrative jupyter notebook is provided that explains about the motivation behind the method with an exemplary application scenario and walks you through a code example step by step. 

In `main.jl`, you find a script to reproduce the experiments based on a 2-dimensional linear ODE system with 2 unknown parameters, while `main_4parameters.jl` contains the code for the more challenging scenario with 4 unknown parameters, where we train the model on batches of individuals with similar trajectories that are inferred simultaneously with the learned latent trajectories. 