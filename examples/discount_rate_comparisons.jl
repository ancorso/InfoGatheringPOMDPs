# The purpose of this file is to evaluate a SARSOP policy solved using one discount rate
# applied to a POMDP with another discount rate. For example, suppose we act according to a 10%
# discount rate, but the true discount is only 7%, we can see how much shortfall we have there.
# We set it up so there is ONE discount rate for evaluation, but the solvers are generated using
# different discount rates. 
using InfoGatheringPOMDPs
using POMDPs
using POMDPTools
using NativeSARSOP
using JLD2
using CSV
using Random
using DataStructures
using Plots
default(framestyle = :box,  color_palette=:seaborn_deep6, fontfamily="Computer Modern", margin=5Plots.mm)


# Define random seeds
fix_solve_and_eval_seed = true # Whether the seed is set before each policy gen and evaluation. Seed is the eval index + test set. It is threadsafe. 
pomdp_gen_seed = 1234 # seed used to control the generation of the pomdps
split_by = :both # Split the test set for ensuring unique :geo, :econ, or :both

# Can parse 1 command line argument for the split_by 
if length(ARGS) > 0 
    if ARGS[1] == "geo"
        split_by = :geo
    elseif ARGS[1] == "econ"
        split_by = :econ
    elseif ARGS[1] == "both"
        split_by = :both
    else
        error("Invalid argument: ", ARGS[1])
    end
end

# eval discount factor
discount_factor = 0.93 # Annual discount factor (options are 0.87, 0.9, 0.93)

# Define the save directory. This will through an error if the savedir already exists
dfstr = "$(round(Int, (1-discount_factor)*100))"
savedir="./results/discount_factors/evaluated_on_$dfstr"
mkpath(savedir)


function get_pomdps(discount_factor)

    # Set the discount factor and load corresponding scenarios
    df_folder = "Discount_$(round(Int, (1-discount_factor)*100))"
    scenarios_path = joinpath("./examples/data/Geothermal Reservoir/", df_folder)
    files = readdir(scenarios_path)
    scenario_csvs = OrderedDict(Symbol("Option $i") => joinpath(scenarios_path, files[i]) for i in 1:length(files))

    # Define the set of geological and economic parameters so geo models can be separated
    geo_params = ["par_P_Std", "par_P_Mean", "par_PMax", "par_AZ", "par_PMin", "par_PV", "par_SEED", "par_Zmax", "par_Zmin", "par_SEEDTrend", "par_C_WATER", "par_P_INIT", "par_FTM", "par_KVKH", "par_C_ROCK", "par_THCOND_RES", "par_HCAP_RES", "par_TempGrad"]
    econ_params = ["par_CAPEXitem1", "par_CAPEXitem2", "par_CAPEXitem5", "par_CAPEXitem6", "par_UnitOPEXWater", "par_UnitOPEXWaterInj", "par_UnitOPEXActiveProducers", "par_UnitOPEXActiveWaterInjectors"]

    # Parameter descriptions
    var_description = OrderedDict( #TODO: check these
        :par_P_Std => "Porosity Std. Dev.",
        :par_P_Mean => "Porosity Mean",
        :par_PMax => "Variogram Anisotropy (major)",
        :par_PMin => "Variogram Anisotropy (minor)",
        :par_PV => "Variogram Anisotropy (vertical)",
        :par_AZ => "Variogram Azimuth",
        :par_FTM => "Fault Transmissibility Multiplier",
        :par_KVKH => "Permeability Ratio (vert/horiz)",
        :par_Zmax => "Surface Trend Z Max",
        :par_Zmin => "Surface Trend Z Min",
        :par_C_WATER => "Water Compressibility",
        :par_P_INIT => "Initial Reservoir Pressure",
        :par_C_ROCK => "Rock Compressibility",
        :par_THCOND_RES => "Rock Thermal Conductivity",
        :par_HCAP_RES => "Rock Heat Capacity",
        :par_TempGrad => "Temperature Gradient",
        :par_CAPEXitem1 => "CAPEX Injection Well",
        :par_CAPEXitem2 => "CAPEX Production Well",
        :par_CAPEXitem3 => "CAPEX Surface Facilities",
        :par_CAPEXitem4 => "CAPEX Flowlines",
        :par_CAPEXitem5 => "CAPEX Production Pump",
        :par_CAPEXitem6 => "CAPEX Injection Pump",
        :par_OPEXfixedtotal => "OPEX Fixed Total",
        :par_UnitOPEXWater => "OPEX Water",
        :par_UnitOPEXWaterInj => "OPEX Water Injectors",
        :par_UnitOPEXActiveProducers => "OPEX Active Producers",
        :par_UnitOPEXActiveWaterInjectors => "OPEX Active Water Injectors"
    )

    # Define which parameters are affected for the three-slim-well case
    pairs_3Wells = [(:par_P_Std, 0.0025), (:par_P_Mean, 0.025), (:par_PMax, 1000), (:par_AZ, 45), (:par_PMin, 200), (:par_PV, 10), (:par_Zmax, 0.045), (:par_Zmin, 0.015)]

    # Define the observation actions
    obs_actions = [
        ObservationAction("Measure Water Compressibility", 14/365, -0.05, uniform(:par_C_WATER, 5e-5)),
        ObservationAction("Measure Initial Reservoir Pressure", 21/365, -0.1, uniform(:par_P_INIT, 5)),
        ObservationAction("Measure Fault Transmissibility Multiplier", 60/365, -2.0, uniform(:par_FTM, 0.015)),
        ObservationAction("Measure Permeability Ratio", 30/365, -0.05, uniform(:par_KVKH, 0.1)),
        ObservationAction("Measure Rock Compressibility", 30/365, -0.05, uniform(:par_C_ROCK, 5e-5)),
        ObservationAction("Measure Rock Thermal Conductivity", 30/365, -0.07, uniform(:par_THCOND_RES, 0.5)),
        ObservationAction("Measure Rock Heat Capacity", 30/365, -0.07, uniform(:par_HCAP_RES, 250)),
        ObservationAction("Measure Temperature Gradient", 21/365, -0.1, uniform(:par_TempGrad, 0.001)),
        ObservationAction("Drill 3 Wells", 270/365, -9, product_uniform(pairs_3Wells)),
        ObservationAction("Assess CAPEX Injection Well", 30/365, -1.2, uniform(:par_CAPEXitem1, 0.3)),
        ObservationAction("Assess CAPEX Production Well", 30/365, -1.2, uniform(:par_CAPEXitem2, 0.3)),
        ObservationAction("Assess CAPEX Surface Facilities", 60/365, -10, scenario_dependent_uniform(:par_CAPEXitem3, keys(scenario_csvs), 18.0)),
        ObservationAction("Assess CAPEX Flowlines", 60/365, -10, scenario_dependent_uniform(:par_CAPEXitem4, keys(scenario_csvs), 5.5)),
        ObservationAction("Assess CAPEX Production Pump", 30/365, -0.03, uniform(:par_CAPEXitem5, 0.01625)),
        ObservationAction("Assess CAPEX Injection Pump", 30/365, -0.02, uniform(:par_CAPEXitem6, 0.01)),
        ObservationAction("Assess OPEX Fixed Total", 30/365, -3.5, scenario_dependent_uniform(:par_OPEXfixedtotal, keys(scenario_csvs), 1.0)),
        ObservationAction("Assess OPEX Water", 30/365, -0.02, uniform(:par_UnitOPEXWater, 0.00975)),
        ObservationAction("Assess OPEX Water Injectors", 30/365, -0.02, uniform(:par_UnitOPEXWaterInj, 0.00975)),
        ObservationAction("Assess OPEX Active Producers", 30/365, -0.01, uniform(:par_UnitOPEXActiveProducers, 0.006)),
        ObservationAction("Assess OPEX Active Water Injectors", 30/365, -0.01, uniform(:par_UnitOPEXActiveWaterInjectors, 0.006)),
    ]

    # Set the number of observation bins for each action
    Nbins = fill(2, length(obs_actions[1:end]))
    Nbins[findall([a.name == "Drill 3 Wells" for a in obs_actions])] .= 5

    # Create the pomdp, the validation and test sets
    pomdps, test_sets = create_pomdps(
        scenario_csvs, 
        geo_params, 
        econ_params, 
        obs_actions, 
        Nbins, 
        rng_seed=pomdp_gen_seed, # Set the pomdp random seed 
        discount=discount_factor,
        split_by=split_by,
    )
    return pomdps, test_sets
end

min_particles = 50
sarsop_policy(pomdp) = EnsureParticleCount(solve(SARSOPSolver(max_time=100.,epsilon=1.0, init_lower = PreSolved(onestep_alphavec_policy(pomdp)), init_upper = NativeSARSOP.FastInformedBound(bel_res=1e-2, init_value = 0.0, max_time=100)), pomdp), BestCurrentOption(pomdp), min_particles)

discount_factors = [0.87, 0.9, 0.93]
policy_names = ["SARSOP (Discount=0.13)", "SARSOP (Discount=0.10)", "SARSOP (Discount=0.07)"]

# Construct the evaluation pomdps
eval_pomdps, eval_test_sets = get_pomdps(discount_factor)

policy_results = []
for γ in discount_factors
    # get the train pomdps
    train_pomdps, _ = get_pomdps(γ)
    res = eval_kfolds_difftest(eval_pomdps, train_pomdps, sarsop_policy, eval_test_sets; rng=Random.GLOBAL_RNG, fix_seed=true)
    push!(policy_results, res)
end

# Save the results
JLD2.@save joinpath(savedir, "results.jld2") policy_results policy_names

# Alternatively, load from file by uncommenting the following lines
# results_file = JLD2.load(joinpath(savedir, "results.jld2")) # <---- Uncomment this line to load the results from file
# policy_results = results_file["policy_results"] # <---- Uncomment this line to load the results from file
# policy_names = results_file["policy_names"] # <---- Uncomment this line to load the results from file

# Plot the results
for (policy_result, policy_name) in zip(policy_results, policy_names)
    # Print out all of the policy metrics (both combined and some individual)
    pobs, pdev, pall = policy_results_summary(eval_pomdps[1], policy_result, policy_name)
    try
        savefig(pall, joinpath(savedir, policy_name * "_summary.pdf"))
        savefig(pobs, joinpath(savedir, policy_name * "_data_acq_actions.pdf"))
        savefig(pdev, joinpath(savedir, policy_name * "_development_selections.pdf"))

        # Plot the sankey diagram that shows the abandon, execute, observe flow
        p = policy_sankey_diagram(eval_pomdps[1], policy_result, policy_name)
        savefig(p, joinpath(savedir, policy_name * "_sankey.pdf"))

        # Similar information to the sankey diagram but also includes expected regret
        df = trajectory_regret_metrics(eval_pomdps[1], policy_result)
        CSV.write(joinpath(savedir,  policy_name * "_trajectory_regret_metrics.csv"), df)
    catch
        println("Error plotting for policy: ", policy_name)
    end
end

# ## Tex figures for the paper:
# gr()
# p = policy_sankey_diagram(pomdps[1], policy_result, policy_name)
# annotate!(p, 11.5, 1.0, text("Walk Away", "Computer Modern", 8, rotation=-90))
# annotate!(p, 11.5, 1.6, text("Develop", "Computer Modern", 8, rotation=-90))
# savefig(p, joinpath(savedir, policy_name * "_sankey.pdf"))

# ENV["PATH"] = ENV["PATH"]*":/Library/TeX/texbin"*":/opt/homebrew/bin"
# pgfplotsx()
# policy_result, policy_name = policy_results[end], policy_names[end]
# pobs, pdev, pall = policy_results_summary(pomdps[1], policy_result, policy_name)
# savefig(pobs, joinpath(savedir, policy_name * "_data_acq_actions.tex"))
# savefig(pdev, joinpath(savedir, policy_name * "_development_selections.tex"))

# Make direct comparisons across policies (figure and table)
p = policy_comparison_summary(policy_results, policy_names)
savefig(p, joinpath(savedir, "policy_comparison.pdf"))
policy_comparison_table(policy_results, policy_names)

# Compare just the PES CDFS across policies
p = pes_comparison(policy_results, policy_names)
savefig(p, joinpath(savedir, "PES_comparison.pdf"))
# savefig(p, joinpath(savedir, "PES_comparison.tex"))

# Compare the expected loss across policies
p = expected_loss_comparison(policy_results, policy_names)
savefig(p, joinpath(savedir, "Expected_Loss_comparison.pdf"))
# savefig(p, joinpath(savedir, "Expected_Loss_comparison.tex"))


policy_results[1]

