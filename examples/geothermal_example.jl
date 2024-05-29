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

# Set the discount factor and load corresponding scenarios
discount_factor = 0.93 # Annual discount factor (options are 0.87, 0.9, 0.93)
df_str = "$(round(Int, (1-discount_factor)*100))"
df_folder = "Discount_$(df_str)"

# Define the save directory. This will through an error if the savedir already exists
savedir="./results/discount_factors/df$df_str"
mkpath(savedir)

# Load the scenario
scenarios_path = joinpath("./examples/data/Geothermal Reservoir/", df_folder)
files = readdir(scenarios_path)
scenario_csvs = OrderedDict(Symbol("Option $i") => joinpath(scenarios_path, files[i]) for i in 1:length(files))

# Define the set of geological and economic parameters so geo models can be separated
geo_params = ["par_P_Std", "par_P_Mean", "par_PMax", "par_AZ", "par_PMin", "par_PV", "par_SEED", "par_Zmax", "par_Zmin", "par_SEEDTrend", "par_C_WATER", "par_P_INIT", "par_FTM", "par_KVKH", "par_C_ROCK", "par_THCOND_RES", "par_HCAP_RES", "par_TempGrad"]
econ_params = ["par_CAPEXitem1", "par_CAPEXitem2", "par_CAPEXitem5", "par_CAPEXitem6", "par_UnitOPEXWater", "par_UnitOPEXWaterInj", "par_UnitOPEXActiveProducers", "par_UnitOPEXActiveWaterInjectors"]

# Plot the scenario returns
p = scenario_returns(scenario_csvs, geo_params, econ_params)
savefig(p, joinpath(savedir, "npvs.pdf"))

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

# Writeout the table of actions
generate_action_table(pomdps[1], var_description)

# Define the rest of the policies
min_particles = 50
option7_pol(pomdp) = FixedPolicy([Symbol("Option 7")])
option10_pol(pomdp) = FixedPolicy([Symbol("Option 10")])
option11_pol(pomdp) = FixedPolicy([Symbol("Option 11")])
all_policy_geo(pomdp) = EnsureParticleCount(FixedPolicy(obs_actions, BestCurrentOption(pomdp)), BestCurrentOption(pomdp), min_particles)
all_policy_econ(pomdp) = EnsureParticleCount(FixedPolicy(reverse(obs_actions), BestCurrentOption(pomdp)), BestCurrentOption(pomdp), min_particles)
random_policy_0_1(pomdp) = EnsureParticleCount(RandPolicy(;pomdp, prob_terminal=0.1), BestCurrentOption(pomdp), min_particles)
random_policy_0_25(pomdp) = EnsureParticleCount(RandPolicy(;pomdp, prob_terminal=0.25), BestCurrentOption(pomdp), min_particles)
random_policy_0_5(pomdp) = EnsureParticleCount(RandPolicy(;pomdp, prob_terminal=0.5), BestCurrentOption(pomdp), min_particles)
# greedy_policy(pomdp) = EnsureParticleCount(OneStepGreedyPolicy(;pomdp), BestCurrentOption(pomdp), min_particles)
sarsop_policy(pomdp) = EnsureParticleCount(solve(SARSOPSolver(max_time=10.,epsilon=1.0, init_lower = PreSolved(onestep_alphavec_policy(pomdp)), init_upper = NativeSARSOP.FastInformedBound(bel_res=1e-2, init_value = 0.0, max_time=100)), pomdp), BestCurrentOption(pomdp), min_particles)

human1 = [19, 20, 17, 18, 15, 14, 10, 11, 12, 13, 6, 7, 8, 3, 5, 9]
human2 = [9, 3, 5, 7, 12, 16]
human3 = [3, 12, 17]
human4 = [2, 6, 7, 8, 9, 10, 11, 12, 16]
human5 = [9, 3, 6, 7, 10, 11, 19, 20]
human6 = [8, 14, 15, 3, 4]
human7 = [3, 9, 8, 7, 2, 4, 5, 12, 10, 11, 13, 16, 17]
human8 = [9, 12, 13, 3, 5, 8, 16, 10, 11, 18]
human9 = [7, 9, 3, 10, 11, 16, 12, 13]

humans = [human1, human2, human3, human4, human5, human6, human7, human8, human9]
human_policies = [(pomdp) -> FixedPolicy([obs_actions[i] for i in h], BestCurrentOption(pomdp)) for h in humans]
human_policy_names = [string("Human Policy ", i) for i in 1:length(humans)]

# print_human_sequences(obs_actions, humans)

# combine policies into a list
policies = [option7_pol, option10_pol, option11_pol, all_policy_geo, all_policy_econ, random_policy_0_1, random_policy_0_25, random_policy_0_5, sarsop_policy] 
policy_names = ["Option 7", "Option 10", "Option 11", "Acquire All (Geo First)", "Acquire All (Econ First)", "Random Policy (Pstop=0.1)", "Random Policy (Pstop=0.25)", "Random Policy (Pstop=0.5)", "SARSOP Policy"]

# policies =  human_policies # <---- Uncomment this line to use the human policies
# policy_names = human_policy_names # <---- Uncomment this line to use the human policies

# Evaluate the policies on the test set 
policy_results = [] # <---- Uncomment this block to evaluate the policies
for (policy, policy_name) in zip(policies, policy_names)
    println("Evaluating policy: ", policy_name, "...")
    push!(policy_results, eval_kfolds(pomdps, policy, test_sets, fix_seed = fix_solve_and_eval_seed))
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
    pobs, pdev, pall = policy_results_summary(pomdps[1], policy_result, policy_name)
    try
        savefig(pall, joinpath(savedir, policy_name * "_summary.pdf"))
        savefig(pobs, joinpath(savedir, policy_name * "_data_acq_actions.pdf"))
        savefig(pdev, joinpath(savedir, policy_name * "_development_selections.pdf"))

        # Plot the sankey diagram that shows the abandon, execute, observe flow
        p = policy_sankey_diagram(pomdps[1], policy_result, policy_name)
        savefig(p, joinpath(savedir, policy_name * "_sankey.pdf"))

        # Similar information to the sankey diagram but also includes expected regret
        df = trajectory_regret_metrics(pomdps[1], policy_result)
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
risk_comparisons = [p in ["Acquire All (Econ First)", "Random Policy (Pstop=0.5)", "SARSOP Policy"] for p in policy_names]
p = pes_comparison(policy_results[risk_comparisons], policy_names[risk_comparisons])
vline!([policy_results[1][:PES][1]], label="Option 7", linestyle=:dash)
vline!([policy_results[2][:PES][1]], label="Option 10", linestyle=:dash)
vline!([policy_results[3][:PES][1]], label="Option 11", linestyle=:dash)
savefig(p, joinpath(savedir, "PES_comparison.pdf"))
# savefig(p, joinpath(savedir, "PES_comparison.tex"))

# Compare the expected loss across policies
p = expected_loss_comparison(policy_results[risk_comparisons], policy_names[risk_comparisons])
vline!([mean(policy_results[1][:expected_loss][1])], label="Option 7", linestyle=:dash)
vline!([mean(policy_results[2][:expected_loss][1])], label="Option 10", linestyle=:dash)
vline!([mean(policy_results[3][:expected_loss][1])], label="Option 11", linestyle=:dash)
savefig(p, joinpath(savedir, "Expected_Loss_comparison.pdf"))
# savefig(p, joinpath(savedir, "Expected_Loss_comparison.tex"))

###########################################################################
# This section is to investigate the number of geological models needed   #
###########################################################################

# Create an array of pomdps, each with a different number of states
fracs = [0.02, 0.04, 0.08, 0.16, 0.32, 0.5, 0.8]
alls = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]

if split_by == :geo
    geo_fracs = fracs
    econ_fracs = alls
elseif split_by == :econ
    geo_fracs = alls
    econ_fracs = fracs
elseif split_by == :both
    geo_fracs = [fracs..., alls[1:end-1]...]
    econ_fracs = [alls..., fracs[1:end-1]...]
end

zip_fracs = [(g,e) for (g, e) in zip(geo_fracs, econ_fracs)]
pomdps_per_geo, test_sets_per_geo = create_pomdps_with_different_training_fractions(
    zip_fracs, 
    scenario_csvs, 
    geo_params, 
    econ_params,
    obs_actions, 
    Nbins; 
    rng_seed=pomdp_gen_seed, 
    discount=discount_factor, 
    split_by)


# Solve the policies and evaluate the results #<---- Uncomment the below lines to solve and eval the policies
target_indices = [p in ["SARSOP Policy"] for p in policy_names]
results = Dict()
for (policy, pol_name) in zip(policies[target_indices], policy_names[target_indices])
    results[pol_name] = Dict(:geo_frac => [], :econ_frac => [], :results =>[])
    for (frac, pomdps, test_sets) in zip(zip_fracs, pomdps_per_geo, test_sets_per_geo)
        geo_frac, econ_frac = frac[1], frac[2]
        println("Solving and evaluating for policy", pol_name, " with geo_frac= ", geo_frac, " econ_frac= ", econ_frac)
        push!(results[pol_name][:geo_frac], geo_frac)
        push!(results[pol_name][:econ_frac], econ_frac)
        push!(results[pol_name][:results], eval_kfolds(pomdps, policy, test_sets, fix_seed = fix_solve_and_eval_seed))
    end
end
JLD2.@save joinpath(savedir, "Nsamples_results.jld2") results

# Alternatively, load from file by uncommenting the following lines
# results = JLD2.load(joinpath(savedir, "Nsamples_results.jld2"))["results"] # <---- Uncomment this line to load the results from file

# Print the same for just the reward for just one policy
reward_vs_ngeolgies(results["SARSOP Policy"], "SARSOP Policy")
savefig(joinpath(savedir, "SARSOP_geological_realizations.pdf"))
# savefig(joinpath(savedir, "SARSOP_geological_realizations.tex")) 

reward_vs_necon(results["SARSOP Policy"], "SARSOP Policy")
savefig(joinpath(savedir, "SARSOP_economic_realizations.pdf")) 
# savefig(joinpath(savedir, "SARSOP_economic_realizations.tex")) 