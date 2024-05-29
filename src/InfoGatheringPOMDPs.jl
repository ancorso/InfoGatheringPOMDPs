module InfoGatheringPOMDPs

using CSV
using DataFrames
using StatsBase
using Printf
using Parameters
using Distributions
using Clustering
using POMDPs
using POMDPTools
using ProgressMeter
using Random
using LinearAlgebra
using Plots.Measures
using Plots
using SankeyPlots
using StatsPlots
using DataStructures

# Default plotting things
default(framestyle = :box,  color_palette=:seaborn_deep6, fontfamily="Computer Modern", margin=5Plots.mm)

export DiscreteUp
include("discrete_updater.jl")

export uniform, product_uniform, scenario_dependent_uniform, 
       InfoGatheringPOMDP, ObservationAction, create_pomdps, 
       create_pomdps_with_different_training_fractions
include("pomdp.jl")

export BestCurrentOption, EnsureParticleCount, FixedPolicy, RandPolicy, OneStepGreedyPolicy, PreSolved, onestep_alphavec_policy, WalkAwayNextLB
include("policies.jl")

export discounted_reward, observation_cost, number_observed, correct_scenario, correct_gonogo, PES, expected_loss, eval_single, eval_kfolds, eval_kfolds_difftest
include("metrics.jl")

export policy_results_summary, policy_comparison_summary, train_states_comparison_summary, 
       policy_sankey_diagram, trajectory_regret_metrics, generate_action_table, scenario_returns, 
       policy_comparison_table, pes_comparison, expected_loss_comparison, reward_vs_ngeolgies, reward_vs_necon, 
       print_human_sequences, action_distribution
include("plotting.jl")

end
