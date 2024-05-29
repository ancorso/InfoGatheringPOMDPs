function Simulators.discounted_reward(history; pomdp)
    γ = discount(pomdp)
    r_total = 0.0
    for s in history
        r_total += s.r*γ^s.t
    end
    return r_total
end

function observation_cost(history; kwargs...)
    if length(history) == 1
        return 0
    end
    return sum(h.r for h in history[1:end-1])
end

function observation_duration(history; kwargs...)
    if length(history) == 1
        return 0
    end
    return sum(h.a.time for h in history[1:end-1])
end

function number_observed(history; kwargs...)
    return length(history) - 1
end

function correct_scenario(history; pomdp, verbose=false)
    true_state = history[1].s
    scenario = history[end].a

    returns = [reward(pomdp, true_state, a) for a in pomdp.terminal_actions]
    correct_scenario = pomdp.terminal_actions[argmax(returns)]

    if verbose
        println("Chosen scenario: ", scenario)
        println("best scenario: ", correct_scenario)
    end
    return scenario == correct_scenario
end


function correct_gonogo(history; pomdp)
    true_state = history[1].s
    scenario = history[end].a

    returns = [reward(pomdp, true_state, a) for a in pomdp.terminal_actions]
    correct_scenario = pomdp.terminal_actions[argmax(returns)]

    if correct_scenario == :abandon
        return scenario == :abandon
    else
        if scenario == :abandon
            return false
        end
        return true_state[scenario] >= 0
    end
end

function PES(history; pomdp)
    last_belief = history[end].b
    final_action = history[end].a

    return sum([b*(reward(pomdp, s, final_action) > 0) for (b, s) in zip(last_belief.b, states(pomdp))])
end

function expected_loss(history; pomdp)
    last_belief = history[end].b
    final_action = history[end].a

    return sum([b*(reward(pomdp, s, final_action) < 0)*reward(pomdp, s, final_action) for (b, s) in zip(last_belief.b, states(pomdp))])
end

function eval_single(pomdp, policy, s, updater = DiscreteUp(pomdp), b0 = initialstate(pomdp); rng=Random.GLOBAL_RNG)
    history = []
    b = b0
    t = 0
    i = 1
    while !isterminal(pomdp, s)
        a = action(policy, b; i)
        sp, o, r = gen(pomdp, s, a, rng)
        bp = nothing
        try
            bp = update(updater, b, a, o)
        catch # This is necessary if the belief is all zeros
            a = action(BestCurrentOption(pomdp), b)
            sp, o, r = gen(pomdp, s, a, rng)
            bp = update(updater, b, a, o)
        end
        push!(history, (;s, a, sp, o, r, b, bp, t))
        dt = a isa ObservationAction ? a.time : 0
        i = i+1
        t = t+dt
        s = sp
        b = bp
    end
    results = Dict()
    results[:reward] = discounted_reward(history; pomdp)
    results[:obs_cost] = observation_cost(history)
    results[:obs_duration] = observation_duration(history)
    results[:num_obs] = number_observed(history)
    results[:correct_scenario] = correct_scenario(history; pomdp)
    results[:correct_gonogo] = correct_gonogo(history; pomdp)
    results[:PES] = PES(history; pomdp)
    results[:expected_loss] = expected_loss(history; pomdp)
    results[:actions] = [s.a for s in history]
    results[:final_action] = history[end].a
    return results
end

function eval_kfolds(pomdps, policy_fn, test_sets; rng=Random.GLOBAL_RNG, fix_seed=true)
    Ntest_sets = length.(test_sets)
    test_starts = [1, (cumsum(Ntest_sets)[1:end-1] .+ 1)...]
    results = Array{Any}(undef, sum(Ntest_sets))
    p = Progress(sum(Ntest_sets), 1, "Evaluating policy on test sets...")
    for (i, (pomdp, test_set)) in collect(enumerate(zip(pomdps, test_sets)))
        fix_seed && Random.seed!(rng, i) # Fix the seed before potentially solving the policy
        policy = policy_fn(pomdp)
        Threads.@threads for (j, s) in collect(enumerate(test_set))
            fix_seed && Random.seed!(rng, i+j) # Fix the seed for evaluation
            results[test_starts[i] + j - 1] = eval_single(pomdp, policy, s; rng)
            next!(p)
        end
    end
    finish!(p)
    return Dict(k => [r[k] for r in results] for k in keys(results[1]))
end

function eval_kfolds_difftest(eval_pomdps, train_pomdps, policy_fn, test_sets; rng=Random.GLOBAL_RNG, fix_seed=true)
    Ntest_sets = length.(test_sets)
    test_starts = [1, (cumsum(Ntest_sets)[1:end-1] .+ 1)...]
    results = Array{Any}(undef, sum(Ntest_sets))
    p = Progress(sum(Ntest_sets), 1, "Evaluating policy on test sets...")
    for (i, (eval_pomdp, train_pomdp, test_set)) in collect(enumerate(zip(eval_pomdps, train_pomdps, test_sets)))
        fix_seed && Random.seed!(rng, i) # Fix the seed before potentially solving the policy
        policy = policy_fn(train_pomdp)
        Threads.@threads for (j, s) in collect(enumerate(test_set))
            fix_seed && Random.seed!(rng, i+j) # Fix the seed for evaluation
            results[test_starts[i] + j - 1] = eval_single(eval_pomdp, policy, s; rng)
            next!(p)
        end
    end
    finish!(p)
    return Dict(k => [r[k] for r in results] for k in keys(results[1]))
end