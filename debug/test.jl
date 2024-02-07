#using EPEC
#using GLMakie
#using Plots
include("../racing/racing.jl")

#probs = setup(; T=10,
#    Δt=0.1,
#    r=1.0,
#    α1=1e-3,
#    α2=1e-6,
#    α3=1e-1,
#    β=1e-1, #.5, # sensitive to high values
#    cd=0.2, #0.25,
#    u_max_nominal=1.0,
#    u_max_drafting=2.5, #2.5, # sensitive to high difference over nominal 
#    box_length=5.0,
#    box_width=2.0,
#    lat_max=2.0);


filtered_keys = Dict()

for (index, res) in results
	filtered_keys[index] = [k for (k, v) in res.steps if v == 50]
end

#intersected_keys = copy(filtered_keys[1])

#for (_, v) in filtered_keys
#	global intersected_keys
#	intersected_keys = intersect(intersected_keys, v);
#end 
intersected_keys = intersect(filtered_keys[3], filtered_keys[9])

# x0s[1000]
#i = rand(1:2000)
#@info "$i"
#i = 177
i = 33 #  32,41,29,24,33
x0 = x0s[i]
mymode = 3;


#x0 = [
#0.21585196022682754
#25.286691700718627
# 0.608529256154026
# 4.381609010493397
# 1.228515598004621
#24.818642861157482
# 1.4561311046522056
# 5.673645746247613
#]

# i = 1000
# step 7
#x0 = [
#	1.6313777749128915
#	2.424405121116234
#	1.0764163851034225-1.0
#	2.3670311446993164
#	0.05661870915779531
#   -0.10512992189500174
#	1.0764163794050767
#	2.294077918939926
#]  

for mymode in [3, 9]
	sim_results = solve_simulation(probs, 150; x0, mode=mymode);
	animate(probs, sim_results; save=true, filename="x0_$(i)_mode$(mymode).mp4");
end

#show_me(safehouse.θ_out, safehouse.w; T=probs.params.T, lat_pos_max=probs.params.lat_max + sqrt(probs.params.r) / 2)

#sim_results = solve_simulation(probs, 1; x0, mode=9);
#animate(probs, sim_results; save=false);

#(f, ax, XA, XB, lat) = visualize(; rad = sqrt(probs.params.r) / 2, lat = probs.params.lat_max + sqrt(probs.params.r) / 2);
#display(f)

# replace P1 and P2
#x0_swapped = copy(x0)
#x0_swapped[1:4] = x0[5:8]
#x0_swapped[5:8] = x0[1:4]

# plot 6
# P1 Leader, P2 Leader
#a_res = solve_seq_adaptive(probs, x0; only_want_gnep=false, only_want_sp=false)
#b_res = solve_seq_adaptive(probs, x0_swapped; only_want_gnep=false, only_want_sp=false)
#r_P1 = a_res.P1
#r_U1 = a_res.U1
#r_P2 = b_res.P1
#r_U2 = b_res.U1

#update_visual!(ax, XA, XB, x0, r_P1, r_P2; T = probs.params.T, lat = lat)
#update_visual!(ax, XA, XB, x0, b_res.P2, a_res.P2; T = probs.params.T, lat = lat)

# plot 10 
# P1 Follower, P2 Follower 
#a_res = solve_seq_adaptive(probs, x0_swapped; only_want_gnep=false, only_want_sp=false)
#b_res = solve_seq_adaptive(probs, x0; only_want_gnep=false, only_want_sp=false)
#r_P1 = a_res.P2
#r_U1 = a_res.U2
#r_P2 = b_res.P2
#r_U2 = b_res.U2

#update_visual!(ax, XA, XB, x0, r_P1, r_P2; T = probs.params.T, lat = lat)
#update_visual!(ax, XA, XB, x0, b_res.P1, a_res.P1; T = probs.params.T, lat = lat)

#prefs = zeros(Int, length(sim_results))
#for key in keys(sim_results)
#    #println("Key: $key, Pref: $(sim_results[key].lowest_preference)")
#	prefs[key] = sim_results[key].lowest_preference;
#end

#histogram(prefs, bins=1:9, xlabel="Type", ylabel="Frequency")    