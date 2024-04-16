using EPEC

probs = setup(; T=10,
    Δt=0.1,
    r=1.0,
    α1=1e-3,
    α2=1e-4,
    β=1e-1, #.5, # sensitive to high values
    cd=0.2, #0.25,
    d=2.0, # actual road width (±)
    u_max_nominal=1.0,
    u_max_drafting=2.5, #2.5, # sensitive to high difference over nominal 
    box_length=5.0,
    box_width=5.0,
    lat_max=4.5 # just used for visulization rn 2024-04-10 (should be fixed)
    );

#x0 = [1.0, 3, 0, 1, -1, 2, 0, 1.5] # it's helpful to start from an initial velocity difference for oscillating behavior but so sensitive
#x0 = [.5, 0, 1, π/2, 0, 2, 1, π/2]	
#x0 = [-.5, 0, 2, π/2, 0, 2, 1, π/2]	
x0 = [0, 2, 1, π/2,-.5, 0, 2, π/2]	
#
#x0 = [-0.9339583266548017
#    0.0
#    2.5321772895121644
#    1.5707963267948966
#    -1.1050599392379927
#    -2.950381877667683
#    1.6054315472628076
#    1.5707963267948966]

#road = Dict(3 => 0, 6 => 0, 9 => -1, 12 => -2.5, 15 => -2.5, 18 => -1, 21 => 0, 24 => .5, 27 => 0);
#road = Dict(0 => 0, 2 => 0, 4=>-.2, 6=>-.5, 8=>-.9, 10=>-1.7, 12=>-2.4, 14=>-2.4, 16=>-1.6, 18=>-.7, 20=>.7, 22=>1.6, 24=>2.4, 26=>2.4, 28=>1.6, 30=>.9,32=>.5,34=>.2,36=>0,38=>0, 40=>-.01);
#road = Dict(0 => 0, 2 => 0, 4=>.1, 6=>0, 8=>-.9, 10=>-1.7, 12=>-2.4, 14=>-2.4, 16=>-1.6, 18=>-.7, 20=>.7, 22=>1.6, 24=>2.4, 26=>2.4, 28=>1.6, 30=>.9,32=>.5,34=>.2,36=>0,38=>0, 40=>-.01);
road = Dict(
    -4 => 0.0,
    -2 => 0.01,
    0 => 0,
    2 => 0.01,
    4 => 0,
    6 => 0.01,
    8 => 0,
    10 => 0.01,
    12 => 0.0, # begin right turn
    12.5 => 0.09,
    13 => 0.25,
    13.5 => 0.5,
    14 => 0.8,
    15 => 1.6,
    16 => 2.3,
    16.5 => 2.45,
    17 => 2.5, # right peak, begin left turn
    17.5 => 2.45,
    18 => 2.3,
    19 => 1.6,
    20 => 0.6,
    21 => -0.6,
    22 => -1.6,
    23 => -2.3,
    23.5 => -2.45,
    24 => -2.5, # left peak
    24.5 => -2.45,
    25 => -2.3,
    26 => -1.6,
    27 => -0.8,
    27.5 => -0.5,
    28 => -0.25,
    28.5 => -0.09,
    29 => -0, # end of turns
    31 => 0.01,
    33 => 0,
    35 => 0.01,
    37 => 0.0,
    39 => 0.01,
    41 => 0.0);
mode = 9;
sim_results = solve_simulation(probs, 50; x0, road, mode);
EPEC.animate(probs, sim_results; save=false, mode, road);

#(; P1, P2, gd_both, h, U1, U2, lowest_preference, sorted_Z) = solve_seq_adaptive(probs, x0);
#(f, ax, XA, XB, lat) = visualize(; rad = sqrt(probs.params.r) / 2, lat = probs.params.lat_max + sqrt(probs.params.r) / 2);
#display(f)
#update_visual!(ax, XA, XB, x0, P1, P2; T = probs.params.T, lat = lat)

#prefs = zeros(Int, length(sim_results))
#for key in keys(sim_results)
#    #println("Key: $key, Pref: $(sim_results[key].lowest_preference)")
#	prefs[key] = sim_results[key].lowest_preference;
#end

#histogram(prefs, bins=1:9, xlabel="Type", ylabel="Frequency")    