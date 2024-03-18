using EPEC
using GLMakie
using Plots
include("../racing/phantom_racing.jl")
include("../racing/visualize_phantom_racing.jl")

# todo ... 
# OP1 -- OP2
#  |     |
# OPa   OPb

# modes...
# gnep: OP1 OP2
# bilevel: OP1 <- OPa
# bilevel: OP2 <- OPb
probs = setup(; T=10,
    Δt=0.1,
    r=1.0,
    α1=1e-3,
    α2=1e-4,
    β=1e-1, #.5, # sensitive to high values
    γ=1.0,
    cd=0.2, #0.25,
    u_max_nominal=1.0,
    u_max_drafting=2.5, #2.5, # sensitive to high difference over nominal 
    box_length=5.0,
    box_width=2.0,
    lat_max=2.0);

# γ=1, no drafting 
x0 = [1.0, 3, 0, 1, -1, 2, 0, 1.5] # it's helpful to start from an initial velocity difference for oscillating behavior but so sensitive

# sometimes I randomly sample from x0s_2000samples_2024-02-01_1739 -- I call include("racing/process_results.jl") first
#i = rand(1:2000)
#@info "$i"
#x0 = x0s[i];

#x0 = [  0.6424983411163723
#0.0
#0.0
#2.456100077007062
#1.017736337568136
#-2.3921907199247454
#0.0
#1.7727958431642168]
#x0 = [1.5966679528041663
#3.0
#0.0
#2.4377286884160085
#-0.3161517746722031
#2.200620080698424
#0.0
#2.75401313424805]

sim_results = solve_simulation(probs, 20; x0);

#animate(probs, sim_results; save=false, update_phantoms=false);
#show_me(probs.extract(θ, probs.gnep.x_inds), x0)
animate(probs, sim_results; save=false, update_phantoms=true);
