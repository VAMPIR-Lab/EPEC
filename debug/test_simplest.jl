
include("../examples/simplest_racing.jl")

probs = simplest_racing.setup(; T=1,
    Δt=0.1,
    r=1.0,
    α1=1e-4,
    α2=0e0,
    β=1e4,
    lat_vel_max=1.0,
    long_vel_max=10.0,
    lat_pos_max=1.5)
#simplest_racing.show_me(safehouse.θ_out, safehouse.w; T=probs.params.T, lat_pos_max=probs.params.lat_pos_max + sqrt(probs.params.r) / 2)

x0 = [0.;0.;.1;-2.0]
sim_results = simplest_racing.solve_simulation(probs, 40; x0);
simplest_racing.animate(probs, sim_results; save=false);