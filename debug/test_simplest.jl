
include("../examples/simplest_racing.jl")

probs = simplest_racing.setup(; T=10,
    Δt=0.1,
    r=1.0,
    α1=1e-3,
    α2=0.,
    β=1e1,
    lat_vel_max=5.0,
    long_vel_max=10.0,
    lat_pos_max=2.0)

x0 = [0.;0.;.5;-1.0]
#x0 =  [-0.0019028890141220287
#13.65974905939654
#-0.9273635894353759
#13.268673804392947]

(; P1, P2, gd_both, h, U1, U2) = simplest_racing.solve_seq(probs, x0);
#simplest_racing.show_me(safehouse.θ_out, safehouse.w; T=probs.params.T, lat_pos_max=probs.params.lat_pos_max + sqrt(probs.params.r) / 2)

sim_results = simplest_racing.solve_simulation(probs, 20; x0);
simplest_racing.animate(probs, sim_results; save=false);