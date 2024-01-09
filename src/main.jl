using EPEC, Plots, LaTeXStrings

include("../examples/simple_ownership.jl")
epec = setup();
x = EPEC.solve(epec);
x[epec.x_inds]

d_range = collect(-1:0.01:1);
plot(d_range, l1.(d_range), label=L"l_1")
plot!(d_range, l2.(d_range), xlabel="h", ylabel="", label=L"l_2")

#(;z)=setup();
#z