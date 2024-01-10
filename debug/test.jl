using EPEC
using GLMakie
include("../examples/simple_racing.jl")

probs = setup(; T=1, 
	Δt = 0.1, 
	r = 1.0, 
	α1 = 1e-3,
	α2 = 0e0,
	cd = 0.01,
	u_max_nominal = 2.0, 
	u_max_drafting = 5.0,
	box_length = 5.0,
	box_width = 1.0,
	lat_max = 1.0);

# fail 1:
#x0 = [0.024854049940194006
#2.0390008441072673
#0.09708099864039535
#5.389942809396402
#-0.9631049795145318
#1.2392045199100574
#0.33790040971200136
#6.392074687707442]

# fail 2:
#x0 = [0.5901398799223905
#2.405885659000482
#-0.7988008366961059
#5.778990383057139
#-0.73723565kee56277414
#2.1558857008413876
#-0.190253823106046
#5.778990405973726]

# fail 3:
#x0 = [0.31645082564541527
#5.976863815351941
#0.38240864202965985
#6.36091235492575
#0.13635980341250545
#4.632676398919672
#0.8101866978722575
#8.267781384678836]

# fail 4:
#x0 = [0.03993038200492152
#4.74345770246394
#0.3789478323350371
#5.973051985373068
#0.1037646806275564
#3.110305384464004
#0.09730542207976449
#7.441960761716991]

# fail 5:
#x0 = [-0.04002988014153725
#5.350464251495313
#-0.3997996010116964
#6.167079052887234
#-0.16343037678236172
#3.877791080397044
#-0.29680851668938474
#7.9222515131287325]

# fail 6:
#x0 = [1, 2, 0, 5, 0, 0, 0, 7]

# fail 7:
x0 = [-0.013275642150166479
5.3719728753939515
0.17631296392717447
8.346953868468791
-0.16959045248661958
3.8808146231625176
-0.997802997238535
10.398356457482667]

# fail 8:
#x0 = [-0.0021594995568859294
#28.650049887286247
#0.04285322107787094
#11.826233418728236
#-0.8293817806661944
#28.04797735124733
#-0.4369283534763312
#12.539518991822035]

# fail 9 (bad initilization?):
#x0 = [ -0.08320811747237386
#10.724680707142564
# 0.1876399601770615
# 9.494661937893104
#-0.8612120474445999
#10.047477657701165
#-0.7481148411504008
#10.198952048696231]


#x0=[0, 0, 0, 7, 0.1, -2.21, 0, 7]
#x0 = [0, 1, 0, 5, -1, 0, 0, 6]

#x0 = [.75, 0, 0, 5.5, -.75, 0, 0, 5]

# fail 10:
x0 =  [-0.0019028890141220287
13.65974905939654
-0.6658240275991629
10.073828363876014
-0.9273635894353759
13.268673804392947
-1.0082598757848815
10.895007101425024]

(; P1, P2, gd_both, h, U1, U2) = solve_seq(probs, x0);
# in case exfiltrated:
# before
#show_me(safehouse.x, safehouse.w)
# after
show_me(safehouse.θ_out, safehouse.w; T=1)

#(f, ax, XA, XB, lat) = visualize(; rad = sqrt(probs.params.r) / 2, lat = probs.params.lat_max + sqrt(probs.params.r) / 2);
#display(f)
#update_visual!(ax, XA, XB, x0, P1, P2; T = probs.params.T, lat = lat)

#sim_results = solve_simulation(probs, 10; x0);
#animate(probs, sim_results; save=false);
