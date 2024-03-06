# z = [x₁ x₂]
function extract_z(z)
    xdim = 2
    ind = Dict()
    idx = 0
    for (len, name) in zip([xdim, xdim, xdim, xdim], ["a", "b", "aa", "bb"])
        ind[name] = (idx+1):(idx+len)
        idx += len
    end
    @inbounds a = @view(z[ind["a"]])
    @inbounds b = @view(z[ind["b"]])
	@inbounds aa = @view(z[ind["aa"]])
    @inbounds bb = @view(z[ind["bb"]])
    (a, b, aa, bb, ind)
end

# e: ego
# o: opponent
function f_ego(xe, xo; g, r)
	(xe[1] - g[1])^2 + (xe[2] - g[2])^2 + (xo[1] - xe[1] - r[1])^2 + (xo[2] - xe[2] - r[2])^2
end

function f1(z; g=[1, 0], r=[1,1])
    a, b, aa, bb = extract_z(z)
	f_ego(a, b; g, r)
end

function f2(z; g=[-1, 0], r=[1,1])
	a, b, aa, bb = extract_z(z)
	f_ego(b, a; g, r)
end

function f3(z; g=[1, 0], r=[1,1])
	a, b, aa, bb = extract_z(z)
	f_ego(aa, b; g, r)
end

function f4(z; g=[-1, 0], r=[1,1])
	a, b, aa, bb = extract_z(z)
	f_ego(bb, a; g, r)
end

function g1(z)
    Vector{eltype(z)}()
end

function g2(z)
    Vector{eltype(z)}()
end

function g3(z)
    Vector{eltype(z)}()
end

function g4(z)
    Vector{eltype(z)}()
end

function setup()
    OPa = OptimizationProblem(8, 1:2, f1, g1, [], [])
    OPb = OptimizationProblem(8, 1:2, f2, g2, [], [])
	OPaa = OptimizationProblem(8, 1:2, f3, g3, [], [])
	OPbb = OptimizationProblem(8, 1:2, f4, g4, [], [])
	gnep = [OPa OPb];
	bilevel = [OPa; OPb];
    all = [OPa OPb; OPaa OPbb]

	(; gnep, bilevel, all)
end

function visualize(z)
    f = Figure(resolution=(500, 500), grid=false)
    ax = Axis(f[1, 1], aspect=DataAspect())

	a, b, aa, bb = extract_z(z)

	GLMakie.lines!(ax, a, b, color=:blue, linewidth=1)
	GLMakie.lines!(ax, a, bb, color=:blue, linewidth=1, linestyle=:dash)
	GLMakie.lines!(ax, b, aa, color=:red, linewidth=1, linestyle=:dash)
	display(f)
end