# Responsibility handling test
# p1, p2 ∈ ℝ² (positions)
# x := [p1; p2]
using EPEC

function f1(x)
    x1 = @view(x[1])
    y1 = @view(x[2])
    return x1.^2+ (y1 .- 1).^2
end

function f2(x)
    x2 = @view(x[3])
    y2 = @view(x[4])
    return x2.^2 + (y2).^2
end

function g1(x)
    x1 = @view(x[1])
    y1 = @view(x[2])
    x2 = @view(x[3])
    y2 = @view(x[4])
    return [(x1 - x2) - (y1 - y2) .- l1(x1 - x2)]
end

function g2(x)
    x1 = @view(x[1])
    y1 = @view(x[2])
    x2 = @view(x[3])
    y2 = @view(x[4])
    [(x1 - x2) - (y1 - y2) .- l2(x1 - x2)]
end

function modified_sigmoid(x; x_offset = 0., y_offset = 0., r = 10.) 
    return 2.0 / (1 + exp(-r.*(x .- x_offset))) - 1.0 + y_offset
end

function l1(x) 
    return 0;#modified_sigmoid(x) -.0
end

function l2(x) 
    return 0;#- modified_sigmoid(x) -.0
end

function setup()
    OP1 = OptimizationProblem(4, [1,2], f1, g1, [-Inf,], [Inf])
    OP2 = OptimizationProblem(4, [3,4], f2, g2, [-Inf,], [Inf])

    MCP = [OP1 OP2]

    p1_init = [-1., 0.0]
    p2_init = [-2., 0.]
    z0 = zeros(MCP.n)
    z0[1:4] = [p1_init; p2_init]
    ret = EPEC.solve(MCP, z0; silent=false)

    p1 = ret.z[1:2]
    p2 = ret.z[3:4]

    λ1 = ret.z[5:6]
    λ2 = ret.z[7:8]

    (; ret.z, MCP, p1, p2, λ1, λ2)
end

