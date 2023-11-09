# X := [x1 x2 x3 x4]
# x1,x2 position of P1
# x3,x4 position of P2

# P1 wants to be ahead of origin (in second dimension)
function f1(x)
    x[1]^2 + (x[2]-1.0)^2
end

# P2 wants to be near origin
function f2(x)
    x[3]^2 + x[4]^2
end

function sigmoid(x, a)
    1.0 / (1.0 + exp(-a*x))
end

# lower bound for P1 -- above zero whenever (x1-x3) ≥ 0, below zero otherwise
function l1(x; a=2.0, b=3.0)
    d = x[1] - x[3]
    sigmoid(d+b, a) - sigmoid(b, a)
end

# lower bound for P2 -- above zero whenever (x1-x3) ≤ 0, below zero otherwise
function l2(x; a=2.0, b=3.0)
    d = x[1] - x[3]
    sigmoid(-d+b, a) - sigmoid(b, a)
end

function g1(x; a=2.0, b=3.0)
    [ (x[1]-x[3]) - (x[2]-x[4]) - l1(x; a, b), ]
end

function g2(x; a=2.0, b=3.0)
    [ (x[1]-x[3]) - (x[2]-x[4]) - l2(x; a, b), ]
end

function setup()
    OP1 = OptimizationProblem(4, [1,2], f1, g1, [0.0,], [Inf,])
    OP2 = OptimizationProblem(4, [3,4], f2, g2, [0.0,], [Inf,])
    (; OP1, OP2)
end
