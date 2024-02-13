function dist2(a,b)
    (a-b)'*(a-b)
end

function dist(a,b)
    sqrt((a-b)'*(a-b))
end

function disthalf(a,b)
    sqrt(sqrt((a-b)'*(a-b)))
end

function sigmoid(x, α)
    1.0 / (1 + exp(-α*x))
end

# Player i chases player h
function f(Z, i, N, m)

    h = mod1(i-1,N)
    j = mod1(i+1,N)
    # h i j

    M = 2*m + m
    Xis = [@view(Z[(i-1)*M+(k-1)*2+1:(i-1)*M+k*2]) for k in 1:m]
    probs_i = @view(Z[(i-1)*M+m*2+1:i*M])
    
    Xhs = [@view(Z[(h-1)*M+(k-1)*2+1:(h-1)*M+k*2]) for k in 1:m]
    probs_h = @view(Z[(h-1)*M+m*2+1:h*M])
    
    Xjs = [@view(Z[(j-1)*M+(k-1)*2+1:(j-1)*M+k*2]) for k in 1:m]
    probs_j = @view(Z[(j-1)*M+m*2+1:j*M])

    A = zeros(Num, m, m)
    for k1 = 1:m
        for k2 = 1:m
            A[k1, k2] = dist2(Xis[k1], Xhs[k2])
        end
    end
    probs_i'*A*probs_h
end

function f_simple(Z, i, N, m)
    M = 2*m + m
    Xis = [@view(Z[(i-1)*M+(k-1)*2+1:(i-1)*M+k*2]) for k in 1:m]
    probs_i = @view(Z[(i-1)*M+m*2+1:i*M])

    sum(xi'*xi for xi in Xis) + (probs_i-fill(0.5, m))'*(probs_i-fill(0.5, m))
end

# Player i avoids player j
function g_col(Z, i, N, m, d2, α)

    h = mod1(i-1,N)
    j = mod1(i+1,N)
    # h i j
    #
    M = 2*m + m
    Xis = [@view(Z[(i-1)*M+(k-1)*2+1:(i-1)*M+k*2]) for k in 1:m]
    probs_i = @view(Z[(i-1)*M+m*2+1:i*M])
    
    Xhs = [@view(Z[(h-1)*M+(k-1)*2+1:(h-1)*M+k*2]) for k in 1:m]
    probs_h = @view(Z[(h-1)*M+m*2+1:h*M])
    
    Xjs = [@view(Z[(j-1)*M+(k-1)*2+1:(j-1)*M+k*2]) for k in 1:m]
    probs_j = @view(Z[(j-1)*M+m*2+1:j*M])

    # Positive elements in A correspond to infeasibilities
    A = zeros(Num, m, m)
    for k1 = 1:m
        for k2 = 1:m
            A[k1, k2] = d2 - dist2(Xis[k1], Xjs[k2])
        end
    end

    # Elements in S are softened indicators of infeasibility
    S = sigmoid.(A, α)

    # Return probability of infeasibility (softened)
    probs_i'*S*probs_j
end

function g_standard(Z, i, N, m)
    M = 2*m + m
    Xis = [@view(Z[(i-1)*M+(k-1)*2+1:(i-1)*M+k*2]) for k in 1:m]
    probs_i = @view(Z[(i-1)*M+m*2+1:i*M])
    [vcat(Xis...); probs_i; sum(probs_i)]
end

function setup(; r=1.0, 
                 x_max = 2.0,
                 pv_max = 0.25,
                 N = 3,
                 m = 2,
                 p1_simple = false,
                 symmetric_start = true,
                 symmetric_end = true,
                 α = 2.0)
    M = 3*m
    lb = [fill(-x_max, 2*m); fill(0.0, m); 1]
    ub = [fill(+x_max, 2*m); fill(Inf, m); 1]

    OPs = map(1:N) do i
        if p1_simple
            fi_pinned = i == 1 && !symmetric_start ? (z -> f_simple(z, i, N, m)) : (z -> f(z, i, N, m))
        else
            fi_pinned = i == 1 && !symmetric_start ? (z -> -f(z, i+1, N, m)) : (z -> f(z, i, N, m))
        end
        gi_pinned = i == N && !symmetric_end ? (z -> g_standard(z, i, N, m)) : (z -> [g_standard(z, i, N, m); g_col(z, i, N, m, r, α)])
        lbi = i == N && !symmetric_end ? lb : [lb; -Inf]
        ubi = i == N && !symmetric_end ? ub : [ub; pv_max]
        OptimizationProblem(M*N, (i-1)*M+1:M*i, fi_pinned, gi_pinned, lbi, ubi)
    end
    gnep = hcat(OPs...)

    function extract_gnep(θ)
        Z = θ[gnep.x_inds]
        vars = Dict()
        for i in 1:N
            Xis = [@view(Z[(i-1)*M+(k-1)*2+1:(i-1)*M+k*2]) for k in 1:m]
            probs_i = @view(Z[(i-1)*M+m*2+1:i*M])
            for k in 1:m
                vars["X$(i)_$(k)"] = Xis[k]
            end
            vars["probs_$(i)"] = probs_i
        end
        vars
    end
    problems = (; gnep, extract_gnep, params=(; r, x_max, pv_max, N, m, α))
end

function solve_lifted(probs, seed)
    if !isnothing(seed)
        rng = MersenneTwister(seed)
    else
        rng = MersenneTwister()
    end
    θ = rand(rng, probs[1].gnep.top_level.n)
    for prob in probs
        #try
            θ = solve(prob.gnep, θ)
        #catch e
        #    Z = prob.extract_gnep(θ)
        #    α = prob.params.α
        #    return Z
        #end
    end
    Z = probs[end].extract_gnep(θ)
end

function timing_results()
    N_range = 2:7
    m_range = 2:5
    num_evals = 10
    times = zeros(length(N_range), length(m_range), num_evals)

    for (eN,N) in enumerate(N_range)
        for (em,m) in enumerate(m_range)
            probs = [EPEC.setup(; N, m, x_max=10.0, α = s, r=1.0, symmetric_start=false, symmetric_end=false, p1_simple=true) for s in [1.0,2,4,8,16,32,64,128]]
            seed = 1  
            tries = 0
            @info "N = $N, m = $m"
            display(sum(times; dims=3))
            while true
                try 
                    t0 = @elapsed (EPEC.solve_lifted(probs, seed))
                    ts = map(1:num_evals) do _
                        t = @elapsed (EPEC.solve_lifted(probs, seed))
                    end
                    times[eN, em, :] = ts
                    @info "Success."
                    break
                catch e
                    if e isa InterruptException
                        @infiltrate
                    end
                    tries += 1
                    if tries == 100
                        times[eN, em,:] .= -1.0
                        @info "Can't get to work. Continuing."
                        break
                    end
                    @info "incrementing seed."
                    seed += 1
                end
            end
        end
    end
    times
end



function visualize(probs, Z)

    f = Figure(resolution=(1000,1000))
    ax = Axis(f[1,1], aspect = DataAspect())
    r = sqrt(probs[1].params.r) / 2
    l = probs[1].params.x_max
    N = probs[1].params.N
    m = probs[1].params.m
    lines!(ax, [-l, -l], [-l, +l], color=:black, linewidth=3)
    lines!(ax, [-l, +l], [+l, +l], color=:black, linewidth=3)
    lines!(ax, [+l, +l], [+l, -l], color=:black, linewidth=3)
    lines!(ax, [+l, -l], [-l, -l], color=:black, linewidth=3)

    circ_x = [r*cos(t) for t in 0:0.1:(2π+0.1)]
    circ_y = [r*sin(t) for t in 0:0.1:(2π+0.1)]

    colors = [:red, :blue, :green, :brown, :orange]
    for i in 1:N 
        p = Z["probs_$i"]
        for k in 1:m
            x, y = Z["X$(i)_$(k)"]
            lines!(ax, circ_x .+ x, circ_y .+ y, color=(colors[i], p[k]), linewidth=3)
            scatter!(ax, x, y, color=(colors[i], p[k]), label="P$(i)_$(k)", markersize=10*(N+2-i))
        end
    end
    
    f[1, 2] = Legend(f, ax)
    save("lifted.png", f)
    f
end

