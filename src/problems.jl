struct OptimizationProblem
    n::Int # domain dimension
    dvars::Vector{Int}
    f::Function
    g::Function
    l::Vector{Float64}
    u::Vector{Float64}
end

function Base.hvcat(blocks_per_row::Tuple{Vararg{Int}}, OPs::OptimizationProblem...)
    @assert length(blocks_per_row) == 2
    @assert allequal(OP.n for OP in OPs)
    N1, N2 = blocks_per_row
    N = N1 + N2
    @assert N1+N2 == length(OPs)
    
    n = first(OPs).n

    top_OPs = OPs[1:N1]
    bot_OPs = OPs[N1+1:end]
    
    n_privates = [length(OP.dvars) for OP in OPs]
    m_privates = [length(OP.l) for OP in OPs]
    
    dim_z_low = sum(n_privates[i]+2m_privates[i] for i in N1+1:N1+N2; init=0)
    dim_total = sum(n_privates[i]+2m_privates[i]+4*dim_z_low for i in 1:N1; init=0)
    # Each top-level player: privates + duals on private cons + slacks on
    # private cons + slacks on z cons + duals on z cons + duals on z agreement
    # + copy of z

    θ = Symbolics.@variables($sym[1:dim_total])[1] |> Symbolics.scalarize
    # θ := [x₁ ... xₙ₁ | z₁ ... zₙ | λ₁ ... λₙ | s₁ ... sₙ | ψ₁ ... ψₙ | r₁ ... rₙ | γ₁ ... γₙ] 
    
    ind = 0
    vars = Dict()
    inds = Dict()

    for (key_base, len_itr) in zip(["x", "z", "λ", "s", "ψ", "r", "γ"], 
                                   [n_privates, dim_z_low, m_privates, m_privates, dim_z_low, dim_z_low, dim_z_low])
        lens = length(len_itr) == 1 ? fill(len_itr, N1) : len_itr
        for i in 1:N1
            len = lens[i]
            vars[key_base, i] = θ[ind+1:ind+len]
            inds[key_base, i] = ind+1:ind+len
            ind+=len
        end
    end

    z = vars["z", 1] # any player's copy of z would do (i.e. z2, z3, ..., or zₙ)
    ind = 0
    for (key_base, len_itr) in zip(["x", "λ", "s"], [n_privates, m_privates, m_privates])
        for i in N1+1:N1+N2
            len = len_itr[i]
            vars[key_base, i] = z[ind+1:ind+len]
            ind += len
        end
    end

    x = vcat((vars["x",i] for i in 1:N1+N2)...)

    # construct F for low-level MCP
    grad_lags = mapreduce(vcat, N1+1:N1+N2) do i
        Lag = OPs[i].f(x) - vars["λ", i]'*OPs[i].g(x)
        grad_lag = Symbolics.gradient(Lag, vars["x", i])
    end 
    cons_s = mapreduce(vcat, N1+1:N1+N2) do i
        OPs[i].g(x) - vars["s", i]
    end
    λs = vcat((vars["λ",i] for i in N1+1:N1+N2)...)

    F = [grad_lags; cons_s; λs]

    F! = Symbolics.build_function(F, θ; expression = Val(false))[2]
    J = Symbolics.sparsejacobian(F, z) 
    (J_rows, J_cols, J_vals) = findnz(J)
    J_vals! = Symbolics.build_function(J_vals, θ; expression = Val(false))[2]

    l = fill(-Inf, length(grad_lags)+length(cons_s))
    u = fill(+Inf, length(grad_lags)+length(cons_s))
    for i in N1+1:N1+N2
        append!(l, OPs[i].l)
        append!(u, OPs[i].u)
    end

    low_level = (; F!, J_rows, J_cols, J_vals!, z_inds=inds["z", 1], l, u)
    
    # reminder :  θ := [x₁ ... xₙ₁ | z₁ ... zₙ | λ₁ ... λₙ | s₁ ... sₙ | ψ₁ ... ψₙ | r₁ ... rₙ | γ₁ ... γₙ] 

    grad_lags_x = mapreduce(vcat, 1:N1) do i
        Lag = OPs[i].f(x) - vars["λ", i]'*OPs[i].g(x) - vars["ψ", i]'*F
        grad_lag = Symbolics.gradient(Lag, vars["x", i])
    end
    grad_lags_z = mapreduce(vcat, 1:N1) do i
        Lag = OPs[i].f(x) - vars["λ", i]'*OPs[i].g(x) - vars["ψ", i]'*F - vars["γ", i]
        grad_lag = Symbolics.gradient(Lag, vars["z", i])
    end
    cons_s_top = mapreduce(vcat, 1:N1) do i
        OPs[i].g(x) - vars["s", i]
    end
    λs_top = vcat((vars["λ",i] for i in 1:N1)...)
    cons_r = mapreduce(vcat, 1:N1) do i
        F - vars["r", i]
    end
    ψs = vcat((vars["ψ",i] for i in 1:N1)...)
    cons_z = mapreduce(vcat, 1:(N1-1)) do i
        -vars["z",i+1] + vars["z", i]
    end
    append!(cons_z, -vars["z",1] + vars["z", N1])

    Ftotal = [grad_lags_x; grad_lags_z; cons_s_top; λs_top; cons_r; ψs; cons_z]
    ltotal = [fill(-Inf, length(grad_lags_x));
              fill(-Inf, length(grad_lags_z)); # will get overwritten by templates
              fill(-Inf, length(cons_s_top));
              vcat((OPs[i].l for i in 1:N1)...);
              fill(-Inf, length(cons_r));
              fill(-Inf, length(ψs)); # will get overwritten by templates
              fill(-Inf, length(cons_z))]
    utotal = [fill(+Inf, length(grad_lags_x));
              fill(+Inf, length(grad_lags_z)); # will get overwritten by templates
              fill(+Inf, length(cons_s_top));
              vcat((OPs[i].u for i in 1:N1)...);
              fill(+Inf, length(cons_r));
              fill(+Inf, length(ψs)); # will get overwritten by templates
              fill(+Inf, length(cons_z))]

    # these are needed for assigning bounds from low-level solutions
    ind = 0
    base = length(grad_lags_x)
    z_inds_top = map(1:N1) do i
        inds = base .+ (ind+1):(ind+dim_z_low)
        ind += dim_z_low
        inds
    end
    ind = 0
    base = length(grad_lags_x)+length(grad_lags_z)+length(cons_s_top)+length(λs_top)+length(cons_r) 
    r_inds_top = map(1:N1) do i
        inds = base .+ (ind+1):(ind:dim_z_low)
        ind += dim_z_low
        inds
    end

    Ftotal! = Symbolics.build_function(Ftotal, θ; expression = Val(false))[2]
    Jtotal = Symbolics.sparsejacobian(Ftotal, θ) 
    (Jtotal_rows, Jtotal_cols, Jtotal_vals) = findnz(Jtotal)
    Jtotal_vals! = Symbolics.build_function(Jtotal_vals, θ; expression = Val(false))[2]

    top_level = (; F! = Ftotal!, 
                   J_rows = Jtotal_rows, 
                   J_cols = Jtotal_cols, 
                   J_vals! = Jtotal_vals!, 
                   z_inds = z_inds_top, 
                   r_inds = r_inds_top, 
                   l = ltotal, 
                   u = utotal)

    (; low_level, top_level)
end

function solve(epec, θ; tol=1e-6)
    low_level = epec.low_level
    top_level = epec.top_level

    converged = false
    while !converged
        (; status, info) = solve_low_level!(low_level, θ) # this should be redundant after the initial iteration
        solution_graph = get_solution_graph(low_level, θ)
        converged = true
        for S in solution_graph
            bounds = convert_recipe(low_level, S)
            (; dθ, status, info) = solve_top_level(top_level, bounds, θ)
            if (norm(dθ) < tol)
                converged = false
                θ += dθ
                break
            end
        end
    end
    return θ
end

function solve_top_level(mcp, bounds, θ; silent=false)
    n = length(mcp.l)
    nnz = length(mcp.J_rows)
    J_shape = sparse(mcp.J_rows, mcp.J_cols, Vector{Cdouble}(undef, nnz), n, n)
    J_col = J_shape.colptr[1:end-1]
    J_len = diff(J_shape.colptr)
    J_row = J_shape.rowval
    function F(n, θ, result)
        mcp.F!(result, θ)
    end
    function J(n, nnz, θ, col, len, row, data)
        mcp.J_vals!(data, θ)
        col .= J_col
        len .= J_len
        row .= J_row
        Cint(0)
    end

    l = mcp.l
    u = mcp.u 

    for ind_set in mcp.z_inds
        l[ind_set] .= bounds.lz
        u[ind_set] .= bounds.uz
    end
    for ind_set in mcp.r_inds
        l[ind_set] .= bounds.lf
        u[ind_set] .= bounds.uf
    end

    status, θ_out, info = PATHSolver.solve_mcp(
         F,
         J,
         l,
         u,
         θ;
         silent,
         nnz,
         jacobian_structure_constant = true,
         jacobian_data_contiguous = true,
     ) 

    dθ = θ_out - θ
    (; dθ, status, info)
end
     


function solve_low_level!(mcp, θ; silent=false)
    n = length(mcp.l)
    θF = copy(θ)
    nnz = length(mcp.J_rows)
    J_shape = sparse(mcp.J_rows, mcp.J_cols, Vector{Cdouble}(undef, nnz), n, n)
    J_col = J_shape.colptr[1:end-1]
    J_len = diff(J_shape.colptr)
    J_row = J_shape.rowval
    z = θF[mcp.z_inds]

    function F(n, z, result)
        θF[mcp.z_inds] .= z
        mcp.F!(result, θF)
        Cint(0)
    end
    function J(n, nnz, z, col, len, row, data)
        θF[mcp.z_inds] .= z
        mcp.J_vals!(data, θF)
        col .= J_col
        len .= J_len
        row .= J_row
        Cint(0)
    end
    
    status, z_out, info = PATHSolver.solve_mcp(
         F,
         J,
         mcp.l,
         mcp.u,
         z;
         silent,
         nnz,
         jacobian_structure_constant = true,
         jacobian_data_contiguous = true,
     ) 

    θ[mcp.z_inds] .= z_out 
    (; status, info)
end

function get_local_solution_graph(mcp, θ; tol=1e-5)
    l = mcp.l  
    u = mcp.u
    n = length(l)
    f = zeros(n)
    mcp.F!(f, θ)
    z = @view θ[mcp.z_inds]
    J = Dict{Int, Vector{Int}}()
    for i in 1:n
        Ji = Int[]
        if f[i] ≥ tol && z[i] < l[i]+tol
            push!(Ji, 1)
        elseif -tol < f[i] < tol && z[i] < l[i]+tol
            push!(Ji, 1)
            push!(Ji, 2)
        elseif -tol < f[i] < tol && l[i]+tol ≤ z[i] ≤ u[i]-tol
            push!(Ji, 2)
        elseif -tol < f[i] < tol && z > u[i]-tol
            push!(Ji, 2)
            push!(Ji, 3)
        elseif f[i] ≤ -tol && z > u[i]-tol
            push!(Ji, 3)
        elseif isapprox(l[i], u[i]; atol=tol)
            push!(Ji, 4)
        end
        J[i] = Ji
    end
    valid_solution = !any(isempty.(Ji for Ji in values(J)))
    !valid_solution && error("Not a valid solution!") 
    recipes = get_all_recipes(J)
end

function get_all_recipes(J)
    Ks = Vector{Dict{Int, Set{Int}}}()
    N = length(J)
    multiples = [i for i in 1:N if length(J[i]) > 1]
    singles = setdiff(1:N, multiples)
    It = Iterators.product([J[i] for i in multiples]...)
    for assignment in It
        K = Dict(j=>Set{Int}() for j in 1:4)
        for (e,ej) in enumerate(assignment)
            push!(K[ej], e)
        end
        for e in singles
            push!(K[J[e]], e)
        end
        push!(Ks, K)
    end
    Ks
end

function convert_recipe(mcp, recipe)
    n = length(mcp.l)

    lf = zeros(n)
    uf = zeros(n)
    lz = zeros(n)
    uz = zeros(n)

    for i in K[1]
        lf[i] = 0.0
        uf[i] = Inf
        lz[i] = mcp.l[i]
        uz[i] = mcp.l[i]
    end
    for i in K[2]
        lf[i] = 0.0
        uf[i] = 0.0
        lz[i] = mcp.l[i]
        uz[i] = mcp.u[i]
    end
    for i in K[3]
        lf[i] = -Inf
        uf[i] = 0.0
        lz[i] = mcp.u[i]
        uz[i] = mcp.u[i]
    end
    for i in K[4]
        lf[i] = -Inf
        uf[i] = Inf
        lz[i] = mcp.l[i]
        uz[i] = mcp.u[i]
    end
    (; lf, uf, lz, uz)
end
