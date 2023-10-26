struct OptimizationProblem{T1<:Integer, T2<:Real}
    n::T1 # domain dimension
    #m::T1 # constraint dimension
    dvars::Vector{T1}
    f::Function
    g::Function
    l::Vector{T2}
    u::Vector{T2}
    #∇ℒ::Function
    #∇²ℒ_vals::Function
    #∇²ℒ_rows::Vector{T1}
    #∇²ℒ_cols::Vector{T1}
    #∇g_vals::Function
    #∇g_rows::Vector{T1}
    #∇g_cols::Vector{T1}
end

#function OptimizationProblem(n, dvars, f, g, l, u)
#    (m = length(l)) == length(u) || error("length of l and u are not equal")
#
#    x_sym, λ_sym = Symbolics.@variables(x_sym[1:n], λ_sym[1:m]) .|> Symbolics.scalarize
#
#    g_sym = g(x_sym)
#    ℒ = f(x_sym) - λ_sym'*g_sym
#    ∇ℒ = Symbolics.gradient(ℒ, x_sym)
#
#    ∇g = Symbolics.sparsejacobian(g_sym, x_sym)
#    (rows_g, cols_g, sym_vals_g) = findnz(∇g)
#
#    ∇²ℒ = Symbolics.sparsejacobian(∇ℒ, x_sym)
#    (rows_ℒ, cols_ℒ, sym_vals_ℒ) = findnz(∇²ℒ)
#
#    ∇ℒ! = Symbolics.build_function(∇ℒ, x_sym, λ_sym; expression = Val(false))[2]
#
#    ∇²ℒ_vals! = Symbolics.build_function(sym_vals_ℒ, x_sym, λ_sym; expression = Val(false))[2]
#    ∇g_vals! = Symbolics.build_function(sym_vals_g, x_sym; expression = Val(false))[2]
#
#    OptimizationProblem(n, m, dvars, f, g, l, u, ∇ℒ!, ∇²ℒ_vals!, rows_ℒ, cols_ℒ, ∇g_vals!, rows_g, cols_g)
#end

function Base.hcat(OPs::OptimizationProblem...)
    @assert allequal(OP.n for OP in OPs)
    @assert isempty(intersect((OP.dvars for OP in OPs)...))
    N = length(OPs)
    n = first(OPs).n
    n_privates = [length(OP.dvars) for OP in OPs]
    m_privates = [length(OP.l) for OP in OPs] 
    m = sum(m_privates)

    #x = Symbolics.@variables(x[1:n])[1] |> Symbolics.scalarize
    xs = map(1:N) do i
        sym = Symbol("x", i)
        Symbolics.@variables($sym[1:n_privates[i]])[1] |> Symbolics.scalarize
    end
    λs = map(1:N) do i
        sym = Symbol("λ", i)
        Symbolics.@variables($sym[1:m_privates[i]])[1] |> Symbolics.scalarize
    end
    ss = map(1:N) do i
        sym = Symbol("s", i)
        Symbolics.@variables($sym[1:m_privates[i]])[1] |> Symbolics.scalarize
    end

    x = Vector{Num}(undef, n)
    for (e, OP) in enumerate(OPs)
        x[OP.dvars] = xs[e]
    end
    λ = vcat(λs...)
    s = vcat(ss...)

    grad_lags = mapreduce(vcat, enumerate(OPs)) do (i, OP)
        Lag = OP.f(x) - λs[i]'*OP.g(x)
        grad_lag = Symbolics.gradient(Lag, xs[i])
    end
    cons_s = mapreduce(vcat, enumerate(OPs)) do (i, OP)
        OP.g(x) - ss[i]
    end
    F = [grad_lags; cons_s; λ]
    F! = Symbolics.build_function(F, x, s, λ; expression = Val(false))[2]
    FF = (result, z) -> begin
        F!(result, @view(z[1:n]), @view(z[n+1:n+m]), @view(z[n+m+1:end]))
    end

    J = Symbolics.sparsejacobian(F, [x; s; λ])
    (rows, cols, vals) = findnz(J)
    J_vals! = Symbolics.build_function(vals, x, s, λ; expression = Val(false))[2]
    JV = (result, z) -> begin
        J_vals!(result, @view(z[1:n]), @view(z[n+1:n+m]), @view(z[n+m+1:end]))
    end

    l = fill(-Inf, n+m)
    append!(l, (OP.l for OP in OPs)...)
    u = fill(Inf, n+m)
    append!(u, (OP.u for OP in OPs)...)

    dvars = collect(1:length(l))
    
    MixedComplementarityProblem(n, FF, JV, rows, cols, l, u, dvars)
end


struct MixedComplementarityProblem
    n::Cint
    F::Function
    J_vals::Function
    J_rows::Vector{Cint}
    J_cols::Vector{Cint}
    l::Vector{Cdouble}
    u::Vector{Cdouble}
    dvars::Vector{Cint}
end

function solve(mcp::MixedComplementarityProblem, z=Vector{Cdouble}; silent=false)
    @assert length(z) == length(mcp.dvars)
    pvars = setdiff(1:mcp.n, mcp.dvars) |> sort

    function F(n, z, result)
        mcp.F(result, z)
        Cint(0)
    end

    d_mask = map(mcp.J_cols) do i
        i ∈ mcp.dvars
    end
    p_mask = .!d_mask

    nnz_d = sum(d_mask)
    nnz_p = sum(p_mask)

    # Jacobian is with respect to all variables, including parameter variables.
    # PATH needs to reason about jacobian of F w.r.t. decision variables only.
    # The following code splits J into the portions w.r.t. dvars and pvars, for
    # use in the function below.
    J_full = sparse(mcp.J_rows, mcp.J_cols, Vector{Cdouble}(undef, length(mcp.J_rows)), length(mcp.dvars), length(mcp.dvars))
    J_dvars = J_full[:, d_mask]
    J_pvars = J_full[:, p_mask]
    val_buffer = zeros(Cdouble, nnz_d+nnz_p)
    J_col = J_dvars.colptr[1:end-1]
    J_len = diff(J_dvars.colptr)
    J_row = J_dvars.rowval

    function J(n, nnz, z, col, len, row, data)
        val_buffer .= 0.0
        mcp.J_vals(val_buffer, z)
        data .= val_buffer[d_mask]
        col .= J_col
        len .= J_len
        row .= J_row
        Cint(0)
    end

    status, z, info = PATHSolver.solve_mcp(
         F,
         J,
         mcp.l,
         mcp.u,
         z;
         silent,
         nnz = nnz_d,
         jacobian_structure_constant = true,
         jacobian_data_contiguous = true,
     ) 
    (; status, z, info)
end
