function visualize(; T=10, rad=0.5, lat_max=2.5)
    f = Figure(resolution=(500, 1000), grid=false)
    ax = Axis(f[1, 1], aspect=DataAspect())

    carsymbol_string = "M 7 9 C 5.897 9 5 9.897 5 11 L 5 11.4375 C 6.328 12.0205 7.324 13.1585 7.75 14.5625 C 7.917 14.8145 8 15.245 8 16 L 8 18 L 11 18 L 11 19 L 13 19 L 13 18 L 15 18 C 16.103 18 17 17.103 17 16 L 17 11 C 17 9.897 16.103 9 15 9 L 7 9 z M 34 10 C 32.897 10 32 10.897 32 12 L 32 16 C 32 17.103 32.897 18 34 18 L 35 18 L 35 19 L 37 19 L 37 18 L 38 18 L 38 19 L 40 19 L 40 18 L 41 18 C 42.103 18 43 17.103 43 16 L 43 12 C 43 10.897 42.103 10 41 10 L 34 10 z M 1 13 C 0.448 13 0 13.448 0 14 L 0 36 C 0 36.553 0.448 37 1 37 L 3 37 C 4.654 37 6 35.654 6 34 L 6 16 C 6 14.346 4.654 13 3 13 L 1 13 z M 45 13 C 44.447 13 44 13.448 44 14 L 44 18.625 C 47.242 19.242 49.80475 20.10125 49.96875 20.15625 C 49.98475 20.16125 49.985 20.1825 50 20.1875 L 50 14 C 50 13.448 49.553 13 49 13 L 45 13 z M 20 15 C 19.649 15 19.33625 15.199 19.15625 15.5 L 16.4375 20 L 8 20 L 8 30 L 16.4375 30 L 19.15625 34.5 C 19.33625 34.8 19.649 35 20 35 L 27 35 C 27.208 35 27.42475 34.9345 27.59375 34.8125 L 34.3125 30 L 38 30 C 43.05 30 49.0605 28.0215 49.3125 27.9375 C 49.7235 27.8025 50 27.432 50 27 L 50 23 C 50 22.57 49.7205 22.1995 49.3125 22.0625 C 49.0635 21.9785 43.157 20 38 20 L 34.3125 20 L 27.59375 15.1875 C 27.42475 15.0665 27.208 15 27 15 L 20 15 z M 22 22 L 27 22 C 28.657 22 30 23.343 30 25 C 30 26.657 28.657 28 27 28 L 22 28 L 22 22 z M 50 29.8125 C 49.981 29.8195 49.9565 29.83775 49.9375 29.84375 C 49.7705 29.89875 47.219 30.72875 44 31.34375 L 44 36 C 44 36.553 44.447 37 45 37 L 49 37 C 49.553 37 50 36.553 50 36 L 50 29.8125 z M 11 31 L 11 32 L 8 32 L 8 34 C 8 34.496 7.97425 34.809 7.90625 35 C 7.58025 36.61 6.469 37.9185 5 38.5625 L 5 39 C 5 40.105 5.895 41 7 41 L 15 41 C 16.105 41 17 40.105 17 39 L 17 34 C 17 32.895 16.105 32 15 32 L 13 32 L 13 31 L 11 31 z M 35 31 L 35 32 L 34 32 C 32.897 32 32 32.897 32 34 L 32 38 C 32 39.103 32.897 40 34 40 L 41 40 C 42.103 40 43 39.103 43 38 L 43 34 C 43 32.897 42.103 32 41 32 L 40 32 L 40 31 L 38 31 L 38 32 L 37 32 L 37 31 L 35 31 z"
    carsymbol = BezierPath(carsymbol_string, fit=true, flipy=true)

    GLMakie.lines!(ax, [-lat_max, -lat_max], [-10.0, 1000.0], color=:black)
    GLMakie.lines!(ax, [+lat_max, +lat_max], [-10.0, 1000.0], color=:black)

    PA = Dict(t => [Observable(0.0), Observable(0.0)] for t in 0:T)
    PB = Dict(t => [Observable(0.0), Observable(0.0)] for t in 0:T)
    Pa = Dict(t => [Observable(0.0), Observable(0.0)] for t in 0:T)
    Pb = Dict(t => [Observable(0.0), Observable(0.0)] for t in 0:T)

    circ_x = [rad * cos(t) for t in 0:0.1:(2π+0.1)]
    circ_y = [rad * sin(t) for t in 0:0.1:(2π+0.1)]
    GLMakie.lines!(ax, @lift(circ_x .+ $(PA[0][1])), @lift(circ_y .+ $(PA[0][2])), color=:blue, linewidth=2, linestyle=:dash, alpha=0.9)
    GLMakie.lines!(ax, @lift(circ_x .+ $(PB[0][1])), @lift(circ_y .+ $(PB[0][2])), color=:red, linewidth=2, linestyle=:dash, alpha=0.9)
    GLMakie.lines!(ax, @lift(circ_x .+ $(Pa[0][1])), @lift(circ_y .+ $(Pa[0][2])), color=:lightblue, linewidth=2, linestyle=:dot, alpha=0.9)
    GLMakie.lines!(ax, @lift(circ_x .+ $(Pb[0][1])), @lift(circ_y .+ $(Pb[0][2])), color=:pink, linewidth=2, linestyle=:dot, alpha=0.9)

    rot1 = map(atan, @lift(($(PA[0][2]) - $(PA[1][2]))), @lift(($(PA[0][1]) - $(PA[1][1]))))
    rot2 = map(atan, @lift(($(PB[0][2]) - $(PB[1][2]))), @lift(($(PB[0][1]) - $(PB[1][1]))))
    rota = map(atan, @lift(($(Pa[0][2]) - $(Pa[1][2]))), @lift(($(Pa[0][1]) - $(Pa[1][1]))))
    rotb = map(atan, @lift(($(Pb[0][2]) - $(Pb[1][2]))), @lift(($(Pb[0][1]) - $(Pb[1][1]))))


    GLMakie.scatter!(ax, @lift([0.0, 0.0] .+ $(PA[0][1])), @lift([0.0, 0.0] .+ $(PA[0][2])), rotations=@lift($(rot1) .+ pi), marker=carsymbol, markersize=80, color=:blue, alpha=0.9)
    GLMakie.scatter!(ax, @lift([0.0, 0.0] .+ $(PB[0][1])), @lift([0.0, 0.0] .+ $(PB[0][2])), rotations=@lift($(rot2) .+ pi), marker=carsymbol, markersize=80, color=:red, alpha=0.9)
    GLMakie.scatter!(ax, @lift([0.0, 0.0] .+ $(Pa[0][1])), @lift([0.0, 0.0] .+ $(Pa[0][2])), rotations=@lift($(rota) .+ pi), marker=carsymbol, markersize=80, color=:lightblue, alpha=0.9)
    GLMakie.scatter!(ax, @lift([0.0, 0.0] .+ $(Pb[0][1])), @lift([0.0, 0.0] .+ $(Pb[0][2])), rotations=@lift($(rotb) .+ pi), marker=carsymbol, markersize=80, color=:pink, alpha=0.9)

    for t in 1:T
        GLMakie.lines!(ax, @lift(circ_x .+ $(PA[t][1])), @lift(circ_y .+ $(PA[t][2])), color=:blue, linewidth=2, linestyle=:dash, alpha=0.9)
        GLMakie.lines!(ax, @lift(circ_x .+ $(PB[t][1])), @lift(circ_y .+ $(PB[t][2])), color=:red, linewidth=2, linestyle=:dash, alpha=0.9)
        GLMakie.lines!(ax, @lift(circ_x .+ $(Pa[t][1])), @lift(circ_y .+ $(Pa[t][2])), color=:lightblue, linewidth=2, linestyle=:dot, alpha=0.9)
        GLMakie.lines!(ax, @lift(circ_x .+ $(Pb[t][1])), @lift(circ_y .+ $(Pb[t][2])), color=:pink, linewidth=2, linestyle=:dot, alpha=0.9)
    end


    function update(XA, XB, x0)
        xdim = 4
        udim = 2
        x0A = x0[1:xdim]
        x0B = x0[xdim+1:2*xdim]
        XXA = [XA[1:4:end] XA[2:4:end] XA[3:4:end] XA[4:4:end]]
        XXB = [XB[1:4:end] XB[2:4:end] XB[3:4:end] XB[4:4:end]]
        XXa = [XA[1:4:end] XA[2:4:end] XA[3:4:end] XA[4:4:end]]
        XXb = [XB[1:4:end] XB[2:4:end] XB[3:4:end] XB[4:4:end]]

        PA[0][1][] = x0A[1]
        PA[0][2][] = x0A[2]
        PB[0][1][] = x0B[1]
        PB[0][2][] = x0B[2]
        Pa[0][1][] = x0A[1]
        Pa[0][2][] = x0A[2]
        Pb[0][1][] = x0B[1]
        Pb[0][2][] = x0B[2]

        for l in 1:T
            PA[l][1][] = XXA[l, 1]
            PA[l][2][] = XXA[l, 2]
            PB[l][1][] = XXB[l, 1]
            PB[l][2][] = XXB[l, 2]
            Pa[l][1][] = XXa[l, 1]
            Pa[l][2][] = XXa[l, 2]
            Pb[l][1][] = XXb[l, 1]
            Pb[l][2][] = XXb[l, 2]
        end

        GLMakie.xlims!(ax, -lat_max, lat_max)
        avg_y_pos = (x0[2] + x0[6]) / 2
        GLMakie.ylims!(ax, avg_y_pos - lat_max, avg_y_pos + 3 * lat_max)
    end

    function update(XA, XB, Xa, Xb, x0)
        xdim = 4
        udim = 2
        x0A = x0[1:xdim]
        x0B = x0[xdim+1:2*xdim]
        PA[0][1][] = x0A[1]
        PA[0][2][] = x0A[2]
        PB[0][1][] = x0B[1]
        PB[0][2][] = x0B[2]
        Pa[0][1][] = x0A[1]
        Pa[0][2][] = x0A[2]
        Pb[0][1][] = x0B[1]
        Pb[0][2][] = x0B[2]

        XXA = [XA[1:4:end] XA[2:4:end] XA[3:4:end] XA[4:4:end]]
        XXB = [XB[1:4:end] XB[2:4:end] XB[3:4:end] XB[4:4:end]]
        XXa = [Xa[1:4:end] Xa[2:4:end] Xa[3:4:end] Xa[4:4:end]]
        XXb = [Xb[1:4:end] Xb[2:4:end] Xb[3:4:end] Xb[4:4:end]]

        for l in 1:T
            PA[l][1][] = XXA[l, 1]
            PA[l][2][] = XXA[l, 2]
            PB[l][1][] = XXB[l, 1]
            PB[l][2][] = XXB[l, 2]
            Pa[l][1][] = XXa[l, 1]
            Pa[l][2][] = XXa[l, 2]
            Pb[l][1][] = XXb[l, 1]
            Pb[l][2][] = XXb[l, 2]
        end

        GLMakie.xlims!(ax, -lat_max, lat_max)
        avg_y_pos = (x0[2] + x0[6]) / 2
        GLMakie.ylims!(ax, avg_y_pos - lat_max, avg_y_pos + 3 * lat_max)
    end

    (f, ax, update)
end

function show_me(XA, XB, x0; T=10, t=0, lat_max=2.5)
    (f, ax, update) = visualize(; T, lat_max)
    display(f)
    update(XA, XB, x0)

    if t > 0
        ax.title = string(t)
    end
end

function show_me(XA, XB, Xa, Xb, x0; T=10, t=0, lat_max=2.5)
    (f, ax, update) = visualize(; T, lat_max)
    display(f)

    update(XA, XB, Xa, Xb, x0)

    if t > 0
        ax.title = string(t)
    end
end

function animate(probs, sim_results; save=false, filename="test.mp4", sleep_duration=1e-1, update_phantoms=false)
    rad = probs.params.r / 2
    lat = probs.params.lat_max + rad
    (f, ax, update) = visualize(; rad=rad, lat_max=lat)
    display(f)
    T = length(sim_results)

    if save
        # extra 5 frames at the end because powerpoint is being weird
        record(f, filename, 1:T+10; framerate=10) do t
            if t <= T
                if update_phantoms
                    update(sim_results[t].XA, sim_results[t].XB, sim_results[t].Xa, sim_results[t].Xb, sim_results[t].x0)
                else
                    update(sim_results[t].XA, sim_results[t].XB, sim_results[t].x0)
                end
                ax.title = "Time step = $(string(t))"
            end
        end
    else
        for t in 1:T
            if update_phantoms
                update(sim_results[t].XA, sim_results[t].XB, sim_results[t].Xa, sim_results[t].Xb, sim_results[t].x0)
            else
                update(sim_results[t].XA, sim_results[t].XB, sim_results[t].x0)
            end
            ax.title = "Time step = $(string(t))"
            sleep(sleep_duration)
        end
    end
end