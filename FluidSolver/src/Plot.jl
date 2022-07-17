using JLD2
using PyPlot

@load "Data/res_glo1.jld2"  res_all
semilogy(res_all, label = "Method#1 global time-stepping")

@load "Data/res_loc1.jld2"  res_all
semilogy(res_all, label = "Method#1 local time-stepping")


@load "Data/res_glo2.jld2"  res_all
semilogy(res_all, label = "Method#2 global time-stepping")


@load "Data/res_loc2.jld2"  res_all
semilogy(res_all, label = "Method#2 local time-stepping")


xlabel("Number of iterations")
ylabel("Rel. residuals")
legend()
savefig("residual.png")


close("all")