#!/usr/bin/env julia

include("./lib/jlBarraCUDA.jl")
using .jlBarraCUDA

x = cuDeviceCount()
println(x)
