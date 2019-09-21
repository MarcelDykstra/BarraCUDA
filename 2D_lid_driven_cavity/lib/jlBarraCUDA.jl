module jlBarraCUDA

export cudaDeviceProperties, cuDeviceCount

function cuDeviceProperties()
    return ccall((:cuDeviceProperties, "./lib/barracuda.so"), Cint, ())
end

function cuDeviceCount()
    return ccall((:cuDeviceCount, "./lib/barracuda.so"), Cint, ())
end

end
