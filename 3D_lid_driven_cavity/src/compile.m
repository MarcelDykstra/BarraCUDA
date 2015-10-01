if strcmpi(computer('arch'), 'win32') || strcmpi(computer('arch'), 'win64')
  % - NVidia Compiler NVCC requires Visual Studio compiler.
  %   Install 'Visual Studio Community 2013' followed by 'NVidia CUDA toolkit', 
  %   'CUDA*.targets' should be available in 
  %   'Visual Studio > Project > Build Customization ...'.
  % - [ http://nl.mathworks.com/help/distcomp/
  %     run-mex-functions-containing-cuda-code.html ]
  % - [ http://www.blinnov.com/en/2010/06/04/
  %     microsoft-was-unexpected-at-this-time/ ] -- 
  %   requires removing two '&quot;' in config XML line
  %   'LIBPATH="&quot;$VSROOT\VC\Lib\amd64&quot;"'
  % - Use 'mex -v' and in config XML 'nvcc -v' for verbose 
  %   information to solve issues.

  mex cuCollidePropagate.cu  barracuda.cu -largeArrayDims -outdir .. -I.
  mex cuDeviceProperties.cu barracuda.cu -largeArrayDims -outdir .. -I.
  mex cuDeviceReset.cu barracuda.cu -largeArrayDims -outdir .. -I.
  mex cuLatticeInit.cu barracuda.cu -largeArrayDims -outdir .. -I.
  mex cuLatticeClear.cu barracuda.cu -largeArrayDims -outdir .. -I.  
elseif strcmpi(computer('arch'), 'glnxa64')
  setenv('MW_NVCC_PATH','/opt/cuda/bin')
  mex cuCollidePropagate.cu  barracuda.cu  -largeArrayDims -outdir ../ ...
    -I/opt/cuda/include -I. -L/opt/cuda/lib64 -lcublas -lcusparse -lcudart
  mex cuDeviceProperties.cu barracuda.cu -largeArrayDims -outdir ../ ...
    -I/opt/cuda/include -I. -L/opt/cuda/lib64 -lcublas -lcusparse -lcudart
  mex cuDeviceReset.cu barracuda.cu -largeArrayDims -outdir ../ ...
    -I/opt/cuda/include  -I. -L/opt/cuda/lib64 -lcublas -lcusparse -lcudart
  mex cuLatticeInit.cu barracuda.cu -largeArrayDims -outdir ../ ...
    -I/opt/cuda/include  -I. -L/opt/cuda/lib64 -lcublas -lcusparse -lcudart
  mex cuLatticeClear.cu barracuda.cu -largeArrayDims -outdir ../ ...
    -I/opt/cuda/include  -I. -L/opt/cuda/lib64 -lcublas -lcusparse -lcudart
else
  error('Architecture not recognised.');
end
