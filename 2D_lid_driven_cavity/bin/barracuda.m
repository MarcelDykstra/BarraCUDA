clc; clear all;

for m = 1:10,
  cuh = cuLatticeInit;

  for n = 1:100,
    [vx, vy, rho] = cuCollidePropagate(cuh);

    figure(2);
    set(gcf, 'MenuBar', 'none');
    set(gcf, 'Color', [1 1 1]);
    imagesc(flipud(vx' .* vx' + vy' .* vy'));
  end

  cuLatticeClear(cuh);
end;

cuDeviceReset;
