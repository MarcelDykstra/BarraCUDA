clc; clear all;

for m = 1:10,
    cuh = cuLatticeInit;

    for n = 1:100,
        [vx, vy, rho] = cuCollidePropagate(cuh);

        figure(1);
        set(gcf, 'MenuBar', 'none');
        set(gcf, 'Color', [1 1 1]);
        imagesc(flipud(vx' .* vx' + vy' .* vy'));
        drawnow;
    end

    cuLatticeClear(cuh);
end;

cuDeviceReset;
