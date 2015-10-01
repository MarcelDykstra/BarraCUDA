clc; clear all; close all;

cuh = cuLatticeInit;
% aviobj = avifile('lbm.avi');

for n = 1:100,
  [vx, vy, vz, rho] = cuCollidePropagate(cuh);
  if (n == 1)
   [x, y, z] = meshgrid(1:size(vx, 2), 1:size(vx, 1), 1:size(vx, 3));
  end

  figure(1);
  set(gcf, 'MenuBar', 'none');
  slice(x, y, z, double(vx - vy), ...
    [(size(vx, 2) / 4) (size(vx, 2) / 2) (3 * size(vx, 2) / 4)], ...
      size(vx, 1) / 2, size(vx, 3) / 2);
  shading interp; axis([1 size(vx, 2) 1 size(vx, 1) 1 size(vx, 3)]);
  set(gcf, 'Color', [1 1 1]); box on;
  set(gca, 'DataAspectRatio', [1 1 1]);
  view(232, -56); lighting gouraud;
  light('Position', [-10 10 -20], 'Style', 'infinite');
  draw now;

  % % frame = getframe(gcf);
  % % aviobj = addframe(aviobj, frame);
end

% % aviobj = close(aviobj);
cuLatticeClear(cuh);
cuDeviceReset;
