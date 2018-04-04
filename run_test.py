#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) Diego Gonzalez <diegogonzalezvidal@gmail.com>
#
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with This program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
# okada 1985's model
from okada_models import forward_point_source_okada85
from okada_models import forward_finite_rectangular_source_okada85
# okada 1992's model
from okada_models import forward_point_source_okada92
from okada_models import forward_finite_rectangular_source_okada92_Uxyz
from okada_models import forward_finite_rectangular_source_okada92_dUdx
from okada_models import forward_finite_rectangular_source_okada92_dUdy
from okada_models import forward_finite_rectangular_source_okada92_dUdz

##############################################################################################################################

# Test fuente puntual caso 2  Okada 1985
Ux, Uy, Uz, dUxdx, dUxdy, dUydx, dUydy, dUzdx, dUzdy = forward_point_source_okada85()
# Test fuente rectangular caso 2  Okada 1985
Ux, Uy, Uz, dUxdx, dUxdy, dUydx, dUydy, dUzdx, dUzdy = forward_finite_rectangular_source_okada85()

# Test fuente rectangular utilizando okada 1992 y z = 0
Ux, Uy, Uz, dUxdx, dUydx, dUzdx, dUxdy, dUydy, dUzdy, dUxdz, dUydz, dUzdz = forward_point_source_okada92()
Ux, Uy, Uz = forward_finite_rectangular_source_okada92_Uxyz()
dUxdx, dUydx, dUzdx = forward_finite_rectangular_source_okada92_dUdx()
dUxdy, dUydy, dUzdy = forward_finite_rectangular_source_okada92_dUdy()
dUxdz, dUydz, dUzdz = forward_finite_rectangular_source_okada92_dUdz()

##############################################################################################################################

# grilla del modelo
x = np.arange(-10, 11)
y = np.arange(-15, 16)
z = np.arange(-15, 1)
# movimiento de la falla
dip = 25
rake = 90
U = 1
# largo, ancho y profundidad del plano de falla
L = 8
W = 6
c = 10
# constantes elasticas de lamme
lmbd = 1
mu = 1
#
U_as_unit = False
verbose = False

ux = np.zeros([len(x), len(y), len(z)])
uy = np.zeros([len(x), len(y), len(z)])
uz = np.zeros([len(x), len(y), len(z)])
duxdx = np.zeros([len(x), len(y), len(z)])
duydx = np.zeros([len(x), len(y), len(z)])
duzdx = np.zeros([len(x), len(y), len(z)])
duxdy = np.zeros([len(x), len(y), len(z)])
duydy = np.zeros([len(x), len(y), len(z)])
duzdy = np.zeros([len(x), len(y), len(z)])
duxdz = np.zeros([len(x), len(y), len(z)])
duydz = np.zeros([len(x), len(y), len(z)])
duzdz = np.zeros([len(x), len(y), len(z)])

for i in range(len(x)):
  for j in range(len(y)):
    for k in range(len(z)):
      xi = x[i]; yi = y[j]; zi = z[k]
      uxi, uyi, uzi = forward_finite_rectangular_source_okada92_Uxyz(x=xi, y=yi, z=zi, dip=dip, rake=rake, c=c, L=L, W=W, U=U, mu=mu, lmbd=lmbd, U_as_unit=U_as_unit, verbose=verbose)
      duxidx, duyidx, duzidx = forward_finite_rectangular_source_okada92_dUdx(x=xi, y=yi, z=zi, dip=dip, rake=rake, c=c, L=L, W=W, U=U, mu=mu, lmbd=lmbd, U_as_unit=U_as_unit, verbose=verbose)
      duxidy, duyidy, duzidy = forward_finite_rectangular_source_okada92_dUdy(x=xi, y=yi, z=zi, dip=dip, rake=rake, c=c, L=L, W=W, U=U, mu=mu, lmbd=lmbd, U_as_unit=U_as_unit, verbose=verbose)
      duxidz, duyidz, duzidz = forward_finite_rectangular_source_okada92_dUdz(x=xi, y=yi, z=zi, dip=dip, rake=rake, c=c, L=L, W=W, U=U, mu=mu, lmbd=lmbd, U_as_unit=U_as_unit, verbose=verbose)
      ux[i,j,k] = uxi
      uy[i,j,k] = uyi
      uz[i,j,k] = uzi
      duxdx[i,j,k] = duxidx
      duydx[i,j,k] = duyidx
      duzdx[i,j,k] = duzidx
      duxdy[i,j,k] = duxidy
      duydy[i,j,k] = duyidy
      duzdy[i,j,k] = duzidy
      duxdz[i,j,k] = duxidz
      duydz[i,j,k] = duyidz
      duzdz[i,j,k] = duzidz


##############################################################################################################################
# fig. desplazamientos en el plano Y-Z a X=xpos
fig = plt.figure()
suptitle = "dip = %.1f°; rake = %.1f°; c = %.1f; L = %.1f; W = %.1f; U = %.1f; lambda = %.1f; mu = %.1f" % (dip, rake, c, L, W, U, lmbd, mu)
fig.suptitle(suptitle, fontsize=10)


# desplazamientos
ax = fig.add_subplot(221)
xpos = 10
ax.set_xlim([y[0],y[-1]])
ax.set_ylim([z[0],z[-1]])
ax.set_xlabel('y-axis')
ax.set_ylabel('z-axis')
ax.set_title( "Vector field for U(X=%.1f, Y, Z)" % (x[xpos]) , fontsize=10, y=1.0)
ax.scatter(0, -c, marker='*', color='C3', s=100, zorder=10, linewidths=1)
ax.plot( [ 0, W*np.cos(np.radians(dip)) ], [ -c, W*np.sin(np.radians(dip))-c ], lw=4, ls='-', zorder=9, color='C3')
Y,Z = np.meshgrid(y,z)
ax.scatter(Y.T, Z.T, c='k', s=3, zorder=0)
qv = ax.quiver(Y.T, Z.T, uy[xpos,:,:], uz[xpos,:,:], np.sqrt(ux[xpos,:,:]**2 + uy[xpos,:,:]**2+uz[xpos,:,:]**2), cmap=plt.cm.viridis_r, clip_on=True, color='C0')
#cbar = plt.colorbar(qv, fraction=0.04, orientation="vertical", format='%.1e', extend='neither')
#cbar.set_label('|U|',size=10)


# desplazamientos y deformaciones en z
ax = fig.add_subplot(222)
ypos = 17
ax.set_ylim([z[0],z[-1]])
ax.set_xlabel("norm. amp." )
ax.set_ylabel('z-axis')
ax.plot(uz[xpos,ypos,:]/abs(uz[xpos,ypos,:]).max(), z, color='C0', clip_on=True, lw='1', label="Uz (X=%.1f, Y=%.1f, Z)" % (x[xpos], y[ypos]))
ax.plot(duzdz[xpos,ypos,:]/abs(duzdz[xpos,ypos,:]).max(), z, color='C1', clip_on=True, lw='1', label="dUz/dz" )
#ax.plot(duzdy[xpos,ypos,:]/abs(duzdy[xpos,ypos,:]).max(), z, color='C2', clip_on=True, lw='1', label="dUz/dy" )
plt.legend()


# desplazamientos y deformaciones en y
ax = fig.add_subplot(223)
zpos = 7
ax.set_xlim([y[0],y[-1]])
ax.set_ylabel("norm. amp." )
ax.set_xlabel('y-axis')
ax.plot(y, uy[xpos,:,zpos]/abs(uy[xpos,:,zpos]).max(), color='C0', clip_on=True, lw='1', label="Uy (X=%.1f, Y, Z=%.1f)" % (x[xpos], z[zpos]))
ax.plot(y, duydy[xpos,:,zpos]/abs(duydy[xpos,:,zpos]).max(), color='C1', clip_on=True, lw='1', label="dUy/dy" )
#ax.plot(y, duydz[xpos,:,zpos]/abs(duydz[xpos,:,zpos]).max(), color='C2', clip_on=True, lw='1', label="dUy/dz" )
plt.legend()


# desplazamientos y deformaciones en x
ax = fig.add_subplot(224)
ax.set_xlim([x[0],x[-1]])
ax.set_ylabel("norm. amp." )
ax.set_xlabel('x-axis')
ax.plot(x, ux[:,ypos,zpos]/abs(ux[:,ypos,zpos]).max(), color='C0', clip_on=True, lw='1', label="Ux (X, Y=%.1f, Z=%.1f)" % (y[ypos], z[zpos]))
ax.plot(x, duxdx[:,ypos,zpos]/abs(duxdx[:,ypos,zpos]).max(), color='C1', clip_on=True, lw='1', label="dUx/dx" )
plt.legend()



# print figure
fig.savefig('tests.jpg', dpi=150, transparent=False)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show(block=False)







