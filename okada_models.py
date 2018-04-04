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



##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
def forward_point_source_okada85(x=2, y=3, d=4, dip=70, rake=10, U=1, mu=1, lmbd=1, U_as_unit=True, verbose=True):
  dip_rad = np.radians(dip)
  rake_rad = np.radians(rake)

  if U_as_unit is True:
    U1 = 1 
    U2 = 1 
    U3 = 1 
    if verbose is True:
      print( "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\nRunning ->  forward_point_source_okada85\nINPUTS  ->  x = %.2f;  y = %.2f; d = %.2f (z=-d); dip = %.1f; rake = (not considered, unit_U is True); U = (not considered, unit_U is True)" % (x, y, d, dip) )
  elif U_as_unit is False:
    U1 = U*np.cos(rake_rad)
    U2 = U*np.sin(rake_rad)
    U3 = 0 # != 0 para intrusiones de fluidos
    if verbose is True:
      print( "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\nRunning ->  forward_point_source_okada85\nINPUTS  ->  x = %.2f;  y = %.2f; d = %.2f (z=-d); dip = %.1f; rake = %.1f; U = %.2f" % (x, y, d, dip, rake, U) )


  ###################################################################################
  ###### Desplazamientos Ux,Uy,Uz ######
  ###################################################################################

  p = y*np.cos(dip_rad) + d*np.sin(dip_rad)
  q = y*np.sin(dip_rad) - d*np.cos(dip_rad)
  R = np.sqrt(x**2 + y**2 + d**2) # np.sqrt(x**2 * p**2 + q**2)

  I1 = (mu/(lmbd+mu)) * y * (1/(R*(R+d)**2) - x**2*( (3*R+d)/(R**3*((R+d)**3))) )
  I2 = (mu/(lmbd+mu)) * x * (1/(R*(R+d)**2) - y**2*( (3*R+d)/(R**3*((R+d)**3))) )
  I3 = (mu/(lmbd+mu)) * (x/R**3) - I2
  I4 = (mu/(lmbd+mu)) * ( -x*y* ((2*R+d)/(R**3*(R+d)**2)) )
  I5 = (mu/(lmbd+mu)) * ( 1/(R*(R+d)) - x**2*((2*R+d)/(R**3*(R+d)**2)) )

  # desplazamientos para strike-slip
  uxs = -U1/(2*np.pi) * ( ((3*x**2*q)/R**5) + (I1*np.sin(dip_rad)) )
  uys = -U1/(2*np.pi) * ( ((3*x*y*q)/R**5) + (I2*np.sin(dip_rad)) )
  uzs = -U1/(2*np.pi) * ( ((3*x*d*q)/R**5) + (I4*np.sin(dip_rad)) )
  # desplazamientos para dip-slip
  uxd = -U2/(2*np.pi) * ( (3*x*p*q/R**5) - (I3*np.sin(dip_rad)*np.cos(dip_rad)) )
  uyd = -U2/(2*np.pi) * ( (3*y*p*q/R**5) - (I1*np.sin(dip_rad)*np.cos(dip_rad)) )
  uzd = -U2/(2*np.pi) * ( (3*d*p*q/R**5) - (I5*np.sin(dip_rad)*np.cos(dip_rad)) )
  # desplazamientos para tensile faults
  uxt = U3/(2*np.pi) * ( (3*x*q**2)/R**5 - I3*(np.sin(dip_rad))**2 )
  uyt = U3/(2*np.pi) * ( (3*y*q**2)/R**5 - I1*(np.sin(dip_rad))**2 )
  uzt = U3/(2*np.pi) * ( (3*d*q**2)/R**5 - I5*(np.sin(dip_rad))**2 )

  # soluciones del desplazamiento para una fuente puntual
  Ux = uxs + uxd + uxt
  Uy = uys + uyd + uyt 
  Uz = uzs + uzd + uzt 


  ###################################################################################
  ###### Deformaciones dUx/dx, dUx/dy, dUy/dx, dUy/dy ######
  ###################################################################################

  s = p*np.sin(dip_rad) + q*np.cos(dip_rad)
  J1 = (mu/(lmbd+mu)) * (-3*x*y*( (3*R+d)/(R**3*(R+d)**3) ) + 3*x**3*y*( (5*R**2 + 4*R*d + d**2)/(R**5*(R+d)**4) ) )
  J2 = (mu/(lmbd+mu)) * (1/R**3 - (3)/(R*(R+d)**2) + 3*x**2*y**2*( (5*R**2 + 4*R*d + d**2)/(R**5*(R+d)**4) ) )
  J3 = (mu/(lmbd+mu)) * (1/R**3 - 3*x**2/R**5) - J2
  J4 = (mu/(lmbd+mu)) * (-3*x*y/R**5) - J1

  # deformaciones para strike-slip
  duxdx_ss = -U1/(2*np.pi) * ( (3*x*q/R**5)*(2 - 5*x**2/R**2) + J1*np.sin(dip_rad) )
  duxdy_ss = -U1/(2*np.pi) * ( -(15*x**2*y*q)/R**7 + (3*x**2/R**5 + J2)*np.sin(dip_rad) )
  duydx_ss = -U1/(2*np.pi) * ( (3*y*q/R**5)*(1-5*x**2/R**2) + J2*np.sin(dip_rad) )
  duydy_ss = -U1/(2*np.pi) * ( (3*x*q/R**5)*(1-5*y**2/R**2) + (3*x*y/R**5 + J4)*np.sin(dip_rad) )
  # deformaciones para dip_slip
  duxdx_ds = -U2/(2*np.pi) * ( (3*p*q/R**5)*(1 - 5*x**2/R**2) - J3*np.sin(dip_rad)*np.cos(dip_rad) )
  duxdy_ds = -U2/(2*np.pi) * ( (3*x/R**5)*(s-5*y*p*q/R**2) - J1*np.sin(dip_rad)*np.cos(dip_rad) )
  duydx_ds = -U2/(2*np.pi) * ( -(15*x*y*p*q)/R**7 - J1*np.sin(dip_rad)*np.cos(dip_rad) )
  duydy_ds = -U2/(2*np.pi) * ( (3*p*q/R**5)*(1-5*y**2/R**2) + 3*y*s/R**5 - J2*np.sin(dip_rad)*np.cos(dip_rad) )
  # deformaciones para tensile fault
  duxdx_tf = U3/(2*np.pi) * ( (3*q**2/R**5)*(1 - 5*x**2/R**2) - J3*(np.sin(dip_rad))**2 )
  duxdy_tf = U3/(2*np.pi) * ( (3*x*q/R**5)*(2*np.sin(dip_rad)-5*y*q/R**2) - J1*(np.sin(dip_rad))**2 )
  duydx_tf = U3/(2*np.pi) * ( -(15*x*y*q**2)/R**7 - J1*(np.sin(dip_rad))**2 )
  duydy_tf = U3/(2*np.pi) * ( (3*q/R**5)*(q + 2*y*np.sin(dip_rad) - 5*y**2*q/R**2) - J2*(np.sin(dip_rad))**2 )

  # soluciones del deformacion para una fuente puntual
  dUxdx = duxdx_ss + duxdx_ds + duxdx_tf
  dUxdy = duxdy_ss + duxdy_ds + duxdy_tf
  dUydx = duydx_ss + duydx_ds + duydx_tf
  dUydy = duydy_ss + duydy_ds + duydy_tf


  ###################################################################################
  ###### Inclinaciones dUz/dx, dUz/dy ######
  ###################################################################################

  K1 = -(mu/(lmbd+mu)) * y * ( (2*R+d)/(R**3*(R+d)**2) - x**2*( (8*R**2+9*R*d+3*d**2)/(R**5*(R+d)**3) ) )
  K2 = -(mu/(lmbd+mu)) * x * ( (2*R+d)/(R**3*(R+d)**2) - y**2*( (8*R**2+9*R*d+3*d**2)/(R**5*(R+d)**3) ) )
  K3 = -(mu/(lmbd+mu)) * (3*x*d/R**5) - K2

  # inclinaciones para strike-slip
  duzdx_ss = -U1/(2*np.pi) * ( (3*d*q/R**5)*(1 - 5*x**2/R**2) + K1*np.sin(dip_rad) )
  duzdy_ss = -U1/(2*np.pi) * ( -(15*x*y*d*q)/R**7 + (3*x*d/R**5 + K2)*np.sin(dip_rad) )
  # inclinaciones para dip_slip
  duzdx_ds = -U2/(2*np.pi) * ( -15*x*d*p*q/R**7 - K3*np.sin(dip_rad)*np.cos(dip_rad)  )
  duzdy_ds = -U2/(2*np.pi) * ( (3*d/R**5)*(s - 5*y*p*q/R**2) - K1*np.sin(dip_rad)*np.cos(dip_rad) )
  # inclinaciones para tensile fault
  duzdx_tf = U3/(2*np.pi) * ( -15*x*d*q**2/R**7 - K3*(np.sin(dip_rad))**2 )
  duzdy_tf = U3/(2*np.pi) * ( (3*d*q/R**5)*(2*np.sin(dip_rad)-5*y*q/R**2) - K1*(np.sin(dip_rad))**2 )

  # soluciones de inclinaciones para una fuente puntual
  dUzdx = duzdx_ss + duzdx_ds + duzdx_tf
  dUzdy = duzdy_ss + duzdy_ds + duzdy_tf



  ###################################################################################
  # print check-list of outputs
  if verbose is True:
    #print( "\n")
    print( "OUTPUTS ->            Ux            Uy          Uz          dUx/dx       dUx/dy       dUy/dx       dUy/dy       dUz/dx       dUz/dy" ) 
    print( "------------------------------------------------------------------------------------------------------------------------------------" )
    print( "Strike-slip   ->  %+.3e   %+.3e   %+.3e   %+.3e   %+.3e   %+.3e   %+.3e   %+.3e   %+.3e" % (uxs, uys, uzs, duxdx_ss, duxdy_ss, duydx_ss, duydy_ss, duzdx_ss, duzdy_ss) )
    print( "Dip-slip      ->  %+.3e   %+.3e   %+.3e   %+.3e   %+.3e   %+.3e   %+.3e   %+.3e   %+.3e" % (uxd, uyd, uzd, duxdx_ds, duxdy_ds, duydx_ds, duydy_ds, duzdx_ds, duzdy_ds) )
    print( "Tensile fault ->  %+.3e   %+.3e   %+.3e   %+.3e   %+.3e   %+.3e   %+.3e   %+.3e   %+.3e" % (uxt, uyt, uzt, duxdx_tf, duxdy_tf, duydx_tf, duydy_tf, duzdx_tf, duzdy_tf) )
    print( "------------------------------------------------------------------------------------------------------------------------------------")
    print( "                  %+.3e   %+.3e   %+.3e   %+.3e   %+.3e   %+.3e   %+.3e   %+.3e   %+.3e" % (Ux, Uy, Uz, dUxdx, dUxdy, dUydx, dUydy , dUzdx, dUzdy) )
    print( "------------------------------------------------------------------------------------------------------------------------------------")
    #print( "EOF\n\n" )


  return Ux, Uy, Uz, dUxdx, dUxdy, dUydx, dUydy , dUzdx, dUzdy






##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
def forward_finite_rectangular_source_okada85(x=2, y=3, dip=70, rake=10, d=4, L=3, W=2, U=1, mu=1, lmbd=1, U_as_unit=True, verbose=True):
  dip_rad = np.radians(dip)
  rake_rad = np.radians(rake)

  if U_as_unit is True:
    U1 = 1 
    U2 = 1 
    U3 = 1 
    if verbose is True:
      print( "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\nRunning ->  forward_finite_rectangular_source_okada85\nINPUTS  ->  x = %.2f;  y = %.2f; d = %.2f (z=-d); dip = %.1f; rake = (not considered, unit_U is True); U = (not considered, unit_U is True); L = %.2f; W = %.2f" % (x, y, d, dip, L, W) )
  elif U_as_unit is False:
    U1 = U*np.cos(rake_rad)
    U2 = U*np.sin(rake_rad)
    U3 = 0 # != 0 para intrusiones de fluidos
    if verbose is True:
      print( "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\nRunning ->  forward_finite_rectangular_source_okada85\nINPUTS  ->  x = %.2f;  y = %.2f; d = %.2f (z=-d); dip = %.1f; rake = %.1f; U = %.2f; L = %.2f; W = %.2f" % (x, y, d, dip, rake, U, L, W) )

  p = y*np.cos(dip_rad) + d*np.sin(dip_rad)
  q = y*np.sin(dip_rad) - d*np.cos(dip_rad)
  qq = np.array([q, q, q, q])    # b = 4

  # Notacion  de Chinnery -> f(e,eta)|| = f(x,p) - f(x,p-W) - f(x-L,p) + f(x-L,p-W)
  e = np.array([x, x, x-L, x-L])
  eta = np.array([p, p-W, p, p-W])
  ytg = eta*np.cos(dip_rad) + qq*np.sin(dip_rad)
  dtg = eta*np.sin(dip_rad) - qq*np.cos(dip_rad)
  R = np.power(e**2 + eta**2 + qq**2, 1/2.)
  X = np.power(e**2 + qq**2, 1/2.)

  ###################################################################################
  ###### Desplazamientos Ux,Uy,Uz ######
  ###################################################################################

  if dip==90. or dip==270. or dip==-90.:# con cos(dip)=0 tenemos que sin(dip)=[+1,-1]
    #print("dip exception for 90 and 270 deg")
    I1 = ( mu/2*(lmbd+mu) ) * ( (e*qq)/(R+dtg)**2 )
    I3 = ( mu/2*(lmbd+mu) ) * ( (eta/(R+dtg)) + ((ytg*qq)/(R+dtg)**2) - np.log(R+eta) )
    I4 = -( mu/(lmbd+mu) ) * ( qq/(R+dtg) )
    I5 = -( mu/(lmbd+mu) ) * ( (e*np.sin(dip_rad))/(R+dtg) )
    I2 = ( mu/(lmbd+mu) ) * ( -np.log(R+eta) ) - I3
  else:
    I5 = ( mu/(lmbd+mu) ) * ( 2/np.cos(dip_rad) ) * (np.arctan((eta*(X+qq*np.cos(dip_rad))+X*(R+X)*np.sin(dip_rad))/(e*(R+X)*np.cos(dip_rad))) )
    I4 = ( mu/(lmbd+mu) ) * ( 1/np.cos(dip_rad) ) * ( np.log(R+dtg)-np.sin(dip_rad)*np.log(R+eta) )
    I3 = ( mu/(lmbd+mu) ) * ( (1/np.cos(dip_rad))*(ytg/(R+dtg))-np.log(R+eta) ) + ( np.sin(dip_rad)/np.cos(dip_rad) )*I4
    I2 = ( mu/(lmbd+mu) ) * ( -np.log(R+eta) ) - I3
    I1 = ( mu/(lmbd+mu) ) * ( (-1./np.cos(dip_rad))*(e/(R+dtg)) ) - ( np.sin(dip_rad)/np.cos(dip_rad) )*I5


  # strike-slip
  Ux_ss = -( U1/(2*np.pi) ) * ( (e*qq)/(R*(R+eta)) + np.arctan((e*eta)/(qq*R)) + I1*np.sin(dip_rad) )
  Uy_ss = -( U1/(2*np.pi) ) * ( (ytg*qq)/(R*(R+eta)) + (qq*np.cos(dip_rad))/(R+eta) + I2*np.sin(dip_rad) )
  Uz_ss = -( U1/(2*np.pi) ) * ( (dtg*qq)/(R*(R+eta)) + (qq*np.sin(dip_rad))/(R+eta) + I4*np.sin(dip_rad) )
  # dip-slip
  Ux_ds = -( U2/(2*np.pi) ) * ( (qq/R) - I3*np.sin(dip_rad)*np.cos(dip_rad) )
  Uy_ds = -( U2/(2*np.pi) ) * ( (ytg*qq)/(R*(R+e)) + np.cos(dip_rad)*np.arctan( (e*eta)/(qq*R) ) - I1*np.sin(dip_rad)*np.cos(dip_rad) )
  Uz_ds = -( U2/(2*np.pi) ) * ( (dtg*qq)/(R*(R+e)) + np.sin(dip_rad)*np.arctan( (e*eta)/(qq*R) ) - I5*np.sin(dip_rad)*np.cos(dip_rad) )
  # tensile fault
  Ux_tf = ( U3/(2*np.pi) ) * ( qq**2/(R*(R+eta)) - I3*(np.sin(dip_rad)**2) )
  Uy_tf = ( U3/(2*np.pi) ) * ( (-dtg*qq)/(R*(R+e)) - np.sin(dip_rad)*( (e*qq)/(R*(R+eta))-np.arctan((e*eta)/(qq*R)) ) - I1*(np.sin(dip_rad)**2) )
  Uz_tf = ( U3/(2*np.pi) ) * ( (ytg*qq)/(R*(R+e)) + np.cos(dip_rad)*( (e*qq)/(R*(R+eta))-np.arctan((e*eta)/(qq*R)) ) - I5*(np.sin(dip_rad)**2) )

  # representacion Chinnery para strike-slip
  Ux_ss_ch = ( Ux_ss[0] - Ux_ss[1] - Ux_ss[2] + Ux_ss[3] )
  Uy_ss_ch = ( Uy_ss[0] - Uy_ss[1] - Uy_ss[2] + Uy_ss[3] )
  Uz_ss_ch = ( Uz_ss[0] - Uz_ss[1] - Uz_ss[2] + Uz_ss[3] )
  # representacion Chinnery para dip-slip
  Ux_ds_ch = ( Ux_ds[0] - Ux_ds[1] - Ux_ds[2] + Ux_ds[3] )
  Uy_ds_ch = ( Uy_ds[0] - Uy_ds[1] - Uy_ds[2] + Uy_ds[3] )
  Uz_ds_ch = ( Uz_ds[0] - Uz_ds[1] - Uz_ds[2] + Uz_ds[3] )
  # representacion Chinnery para tensile fault
  Ux_tf_ch = ( Ux_tf[0] - Ux_tf[1] - Ux_tf[2] + Ux_tf[3] )
  Uy_tf_ch = ( Uy_tf[0] - Uy_tf[1] - Uy_tf[2] + Uy_tf[3] )
  Uz_tf_ch = ( Uz_tf[0] - Uz_tf[1] - Uz_tf[2] + Uz_tf[3] )

  # soluciones del desplazamiento Ux, Uy, Uz
  Ux = Ux_ss_ch + Ux_ds_ch + Ux_tf_ch
  Uy = Uy_ss_ch + Uy_ds_ch + Uy_tf_ch
  Uz = Uz_ss_ch + Uz_ds_ch + Uz_tf_ch

  ###################################################################################
  ###### Deformaciones dUx/dx, dUx/dy, dUy/dx, dUy/dy ######
  ###################################################################################

  Ae = (2*R+e)/((R**3)*((R+e)**2))
  Aeta = (2*R+eta)/((R**3)*((R+eta)**2))

  if dip==90. or dip==270 or dip==-90.:# con cos(dip)=0 tenemos que sin(dip)=[+1,-1]
    K3 = ( mu/(lmbd+mu) ) * ( np.sin(dip_rad)/(R+dtg) ) * ( e**2/(R*(R+dtg))-1 )
    K2 = ( mu/(lmbd+mu) ) * ( -np.sin(dip_rad)/R + (qq*np.cos(dip_rad))/(R*(R+eta)) ) - K3
    K1 = ( mu/(lmbd+mu) ) * ( (e*qq)/( R*(R+dtg)**2 ) )
  else:
    K3 = ( mu/(lmbd+mu) ) * ( 1/np.cos(dip_rad) ) * ( qq/(R*(R+eta)) - ytg/(R*(R+dtg)) )
    K2 = ( mu/(lmbd+mu) ) * ( -np.sin(dip_rad)/R + (qq*np.cos(dip_rad))/(R*(R+eta)) ) - K3
    K1 = ( mu/(lmbd+mu) ) * ( e/np.cos(dip_rad) ) * ( 1/(R*(R+dtg)) - np.sin(dip_rad)/(R*(R+eta)) )


  if dip==90. or dip==270 or dip==-90.:# con cos(dip)=0 tenemos que sin(dip)=[+1,-1]
    J1 = ( mu/(2*(lmbd+mu)) ) * ( qq/((R+dtg)**2) ) * ( (2*e**2)/(R*(R+dtg)) - 1 )
    J2 = ( mu/(2*(lmbd+mu)) ) * ( (e*np.sin(dip_rad))/((R+dtg)**2) ) * ( (2*qq**2)/(R*(R+dtg)) - 1 )
    J3 = ( mu/(lmbd+mu) ) * ( -e/(R*(R+eta))) - J2
    J4 = ( mu/(lmbd+mu) ) * ( -np.cos(dip_rad)/R - (qq*np.sin(dip_rad))/(R*(R+eta)) ) - J1
  else:
    J1 = ( mu/(lmbd+mu) ) * ( 1/np.cos(dip_rad) ) * ( e**2/(R*(R+dtg)**2) - 1/(R+dtg) ) - ( np.sin(dip_rad)/np.cos(dip_rad) )*K3
    J2 = ( mu/(lmbd+mu) ) * ( 1/np.cos(dip_rad) ) * ( (e*ytg)/(R*(R+dtg)**2) ) - ( np.sin(dip_rad)/np.cos(dip_rad) )*K1
    J3 = ( mu/(lmbd+mu) ) * ( -e/(R*(R+eta))) - J2
    J4 = ( mu/(lmbd+mu) ) * ( -np.cos(dip_rad)/R - (qq*np.sin(dip_rad))/(R*(R+eta)) ) - J1


  # strike-slip
  dUxdx_ss = ( U1/(2*np.pi) ) * ( e**2*qq*Aeta - J1*np.sin(dip_rad) )
  dUxdy_ss = ( U1/(2*np.pi) ) * ( (e**3*dtg)/((R**3)*(eta**2+qq**2)) - (e**3*Aeta+J2)*np.sin(dip_rad) )
  dUydx_ss = ( U1/(2*np.pi) ) * ( ((e*qq)/R**3)*np.cos(dip_rad) + (e*qq**2*Aeta-J2)*np.sin(dip_rad) )
  dUydy_ss = ( U1/(2*np.pi) ) * ( ((ytg*qq)/R**3)*np.cos(dip_rad) + ( qq**3*Aeta*np.sin(dip_rad) - (2*qq*np.sin(dip_rad))/(R*(R+eta)) - ((e**2+eta**2)/(R**3))*np.cos(dip_rad) - J4 )*np.sin(dip_rad) )
  # dip-slip
  dUxdx_ds = ( U2/(2*np.pi) ) * ( (e*qq)/R**3 + J3*np.sin(dip_rad)*np.cos(dip_rad) )
  dUxdy_ds = ( U2/(2*np.pi) ) * ( (ytg*qq)/R**3 - np.sin(dip_rad)/R + J1*np.sin(dip_rad)*np.cos(dip_rad) )
  dUydx_ds = ( U2/(2*np.pi) ) * ( (ytg*qq)/R**3 + (q*np.cos(dip_rad))/(R*(R+eta)) + J1*np.sin(dip_rad)*np.cos(dip_rad) )
  dUydy_ds = ( U2/(2*np.pi) ) * ( ytg**2*qq*Ae - ( (2*ytg)/(R*(R+e)) + (e*np.cos(dip_rad))/(R*(R+eta)) )*np.sin(dip_rad) + J2*np.sin(dip_rad)*np.cos(dip_rad) )
  # tensile fault
  dUxdx_tf = -( U3/(2*np.pi) ) * ( e*qq**2*Aeta + J3*(np.sin(dip_rad))**2 )
  dUxdy_tf = -( U3/(2*np.pi) ) * ( -(dtg*qq)/R**3 - e**2*qq*Aeta*np.sin(dip_rad) + J1*(np.sin(dip_rad))**2 )
  dUydx_tf = -( U3/(2*np.pi) ) * ( (qq**2/R**3)*np.cos(dip_rad) + qq**3*Aeta*np.sin(dip_rad) + J1*(np.sin(dip_rad))**2 )
  dUydy_tf = -( U3/(2*np.pi) ) * ( (ytg*np.cos(dip_rad) - dtg*np.sin(dip_rad))*qq**2*Ae - (qq*np.sin(2*dip_rad))/(R*(R+e)) - (e*qq**2*Aeta-J2)*(np.sin(dip_rad))**2 )

  # representacion Chinnery para strike-slip
  dUxdx_ss_ch = ( dUxdx_ss[0] - dUxdx_ss[1] - dUxdx_ss[2] + dUxdx_ss[3] )
  dUxdy_ss_ch = ( dUxdy_ss[0] - dUxdy_ss[1] - dUxdy_ss[2] + dUxdy_ss[3] )
  dUydx_ss_ch = ( dUydx_ss[0] - dUydx_ss[1] - dUydx_ss[2] + dUydx_ss[3] )
  dUydy_ss_ch = ( dUydy_ss[0] - dUydy_ss[1] - dUydy_ss[2] + dUydy_ss[3] )
  # representacion Chinnery para dip-slip
  dUxdx_ds_ch = ( dUxdx_ds[0] - dUxdx_ds[1] - dUxdx_ds[2] + dUxdx_ds[3] )
  dUxdy_ds_ch = ( dUxdy_ds[0] - dUxdy_ds[1] - dUxdy_ds[2] + dUxdy_ds[3] )
  dUydx_ds_ch = ( dUydx_ds[0] - dUydx_ds[1] - dUydx_ds[2] + dUydx_ds[3] )
  dUydy_ds_ch = ( dUydy_ds[0] - dUydy_ds[1] - dUydy_ds[2] + dUydy_ds[3] )
  # representacion Chinnery para tensile fault
  dUxdx_tf_ch = ( dUxdx_tf[0] - dUxdx_tf[1] - dUxdx_tf[2] + dUxdx_tf[3] )
  dUxdy_tf_ch = ( dUxdy_tf[0] - dUxdy_tf[1] - dUxdy_tf[2] + dUxdy_tf[3] )
  dUydx_tf_ch = ( dUydx_tf[0] - dUydx_tf[1] - dUydx_tf[2] + dUydx_tf[3] )
  dUydy_tf_ch = ( dUydy_tf[0] - dUydy_tf[1] - dUydy_tf[2] + dUydy_tf[3] )

  # soluciones del deformaciones dUx/dx, dUx/dy, dUy/dx, dUy/dy
  dUxdx = dUxdx_ss_ch + dUxdx_ds_ch + dUxdx_tf_ch
  dUxdy = dUxdy_ss_ch + dUxdy_ds_ch + dUxdy_tf_ch
  dUydx = dUydx_ss_ch + dUydx_ds_ch + dUydx_tf_ch
  dUydy = dUydy_ss_ch + dUydy_ds_ch + dUydy_tf_ch


  ###################################################################################
  ###### Inclinaciones dUz/dx, dUz/dy ######
  ###################################################################################

  # strike-slip
  dUzdx_ss = ( U1/(2*np.pi) ) * ( -e*qq**2*Aeta*np.cos(dip_rad) + ( (e*qq)/R**3 -K1 )*np.sin(dip_rad) )
  dUzdy_ss = ( U1/(2*np.pi) ) * ( ((dtg*qq)/R**3)*np.cos(dip_rad) + ( e**2*qq*Aeta*np.cos(dip_rad) - np.sin(dip_rad)/R + (ytg*qq)/R**3 - K2 )*np.sin(dip_rad) )
  # dip-slip
  dUzdx_ds = ( U2/(2*np.pi) ) * ( (dtg*qq)/R**3 + (qq*np.sin(dip_rad))/(R*(R+eta)) + K3*np.sin(dip_rad)*np.cos(dip_rad) )
  dUzdy_ds = ( U2/(2*np.pi) ) * ( ytg*dtg*qq*Ae - ( (2*dtg)/(R*(R+e)) + (e*np.sin(dip_rad))/(R*(R+eta)) )*np.sin(dip_rad) + K1*np.sin(dip_rad)*np.cos(dip_rad) )
  # tensile fault
  dUzdx_tf = -( U3/(2*np.pi) ) * ( (qq**2/R**3)*np.sin(dip_rad) - qq**3*Aeta*np.cos(dip_rad) + K3*(np.sin(dip_rad))**2 )
  dUzdy_tf = -( U3/(2*np.pi) ) * ( (ytg*np.sin(dip_rad) + dtg*np.cos(dip_rad))*qq**2*Ae + e*qq**2*Aeta*np.sin(dip_rad)*np.cos(dip_rad) - ( (2*qq)/(R*(R+e)) - K1)*(np.sin(dip_rad))**2 )

  # representacion Chinnery para strike-slip
  dUzdx_ss_ch = ( dUzdx_ss[0] - dUzdx_ss[1] - dUzdx_ss[2] + dUzdx_ss[3] )
  dUzdy_ss_ch = ( dUzdy_ss[0] - dUzdy_ss[1] - dUzdy_ss[2] + dUzdy_ss[3] )
  # representacion Chinnery para dip-slip
  dUzdx_ds_ch = ( dUzdx_ds[0] - dUzdx_ds[1] - dUzdx_ds[2] + dUzdx_ds[3] )
  dUzdy_ds_ch = ( dUzdy_ds[0] - dUzdy_ds[1] - dUzdy_ds[2] + dUzdy_ds[3] )
  # representacion Chinnery para tensile fault
  dUzdx_tf_ch = ( dUzdx_tf[0] - dUzdx_tf[1] - dUzdx_tf[2] + dUzdx_tf[3] )
  dUzdy_tf_ch = ( dUzdy_tf[0] - dUzdy_tf[1] - dUzdy_tf[2] + dUzdy_tf[3] )

  # soluciones del deformaciones dUz/dx, dUz/dy
  dUzdx = dUzdx_ss_ch + dUzdx_ds_ch + dUzdx_tf_ch
  dUzdy = dUzdy_ss_ch + dUzdy_ds_ch + dUzdy_tf_ch



  ###################################################################################
  # print check-list of outputs
  if verbose is True:
    #print( "\n")
    print( "OUTPUTS ->            Ux            Uy          Uz          dUx/dx       dUx/dy       dUy/dx       dUy/dy       dUz/dx       dUz/dy" ) 
    print( "------------------------------------------------------------------------------------------------------------------------------------" )
    print( "Strike-slip   ->  %+.3e   %+.3e   %+.3e   %+.3e   %+.3e   %+.3e   %+.3e   %+.3e   %+.3e" % (Ux_ss_ch, Uy_ss_ch, Uz_ss_ch, dUxdx_ss_ch, dUxdy_ss_ch, dUydx_ss_ch, dUydy_ss_ch, dUzdx_ss_ch, dUzdy_ss_ch) )
    print( "Dip-slip      ->  %+.3e   %+.3e   %+.3e   %+.3e   %+.3e   %+.3e   %+.3e   %+.3e   %+.3e" % (Ux_ds_ch, Uy_ds_ch, Uz_ds_ch, dUxdx_ds_ch, dUxdy_ds_ch, dUydx_ds_ch, dUydy_ds_ch, dUzdx_ds_ch, dUzdy_ds_ch) )
    print( "Tensile fault ->  %+.3e   %+.3e   %+.3e   %+.3e   %+.3e   %+.3e   %+.3e   %+.3e   %+.3e" % (Ux_tf_ch, Uy_tf_ch, Uz_tf_ch, dUxdx_tf_ch, dUxdy_tf_ch, dUydx_tf_ch, dUydy_tf_ch, dUzdx_tf_ch, dUzdy_tf_ch) )
    print( "------------------------------------------------------------------------------------------------------------------------------------")
    print( "                  %+.3e   %+.3e   %+.3e   %+.3e   %+.3e   %+.3e   %+.3e   %+.3e   %+.3e" % (Ux, Uy, Uz, dUxdx, dUxdy, dUydx, dUydy, dUzdx, dUzdy) )
    print( "------------------------------------------------------------------------------------------------------------------------------------")
    #print( "EOF\n\n" )

  return Ux, Uy, Uz, dUxdx, dUxdy, dUydx, dUydy, dUzdx, dUzdy








##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
def forward_point_source_okada92(x=2, y=3, z=0, dip=70, rake=10, c=4, U=1, mu=1, lmbd=1, U_as_unit=True, verbose=True):
  dip_rad = np.radians(dip)
  rake_rad = np.radians(rake)

  if U_as_unit is True:
    U1 = 1 
    U2 = 1 
    U3 = 1 
    U4 = 1
    if verbose is True:
      print( "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\nRunning ->  forward_point_source_okada92\nINPUTS  ->  x = %.2f;  y = %.2f; z = %.2f; c = %.2f (z=-c); dip = %.1f; rake = (not considered, unit_U is True); U = (not considered, unit_U is True)" % (x, y, z, c, dip) )
  elif U_as_unit is False:
    U1 = U*np.cos(rake_rad)
    U2 = U*np.sin(rake_rad)
    U3 = 0 # != 0 para intrusiones de fluidos
    U4 = 0
    if verbose is True:
      print( "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\nRunning ->  forward_point_source_okada92\nINPUTS  ->  x = %.2f;  y = %.2f; z = %.2f; c = %.2f (z=-c); dip = %.1f; rake = %.1f; U = %.2f" % (x, y, z, c, dip, rake, U) )



  ###################################################################################
  ###### Desplazamientos Ux,Uy,Uz ######
  ###################################################################################

  def get_u123ABC(x, y, z, dip, c, mu, lmbd):
    #print("Running -> evaluate_uABC_for_xyz ")
    d = c-z
    R = np.sqrt(x**2 + y**2 + d**2)
    alph = (lmbd+mu)/(lmbd+2*mu)
    p = y*np.cos(dip_rad) + d*np.sin(dip_rad)
    q = y*np.sin(dip_rad) - d*np.cos(dip_rad)
    s = p*np.sin(dip_rad) + q*np.cos(dip_rad)
    t = p*np.cos(dip_rad) - q*np.sin(dip_rad)

    A3 = 1 - 3*x**2/R**2 
    A5 = 1 - 5*x**2/R**2
    A7 = 1 - 7*x**2/R**2
    B3 = 1 - 3*y**2/R**2 
    B5 = 1 - 5*y**2/R**2
    B7 = 1 - 7*y**2/R**2
    C3 = 1 - 3*d**2/R**2 
    C5 = 1 - 5*d**2/R**2
    C7 = 1 - 7*d**2/R**2

    I1 = y*( 1/(R*(R+d)**2) - x**2*( (3*R+d)/(R**3*(R+d)**3) ) )
    I2 = x*( 1/(R*(R+d)**2) - y**2*( (3*R+d)/(R**3*(R+d)**3) ) )
    I3 = x/R**3 - I2
    I4 = -x*y*( (2*R+d)/(R**3*(R+d)**2) )
    I5 = 1/(R*(R+d)) - x**2*( (2*R+d)/(R**3*(R+d)**2) )

    # strike-slip uA(x,y,z)
    u1A_ss = ( (1-alph)/2 )*( q/R**3 ) + ( alph/2 )*( (3*x**2*q)/R**5 )
    u2A_ss = ( (1-alph)/2 )*( x/R**3 )*np.sin(dip_rad) + ( alph/2 )*( (3*x*y*q)/R**5 )
    u3A_ss = -( (1-alph)/2 )*( x/R**3 )*np.cos(dip_rad) + ( alph/2 )*( (3*x*d*q)/R**5 )
    # dip-slip uA(x,y,z)
    u1A_ds = ( alph/2 )*( (3*x*p*q)/R**5 )
    u2A_ds = ( (1-alph)/2 )*( s/R**3 ) + ( alph/2 )*( (3*y*p*q)/R**5 )
    u3A_ds = -( (1-alph)/2 )*( t/R**3 ) + ( alph/2 )*( (3*d*p*q)/R**5 )
    # tensile fault uA(x,y,z)
    u1A_tf = ( (1-alph)/2 )*( x/R**3 ) - ( alph/2 )*( (3*x*q**2)/R**5 )
    u2A_tf = ( (1-alph)/2 )*( t/R**3 ) - ( alph/2 )*( (3*y*q**2)/R**5 )
    u3A_tf = ( (1-alph)/2 )*( s/R**3 ) - ( alph/2 )*( (3*d*q**2)/R**5 )
    # inflation uA(x,y,z)
    u1A_in = -( (1-alph)/2 )*( x/R**3 ) 
    u2A_in = -( (1-alph)/2 )*( y/R**3 ) 
    u3A_in = -( (1-alph)/2 )*( d/R**3 ) 

    # strike-slip uB(x,y,z)
    u1B_ss = -(3*x**2*q)/R**5 - ( (1-alph)/alph )*I1*np.sin(dip_rad)
    u2B_ss = -(3*x*y*q)/R**5 - ( (1-alph)/alph )*I2*np.sin(dip_rad)
    u3B_ss = -(3*c*x*q)/R**5 - ( (1-alph)/alph )*I4*np.sin(dip_rad)
    # dip_slip uB(x,y,z)
    u1B_ds = -(3*x*p*q)/R**5 + ( (1-alph)/alph )*I3*np.sin(dip_rad)*np.cos(dip_rad)
    u2B_ds = -(3*y*p*q)/R**5 + ( (1-alph)/alph )*I1*np.sin(dip_rad)*np.cos(dip_rad)
    u3B_ds = -(3*c*p*q)/R**5 + ( (1-alph)/alph )*I5*np.sin(dip_rad)*np.cos(dip_rad)
    # tensile fault uB(x,y,z)
    u1B_tf = (3*x*q**2)/R**5 - ( (1-alph)/alph )*I3*(np.sin(dip_rad))**2
    u2B_tf = (3*y*q**2)/R**5 - ( (1-alph)/alph )*I1*(np.sin(dip_rad))**2
    u3B_tf = (3*c*q**2)/R**5 - ( (1-alph)/alph )*I5*(np.sin(dip_rad))**2
    # inflation uB(x,y,z)
    u1B_in = ( (1-alph)/alph )*(x/R**3)
    u2B_in = ( (1-alph)/alph )*(y/R**3)
    u3B_in = ( (1-alph)/alph )*(d/R**3)

    # strike-slip uC(x,y,z)
    u1C_ss = -(1-alph)*(A3/R**3)*np.cos(dip_rad) + alph*(3*c*q/R**5)*A5
    u2C_ss = (1-alph)*((3*x*y)/R**5)*np.cos(dip_rad) + alph*(3*c*x/R**5)*( np.sin(dip_rad) - (5*y*q)/R**2 )
    u3C_ss = -(1-alph)*((3*x*y)/R**5)*np.sin(dip_rad) + alph*(3*c*x/R**5)*( np.cos(dip_rad) + (5*d*q)/R**2 )
    # dip_slip uC(x,y,z)
    u1C_ds = (1-alph)*(3*x*t/R**5) - alph*(15*c*x*p*q)/R**7
    u2C_ds = -(1-alph)*(1/R**3)*(np.cos(2*dip_rad) - 3*y*t/R**2) + alph*(3*c/R**5)*(s - (5*y*p*q)/R**2 )
    u3C_ds = -(1-alph)*(A3/R**3)*np.sin(dip_rad)*np.cos(dip_rad) + alph*(3*c/R**5)*( t + (5*d*p*q)/R**2 )
    # tensile fault uC(x,y,z)
    u1C_tf = -(1-alph)*(3*x**5/R**5) + alph*(15*c*x*q**2)/R**7 - alph*(3*x*z)/R**5
    u2C_tf = (1-alph)*(1/R**3)*(np.sin(2*dip_rad) - 3*y*s/R**2) + alph*(3*c/R**5)*(t-y+(5*y*q**2)/R**2 ) - alph*(3*y*z)/R**5
    u3C_tf = -(1-alph)*(1/R**3)*(1-A3*(np.sin(dip_rad)**2)) - alph*(3*c/R**5)*( s-d+(5*d*q**2)/R**2 ) + alph*(3*d*z)/R**5
    # inflation uC(x,y,z)
    u1C_in = (1-alph)*(3*x*d/R**5)
    u2C_in = (1-alph)*(3*y*d/R**5) 
    u3C_in = (1-alph)*(C3/R**3) 

    return u1A_ss, u2A_ss, u3A_ss, u1B_ss, u2B_ss, u3B_ss, u1C_ss, u2C_ss, u3C_ss, u1A_ds, u2A_ds, u3A_ds, u1B_ds, u2B_ds, u3B_ds, u1C_ds, u2C_ds, u3C_ds, u1A_tf, u2A_tf, u3A_tf, u1B_tf, u2B_tf, u3B_tf, u1C_tf, u2C_tf, u3C_tf, u1A_in, u2A_in, u3A_in, u1B_in, u2B_in, u3B_in, u1C_in, u2C_in, u3C_in


  u1A_ss_1, u2A_ss_1, u3A_ss_1, u1B_ss_1, u2B_ss_1, u3B_ss_1, u1C_ss_1, u2C_ss_1, u3C_ss_1, u1A_ds_1, u2A_ds_1, u3A_ds_1, u1B_ds_1, u2B_ds_1, u3B_ds_1, u1C_ds_1, u2C_ds_1, u3C_ds_1, u1A_tf_1, u2A_tf_1, u3A_tf_1, u1B_tf_1, u2B_tf_1, u3B_tf_1, u1C_tf_1, u2C_tf_1, u3C_tf_1, u1A_in_1, u2A_in_1, u3A_in_1, u1B_in_1, u2B_in_1, u3B_in_1, u1C_in_1, u2C_in_1, u3C_in_1 = get_u123ABC(x, y, -z, dip, c, mu, lmbd)
  u1A_ss, u2A_ss, u3A_ss, u1B_ss, u2B_ss, u3B_ss, u1C_ss, u2C_ss, u3C_ss, u1A_ds, u2A_ds, u3A_ds, u1B_ds, u2B_ds, u3B_ds, u1C_ds, u2C_ds, u3C_ds, u1A_tf, u2A_tf, u3A_tf, u1B_tf, u2B_tf, u3B_tf, u1C_tf, u2C_tf, u3C_tf, u1A_in, u2A_in, u3A_in, u1B_in, u2B_in, u3B_in, u1C_in, u2C_in, u3C_in = get_u123ABC(x, y, z, dip, c, mu, lmbd)


  # soluciones strike-slip
  Ux_ss = ( U1/(2*np.pi*mu) ) * ( u1A_ss - u1A_ss_1 + u1B_ss + z*u1C_ss )
  Uy_ss = ( U1/(2*np.pi*mu) ) * ( u2A_ss - u2A_ss_1 + u2B_ss + z*u2C_ss )
  Uz_ss = ( U1/(2*np.pi*mu) ) * ( u3A_ss - u3A_ss_1 + u3B_ss + z*u3C_ss )
  # soluciones dip-slip
  Ux_ds = ( U2/(2*np.pi*mu) ) * ( u1A_ds - u1A_ds_1 + u1B_ds + z*u1C_ds )
  Uy_ds = ( U2/(2*np.pi*mu) ) * ( u2A_ds - u2A_ds_1 + u2B_ds + z*u2C_ds )
  Uz_ds = ( U2/(2*np.pi*mu) ) * ( u3A_ds - u3A_ds_1 + u3B_ds + z*u3C_ds )
  # soluciones tensile fault
  Ux_tf = ( U3/(2*np.pi*mu) ) * ( u1A_tf - u1A_tf_1 + u1B_tf + z*u1C_tf )
  Uy_tf = ( U3/(2*np.pi*mu) ) * ( u2A_tf - u2A_tf_1 + u2B_tf + z*u2C_tf )
  Uz_tf = ( U3/(2*np.pi*mu) ) * ( u3A_tf - u3A_tf_1 + u3B_tf + z*u3C_tf )
  # soluciones inflation
  Ux_in = ( U4/(2*np.pi*mu) ) * ( u1A_in - u1A_in_1 + u1B_in + z*u1C_in )
  Uy_in = ( U4/(2*np.pi*mu) ) * ( u2A_in - u2A_in_1 + u2B_in + z*u2C_in )
  Uz_in = ( U4/(2*np.pi*mu) ) * ( u3A_in - u3A_in_1 + u3B_in + z*u3C_in )

  Ux = Ux_ss + Ux_ds + Ux_tf + Ux_in
  Uy = Uy_ss + Uy_ds + Uy_tf + Uy_in
  Uz = Uz_ss + Uz_ds + Uz_tf + Uz_in


  ###################################################################################
  ###### Deformaciones dUx/dx, dUy/dx, dUz/dx ######
  ###################################################################################

  def get_dudx123ABC(x, y, z, dip, c, mu, lmbd):
    d = c-z
    R = np.sqrt(x**2 + y**2 + d**2)
    alph = (lmbd+mu)/(lmbd+2*mu)
    p = y*np.cos(dip_rad) + d*np.sin(dip_rad)
    q = y*np.sin(dip_rad) - d*np.cos(dip_rad)
    s = p*np.sin(dip_rad) + q*np.cos(dip_rad)
    t = p*np.cos(dip_rad) - q*np.sin(dip_rad)
    
    A3 = 1 - 3*x**2/R**2 
    A5 = 1 - 5*x**2/R**2
    A7 = 1 - 7*x**2/R**2
    B3 = 1 - 3*y**2/R**2 
    B5 = 1 - 5*y**2/R**2
    B7 = 1 - 7*y**2/R**2
    C3 = 1 - 3*d**2/R**2 
    C5 = 1 - 5*d**2/R**2
    C7 = 1 - 7*d**2/R**2

    J1 = -3*x*y*( (3*R+d)/(R**3*(R+d)**3) - x**2*(5*R**2+4*R*d+d**2)/(R**5*(R+d)**4)  )
    J2 = 1./R**3 - 3/(R*(R+d)**2) + 3*x**2*y**2*( (5*R**2 + 4*R*d + d**2)/(R**5*(R+d)**4) )
    J3 = A3/R**3 - J2
    J4 = -3*x*y/R**5 - J1
    K1 = -y*( (2*R+d)/(R**3*(R+d)**2) - x**2*(8*R**2+9*R*d+3*d**2)/(R**5*(R+d)**3) )
    K2 = -x*( (2*R+d)/(R**3*(R+d)**2) - y**2*(8*R**2+9*R*d+3*d**2)/(R**5*(R+d)**3) )
    K3 = -3*x*d/R**5 - K2

    # strike-slip duAdx(x,y,z)
    du1dxA_ss = -( (1-alph)/2 )*( 3*x*q/R**5 ) + ( alph/2 )*( (3*x*q)/R**5 )*( 1 + A5 )
    du2dxA_ss = ( (1-alph)/2 )*( A3/R**3 )*np.sin(dip_rad) + ( alph/2 )*( (3*y*q)/R**5 )*A5
    du3dxA_ss = -( (1-alph)/2 )*( A3/R**3 )*np.cos(dip_rad) + ( alph/2 )*( (3*d*q)/R**5 )*A5
    # dip-slip dudxA(x,y,z)
    du1dxA_ds = ( alph/2 )*( (3*p*q)/R**5 )*A5
    du2dxA_ds = -( (1-alph)/2 )*( (x*s)/R**5 ) - ( alph/2 )*( (15*x*d*p*q)/R**7 )
    du3dxA_ds = ( (1-alph)/2 )*( (x*t)/R**5 ) - ( alph/2 )*( (15*x*d*p*q)/R**7 )
    # tensile fault dudxA(x,y,z)
    du1dxA_tf = ( (1-alph)/2 )*( A3/R**3 ) - ( alph/2 )*( (3*q**2)/R**5 )*A5
    du2dxA_tf = -( (1-alph)/2 )*( (x*t)/R**5 ) + ( alph/2 )*( (15*x*y*q**2)/R**7 )
    du3dxA_tf = -( (1-alph)/2 )*( (x*s)/R**5 ) + ( alph/2 )*( (15*x*d*q**2)/R**7 )
    # inflation dudxA(x,y,z)
    du1dxA_in = -( (1-alph)/2 )*( A3/R**3 ) 
    du2dxA_in = ( (1-alph)/2 )*( (3*x*y)/R**5 ) 
    du3dxA_in = ( (1-alph)/2 )*( (3*x*d)/R**5 ) 

    # strike-slip dudxB(x,y,z)
    du1dxB_ss = -( (3*x*q)/R**5 )*( 1 + A5 ) - ( (1-alph)/alph )*J1*np.sin(dip_rad)
    du2dxB_ss = -( (3*y*q)/R**5 )*A5 - ( (1-alph)/alph )*J2*np.sin(dip_rad)
    du3dxB_ss = -( (3*c*q)/R**5 )*A5 - ( (1-alph)/alph )*K1*np.sin(dip_rad)
    # dip_slip dudxB(x,y,z)
    du1dxB_ds = -( (3*p*q)/R**5 )*A5 + ( (1-alph)/alph )*J3*np.sin(dip_rad)*np.cos(dip_rad)
    du2dxB_ds = (15*x*y*p*q)/R**7 + ( (1-alph)/alph )*J1*np.sin(dip_rad)*np.cos(dip_rad)
    du3dxB_ds = (15*c*x*p*q)/R**7 + ( (1-alph)/alph )*K3*np.sin(dip_rad)*np.cos(dip_rad)
    # tensile fault dudxB(x,y,z)
    du1dxB_tf = ( (3*q**2)/R**5 )*A5 - ( (1-alph)/alph )*J3*(np.sin(dip_rad))**2
    du2dxB_tf = -(15*x*y*q**2)/R**7 - ( (1-alph)/alph )*J1*(np.sin(dip_rad))**2
    du3dxB_tf = -(15*c*x*q**2)/R**7 - ( (1-alph)/alph )*K3*(np.sin(dip_rad))**2
    # inflation dudxB(x,y,z)
    du1dxB_in = ( (1-alph)/alph )*(A3/R**3)
    du2dxB_in = -( (1-alph)/alph )*((2*x*y)/R**5)
    du3dxB_in = -( (1-alph)/alph )*((3*x*d)/R**5)

    # strike-slip dudxC(x,y,z)
    du1dxC_ss = (1-alph)*(3*x/R**3)*(2 + A5)*np.cos(dip_rad) - alph*(15*c*x*q/R**5)*(2 + A7)
    du2dxC_ss = (1-alph)*((3*y)/R**5)*A5*np.cos(dip_rad) + alph*(3*c/R**5)*( A5*np.sin(dip_rad) - ((5*y*q)/R**2)*A7 )
    du3dxC_ss = -(1-alph)*((3*y)/R**5)*A5*np.sin(dip_rad) + alph*(3*c/R**5)*( A5*np.cos(dip_rad) + ((5*d*q)/R**2)*A7 )
    # dip_slip dudxC(x,y,z)
    du1dxC_ds = (1-alph)*(3*t/R**5)*A5 - alph*A7*(15*c*p*q)/R**7
    du2dxC_ds = (1-alph)*(3*x/R**5)*(np.cos(2*dip_rad) - 5*y*t/R**2) - alph*(15*c*x/R**7)*(s - (7*y*p*q)/R**2 )
    du3dxC_ds = (1-alph)*(A3/R**3)*np.sin(dip_rad)*np.cos(dip_rad) + alph*(3*c/R**5)*( t + (5*d*p*q)/R**2 )
    # tensile fault dudxC(x,y,z)
    du1dxC_tf = -(1-alph)*(3*s/R**5) + alph*A7*(15*c*q**2)/R**7 - alph*A5*(3*z)/R**5
    du2dxC_tf = -(1-alph)*(3*x/R**5)*(np.sin(2*dip_rad) - 5*y*s/R**2) - alph*(15*c*x/R**7)*(t-y+(7*y*q**2)/R**2 ) + alph*(15*x*y*z)/R**7
    du3dxC_tf = (1-alph)*(3*x/R**5)*(1 - (2+A5)*(np.sin(dip_rad)**2)) + alph*(15*c*x/R**7)*(s-d+(7*d*q**2)/R**2 ) - alph*(15*x*d*z)/R**7
    # inflation dudxC(x,y,z)
    du1dxC_in = (1-alph)*A5*(3*d/R**5)
    du2dxC_in = -(1-alph)*(15*x*y*d/R**7) 
    du3dxC_in = -(1-alph)*(3*x/R**5)*C5 

    return du1dxA_ss, du2dxA_ss, du3dxA_ss, du1dxB_ss, du2dxB_ss, du3dxB_ss, du1dxC_ss, du2dxC_ss, du3dxC_ss, du1dxA_ds, du2dxA_ds, du3dxA_ds, du1dxB_ds, du2dxB_ds, du3dxB_ds, du1dxC_ds, du2dxC_ds, du3dxC_ds, du1dxA_tf, du2dxA_tf, du3dxA_tf, du1dxB_tf, du2dxB_tf, du3dxB_tf, du1dxC_tf, du2dxC_tf, du3dxC_tf, du1dxA_in, du2dxA_in, du3dxA_in, du1dxB_in, du2dxB_in, du3dxB_in, du1dxC_in, du2dxC_in, du3dxC_in


  du1dxA_ss_1, du2dxA_ss_1, du3dxA_ss_1, du1dxB_ss_1, du2dxB_ss_1, du3dxB_ss_1, du1dxC_ss_1, du2dxC_ss_1, du3dxC_ss_1, du1dxA_ds_1, du2dxA_ds_1, du3dxA_ds_1, du1dxB_ds_1, du2dxB_ds_1, du3dxB_ds_1, du1dxC_ds_1, du2dxC_ds_1, du3dxC_ds_1, du1dxA_tf_1, du2dxA_tf_1, du3dxA_tf_1, du1dxB_tf_1, du2dxB_tf_1, du3dxB_tf_1, du1dxC_tf_1, du2dxC_tf_1, du3dxC_tf_1, du1dxA_in_1, du2dxA_in_1, du3dxA_in_1, du1dxB_in_1, du2dxB_in_1, du3dxB_in_1, du1dxC_in_1, du2dxC_in_1, du3dxC_in_1 = get_dudx123ABC(x, y, -z, dip, c, mu, lmbd)
  du1dxA_ss, du2dxA_ss, du3dxA_ss, du1dxB_ss, du2dxB_ss, du3dxB_ss, du1dxC_ss, du2dxC_ss, du3dxC_ss, du1dxA_ds, du2dxA_ds, du3dxA_ds, du1dxB_ds, du2dxB_ds, du3dxB_ds, du1dxC_ds, du2dxC_ds, du3dxC_ds, du1dxA_tf, du2dxA_tf, du3dxA_tf, du1dxB_tf, du2dxB_tf, du3dxB_tf, du1dxC_tf, du2dxC_tf, du3dxC_tf, du1dxA_in, du2dxA_in, du3dxA_in, du1dxB_in, du2dxB_in, du3dxB_in, du1dxC_in, du2dxC_in, du3dxC_in = get_dudx123ABC(x, y, z, dip, c, mu, lmbd)

  # soluciones strike-slip
  dUxdx_ss = ( U1/(2*np.pi*mu) ) * ( du1dxA_ss - du1dxA_ss_1 + du1dxB_ss + z*du1dxC_ss )
  dUydx_ss = ( U1/(2*np.pi*mu) ) * ( du2dxA_ss - du2dxA_ss_1 + du2dxB_ss + z*du2dxC_ss )
  dUzdx_ss = ( U1/(2*np.pi*mu) ) * ( du3dxA_ss - du3dxA_ss_1 + du3dxB_ss + z*du3dxC_ss )
  # soluciones dip-slip
  dUxdx_ds = ( U2/(2*np.pi*mu) ) * ( du1dxA_ds - du1dxA_ds_1 + du1dxB_ds + z*du1dxC_ds )
  dUydx_ds = ( U2/(2*np.pi*mu) ) * ( du2dxA_ds - du2dxA_ds_1 + du2dxB_ds + z*du2dxC_ds )
  dUzdx_ds = ( U2/(2*np.pi*mu) ) * ( du3dxA_ds - du3dxA_ds_1 + du3dxB_ds + z*du3dxC_ds )
  # soluciones tensile fault
  dUxdx_tf = ( U3/(2*np.pi*mu) ) * ( du1dxA_tf - du1dxA_tf_1 + du1dxB_tf + z*du1dxC_tf )
  dUydx_tf = ( U3/(2*np.pi*mu) ) * ( du2dxA_tf - du2dxA_tf_1 + du2dxB_tf + z*du2dxC_tf )
  dUzdx_tf = ( U3/(2*np.pi*mu) ) * ( du3dxA_tf - du3dxA_tf_1 + du3dxB_tf + z*du3dxC_tf )
  # soluciones inflation
  dUxdx_in = ( U4/(2*np.pi*mu) ) * ( du1dxA_in - du1dxA_in_1 + du1dxB_in + z*du1dxC_in )
  dUydx_in = ( U4/(2*np.pi*mu) ) * ( du2dxA_in - du2dxA_in_1 + du2dxB_in + z*du2dxC_in )
  dUzdx_in = ( U4/(2*np.pi*mu) ) * ( du3dxA_in - du3dxA_in_1 + du3dxB_in + z*du3dxC_in )

  dUxdx = dUxdx_ss + dUxdx_ds + dUxdx_tf + dUxdx_in
  dUydx = dUydx_ss + dUydx_ds + dUydx_tf + dUydx_in
  dUzdx = dUzdx_ss + dUzdx_ds + dUzdx_tf + dUzdx_in



  ###################################################################################
  ###### Deformaciones dUx/dy, dUy/dy, dUz/dy ######
  ###################################################################################

  def get_dudy123ABC(x, y, z, dip, c, mu, lmbd):
    d = c-z
    R = np.sqrt(x**2 + y**2 + d**2)
    alph = (lmbd+mu)/(lmbd+2*mu)
    p = y*np.cos(dip_rad) + d*np.sin(dip_rad)
    q = y*np.sin(dip_rad) - d*np.cos(dip_rad)
    s = p*np.sin(dip_rad) + q*np.cos(dip_rad)
    t = p*np.cos(dip_rad) - q*np.sin(dip_rad)
    U = np.sin(dip_rad) - 5*y*q/R**2
    V = s - 5*y*p*q/R**2
    W = np.sin(dip_rad) + U

    A3 = 1 - 3*x**2/R**2 
    A5 = 1 - 5*x**2/R**2
    A7 = 1 - 7*x**2/R**2
    B3 = 1 - 3*y**2/R**2 
    B5 = 1 - 5*y**2/R**2
    B7 = 1 - 7*y**2/R**2
    C3 = 1 - 3*d**2/R**2 
    C5 = 1 - 5*d**2/R**2
    C7 = 1 - 7*d**2/R**2

    J1 = -3*x*y*( (3*R+d)/(R**3*(R+d)**3) - x**2*(5*R**2+4*R*d+d**2)/(R**5*(R+d)**4)  )
    J2 = 1./R**3 - 3/(R*(R+d)**2) + 3*x**2*y**2*( (5*R**2 + 4*R*d + d**2)/(R**5*(R+d)**4) )
    J3 = A3/R**3 - J2
    J4 = -3*x*y/R**5 - J1
    K1 = -y*( (2*R+d)/(R**3*(R+d)**2) - x**2*(8*R**2+9*R*d+3*d**2)/(R**5*(R+d)**3) )
    K2 = -x*( (2*R+d)/(R**3*(R+d)**2) - y**2*(8*R**2+9*R*d+3*d**2)/(R**5*(R+d)**3) )
    K3 = -3*x*d/R**5 - K2

    # strike-slip dudyA(x,y,z)
    du1dyA_ss = ( (1-alph)/2 )*( 1/R**3 )*( np.sin(dip_rad) - 3*y*q/R**2 ) + ( alph/2 )*( (3*x**2)/R**5 )*U
    du2dyA_ss = -( (1-alph)/2 )*( 3*x*y/R**5 )*np.sin(dip_rad) + ( alph/2 )*( (3*x*y)/R**5 )*U + ( alph/2 )*(3*x*q/R**5)
    du3dyA_ss = ( (1-alph)/2 )*( 3*x*y/R**5 )*np.cos(dip_rad) + ( alph/2 )*( (3*x*d)/R**5 )*U
    # dip-slip dudyA(x,y,z)
    du1dyA_ds = ( alph/2 )*( (3*x)/R**5 )*V
    du2dyA_ds = ( (1-alph)/2 )*( 1/R**3 )*( np.sin(2*dip_rad) - 3*y*s/R**2 ) + ( alph/2 )*( (3*y)/R**5 )*V + ( alph/2 )*( (3*p*q)/R**5 )
    du3dyA_ds = -( (1-alph)/2 )*( 1/R**3 )*( np.cos(2*dip_rad) - 3*y*t/R**2 ) + ( alph/2 )*( 3*d/R**5 )*V
    # tensile fault dudyA(x,y,z)
    du1dyA_tf = -( (1-alph)/2 )*( 3*x*y/R**5 ) - ( alph/2 )*( (3*x*q)/R**5 )*W
    du2dyA_tf = ( (1-alph)/2 )*( 1/R**3 )*( np.cos(2*dip_rad) - 3*y*t/R**2 ) - ( alph/2 )*( 3*y*q/R**5 )*W - ( alph/2 )*( 3*q**2/R**5 )
    du3dyA_tf = ( (1-alph)/2 )*( 1/R**3 )*( np.sin(2*dip_rad) - 3*y*s/R**2 ) - ( alph/2 )*( 3*d*q/R**5 )*W
    # inflation dudyA(x,y,z)
    du1dyA_in = ( (1-alph)/2 )*( (3*x*y)/R**5 ) 
    du2dyA_in = -( (1-alph)/2 )*( B3/R**3 ) 
    du3dyA_in = ( (1-alph)/2 )*( (3*y*d)/R**5 ) 

    # strike-slip dudyB(x,y,z)
    du1dyB_ss = -( (3*x**2)/R**5 )*U - ( (1-alph)/alph )*J2*np.sin(dip_rad)
    du2dyB_ss = -( (3*x*y)/R**5 )*U - 3*x*q/R**5 - ( (1-alph)/alph )*J4*np.sin(dip_rad)
    du3dyB_ss = -( (3*c*x)/R**5 )*U - ( (1-alph)/alph )*K2*np.sin(dip_rad)
    # dip_slip dudyB(x,y,z)
    du1dyB_ds = -( 3*x/R**5 )*V + ( (1-alph)/alph )*J1*np.sin(dip_rad)*np.cos(dip_rad)
    du2dyB_ds = -( 3*y/R**5 )*V - 3*p*q/R**5 + ( (1-alph)/alph )*J2*np.sin(dip_rad)*np.cos(dip_rad)
    du3dyB_ds = -( 3*c/R**5 )*V + ( (1-alph)/alph )*K1*np.sin(dip_rad)*np.cos(dip_rad)
    # tensile fault dudyB(x,y,z)
    du1dyB_tf = ( 3*x*q/R**5 )*W - ( (1-alph)/alph )*J1*(np.sin(dip_rad))**2
    du2dyB_tf = ( 3*y*q/R**5 )*W + 3*q**2/R**5 - ( (1-alph)/alph )*J2*(np.sin(dip_rad))**2
    du3dyB_tf = ( 3*c*q/R**5 )*W - ( (1-alph)/alph )*K1*(np.sin(dip_rad))**2
    # inflation dudyB(x,y,z)
    du1dyB_in = -( (1-alph)/alph )*(3*x*y/R**5)
    du2dyB_in = ( (1-alph)/alph )*(B3/R**3)
    du3dyB_in = -( (1-alph)/alph )*(3*y*d/R**5)

    # strike-slip dudyC(x,y,z)
    du1dyC_ss = (1-alph)*(3*y/R**5)*A5*np.cos(dip_rad) + alph*( 3*c/R**5 )*( A5*np.sin(dip_rad) - (5*y*q/R**2)*A7 )
    du2dyC_ss = (1-alph)*(3*x/R**5)*B5*np.cos(dip_rad) - alph*(15*c*x/R**7)*( 2*y*np.sin(dip_rad) + q*B7 )
    du3dyC_ss = -(1-alph)*(3*x/R**5)*B5*np.sin(dip_rad) + alph*(15*c*x/R**7)*( d*B7*np.sin(dip_rad) - y*C7*np.cos(dip_rad) )
    # dip_slip dudyC(x,y,z)
    du1dyC_ds = (1-alph)*(3*x/R**5)*(np.cos(2*dip_rad) - 5*y*t/R**2 ) - alph*(15*c*x/R**7)*(s - 7*y*p*q/R**2)
    du2dyC_ds = (1-alph)*(3/R**5)*(2*y*np.cos(2*dip_rad) + t*B5) + alph*(3*c/R**5)*(np.sin(2*dip_rad) - 10*y*s/R**2 - (5*p*q/R**2)*B7 )
    du3dyC_ds = (1-alph)*(3*y/R**5)*A5*np.sin(dip_rad)*np.cos(dip_rad) - alph*(3*c/R**5)*( (3+A5)*np.cos(2*dip_rad) + (35*y*d*p*q)/R**4 )
    # tensile fault dudyC(x,y,z)
    du1dyC_tf = -(1-alph)*(3*x/R**5)*(np.sin(2*dip_rad) - 5*y*s/R**2) - alph*(15*c*x/R**7)*(t - y + 7*y*q**2/R**2) + alph*(15*x*y*z)/R**7
    du2dyC_tf = -(1-alph)*(3/R**5)*(2*y*np.sin(2*dip_rad) + s*B5) - alph*(3*c/R**5)*( 2*np.sin(dip_rad)**2 + 10*y*(t-y)/R**2 - (5*q**2/R**2)*B7 ) - alph*(3*z/R**5)*B5
    du3dyC_tf = (1-alph)*(3*y/R**5)*(1 - A5*(np.sin(dip_rad)**2)) + alph*(3*c/R**5)*( (3+A5)*np.sin(2*dip_rad) - (5*y*d/R**2)*(2-7*q**2/R**2) ) - alph*(15*y*d*z/R**7)
    # inflation dudyC(x,y,z)
    du1dyC_in = -(1-alph)*(15*x*y*d/R**7)
    du2dyC_in = (1-alph)*(3*d/R**5)*B5 
    du3dyC_in = -(1-alph)*(3*y/R**5)*C5 

    return du1dyA_ss, du2dyA_ss, du3dyA_ss, du1dyB_ss, du2dyB_ss, du3dyB_ss, du1dyC_ss, du2dyC_ss, du3dyC_ss, du1dyA_ds, du2dyA_ds, du3dyA_ds, du1dyB_ds, du2dyB_ds, du3dyB_ds, du1dyC_ds, du2dyC_ds, du3dyC_ds, du1dyA_tf, du2dyA_tf, du3dyA_tf, du1dyB_tf, du2dyB_tf, du3dyB_tf, du1dyC_tf, du2dyC_tf, du3dyC_tf, du1dyA_in, du2dyA_in, du3dyA_in, du1dyB_in, du2dyB_in, du3dyB_in, du1dyC_in, du2dyC_in, du3dyC_in


  du1dyA_ss_1, du2dyA_ss_1, du3dyA_ss_1, du1dyB_ss_1, du2dyB_ss_1, du3dyB_ss_1, du1dyC_ss_1, du2dyC_ss_1, du3dyC_ss_1, du1dyA_ds_1, du2dyA_ds_1, du3dyA_ds_1, du1dyB_ds_1, du2dyB_ds_1, du3dyB_ds_1, du1dyC_ds_1, du2dyC_ds_1, du3dyC_ds_1, du1dyA_tf_1, du2dyA_tf_1, du3dyA_tf_1, du1dyB_tf_1, du2dyB_tf_1, du3dyB_tf_1, du1dyC_tf_1, du2dyC_tf_1, du3dyC_tf_1, du1dyA_in_1, du2dyA_in_1, du3dyA_in_1, du1dyB_in_1, du2dyB_in_1, du3dyB_in_1, du1dyC_in_1, du2dyC_in_1, du3dyC_in_1 = get_dudy123ABC(x, y, -z, dip, c, mu, lmbd)
  du1dyA_ss, du2dyA_ss, du3dyA_ss, du1dyB_ss, du2dyB_ss, du3dyB_ss, du1dyC_ss, du2dyC_ss, du3dyC_ss, du1dyA_ds, du2dyA_ds, du3dyA_ds, du1dyB_ds, du2dyB_ds, du3dyB_ds, du1dyC_ds, du2dyC_ds, du3dyC_ds, du1dyA_tf, du2dyA_tf, du3dyA_tf, du1dyB_tf, du2dyB_tf, du3dyB_tf, du1dyC_tf, du2dyC_tf, du3dyC_tf, du1dyA_in, du2dyA_in, du3dyA_in, du1dyB_in, du2dyB_in, du3dyB_in, du1dyC_in, du2dyC_in, du3dyC_in = get_dudy123ABC(x, y, z, dip, c, mu, lmbd)

  # soluciones strike-slip
  dUxdy_ss = ( U1/(2*np.pi*mu) ) * ( du1dyA_ss - du1dyA_ss_1 + du1dyB_ss + z*du1dyC_ss )
  dUydy_ss = ( U1/(2*np.pi*mu) ) * ( du2dyA_ss - du2dyA_ss_1 + du2dyB_ss + z*du2dyC_ss )
  dUzdy_ss = ( U1/(2*np.pi*mu) ) * ( du3dyA_ss - du3dyA_ss_1 + du3dyB_ss + z*du3dyC_ss )
  # soluciones dip-slip
  dUxdy_ds = ( U2/(2*np.pi*mu) ) * ( du1dyA_ds - du1dyA_ds_1 + du1dyB_ds + z*du1dyC_ds )
  dUydy_ds = ( U2/(2*np.pi*mu) ) * ( du2dyA_ds - du2dyA_ds_1 + du2dyB_ds + z*du2dyC_ds )
  dUzdy_ds = ( U2/(2*np.pi*mu) ) * ( du3dyA_ds - du3dyA_ds_1 + du3dyB_ds + z*du3dyC_ds )
  # soluciones tensile fault
  dUxdy_tf = ( U3/(2*np.pi*mu) ) * ( du1dyA_tf - du1dyA_tf_1 + du1dyB_tf + z*du1dyC_tf )
  dUydy_tf = ( U3/(2*np.pi*mu) ) * ( du2dyA_tf - du2dyA_tf_1 + du2dyB_tf + z*du2dyC_tf )
  dUzdy_tf = ( U3/(2*np.pi*mu) ) * ( du3dyA_tf - du3dyA_tf_1 + du3dyB_tf + z*du3dyC_tf )
  # soluciones inflation
  dUxdy_in = ( U4/(2*np.pi*mu) ) * ( du1dyA_in - du1dyA_in_1 + du1dyB_in + z*du1dyC_in )
  dUydy_in = ( U4/(2*np.pi*mu) ) * ( du2dyA_in - du2dyA_in_1 + du2dyB_in + z*du2dyC_in )
  dUzdy_in = ( U4/(2*np.pi*mu) ) * ( du3dyA_in - du3dyA_in_1 + du3dyB_in + z*du3dyC_in )

  dUxdy = dUxdy_ss + dUxdy_ds + dUxdy_tf + dUxdy_in
  dUydy = dUydy_ss + dUydy_ds + dUydy_tf + dUydy_in
  dUzdy = dUzdy_ss + dUzdy_ds + dUzdy_tf + dUzdy_in


  ###################################################################################
  ###### Deformaciones dUx/dz, dUy/dz, dUz/dz ######
  ###################################################################################

  def get_dudz123ABC(x, y, z, dip, c, mu, lmbd):
    d = c-z
    R = np.sqrt(x**2 + y**2 + d**2)
    alph = (lmbd+mu)/(lmbd+2*mu)
    p = y*np.cos(dip_rad) + d*np.sin(dip_rad)
    q = y*np.sin(dip_rad) - d*np.cos(dip_rad)
    s = p*np.sin(dip_rad) + q*np.cos(dip_rad)
    t = p*np.cos(dip_rad) - q*np.sin(dip_rad)
    UU = np.cos(dip_rad) + 5*d*q/R**2
    VV = t + 5*d*p*q/R**2
    WW = np.cos(dip_rad) + UU

    A3 = 1 - 3*x**2/R**2 
    A5 = 1 - 5*x**2/R**2
    A7 = 1 - 7*x**2/R**2
    B3 = 1 - 3*y**2/R**2 
    B5 = 1 - 5*y**2/R**2
    B7 = 1 - 7*y**2/R**2
    C3 = 1 - 3*d**2/R**2 
    C5 = 1 - 5*d**2/R**2
    C7 = 1 - 7*d**2/R**2

    J1 = -3*x*y*( (3*R+d)/(R**3*(R+d)**3) - x**2*(5*R**2+4*R*d+d**2)/(R**5*(R+d)**4)  )
    J2 = 1./R**3 - 3/(R*(R+d)**2) + 3*x**2*y**2*( (5*R**2 + 4*R*d + d**2)/(R**5*(R+d)**4) )
    J3 = A3/R**3 - J2
    J4 = -3*x*y/R**5 - J1
    K1 = -y*( (2*R+d)/(R**3*(R+d)**2) - x**2*(8*R**2+9*R*d+3*d**2)/(R**5*(R+d)**3) )
    K2 = -x*( (2*R+d)/(R**3*(R+d)**2) - y**2*(8*R**2+9*R*d+3*d**2)/(R**5*(R+d)**3) )
    K3 = -3*x*d/R**5 - K2

    # strike-slip dudzA(x,y,z)
    du1dzA_ss = ( (1-alph)/2 )*( 1/R**3 )*( np.cos(dip_rad) + 3*d*q/R**2 ) + ( alph/2 )*( (3*x**2)/R**5 )*UU
    du2dzA_ss = ( (1-alph)/2 )*( 3*x*d/R**5 )*np.sin(dip_rad) + ( alph/2 )*( 3*x*y/R**5 )*UU
    du3dzA_ss = -( (1-alph)/2 )*( 3*x*d/R**5 )*np.cos(dip_rad) + ( alph/2 )*( 3*x*d/R**5 )*UU - ( alph/2 )*( 3*x*q/R**5 )
    # dip-slip dudzA(x,y,z)
    du1dzA_ds = ( alph/2 )*( (3*x)/R**5 )*VV
    du2dzA_ds = ( (1-alph)/2 )*( 1/R**3 )*( np.cos(2*dip_rad) + 3*d*s/R**2 ) + ( alph/2 )*( 3*y/R**5 )*VV
    du3dzA_ds = ( (1-alph)/2 )*( 1/R**3 )*( np.sin(2*dip_rad) - 3*d*t/R**2 ) + ( alph/2 )*( 3*d/R**5 )*VV - ( alph/2 )*( (3*p*q)/R**5 )
    # tensile fault dudzA(x,y,z)
    du1dzA_tf = ( (1-alph)/2 )*( 3*x*d/R**5 ) - ( alph/2 )*( (3*x*q)/R**5 )*WW
    du2dzA_tf = -( (1-alph)/2 )*( 1/R**3 )*( np.sin(2*dip_rad) - 3*d*t/R**2 ) - ( alph/2 )*( 3*y*q/R**5 )*WW 
    du3dzA_tf = ( (1-alph)/2 )*( 1/R**3 )*( np.cos(2*dip_rad) + 3*d*s/R**2 ) - ( alph/2 )*( 3*d*q/R**5 )*WW + ( alph/2 )*( 3*q**2/R**5 )
    # inflation dudzA(x,y,z)
    du1dzA_in = -( (1-alph)/2 )*( 3*x*d/R**5 ) 
    du2dzA_in = -( (1-alph)/2 )*( 3*y*d/R**5 ) 
    du3dzA_in = ( (1-alph)/2 )*( C3/R**3 ) 

    # strike-slip dudzB(x,y,z)
    du1dzB_ss = -( (3*x**2)/R**5 )*UU + ( (1-alph)/alph )*K1*np.sin(dip_rad)
    du2dzB_ss = -( (3*x*y)/R**5 )*UU + ( (1-alph)/alph )*K2*np.sin(dip_rad)
    du3dzB_ss = -( (3*c*x)/R**5 )*UU + ( (1-alph)/alph )*(3*x*y/R**5)*np.sin(dip_rad)
    # dip_slip dudzB(x,y,z)
    du1dzB_ds = -( 3*x/R**5 )*VV - ( (1-alph)/alph )*K3*np.sin(dip_rad)*np.cos(dip_rad)
    du2dzB_ds = -( 3*y/R**5 )*VV - ( (1-alph)/alph )*K1*np.sin(dip_rad)*np.cos(dip_rad)
    du3dzB_ds = -( 3*c/R**5 )*VV + ( (1-alph)/alph )*(A3/R**3)*np.sin(dip_rad)*np.cos(dip_rad)
    # tensile fault dudzB(x,y,z)
    du1dzB_tf = ( 3*x*q/R**5 )*WW + ( (1-alph)/alph )*K3*(np.sin(dip_rad))**2
    du2dzB_tf = ( 3*y*q/R**5 )*WW + ( (1-alph)/alph )*K1*(np.sin(dip_rad))**2
    du3dzB_tf = ( 3*c*q/R**5 )*WW - ( (1-alph)/alph )*(A3/R**3)*(np.sin(dip_rad))**2
    # inflation dudzB(x,y,z)
    du1dzB_in = ( (1-alph)/alph )*(3*x*d/R**5)
    du2dzB_in = ( (1-alph)/alph )*(3*y*d/R**5)
    du3dzB_in = -( (1-alph)/alph )*(C3/R**3)

    # strike-slip dudzC(x,y,z)
    du1dzC_ss = -(1-alph)*(3*d/R**5)*A5*np.cos(dip_rad) + alph*( 3*c/R**5 )*( A5*np.cos(dip_rad) + (5*d*q/R**2)*A7 )
    du2dzC_ss = (1-alph)*(15*x*y*d/R**7)*np.cos(dip_rad) + alph*(15*c*x/R**7)*( d*B7*np.sin(dip_rad) - y*C7*np.cos(dip_rad) )
    du3dzC_ss = -(1-alph)*(15*x*y*d/R**7)*np.sin(dip_rad) + alph*(15*c*x/R**7)*( 2*d*np.cos(dip_rad) - q*C7 )
    # dip_slip dudzC(x,y,z)
    du1dzC_ds = -(1-alph)*(3*x/R**5)*(np.sin(2*dip_rad) - 5*d*t/R**2 ) - alph*(15*c*x/R**7)*(t + 7*d*p*q/R**2)
    du2dzC_ds = -(1-alph)*(3/R**5)*(d*B5*np.cos(2*dip_rad) + y*C5*np.sin(2*dip_rad)) - alph*(3*c/R**5)*((3+A5)*np.cos(2*dip_rad) + 35*y*d*p*q/R**4 )
    du3dzC_ds = -(1-alph)*(3*d/R**5)*A5*np.sin(dip_rad)*np.cos(dip_rad) - alph*(3*c/R**5)*( np.sin(2*dip_rad) - (10*d*t)/R**2  + (5*p*q/R**2)*C7)
    # tensile fault dudzC(x,y,z)
    du1dzC_tf = -(1-alph)*(3*x/R**5)*(np.cos(2*dip_rad) + 5*d*s/R**2) + alph*(15*c*x/R**7)*(s - d + 7*d*q**2/R**2) - alph*(3*x/R**5)*(1+5*d*z/R**2)
    du2dzC_tf = (1-alph)*(3/R**5)*(d*B5*np.sin(2*dip_rad) - y*C5*np.cos(2*dip_rad)) + alph*(3*c/R**5)*( (3+A5)*np.sin(2*dip_rad) - (5*y*d/R**2)*(2-7*q**2/R**2) ) - alph*(3*y/R**5)*(1+5*d*z/R**2)
    du3dzC_tf = -(1-alph)*(3*d/R**5)*(1 - A5*(np.sin(dip_rad)**2)) - alph*(3*c/R**5)*( np.cos(2*dip_rad) + (10*d*(s-d)/R**2) - (5*q**2/R**2)*C7 ) - alph*(3*z/R**5)*(1+C5)
    # inflation dudzC(x,y,z)
    du1dzC_in = -(1-alph)*(3*x/R**5)*C5
    du2dzC_in = -(1-alph)*(3*y/R**5)*C5 
    du3dzC_in = (1-alph)*(3*d/R**5)*(2+C5) 

    return du1dzA_ss, du2dzA_ss, du3dzA_ss, du1dzB_ss, du2dzB_ss, du3dzB_ss, du1dzC_ss, du2dzC_ss, du3dzC_ss, du1dzA_ds, du2dzA_ds, du3dzA_ds, du1dzB_ds, du2dzB_ds, du3dzB_ds, du1dzC_ds, du2dzC_ds, du3dzC_ds, du1dzA_tf, du2dzA_tf, du3dzA_tf, du1dzB_tf, du2dzB_tf, du3dzB_tf, du1dzC_tf, du2dzC_tf, du3dzC_tf, du1dzA_in, du2dzA_in, du3dzA_in, du1dzB_in, du2dzB_in, du3dzB_in, du1dzC_in, du2dzC_in, du3dzC_in


  du1dzA_ss_1, du2dzA_ss_1, du3dzA_ss_1, du1dzB_ss_1, du2dzB_ss_1, du3dzB_ss_1, du1dzC_ss_1, du2dzC_ss_1, du3dzC_ss_1, du1dzA_ds_1, du2dzA_ds_1, du3dzA_ds_1, du1dzB_ds_1, du2dzB_ds_1, du3dzB_ds_1, du1dzC_ds_1, du2dzC_ds_1, du3dzC_ds_1, du1dzA_tf_1, du2dzA_tf_1, du3dzA_tf_1, du1dzB_tf_1, du2dzB_tf_1, du3dzB_tf_1, du1dzC_tf_1, du2dzC_tf_1, du3dzC_tf_1, du1dzA_in_1, du2dzA_in_1, du3dzA_in_1, du1dzB_in_1, du2dzB_in_1, du3dzB_in_1, du1dzC_in_1, du2dzC_in_1, du3dzC_in_1 = get_dudz123ABC(x, y, -z, dip, c, mu, lmbd)
  du1dzA_ss, du2dzA_ss, du3dzA_ss, du1dzB_ss, du2dzB_ss, du3dzB_ss, du1dzC_ss, du2dzC_ss, du3dzC_ss, du1dzA_ds, du2dzA_ds, du3dzA_ds, du1dzB_ds, du2dzB_ds, du3dzB_ds, du1dzC_ds, du2dzC_ds, du3dzC_ds, du1dzA_tf, du2dzA_tf, du3dzA_tf, du1dzB_tf, du2dzB_tf, du3dzB_tf, du1dzC_tf, du2dzC_tf, du3dzC_tf, du1dzA_in, du2dzA_in, du3dzA_in, du1dzB_in, du2dzB_in, du3dzB_in, du1dzC_in, du2dzC_in, du3dzC_in = get_dudz123ABC(x, y, z, dip, c, mu, lmbd)

  # soluciones strike-slip
  dUxdz_ss = ( U1/(2*np.pi*mu) ) * ( du1dzA_ss - du1dzA_ss_1 + du1dzB_ss + z*du1dzC_ss )
  dUydz_ss = ( U1/(2*np.pi*mu) ) * ( du2dzA_ss - du2dzA_ss_1 + du2dzB_ss + z*du2dzC_ss )
  dUzdz_ss = ( U1/(2*np.pi*mu) ) * ( du3dzA_ss - du3dzA_ss_1 + du3dzB_ss + z*du3dzC_ss )
  # soluciones dip-slip
  dUxdz_ds = ( U2/(2*np.pi*mu) ) * ( du1dzA_ds - du1dzA_ds_1 + du1dzB_ds + z*du1dzC_ds )
  dUydz_ds = ( U2/(2*np.pi*mu) ) * ( du2dzA_ds - du2dzA_ds_1 + du2dzB_ds + z*du2dzC_ds )
  dUzdz_ds = ( U2/(2*np.pi*mu) ) * ( du3dzA_ds - du3dzA_ds_1 + du3dzB_ds + z*du3dzC_ds )
  # soluciones tensile fault
  dUxdz_tf = ( U3/(2*np.pi*mu) ) * ( du1dzA_tf - du1dzA_tf_1 + du1dzB_tf + z*du1dzC_tf )
  dUydz_tf = ( U3/(2*np.pi*mu) ) * ( du2dzA_tf - du2dzA_tf_1 + du2dzB_tf + z*du2dzC_tf )
  dUzdz_tf = ( U3/(2*np.pi*mu) ) * ( du3dzA_tf - du3dzA_tf_1 + du3dzB_tf + z*du3dzC_tf )
  # soluciones inflation
  dUxdz_in = ( U4/(2*np.pi*mu) ) * ( du1dzA_in - du1dzA_in_1 + du1dzB_in + z*du1dzC_in )
  dUydz_in = ( U4/(2*np.pi*mu) ) * ( du2dzA_in - du2dzA_in_1 + du2dzB_in + z*du2dzC_in )
  dUzdz_in = ( U4/(2*np.pi*mu) ) * ( du3dzA_in - du3dzA_in_1 + du3dzB_in + z*du3dzC_in )

  dUxdz = dUxdz_ss + dUxdz_ds + dUxdz_tf + dUxdz_in
  dUydz = dUydz_ss + dUydz_ds + dUydz_tf + dUydz_in
  dUzdz = dUzdz_ss + dUzdz_ds + dUzdz_tf + dUzdz_in




  ###################################################################################
  # print check-list of outputs
  if verbose is True:
    print( "OUTPUTS ->         Ux          Uy          Uz        dUx/dx       dUy/dx       dUz/dx       dUx/dy       dUy/dy       dUz/dy       dUx/dz       dUy/dz       dUz/dz" ) 
    print( "------------------------------------------------------------------------------------------------------------------------------------" )
    print( "Strike-slip   -> %+.3e  %+.3e  %+.3e  %+.3e  %+.3e  %+.3e  %+.3e  %+.3e  %+.3e  %+.3e  %+.3e  %+.3e" % (Ux_ss, Uy_ss, Uz_ss, dUxdx_ss, dUydx_ss, dUzdx_ss, dUxdy_ss, dUydy_ss, dUzdy_ss, dUxdz_ss, dUydz_ss, dUzdz_ss) )
    print( "Dip-slip      -> %+.3e  %+.3e  %+.3e  %+.3e  %+.3e  %+.3e  %+.3e  %+.3e  %+.3e  %+.3e  %+.3e  %+.3e" % (Ux_ds, Uy_ds, Uz_ds, dUxdx_ds, dUydx_ds, dUzdx_ds, dUxdy_ds, dUydy_ds, dUzdy_ds, dUxdz_ds, dUydz_ds, dUzdz_ds) )
    print( "Tensile fault -> %+.3e  %+.3e  %+.3e  %+.3e  %+.3e  %+.3e  %+.3e  %+.3e  %+.3e  %+.3e  %+.3e  %+.3e" % (Ux_tf, Uy_tf, Uz_tf, dUxdx_tf, dUydx_tf, dUzdx_tf, dUxdy_tf, dUydy_tf, dUzdy_tf, dUxdz_tf, dUydz_tf, dUzdz_tf) )
    print( "Inflation     -> %+.3e  %+.3e  %+.3e  %+.3e  %+.3e  %+.3e  %+.3e  %+.3e  %+.3e  %+.3e  %+.3e  %+.3e" % (Ux_in, Uy_in, Uz_in, dUxdx_in, dUydx_in, dUzdx_in, dUxdy_in, dUydy_in, dUzdy_in, dUxdz_in, dUydz_in, dUzdz_in) )
    print( "------------------------------------------------------------------------------------------------------------------------------------")
    print( "                 %+.3e  %+.3e  %+.3e  %+.3e  %+.3e  %+.3e  %+.3e  %+.3e  %+.3e  %+.3e  %+.3e  %+.3e" % (Ux, Uy, Uz, dUxdx, dUydx, dUzdx, dUxdy, dUydy, dUzdy, dUxdz, dUydz, dUzdz) )
    print( "------------------------------------------------------------------------------------------------------------------------------------")
    #print( "EOF\n\n" )

  return Ux, Uy, Uz, dUxdx, dUydx, dUzdx, dUxdy, dUydy, dUzdy, dUxdz, dUydz, dUzdz






##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
def forward_finite_rectangular_source_okada92_Uxyz(x=2, y=3, z=0, dip=70, rake=10, c=4, L=3, W=2, U=1, mu=1, lmbd=1, U_as_unit=True, verbose=True):
  dip_rad = np.radians(dip)
  rake_rad = np.radians(rake)

  if U_as_unit is True:
    U1 = 1 
    U2 = 1 
    U3 = 1 
    if verbose is True:
      print( "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\nRunning ->  forward_finite_rectangular_source_okada92_Uxyz\nINPUTS  ->  x = %.2f;  y = %.2f; z = %.2f; c = %.2f (z=-c); dip = %.1f; rake = (not considered, unit_U is True); L = %.2f; W = %.2f; U = (not considered, unit_U is True)" % (x, y, z, c, dip, L, W) )
  elif U_as_unit is False:
    U1 = U*np.cos(rake_rad)
    U2 = U*np.sin(rake_rad)
    U3 = 0 # != 0 para intrusiones de fluidos
    if verbose is True:
      print( "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\nRunning ->  forward_finite_rectangular_source_okada92_Uxyz\nINPUTS  ->  x = %.2f;  y = %.2f; z = %.2f; c = %.2f (z=-c); dip = %.1f; rake = %.1f; L = %.2f; W = %.2f; U = %.2f" % (x, y, z, c, dip, rake, L, W, U) )

  ###################################################################################
  ###### Desplazamientos Ux,Uy,Uz ######
  ###################################################################################

  # para calcular u1, u2, u3 para las partes A,B,C utilizando evaluando la representacion de chinnery
  def get_u123ABC_from_Chinnerys_representation(x, y, z, dip, c, L, W, U, mu, lmbd):
    d = c-z
    p = y*np.cos(dip_rad) + d*np.sin(dip_rad)
    q = y*np.sin(dip_rad) - d*np.cos(dip_rad)
    qq = np.array([q, q, q, q])    # b = 4

    # Notacion  de Chinnery -> f(e,eta)|| = f(x,p) - f(x,p-W) - f(x-L,p) + f(x-L,p-W)
    e = np.array([x, x, x-L, x-L])
    eta = np.array([p, p-W, p, p-W])
    alph = (lmbd+mu)/(lmbd+2*mu)
    R = np.sqrt(e**2 + eta**2 + qq**2)
    ytg = eta*np.cos(dip_rad) + qq*np.sin(dip_rad)
    dtg = eta*np.sin(dip_rad) - qq*np.cos(dip_rad)
    ctg = dtg + z

    ######
    Theta = np.arctan((e*eta)/(qq*R))
    X = np.sqrt(e**2+qq**2)
    X11 = 1/(R*(R+e))
    X32 = (2*R + e)/(R**3*(R+e)**2)
    X53 = (8*R**2 + 9*R*e + 3*e**2)/(R**3*(R+e)**2)
    Y11 = 1/(R*(R+eta))
    Y32 = (2*R + eta)/(R**3*(R+eta)**2)
    Y53 = (8*R**2 + 9*R*eta + 3*eta**2)/(R**5*(R+eta)**3)
    h = qq*np.cos(dip_rad) - z
    Z32 = np.sin(dip_rad)/R**3 - h*Y32
    Z53 = 3*np.sin(dip_rad)/R**5 - h*Y53
    Y0 = Y11 - e**2*Y32
    Z0 = Z32 - e**2*Z53

    if dip==90. or dip==270 or dip==-90.:# con cos(dip)=0 tenemos que sin(dip)=[+1,-1]
      I4 = 0.5*( (e*ytg)/(R+dtg)**2 )
      I3 =  0.5*(eta/(R+dtg) + (ytg*qq)/(R+dtg)**2 - np.log(R+eta) )
      I2 = np.log(R+dtg) + I3*np.sin(dip_rad)
      I1 = -(e/(R+dtg))*np.cos(dip_rad) - I4*np.sin(dip_rad)
    else:
      I4 = (np.sin(dip_rad)/np.cos(dip_rad)) * (e/(R+dtg)) + (2/np.cos(dip_rad)**2)*np.arctan( (eta*(X+qq*np.cos(dip_rad)) + X*(R+X)*np.sin(dip_rad))/(e*(R+X)*np.cos(dip_rad)) )
      I3 = (1./np.cos(dip_rad)) * (ytg/(R+dtg)) - (1/np.cos(dip_rad)**2)*(np.log(R+eta) - np.sin(dip_rad)*np.log(R+dtg) )
      I2 = np.log(R+dtg) + I3*np.sin(dip_rad)
      I1 = -(e/(R+dtg))*np.cos(dip_rad) - I4*np.sin(dip_rad)

    ######
    # strike-slip f123A
    f1A_ss = Theta/2. + (alph/2.)*e*qq*Y11
    f2A_ss = (alph/2.)*(qq/R)
    f3A_ss = ((1-alph)/2.)*np.log(R+eta) - (alph/2.)*qq**2*Y11
    # dip-slip f123A
    f1A_ds = (alph/2.)*(qq/R)
    f2A_ds = Theta/2. + (alph/2.)*eta*qq*X11
    f3A_ds = ((1-alph)/2.)*np.log(R+e) - (alph/2.)*qq**2*X11
    # tensile fault f123A
    f1A_tf = -((1-alph)/2.)*np.log(R+eta) - (alph/2.)*qq**2*Y11 
    f2A_tf = -((1-alph)/2.)*np.log(R+e) - (alph/2.)*qq**2*X11 
    f3A_tf = Theta/2. - (alph/2.)*qq*(eta*X11 + e*Y11)

    # dip-slip f123B
    f1B_ss = -e*qq*Y11 - Theta - ((1-alph)/alph)*I1*np.sin(dip_rad)
    f2B_ss = -qq/R + ((1-alph)/alph)*(ytg/(R+dtg))*np.sin(dip_rad)
    f3B_ss = qq**2*Y11 - ((1-alph)/alph)*I2*np.sin(dip_rad)
    # dip-slip f123B
    f1B_ds = -qq/R + ((1-alph)/alph)*I3*np.sin(dip_rad)*np.cos(dip_rad)
    f2B_ds = -eta*qq*X11 - Theta - ((1-alph)/alph)*(e/(R+dtg))*np.sin(dip_rad)*np.cos(dip_rad)
    f3B_ds = qq**2*X11 + ((1-alph)/alph)*I4*np.sin(dip_rad)*np.cos(dip_rad)
    # tensile fault f123B
    f1B_tf = qq**2*Y11 - ((1-alph)/alph)*I3*(np.sin(dip_rad))**2
    f2B_tf = qq**2*X11 + ((1-alph)/alph)*(e/(R+dtg))*(np.sin(dip_rad))**2
    f3B_tf = qq*(eta*X11 + e*Y11) - Theta - ((1-alph)/alph)*I4*(np.sin(dip_rad))**2

    # dip-slip f123C
    f1C_ss = (1-alph)*e*Y11*np.cos(dip_rad) - alph*e*qq*Z32
    f2C_ss = (1-alph)*(np.cos(dip_rad)/R + 2*qq*Y11*np.sin(dip_rad) )
    f3C_ss = (1-alph)*qq*Y11*np.cos(dip_rad) - alph*((ctg*eta)/(R**3) - z*Y11 + e**2*Z32 )
    # dip-slip f123C
    f1C_ds = (1-alph)*(np.cos(dip_rad)/R) - qq*Y11*np.sin(dip) - alph*(ctg*qq)/R**3
    f2C_ds = (1-alph)*ytg*X11 - alph*ctg*eta*qq*X32
    f3C_ds = -dtg*X11 - e*Y11*np.sin(dip_rad) - alph*ctg*(X11 - qq**2*X32)
    # tensile fault f123C
    f1C_tf = -(1-alph)*(np.sin(dip_rad)/R + qq*Y11*np.cos(dip_rad)) - alph*(z*Y11 - qq**2*X32)
    f2C_tf = (1-alph)*2*e*Y11*np.sin(dip_rad) + dtg*X11 - alph*ctg*(X11 - qq**2*X32)
    f3C_tf = (1-alph)*(ytg*X11 + e*Y11*np.cos(dip_rad)) + alph*qq*(ctg*eta*X32 + e*Z32)


    ######
    # representacion Chinnery para strike-slip u123A
    u1A_ss = ( f1A_ss[0] - f1A_ss[1] - f1A_ss[2] + f1A_ss[3] )
    u2A_ss = ( f2A_ss[0] - f2A_ss[1] - f2A_ss[2] + f2A_ss[3] )
    u3A_ss = ( f3A_ss[0] - f3A_ss[1] - f3A_ss[2] + f3A_ss[3] )
    # representacion Chinnery para dip-slip u123A
    u1A_ds = ( f1A_ds[0] - f1A_ds[1] - f1A_ds[2] + f1A_ds[3] )
    u2A_ds = ( f2A_ds[0] - f2A_ds[1] - f2A_ds[2] + f2A_ds[3] )
    u3A_ds = ( f3A_ds[0] - f3A_ds[1] - f3A_ds[2] + f3A_ds[3] )
    # representacion Chinnery para dip-slip u123A
    u1A_tf = ( f1A_tf[0] - f1A_tf[1] - f1A_tf[2] + f1A_tf[3] )
    u2A_tf = ( f2A_tf[0] - f2A_tf[1] - f2A_tf[2] + f2A_tf[3] )
    u3A_tf = ( f3A_tf[0] - f3A_tf[1] - f3A_tf[2] + f3A_tf[3] )

    # representacion Chinnery para strike-slip u123B
    u1B_ss = ( f1B_ss[0] - f1B_ss[1] - f1B_ss[2] + f1B_ss[3] )
    u2B_ss = ( f2B_ss[0] - f2B_ss[1] - f2B_ss[2] + f2B_ss[3] )
    u3B_ss = ( f3B_ss[0] - f3B_ss[1] - f3B_ss[2] + f3B_ss[3] )
    # representacion Chinnery para dip-slip u123B
    u1B_ds = ( f1B_ds[0] - f1B_ds[1] - f1B_ds[2] + f1B_ds[3] )
    u2B_ds = ( f2B_ds[0] - f2B_ds[1] - f2B_ds[2] + f2B_ds[3] )
    u3B_ds = ( f3B_ds[0] - f3B_ds[1] - f3B_ds[2] + f3B_ds[3] )
    # representacion Chinnery para dip-slip u123B
    u1B_tf = ( f1B_tf[0] - f1B_tf[1] - f1B_tf[2] + f1B_tf[3] )
    u2B_tf = ( f2B_tf[0] - f2B_tf[1] - f2B_tf[2] + f2B_tf[3] )
    u3B_tf = ( f3B_tf[0] - f3B_tf[1] - f3B_tf[2] + f3B_tf[3] )

    # representacion Chinnery para strike-slip u123C
    u1C_ss = ( f1C_ss[0] - f1C_ss[1] - f1C_ss[2] + f1C_ss[3] )
    u2C_ss = ( f2C_ss[0] - f2C_ss[1] - f2C_ss[2] + f2C_ss[3] )
    u3C_ss = ( f3C_ss[0] - f3C_ss[1] - f3C_ss[2] + f3C_ss[3] )
    # representacion Chinnery para dip-slip u123C
    u1C_ds = ( f1C_ds[0] - f1C_ds[1] - f1C_ds[2] + f1C_ds[3] )
    u2C_ds = ( f2C_ds[0] - f2C_ds[1] - f2C_ds[2] + f2C_ds[3] )
    u3C_ds = ( f3C_ds[0] - f3C_ds[1] - f3C_ds[2] + f3C_ds[3] )
    # representacion Chinnery para dip-slip u123C
    u1C_tf = ( f1C_tf[0] - f1C_tf[1] - f1C_tf[2] + f1C_tf[3] )
    u2C_tf = ( f2C_tf[0] - f2C_tf[1] - f2C_tf[2] + f2C_tf[3] )
    u3C_tf = ( f3C_tf[0] - f3C_tf[1] - f3C_tf[2] + f3C_tf[3] )

    return u1A_ss, u2A_ss, u3A_ss, u1B_ss, u2B_ss, u3B_ss, u1C_ss, u2C_ss, u3C_ss, u1A_ds, u2A_ds, u3A_ds, u1B_ds, u2B_ds, u3B_ds, u1C_ds, u2C_ds, u3C_ds, u1A_tf, u2A_tf, u3A_tf, u1B_tf, u2B_tf, u3B_tf, u1C_tf, u2C_tf, u3C_tf


  ###
  u1A_ss1, u2A_ss1, u3A_ss1, u1B_ss, u2B_ss, u3B_ss, u1C_ss, u2C_ss, u3C_ss, u1A_ds1, u2A_ds1, u3A_ds1, u1B_ds, u2B_ds, u3B_ds, u1C_ds, u2C_ds, u3C_ds, u1A_tf1, u2A_tf1, u3A_tf1, u1B_tf, u2B_tf, u3B_tf, u1C_tf, u2C_tf, u3C_tf = get_u123ABC_from_Chinnerys_representation(x, y, -z, dip, c, L, W, U, mu, lmbd)
  u1A_ss, u2A_ss, u3A_ss, u1B_ss, u2B_ss, u3B_ss, u1C_ss, u2C_ss, u3C_ss, u1A_ds, u2A_ds, u3A_ds, u1B_ds, u2B_ds, u3B_ds, u1C_ds, u2C_ds, u3C_ds, u1A_tf, u2A_tf, u3A_tf, u1B_tf, u2B_tf, u3B_tf, u1C_tf, u2C_tf, u3C_tf = get_u123ABC_from_Chinnerys_representation(x, y, z, dip, c, L, W, U, mu, lmbd)

  # Uxyz for strike-slip
  Ux_ss = (U1/(2*np.pi)) * (  u1A_ss - u1A_ss1 + u1B_ss + z*u1C_ss ) 
  Uy_ss = (U1/(2*np.pi)) * ( (u2A_ss - u2A_ss1 + u2B_ss + z*u2C_ss)*np.cos(dip_rad) - (u3A_ss - u3A_ss1 + u3B_ss + z*u3C_ss)*np.sin(dip_rad) ) 
  Uz_ss = (U1/(2*np.pi)) * ( (u2A_ss - u2A_ss1 + u2B_ss - z*u2C_ss)*np.sin(dip_rad) + (u3A_ss - u3A_ss1 + u3B_ss - z*u3C_ss)*np.cos(dip_rad) ) 
  # Uxyz for dip-slip
  Ux_ds = (U2/(2*np.pi)) * (  u1A_ds - u1A_ds1 + u1B_ds + z*u1C_ds ) 
  Uy_ds = (U2/(2*np.pi)) * ( (u2A_ds - u2A_ds1 + u2B_ds + z*u2C_ds)*np.cos(dip_rad) - (u3A_ds - u3A_ds1 + u3B_ds + z*u3C_ds)*np.sin(dip_rad) ) 
  Uz_ds = (U2/(2*np.pi)) * ( (u2A_ds - u2A_ds1 + u2B_ds - z*u2C_ds)*np.sin(dip_rad) + (u3A_ds - u3A_ds1 + u3B_ds - z*u3C_ds)*np.cos(dip_rad) ) 
  # Uxyz for tensile fault
  Ux_tf = (U3/(2*np.pi)) * (  u1A_tf - u1A_tf1 + u1B_tf + z*u1C_tf ) 
  Uy_tf = (U3/(2*np.pi)) * ( (u2A_tf - u2A_tf1 + u2B_tf + z*u2C_tf)*np.cos(dip_rad) - (u3A_tf - u3A_tf1 + u3B_tf + z*u3C_tf)*np.sin(dip_rad) ) 
  Uz_tf = (U3/(2*np.pi)) * ( (u2A_tf - u2A_tf1 + u2B_tf - z*u2C_tf)*np.sin(dip_rad) + (u3A_tf - u3A_tf1 + u3B_tf - z*u3C_tf)*np.cos(dip_rad) ) 

  # soluciones
  Ux = Ux_ss + Ux_ds + Ux_tf
  Uy = Uy_ss + Uy_ds + Uy_tf
  Uz = Uz_ss + Uz_ds + Uz_tf



  ###################################################################################
  # print check-list of outputs
  if verbose is True:
    print( "OUTPUTS ->           Ux           Uy           Uz         " ) 
    print( "------------------------------------------------------------------------------------------------------------------------------------" )
    print( "Strike-slip   ->   %+.3e   %+.3e   %+.3e" % (Ux_ss, Uy_ss, Uz_ss) )
    print( "Dip-slip      ->   %+.3e   %+.3e   %+.3e" % (Ux_ds, Uy_ds, Uz_ds) )
    print( "Tensile fault ->   %+.3e   %+.3e   %+.3e" % (Ux_tf, Uy_tf, Uz_tf) )
    print( "------------------------------------------------------------------------------------------------------------------------------------")
    print( "                   %+.3e   %+.3e   %+.3e" % (Ux, Uy, Uz) )
    print( "------------------------------------------------------------------------------------------------------------------------------------")
    #print( "EOF\n\n" )

  return Ux, Uy, Uz






##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
def forward_finite_rectangular_source_okada92_dUdx(x=2, y=3, z=0, dip=70, rake=10, c=4, L=3, W=2, U=1, mu=1, lmbd=1, U_as_unit=True, verbose=True):
  dip_rad = np.radians(dip)
  rake_rad = np.radians(rake)

  if U_as_unit is True:
    U1 = 1 
    U2 = 1 
    U3 = 1 
    if verbose is True:
      print( "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\nRunning ->  forward_finite_rectangular_source_okada92_dUdx\nINPUTS  ->  x = %.2f;  y = %.2f; z = %.2f; c = %.2f (z=-c); dip = %.1f; rake = (not considered, unit_U is True); L = %.2f; W = %.2f; U = (not considered, unit_U is True)" % (x, y, z, c, dip, L, W) )
  elif U_as_unit is False:
    U1 = U*np.cos(rake_rad)
    U2 = U*np.sin(rake_rad)
    U3 = 0 # != 0 para intrusiones de fluidos
    if verbose is True:
      print( "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\nRunning ->  forward_finite_rectangular_source_okada92_dUdx\nINPUTS  ->  x = %.2f;  y = %.2f; z = %.2f; c = %.2f (z=-c); dip = %.1f; rake = %.1f; L = %.2f; W = %.2f; U = %.2f" % (x, y, z, c, dip, rake, L, W, U) )


  ###################################################################################
  ###### Derivadas dUx/dx, dUy/dx, dUz/dx ######
  ###################################################################################

  # para calcular df1dx, df2dx, df3dx para las partes A,B,C utilizando evaluando la representacion de chinnery
  def get_df123ABC_from_Chinnerys_representation(x, y, z, dip, c, L, W, mu, lmbd):
    d = c-z
    p = y*np.cos(dip_rad) + d*np.sin(dip_rad)
    q = y*np.sin(dip_rad) - d*np.cos(dip_rad)
    qq = np.array([q, q, q, q])    # b = 4

    # Notacion  de Chinnery -> f(e,eta)|| = f(x,p) - f(x,p-W) - f(x-L,p) + f(x-L,p-W)
    e = np.array([x, x, x-L, x-L])
    eta = np.array([p, p-W, p, p-W])
    alph = (lmbd+mu)/(lmbd+2*mu)
    R = np.sqrt(e**2 + eta**2 + qq**2)
    ytg = eta*np.cos(dip_rad) + qq*np.sin(dip_rad)
    dtg = eta*np.sin(dip_rad) - qq*np.cos(dip_rad)
    ctg = dtg + z

    ######
    X11 = 1/(R*(R+e))
    X32 = (2*R + e)/(R**3*(R+e)**2)
    X53 = (8*R**2 + 9*R*e + 3*e**2)/(R**3*(R+e)**2)
    Y11 = 1/(R*(R+eta))
    Y32 = (2*R + eta)/(R**3*(R+eta)**2)
    Y53 = (8*R**2 + 9*R*eta + 3*eta**2)/(R**5*(R+eta)**3)
    h = qq*np.cos(dip_rad) - z
    Z32 = np.sin(dip_rad)/R**3 - h*Y32
    Z53 = 3*np.sin(dip_rad)/R**5 - h*Y53
    Y0 = Y11 - e**2*Y32
    Z0 = Z32 - e**2*Z53
    D11 = 1/(R*(R+dtg))

    K1 = (e/np.cos(dip_rad))*(D11-Y11*np.sin(dip_rad))
    K3 = (1./np.cos(dip_rad)) * (qq*Y11 - ytg*D11)
    if dip==90. or dip==270 or dip==-90.:# con cos(dip)=0 tenemos que sin(dip)=[+1,-1]
      K1 = ( (e*qq)/(R+dtg) )*D11
      K3 = (np.sin(dip_rad)/(R+dtg)) * (e**2*D11 -1)

    K2 = 1./R + K3*np.sin(dip_rad)
    K4 = e*Y11*np.cos(dip_rad) - K1*np.sin(dip_rad)

    J2 = ((e*ytg)/(R+dtg))*D11
    J5 = -(dtg + (ytg**2)/(R+dtg))*D11
    J3 = (1/np.cos(dip_rad))*(K1 - J2*np.sin(dip_rad))
    J6 = (1./np.cos(dip_rad))*(K3 - J5*np.sin(dip_rad))    
    if dip==90. or dip==270 or dip==-90.:# con cos(dip)=0 tenemos que sin(dip)=[+1,-1]
      J3 = -(e/(R+dtg)**2)*(qq**2*D11 - 0.5)
      J6 = -(ytg/(R+dtg)**2)*(e**2*D11 - 0.5)

    J4 = -e*Y11 - J2*np.cos(dip_rad) + J3*np.sin(dip_rad)
    J1 = J5*np.cos(dip_rad) - J6*np.sin(dip_rad)

    ######
    # strike-slip j123A
    df1dxA_ss = -((1-alph)/2.)*qq*Y11 - (alph/2.)*e**2*qq*Y32
    df2dxA_ss = -(alph/2.)*e*qq/R**3
    df3dxA_ss = ((1-alph)/2.)*e*Y11 + (alph/2.)*e*q**2*Y32
    # dip-slip j123A
    df1dxA_ds = -(alph/2.)*(e*qq)/R**3
    df2dxA_ds = -(qq/2.)*Y11 - (alph/2.)*(eta*qq)/R**3
    df3dxA_ds = ((1-alph)/2.)/R + (alph/2.)*(qq**2/R**3)
    # tensile fault j123A
    df1dxA_tf = -((1-alph)/2.)*e*Y11 + (alph/2.)*e*q**2*Y32
    df2dxA_tf = -((1-alph)/2.)/R + (alph/2.)*(qq**2/R**3)
    df3dxA_tf = -((1-alph)/2.)*qq*Y11 - (alph/2.)*q**3*Y32

    # strike-slip j123B
    df1dxB_ss = e**2*qq*Y32 - ((1-alph)/alph)*J1*np.sin(dip_rad)
    df2dxB_ss = (e*qq)/R**3 - ((1-alph)/alph)*J2*np.sin(dip_rad)
    df3dxB_ss = -e*qq**2*Y32 - ((1-alph)/alph)*J3*np.sin(dip_rad)
    # dip-slip j123B
    df1dxB_ds = (e*qq)/R**3 + ((1-alph)/alph)*J4*np.sin(dip_rad)*np.cos(dip_rad)
    df2dxB_ds = (eta*qq)/R**3 + qq*Y11 + ((1-alph)/alph)*J5*np.sin(dip_rad)*np.cos(dip_rad)
    df3dxB_ds = -qq**2/R**3 + ((1-alph)/alph)*J6*np.sin(dip_rad)*np.cos(dip_rad)
    # tensile fault j123B
    df1dxB_tf = -e*qq**2*Y32 - ((1-alph)/alph)*J4*(np.sin(dip_rad))**2
    df2dxB_tf = -qq**2/R**3 - ((1-alph)/alph)*J5*(np.sin(dip_rad))**2
    df3dxB_tf = qq**3*Y32 - ((1-alph)/alph)*J6*(np.sin(dip_rad))**2

    # strike-slip j123C
    df1dxC_ss = (1-alph)*Y0*np.cos(dip_rad) - alph*qq*Z0
    df2dxC_ss = -(1-alph)*e*( np.cos(dip_rad)/R**3 + 2*qq*Y32*np.sin(dip_rad) ) + alph*(3*ctg*e*qq/R**5)
    df3dxC_ss = -(1-alph)*e*qq*Y32*np.cos(dip_rad) + alph*e*( (3*ctg*eta)/R**5 - z*Y32 - Z32 - Z0)
    # dip-slip j123C
    df1dxC_ds = -(1-alph)*(e/R**3)*np.cos(dip_rad) + e*qq*Y32*np.sin(dip_rad) + alph*(3*ctg*e*qq)/R**5
    df2dxC_ds = -(1-alph)*(ytg/R**3) + alph*(3*ctg*eta*qq)/R**5
    df3dxC_ds = (dtg/R**3) - Y0*np.sin(dip_rad) + alph*(ctg/R**3)*(1-(3*qq**2)/R**2)
    # tensile fault j123C
    df1dxC_tf = (1-alph)*(e/R**3)*np.sin(dip_rad) + e*qq*Y32*np.cos(dip_rad) + alph*e*( (3*ctg*eta)/R**5 - 2*Z32 - Z0)
    df2dxC_tf = (1-alph)*2*Y0*np.sin(dip_rad) - dtg/R**3 + alph*(ctg/R**3)*(1-(3*qq**2)/R**2)
    df3dxC_tf = -(1-alph)*(ytg/R**3 - Y0*np.cos(dip_rad)) - alph*( (3*ctg*eta*qq)/R**5 - qq*Z0 )


    ######
    # representacion Chinnery para strike-slip u123A
    j1A_ss = ( df1dxA_ss[0] - df1dxA_ss[1] - df1dxA_ss[2] + df1dxA_ss[3] )
    j2A_ss = ( df2dxA_ss[0] - df2dxA_ss[1] - df2dxA_ss[2] + df2dxA_ss[3] )
    j3A_ss = ( df3dxA_ss[0] - df3dxA_ss[1] - df3dxA_ss[2] + df3dxA_ss[3] )
    # representacion Chinnery para dip-slip u123A
    j1A_ds = ( df1dxA_ds[0] - df1dxA_ds[1] - df1dxA_ds[2] + df1dxA_ds[3] )
    j2A_ds = ( df2dxA_ds[0] - df2dxA_ds[1] - df2dxA_ds[2] + df2dxA_ds[3] )
    j3A_ds = ( df3dxA_ds[0] - df3dxA_ds[1] - df3dxA_ds[2] + df3dxA_ds[3] )
    # representacion Chinnery para dip-slip u123A
    j1A_tf = ( df1dxA_tf[0] - df1dxA_tf[1] - df1dxA_tf[2] + df1dxA_tf[3] )
    j2A_tf = ( df2dxA_tf[0] - df2dxA_tf[1] - df2dxA_tf[2] + df2dxA_tf[3] )
    j3A_tf = ( df3dxA_tf[0] - df3dxA_tf[1] - df3dxA_tf[2] + df3dxA_tf[3] )

    # representacion Chinnery para strike-slip u123B
    j1B_ss = ( df1dxB_ss[0] - df1dxB_ss[1] - df1dxB_ss[2] + df1dxB_ss[3] )
    j2B_ss = ( df2dxB_ss[0] - df2dxB_ss[1] - df2dxB_ss[2] + df2dxB_ss[3] )
    j3B_ss = ( df3dxB_ss[0] - df3dxB_ss[1] - df3dxB_ss[2] + df3dxB_ss[3] )
    # representacion Chinnery para dip-slip u123B
    j1B_ds = ( df1dxB_ds[0] - df1dxB_ds[1] - df1dxB_ds[2] + df1dxB_ds[3] )
    j2B_ds = ( df2dxB_ds[0] - df2dxB_ds[1] - df2dxB_ds[2] + df2dxB_ds[3] )
    j3B_ds = ( df3dxB_ds[0] - df3dxB_ds[1] - df3dxB_ds[2] + df3dxB_ds[3] )
    # representacion Chinnery para dip-slip u123B
    j1B_tf = ( df1dxB_tf[0] - df1dxB_tf[1] - df1dxB_tf[2] + df1dxB_tf[3] )
    j2B_tf = ( df2dxB_tf[0] - df2dxB_tf[1] - df2dxB_tf[2] + df2dxB_tf[3] )
    j3B_tf = ( df3dxB_tf[0] - df3dxB_tf[1] - df3dxB_tf[2] + df3dxB_tf[3] )

    # representacion Chinnery para strike-slip u123C
    j1C_ss = ( df1dxC_ss[0] - df1dxC_ss[1] - df1dxC_ss[2] + df1dxC_ss[3] )
    j2C_ss = ( df2dxC_ss[0] - df2dxC_ss[1] - df2dxC_ss[2] + df2dxC_ss[3] )
    j3C_ss = ( df3dxC_ss[0] - df3dxC_ss[1] - df3dxC_ss[2] + df3dxC_ss[3] )
    # representacion Chinnery para dip-slip u123C
    j1C_ds = ( df1dxC_ds[0] - df1dxC_ds[1] - df1dxC_ds[2] + df1dxC_ds[3] )
    j2C_ds = ( df2dxC_ds[0] - df2dxC_ds[1] - df2dxC_ds[2] + df2dxC_ds[3] )
    j3C_ds = ( df3dxC_ds[0] - df3dxC_ds[1] - df3dxC_ds[2] + df3dxC_ds[3] )
    # representacion Chinnery para dip-slip u123C
    j1C_tf = ( df1dxC_tf[0] - df1dxC_tf[1] - df1dxC_tf[2] + df1dxC_tf[3] )
    j2C_tf = ( df2dxC_tf[0] - df2dxC_tf[1] - df2dxC_tf[2] + df2dxC_tf[3] )
    j3C_tf = ( df3dxC_tf[0] - df3dxC_tf[1] - df3dxC_tf[2] + df3dxC_tf[3] )

    return j1A_ss, j2A_ss, j3A_ss, j1B_ss, j2B_ss, j3B_ss, j1C_ss, j2C_ss, j3C_ss, j1A_ds, j2A_ds, j3A_ds, j1B_ds, j2B_ds, j3B_ds, j1C_ds, j2C_ds, j3C_ds, j1A_tf, j2A_tf, j3A_tf, j1B_tf, j2B_tf, j3B_tf, j1C_tf, j2C_tf, j3C_tf


  ######
  j1A_ss1, j2A_ss1, j3A_ss1, j1B_ss, j2B_ss, j3B_ss, j1C_ss, j2C_ss, j3C_ss, j1A_ds1, j2A_ds1, j3A_ds1, j1B_ds, j2B_ds, j3B_ds, j1C_ds, j2C_ds, j3C_ds, j1A_tf1, j2A_tf1, j3A_tf1, j1B_tf, j2B_tf, j3B_tf, j1C_tf, j2C_tf, j3C_tf = get_df123ABC_from_Chinnerys_representation(x, y, -z, dip, c, L, W, mu, lmbd)
  j1A_ss, j2A_ss, j3A_ss, j1B_ss, j2B_ss, j3B_ss, j1C_ss, j2C_ss, j3C_ss, j1A_ds, j2A_ds, j3A_ds, j1B_ds, j2B_ds, j3B_ds, j1C_ds, j2C_ds, j3C_ds, j1A_tf, j2A_tf, j3A_tf, j1B_tf, j2B_tf, j3B_tf, j1C_tf, j2C_tf, j3C_tf = get_df123ABC_from_Chinnerys_representation(x, y, z, dip, c, L, W, mu, lmbd)

  # dUxyzdx for strike-slip
  dUxdx_ss = (U1/(2*np.pi)) * (  j1A_ss - j1A_ss1 + j1B_ss + z*j1C_ss ) 
  dUydx_ss = (U1/(2*np.pi)) * ( (j2A_ss - j2A_ss1 + j2B_ss + z*j2C_ss)*np.cos(dip_rad) - (j3A_ss - j3A_ss1 + j3B_ss + z*j3C_ss)*np.sin(dip_rad) ) 
  dUzdx_ss = (U1/(2*np.pi)) * ( (j2A_ss - j2A_ss1 + j2B_ss - z*j2C_ss)*np.sin(dip_rad) + (j3A_ss - j3A_ss1 + j3B_ss - z*j3C_ss)*np.cos(dip_rad) ) 
  # dUxyzdx for dip-slip
  dUxdx_ds = (U2/(2*np.pi)) * (  j1A_ds - j1A_ds1 + j1B_ds + z*j1C_ds ) 
  dUydx_ds = (U2/(2*np.pi)) * ( (j2A_ds - j2A_ds1 + j2B_ds + z*j2C_ds)*np.cos(dip_rad) - (j3A_ds - j3A_ds1 + j3B_ds + z*j3C_ds)*np.sin(dip_rad) ) 
  dUzdx_ds = (U2/(2*np.pi)) * ( (j2A_ds - j2A_ds1 + j2B_ds - z*j2C_ds)*np.sin(dip_rad) + (j3A_ds - j3A_ds1 + j3B_ds - z*j3C_ds)*np.cos(dip_rad) ) 
  # dUxyzdx for tensile fault
  dUxdx_tf = (U3/(2*np.pi)) * (  j1A_tf - j1A_tf1 + j1B_tf + z*j1C_tf ) 
  dUydx_tf = (U3/(2*np.pi)) * ( (j2A_tf - j2A_tf1 + j2B_tf + z*j2C_tf)*np.cos(dip_rad) - (j3A_tf - j3A_tf1 + j3B_tf + z*j3C_tf)*np.sin(dip_rad) ) 
  dUzdx_tf = (U3/(2*np.pi)) * ( (j2A_tf - j2A_tf1 + j2B_tf - z*j2C_tf)*np.sin(dip_rad) + (j3A_tf - j3A_tf1 + j3B_tf - z*j3C_tf)*np.cos(dip_rad) ) 

  # soluciones de deformacion dUxyzdx
  dUxdx = dUxdx_ss + dUxdx_ds + dUxdx_tf
  dUydx = dUydx_ss + dUydx_ds + dUydx_tf
  dUzdx = dUzdx_ss + dUzdx_ds + dUzdx_tf


  ###################################################################################
  # print check-list of outputs
  if verbose is True:
    print( "OUTPUTS ->          dUx/dx       dUy/dx       dUz/dx" ) 
    print( "------------------------------------------------------------------------------------------------------------------------------------" )
    print( "Strike-slip   ->  %+.3e   %+.3e   %+.3e   " % (dUxdx_ss, dUydx_ss, dUzdx_ss) )
    print( "Dip-slip      ->  %+.3e   %+.3e   %+.3e   " % (dUxdx_ds, dUydx_ds, dUzdx_ds) )
    print( "Tensile fault ->  %+.3e   %+.3e   %+.3e   " % (dUxdx_tf, dUydx_tf, dUzdx_tf) )
    print( "------------------------------------------------------------------------------------------------------------------------------------")
    print( "                  %+.3e   %+.3e   %+.3e   " % (dUxdx, dUydx, dUzdx) )
    print( "------------------------------------------------------------------------------------------------------------------------------------")
    #print( "EOF\n\n" )

  return dUxdx, dUydx, dUzdx






##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
def forward_finite_rectangular_source_okada92_dUdy(x=2, y=3, z=0, dip=70, rake=10, c=4, L=3, W=2, U=1, mu=1, lmbd=1, U_as_unit=True, verbose=True):
  dip_rad = np.radians(dip)
  rake_rad = np.radians(rake)

  if U_as_unit is True:
    U1 = 1 
    U2 = 1 
    U3 = 1 
    if verbose is True:
      print( "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\nRunning ->  forward_finite_rectangular_source_okada92_dUdy\nINPUTS  ->  x = %.2f;  y = %.2f; z = %.2f; c = %.2f (z=-c); dip = %.1f; rake = (not considered, unit_U is True); L = %.2f; W = %.2f; U = (not considered, unit_U is True)" % (x, y, z, c, dip, L, W) )
  elif U_as_unit is False:
    U1 = U*np.cos(rake_rad)
    U2 = U*np.sin(rake_rad)
    U3 = 0 # != 0 para intrusiones de fluidos
    if verbose is True:
      print( "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\nRunning ->  forward_finite_rectangular_source_okada92_dUdy\nINPUTS  ->  x = %.2f;  y = %.2f; z = %.2f; c = %.2f (z=-c); dip = %.1f; rake = %.1f; L = %.2f; W = %.2f; U = %.2f" % (x, y, z, c, dip, rake, L, W, U) )


  ###################################################################################
  ###### Derivadas dUx/dy, dUy/dy, dUz/dy ######
  ###################################################################################

  # para calcular df1dy, df2dy, df3dy para las partes A,B,C utilizando evaluando la representacion de chinnery
  def get_df123ABC_from_Chinnerys_representation(x, y, z, dip, c, L, W, mu, lmbd):
    d = c-z
    p = y*np.cos(dip_rad) + d*np.sin(dip_rad)
    q = y*np.sin(dip_rad) - d*np.cos(dip_rad)
    qq = np.array([q, q, q, q])    # b = 4

    # Notacion  de Chinnery -> f(e,eta)|| = f(x,p) - f(x,p-W) - f(x-L,p) + f(x-L,p-W)
    e = np.array([x, x, x-L, x-L])
    eta = np.array([p, p-W, p, p-W])
    alph = (lmbd+mu)/(lmbd+2*mu)
    R = np.sqrt(e**2 + eta**2 + qq**2)
    ytg = eta*np.cos(dip_rad) + qq*np.sin(dip_rad)
    dtg = eta*np.sin(dip_rad) - qq*np.cos(dip_rad)
    ctg = dtg + z

    ######
    X11 = 1/(R*(R+e))
    X32 = (2*R + e)/(R**3*(R+e)**2)
    X53 = (8*R**2 + 9*R*e + 3*e**2)/(R**3*(R+e)**2)
    Y11 = 1/(R*(R+eta))
    Y32 = (2*R + eta)/(R**3*(R+eta)**2)
    Y53 = (8*R**2 + 9*R*eta + 3*eta**2)/(R**5*(R+eta)**3)
    h = qq*np.cos(dip_rad) - z
    Z32 = np.sin(dip_rad)/R**3 - h*Y32
    Z53 = 3*np.sin(dip_rad)/R**5 - h*Y53
    Y0 = Y11 - e**2*Y32
    Z0 = Z32 - e**2*Z53
    D11 = 1/(R*(R+dtg))

    K1 = (e/np.cos(dip_rad))*(D11-Y11*np.sin(dip_rad))
    K3 = (1./np.cos(dip_rad)) * (qq*Y11 - ytg*D11)
    if dip==90. or dip==270 or dip==-90.:# con cos(dip)=0 tenemos que sin(dip)=[+1,-1]
      K1 = ( (e*qq)/(R+dtg) )*D11
      K3 = (np.sin(dip_rad)/(R+dtg)) * (e**2*D11 -1)

    K2 = 1./R + K3*np.sin(dip_rad)
    K4 = e*Y11*np.cos(dip_rad) - K1*np.sin(dip_rad)

    J2 = ((e*ytg)/(R+dtg))*D11
    J5 = -(dtg + (ytg**2)/(R+dtg))*D11
    J3 = (1/np.cos(dip_rad))*(K1 - J2*np.sin(dip_rad))
    J6 = (1./np.cos(dip_rad))*(K3 - J5*np.sin(dip_rad))    
    if dip==90. or dip==270 or dip==-90.:# con cos(dip)=0 tenemos que sin(dip)=[+1,-1]
      J3 = -(e/(R+dtg)**2)*(qq**2*D11 - 0.5)
      J6 = -(ytg/(R+dtg)**2)*(e**2*D11 - 0.5)

    J4 = -e*Y11 - J2*np.cos(dip_rad) + J3*np.sin(dip_rad)
    J1 = J5*np.cos(dip_rad) - J6*np.sin(dip_rad)

    E = np.sin(dip_rad)/R - (ytg*qq)/R**3
    F = dtg/R**3 + e**2*Y32*np.sin(dip_rad)
    G = 2*X11*np.sin(dip_rad) - ytg*qq*X32
    H = dtg*qq*X32 + e*qq*Y32*np.sin(dip_rad)
    P = np.cos(dip_rad)/R**3 + qq*Y32*np.sin(dip_rad)
    Q = (3*ctg*dtg)/R**5 - (z*Y32 + Z32 + Z0)*np.sin(dip_rad)




    ######
    # strike-slip j123A
    df1dyA_ss = ((1-alph)/2.)*e*Y11*np.sin(dip_rad) + (dtg/2.)*X11 - (alph/2.)*e*F
    df2dyA_ss = (alph/2.)*E
    df3dyA_ss = ((1-alph)/2.)*(np.cos(dip_rad)/R + qq*Y11*np.sin(dip_rad)) - (alph/2.)*qq*F
    # dip-slip j123A
    df1dyA_ds = (alph/2.)*E
    df2dyA_ds = ((1-alph)/2.)*dtg*X11 + (e/2.)*Y11*np.sin(dip_rad) + (alph/2.)*eta*G
    df3dyA_ds = ((1-alph)/2.)*ytg*X11 - (alph/2.)*qq*G
    # tensile fault j123A
    df1dyA_tf = -((1-alph)/2.)*(np.cos(dip_rad)/R + qq*Y11*np.sin(dip_rad)) - (alph/2.)*qq*F
    df2dyA_tf = -((1-alph)/2.)*ytg*X11 - (alph/2.)*qq*G
    df3dyA_tf = ((1-alph)/2.)*(dtg*X11 + e*Y11*np.sin(dip_rad)) + (alph/2.)*qq*H

    # strike-slip j123B
    df1dyB_ss = -e*F - dtg*X11 + ((1-alph)/alph)*( e*Y11 + J4 )*np.sin(dip_rad)
    df2dyB_ss = -E + ((1-alph)/alph)*( 1./R + J5 )*np.sin(dip_rad)
    df3dyB_ss = qq*F - ((1-alph)/alph)*( qq*Y11 - J6 )*np.sin(dip_rad)
    # dip-slip j123B
    df1dyB_ds = -E + ((1-alph)/alph)*J1*np.sin(dip_rad)*np.cos(dip_rad)
    df2dyB_ds = -eta*G-e*Y11*np.sin(dip_rad) + ((1-alph)/alph)*J2*np.sin(dip_rad)*np.cos(dip_rad)
    df3dyB_ds = qq*G + ((1-alph)/alph)*J3*np.sin(dip_rad)*np.cos(dip_rad)
    # tensile fault j123B
    df1dyB_tf = qq*F - ((1-alph)/alph)*J1*(np.sin(dip_rad))**2
    df2dyB_tf = qq*G - ((1-alph)/alph)*J2*(np.sin(dip_rad))**2
    df3dyB_tf = -qq*H - ((1-alph)/alph)*J3*(np.sin(dip_rad))**2

    # strike-slip j123C
    df1dyC_ss = -(1-alph)*e*P*np.cos(dip_rad) - alph*e*Q
    df2dyC_ss = 2*(1-alph)*(dtg/R**3 - Y0*np.sin(dip_rad))*np.sin(dip_rad) - (ytg/R**3)*np.cos(dip_rad) - alph*(((ctg+dtg)/R**3 )*np.sin(dip_rad) - eta/R**3 - (3*ctg*ytg*qq)/R**5 )
    df3dyC_ss = -(1-alph)*qq/R**3 + (ytg/R**3 - Y0*np.cos(dip_rad))*np.sin(dip_rad) + alph*( ((ctg+dtg)/R**3 )*np.cos(dip_rad) + (3*ctg*dtg*qq)/R**5 - (Y0*np.cos(dip_rad) + qq*Z0)*np.sin(dip_rad) )
    # dip-slip j123C
    df1dyC_ds = -(1-alph)*eta/R**3 + Y0*(np.sin(dip_rad))**2 - alph*( ((ctg+dtg)/R**3 )*np.sin(dip_rad) - (3*ctg*ytg*qq)/R**5 )
    df2dyC_ds = (1-alph)*(X11 - ytg**2*X32) - alph*ctg*( (dtg + 2*qq*np.cos(dip_rad))*X32 - ytg*eta*qq*X53 )
    df3dyC_ds = e*P*np.sin(dip_rad) + ytg*dtg*X32 + alph*ctg*( (ytg + 2*qq*np.sin(dip_rad))*X32 - ytg*qq**2*X53  )
    # tensile fault j123C
    df1dyC_tf = (1-alph)*(e/R**3 + Y0*np.sin(dip_rad)*np.cos(dip_rad)) + alph*( (z/R**3)*np.cos(dip_rad) + (3*ctg*dtg*qq)/R**5 - qq*Z0*np.sin(dip_rad) )
    df2dyC_tf = -(1-alph)*2*e*P*np.sin(dip_rad) - ytg*dtg*X32 + alph*ctg*( (ytg + 2*qq*np.sin(dip_rad))*X32 - ytg*qq**2*X53 ) 
    df3dyC_tf = -(1-alph)*(e*P*np.cos(dip_rad) - X11 + ytg**2*X32) + alph*ctg*( (dtg + 2*qq*np.cos(dip_rad))*X32 - ytg*eta*qq*X53 ) + alph*e*Q


    ######
    # representacion Chinnery para strike-slip u123A
    k1A_ss = ( df1dyA_ss[0] - df1dyA_ss[1] - df1dyA_ss[2] + df1dyA_ss[3] )
    k2A_ss = ( df2dyA_ss[0] - df2dyA_ss[1] - df2dyA_ss[2] + df2dyA_ss[3] )
    k3A_ss = ( df3dyA_ss[0] - df3dyA_ss[1] - df3dyA_ss[2] + df3dyA_ss[3] )
    # representacion Chinnery para dip-slip u123A
    k1A_ds = ( df1dyA_ds[0] - df1dyA_ds[1] - df1dyA_ds[2] + df1dyA_ds[3] )
    k2A_ds = ( df2dyA_ds[0] - df2dyA_ds[1] - df2dyA_ds[2] + df2dyA_ds[3] )
    k3A_ds = ( df3dyA_ds[0] - df3dyA_ds[1] - df3dyA_ds[2] + df3dyA_ds[3] )
    # representacion Chinnery para dip-slip u123A
    k1A_tf = ( df1dyA_tf[0] - df1dyA_tf[1] - df1dyA_tf[2] + df1dyA_tf[3] )
    k2A_tf = ( df2dyA_tf[0] - df2dyA_tf[1] - df2dyA_tf[2] + df2dyA_tf[3] )
    k3A_tf = ( df3dyA_tf[0] - df3dyA_tf[1] - df3dyA_tf[2] + df3dyA_tf[3] )

    # representacion Chinnery para strike-slip u123B
    k1B_ss = ( df1dyB_ss[0] - df1dyB_ss[1] - df1dyB_ss[2] + df1dyB_ss[3] )
    k2B_ss = ( df2dyB_ss[0] - df2dyB_ss[1] - df2dyB_ss[2] + df2dyB_ss[3] )
    k3B_ss = ( df3dyB_ss[0] - df3dyB_ss[1] - df3dyB_ss[2] + df3dyB_ss[3] )
    # representacion Chinnery para dip-slip u123B
    k1B_ds = ( df1dyB_ds[0] - df1dyB_ds[1] - df1dyB_ds[2] + df1dyB_ds[3] )
    k2B_ds = ( df2dyB_ds[0] - df2dyB_ds[1] - df2dyB_ds[2] + df2dyB_ds[3] )
    k3B_ds = ( df3dyB_ds[0] - df3dyB_ds[1] - df3dyB_ds[2] + df3dyB_ds[3] )
    # representacion Chinnery para dip-slip u123B
    k1B_tf = ( df1dyB_tf[0] - df1dyB_tf[1] - df1dyB_tf[2] + df1dyB_tf[3] )
    k2B_tf = ( df2dyB_tf[0] - df2dyB_tf[1] - df2dyB_tf[2] + df2dyB_tf[3] )
    k3B_tf = ( df3dyB_tf[0] - df3dyB_tf[1] - df3dyB_tf[2] + df3dyB_tf[3] )

    # representacion Chinnery para strike-slip u123C
    k1C_ss = ( df1dyC_ss[0] - df1dyC_ss[1] - df1dyC_ss[2] + df1dyC_ss[3] )
    k2C_ss = ( df2dyC_ss[0] - df2dyC_ss[1] - df2dyC_ss[2] + df2dyC_ss[3] )
    k3C_ss = ( df3dyC_ss[0] - df3dyC_ss[1] - df3dyC_ss[2] + df3dyC_ss[3] )
    # representacion Chinnery para dip-slip u123C
    k1C_ds = ( df1dyC_ds[0] - df1dyC_ds[1] - df1dyC_ds[2] + df1dyC_ds[3] )
    k2C_ds = ( df2dyC_ds[0] - df2dyC_ds[1] - df2dyC_ds[2] + df2dyC_ds[3] )
    k3C_ds = ( df3dyC_ds[0] - df3dyC_ds[1] - df3dyC_ds[2] + df3dyC_ds[3] )
    # representacion Chinnery para dip-slip u123C
    k1C_tf = ( df1dyC_tf[0] - df1dyC_tf[1] - df1dyC_tf[2] + df1dyC_tf[3] )
    k2C_tf = ( df2dyC_tf[0] - df2dyC_tf[1] - df2dyC_tf[2] + df2dyC_tf[3] )
    k3C_tf = ( df3dyC_tf[0] - df3dyC_tf[1] - df3dyC_tf[2] + df3dyC_tf[3] )

    return k1A_ss, k2A_ss, k3A_ss, k1B_ss, k2B_ss, k3B_ss, k1C_ss, k2C_ss, k3C_ss, k1A_ds, k2A_ds, k3A_ds, k1B_ds, k2B_ds, k3B_ds, k1C_ds, k2C_ds, k3C_ds, k1A_tf, k2A_tf, k3A_tf, k1B_tf, k2B_tf, k3B_tf, k1C_tf, k2C_tf, k3C_tf


  ######
  k1A_ss1, k2A_ss1, k3A_ss1, k1B_ss, k2B_ss, k3B_ss, k1C_ss, k2C_ss, k3C_ss, k1A_ds1, k2A_ds1, k3A_ds1, k1B_ds, k2B_ds, k3B_ds, k1C_ds, k2C_ds, k3C_ds, k1A_tf1, k2A_tf1, k3A_tf1, k1B_tf, k2B_tf, k3B_tf, k1C_tf, k2C_tf, k3C_tf = get_df123ABC_from_Chinnerys_representation(x, y, -z, dip, c, L, W, mu, lmbd)
  k1A_ss, k2A_ss, k3A_ss, k1B_ss, k2B_ss, k3B_ss, k1C_ss, k2C_ss, k3C_ss, k1A_ds, k2A_ds, k3A_ds, k1B_ds, k2B_ds, k3B_ds, k1C_ds, k2C_ds, k3C_ds, k1A_tf, k2A_tf, k3A_tf, k1B_tf, k2B_tf, k3B_tf, k1C_tf, k2C_tf, k3C_tf = get_df123ABC_from_Chinnerys_representation(x, y, z, dip, c, L, W, mu, lmbd)

  # dUxyzdy for strike-slip
  dUxdy_ss = (U1/(2*np.pi)) * (  k1A_ss - k1A_ss1 + k1B_ss + z*k1C_ss ) 
  dUydy_ss = (U1/(2*np.pi)) * ( (k2A_ss - k2A_ss1 + k2B_ss + z*k2C_ss)*np.cos(dip_rad) - (k3A_ss - k3A_ss1 + k3B_ss + z*k3C_ss)*np.sin(dip_rad) ) 
  dUzdy_ss = (U1/(2*np.pi)) * ( (k2A_ss - k2A_ss1 + k2B_ss - z*k2C_ss)*np.sin(dip_rad) + (k3A_ss - k3A_ss1 + k3B_ss - z*k3C_ss)*np.cos(dip_rad) ) 
  # dUxyzdy for dip-slip
  dUxdy_ds = (U2/(2*np.pi)) * (  k1A_ds - k1A_ds1 + k1B_ds + z*k1C_ds ) 
  dUydy_ds = (U2/(2*np.pi)) * ( (k2A_ds - k2A_ds1 + k2B_ds + z*k2C_ds)*np.cos(dip_rad) - (k3A_ds - k3A_ds1 + k3B_ds + z*k3C_ds)*np.sin(dip_rad) ) 
  dUzdy_ds = (U2/(2*np.pi)) * ( (k2A_ds - k2A_ds1 + k2B_ds - z*k2C_ds)*np.sin(dip_rad) + (k3A_ds - k3A_ds1 + k3B_ds - z*k3C_ds)*np.cos(dip_rad) ) 
  # dUxyzdy for tensile fault
  dUxdy_tf = (U3/(2*np.pi)) * (  k1A_tf - k1A_tf1 + k1B_tf + z*k1C_tf ) 
  dUydy_tf = (U3/(2*np.pi)) * ( (k2A_tf - k2A_tf1 + k2B_tf + z*k2C_tf)*np.cos(dip_rad) - (k3A_tf - k3A_tf1 + k3B_tf + z*k3C_tf)*np.sin(dip_rad) ) 
  dUzdy_tf = (U3/(2*np.pi)) * ( (k2A_tf - k2A_tf1 + k2B_tf - z*k2C_tf)*np.sin(dip_rad) + (k3A_tf - k3A_tf1 + k3B_tf - z*k3C_tf)*np.cos(dip_rad) ) 

  # soluciones de deformacion dUxyzdx
  dUxdy = dUxdy_ss + dUxdy_ds + dUxdy_tf
  dUydy = dUydy_ss + dUydy_ds + dUydy_tf
  dUzdy = dUzdy_ss + dUzdy_ds + dUzdy_tf


  ###################################################################################
  # print check-list of outputs
  if verbose is True:
    print( "OUTPUTS ->          dUx/dy       dUy/dy       dUz/dy" ) 
    print( "------------------------------------------------------------------------------------------------------------------------------------" )
    print( "Strike-slip   ->  %+.3e   %+.3e   %+.3e   " % (dUxdy_ss, dUydy_ss, dUzdy_ss) )
    print( "Dip-slip      ->  %+.3e   %+.3e   %+.3e   " % (dUxdy_ds, dUydy_ds, dUzdy_ds) )
    print( "Tensile fault ->  %+.3e   %+.3e   %+.3e   " % (dUxdy_tf, dUydy_tf, dUzdy_tf) )
    print( "------------------------------------------------------------------------------------------------------------------------------------")
    print( "                  %+.3e   %+.3e   %+.3e   " % (dUxdy, dUydy, dUzdy) )
    print( "------------------------------------------------------------------------------------------------------------------------------------")
    #print( "EOF\n\n" )

  return dUxdy, dUydy, dUzdy





##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
def forward_finite_rectangular_source_okada92_dUdz(x=2, y=3, z=0, dip=70, rake=10, c=4, L=3, W=2, U=1, mu=1, lmbd=1, U_as_unit=True, verbose=True):
  dip_rad = np.radians(dip)
  rake_rad = np.radians(rake)

  if U_as_unit is True:
    U1 = 1 
    U2 = 1 
    U3 = 1 
    if verbose is True:
      print( "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\nRunning ->  forward_finite_rectangular_source_okada92_dUdz\nINPUTS  ->  x = %.2f;  y = %.2f; z = %.2f; c = %.2f (z=-c); dip = %.1f; rake = (not considered, unit_U is True); L = %.2f; W = %.2f; U = (not considered, unit_U is True)" % (x, y, z, c, dip, L, W) )
  elif U_as_unit is False:
    U1 = U*np.cos(rake_rad)
    U2 = U*np.sin(rake_rad)
    U3 = 0 # != 0 para intrusiones de fluidos
    if verbose is True:
      print( "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\nRunning ->  forward_finite_rectangular_source_okada92_dUdz\nINPUTS  ->  x = %.2f;  y = %.2f; z = %.2f; c = %.2f (z=-c); dip = %.1f; rake = %.1f; L = %.2f; W = %.2f; U = %.2f" % (x, y, z, c, dip, rake, L, W, U) )


  ###################################################################################
  ###### Derivadas dUx/dz, dUy/dz, dUz/dz ######
  ###################################################################################

  # para calcular df1dz, df2dz, df3dz para las partes A,B,C utilizando evaluando la representacion de chinnery
  def get_df123ABC_from_Chinnerys_representation(x, y, z, dip, c, L, W, mu, lmbd):
    d = c-z
    p = y*np.cos(dip_rad) + d*np.sin(dip_rad)
    q = y*np.sin(dip_rad) - d*np.cos(dip_rad)
    qq = np.array([q, q, q, q])    # b = 4

    # Notacion  de Chinnery -> f(e,eta)|| = f(x,p) - f(x,p-W) - f(x-L,p) + f(x-L,p-W)
    e = np.array([x, x, x-L, x-L])
    eta = np.array([p, p-W, p, p-W])
    alph = (lmbd+mu)/(lmbd+2*mu)
    R = np.sqrt(e**2 + eta**2 + qq**2)
    ytg = eta*np.cos(dip_rad) + qq*np.sin(dip_rad)
    dtg = eta*np.sin(dip_rad) - qq*np.cos(dip_rad)
    ctg = dtg + z

    ######
    X11 = 1/(R*(R+e))
    X32 = (2*R + e)/(R**3*(R+e)**2)
    X53 = (8*R**2 + 9*R*e + 3*e**2)/(R**3*(R+e)**2)
    Y11 = 1/(R*(R+eta))
    Y32 = (2*R + eta)/(R**3*(R+eta)**2)
    Y53 = (8*R**2 + 9*R*eta + 3*eta**2)/(R**5*(R+eta)**3)
    h = qq*np.cos(dip_rad) - z
    Z32 = np.sin(dip_rad)/R**3 - h*Y32
    Z53 = 3*np.sin(dip_rad)/R**5 - h*Y53
    Y0 = Y11 - e**2*Y32
    Z0 = Z32 - e**2*Z53
    D11 = 1/(R*(R+dtg))

    K1 = (e/np.cos(dip_rad))*(D11-Y11*np.sin(dip_rad))
    K3 = (1./np.cos(dip_rad)) * (qq*Y11 - ytg*D11)
    if dip==90. or dip==270 or dip==-90.:# con cos(dip)=0 tenemos que sin(dip)=[+1,-1]
      K1 = ( (e*qq)/(R+dtg) )*D11
      K3 = (np.sin(dip_rad)/(R+dtg)) * (e**2*D11 -1)

    K2 = 1./R + K3*np.sin(dip_rad)
    K4 = e*Y11*np.cos(dip_rad) - K1*np.sin(dip_rad)

    J2 = ((e*ytg)/(R+dtg))*D11
    J5 = -(dtg + (ytg**2)/(R+dtg))*D11
    J3 = (1/np.cos(dip_rad))*(K1 - J2*np.sin(dip_rad))
    J6 = (1./np.cos(dip_rad))*(K3 - J5*np.sin(dip_rad))    
    if dip==90. or dip==270 or dip==-90.:# con cos(dip)=0 tenemos que sin(dip)=[+1,-1]
      J3 = -(e/(R+dtg)**2)*(qq**2*D11 - 0.5)
      J6 = -(ytg/(R+dtg)**2)*(e**2*D11 - 0.5)

    J4 = -e*Y11 - J2*np.cos(dip_rad) + J3*np.sin(dip_rad)
    J1 = J5*np.cos(dip_rad) - J6*np.sin(dip_rad)

    EE = np.cos(dip_rad)/R + (dtg*qq)/R**3
    FF = ytg/R**3 + e**2*Y32*np.cos(dip_rad)
    GG = 2*X11*np.cos(dip_rad) + dtg*qq*X32
    HH = ytg*qq*X32 + e*qq*Y32*np.cos(dip_rad)
    PP = np.sin(dip_rad)/R**3 - qq*Y32*np.cos(dip_rad)
    QQ = (3*ctg*ytg)/R**5 - (z*Y32 + Z32 + Z0)*np.cos(dip_rad)



    ######
    # strike-slip l123A
    df1dzA_ss = ((1-alph)/2.)*e*Y11*np.cos(dip_rad) + (ytg/2.)*X11 - (alph/2.)*e*FF
    df2dzA_ss = (alph/2.)*EE
    df3dzA_ss = -((1-alph)/2.)*(np.sin(dip_rad)/R - qq*Y11*np.cos(dip_rad)) - (alph/2.)*qq*FF
    # dip-slip l123A
    df1dzA_ds = (alph/2.)*EE
    df2dzA_ds = ((1-alph)/2.)*ytg*X11 + (e/2.)*Y11*np.cos(dip_rad) + (alph/2.)*eta*GG
    df3dzA_ds = -((1-alph)/2.)*dtg*X11 - (alph/2.)*qq*GG
    # tensile fault l123A
    df1dzA_tf = ((1-alph)/2.)*(np.sin(dip_rad)/R - qq*Y11*np.cos(dip_rad)) - (alph/2.)*qq*FF
    df2dzA_tf = ((1-alph)/2.)*dtg*X11 - (alph/2.)*qq*GG
    df3dzA_tf = ((1-alph)/2.)*(ytg*X11 + e*Y11*np.cos(dip_rad)) + (alph/2.)*qq*HH

    # strike-slip l123B
    df1dzB_ss = -e*FF - ytg*X11 + ((1-alph)/alph)*K1*np.sin(dip_rad)
    df2dzB_ss = -EE + ((1-alph)/alph)*ytg*D11*np.sin(dip_rad)
    df3dzB_ss = qq*FF + ((1-alph)/alph)*K2*np.sin(dip_rad)
    # dip-slip l123B
    df1dzB_ds = -EE - ((1-alph)/alph)*K3*np.sin(dip_rad)*np.cos(dip_rad)
    df2dzB_ds = -eta*GG - e*Y11*np.cos(dip_rad) - ((1-alph)/alph)*e*D11*np.sin(dip_rad)*np.cos(dip_rad)
    df3dzB_ds = qq*GG - ((1-alph)/alph)*K4*np.sin(dip_rad)*np.cos(dip_rad)
    # tensile fault l123B
    df1dzB_tf = qq*FF + ((1-alph)/alph)*K3*(np.sin(dip_rad))**2
    df2dzB_tf = qq*GG + ((1-alph)/alph)*e*D11*(np.sin(dip_rad))**2
    df3dzB_tf = -qq*HH + ((1-alph)/alph)*K4*(np.sin(dip_rad))**2

    # strike-slip l123C
    df1dzC_ss = (1-alph)*e*PP*np.cos(dip_rad) - alph*e*QQ
    df2dzC_ss = 2*(1-alph)*(ytg/R**3 - Y0*np.cos(dip_rad))*np.sin(dip_rad) + (dtg/R**3)*np.cos(dip_rad) - alph*(((ctg+dtg)/R**3 )*np.cos(dip_rad) + (3*ctg*dtg*qq)/R**5 )
    df3dzC_ss = (ytg/R**3 - Y0*np.cos(dip_rad))*np.cos(dip_rad) - alph*( ((ctg+dtg)/R**3 )*np.sin(dip_rad) - (3*ctg*ytg*qq)/R**5 - Y0*(np.sin(dip_rad)**2) + qq*Z0*np.cos(dip_rad) )
    # dip-slip l123C
    df1dzC_ds = -qq/R**3 + Y0*np.sin(dip_rad)*np.cos(dip_rad) - alph*( ((ctg+dtg)/R**3 )*np.cos(dip_rad) + (3*ctg*dtg*qq)/R**5 )
    df2dzC_ds = (1-alph)*ytg*dtg*X32 - alph*ctg*( (ytg - 2*qq*np.sin(dip_rad))*X32 + dtg*eta*qq*X53 )
    df3dzC_ds = -e*PP*np.sin(dip_rad) + X11 - dtg**2*X32 - alph*ctg*( (dtg + 2*qq*np.cos(dip_rad))*X32 - dtg*qq**2*X53  )
    # tensile fault l123C
    df1dzC_tf = -eta/R**3 + Y0*(np.cos(dip_rad)**2) - alph*( (z/R**3)*np.sin(dip_rad) - (3*ctg*ytg*qq)/R**5 - Y0*(np.sin(dip_rad)**2) + qq*Z0*np.cos(dip_rad) )
    df2dzC_tf = (1-alph)*2*e*PP*np.sin(dip_rad) - X11 + dtg**2*X32 - alph*ctg*( (dtg - 2*qq*np.cos(dip_rad))*X32 - dtg*qq**2*X53 ) 
    df3dzC_tf = (1-alph)*(e*PP*np.cos(dip_rad) + ytg*dtg*X32) + alph*ctg*( (ytg - 2*qq*np.sin(dip_rad))*X32 + dtg*eta*qq*X53 ) + alph*e*QQ


    ######
    # representacion Chinnery para strike-slip u123A
    l1A_ss = ( df1dzA_ss[0] - df1dzA_ss[1] - df1dzA_ss[2] + df1dzA_ss[3] )
    l2A_ss = ( df2dzA_ss[0] - df2dzA_ss[1] - df2dzA_ss[2] + df2dzA_ss[3] )
    l3A_ss = ( df3dzA_ss[0] - df3dzA_ss[1] - df3dzA_ss[2] + df3dzA_ss[3] )
    # representacion Chinnery para dip-slip u123A
    l1A_ds = ( df1dzA_ds[0] - df1dzA_ds[1] - df1dzA_ds[2] + df1dzA_ds[3] )
    l2A_ds = ( df2dzA_ds[0] - df2dzA_ds[1] - df2dzA_ds[2] + df2dzA_ds[3] )
    l3A_ds = ( df3dzA_ds[0] - df3dzA_ds[1] - df3dzA_ds[2] + df3dzA_ds[3] )
    # representacion Chinnery para dip-slip u123A
    l1A_tf = ( df1dzA_tf[0] - df1dzA_tf[1] - df1dzA_tf[2] + df1dzA_tf[3] )
    l2A_tf = ( df2dzA_tf[0] - df2dzA_tf[1] - df2dzA_tf[2] + df2dzA_tf[3] )
    l3A_tf = ( df3dzA_tf[0] - df3dzA_tf[1] - df3dzA_tf[2] + df3dzA_tf[3] )

    # representacion Chinnery para strike-slip u123B
    l1B_ss = ( df1dzB_ss[0] - df1dzB_ss[1] - df1dzB_ss[2] + df1dzB_ss[3] )
    l2B_ss = ( df2dzB_ss[0] - df2dzB_ss[1] - df2dzB_ss[2] + df2dzB_ss[3] )
    l3B_ss = ( df3dzB_ss[0] - df3dzB_ss[1] - df3dzB_ss[2] + df3dzB_ss[3] )
    # representacion Chinnery para dip-slip u123B
    l1B_ds = ( df1dzB_ds[0] - df1dzB_ds[1] - df1dzB_ds[2] + df1dzB_ds[3] )
    l2B_ds = ( df2dzB_ds[0] - df2dzB_ds[1] - df2dzB_ds[2] + df2dzB_ds[3] )
    l3B_ds = ( df3dzB_ds[0] - df3dzB_ds[1] - df3dzB_ds[2] + df3dzB_ds[3] )
    # representacion Chinnery para dip-slip u123B
    l1B_tf = ( df1dzB_tf[0] - df1dzB_tf[1] - df1dzB_tf[2] + df1dzB_tf[3] )
    l2B_tf = ( df2dzB_tf[0] - df2dzB_tf[1] - df2dzB_tf[2] + df2dzB_tf[3] )
    l3B_tf = ( df3dzB_tf[0] - df3dzB_tf[1] - df3dzB_tf[2] + df3dzB_tf[3] )

    # representacion Chinnery para strike-slip u123C
    l1C_ss = ( df1dzC_ss[0] - df1dzC_ss[1] - df1dzC_ss[2] + df1dzC_ss[3] )
    l2C_ss = ( df2dzC_ss[0] - df2dzC_ss[1] - df2dzC_ss[2] + df2dzC_ss[3] )
    l3C_ss = ( df3dzC_ss[0] - df3dzC_ss[1] - df3dzC_ss[2] + df3dzC_ss[3] )
    # representacion Chinnery para dip-slip u123C
    l1C_ds = ( df1dzC_ds[0] - df1dzC_ds[1] - df1dzC_ds[2] + df1dzC_ds[3] )
    l2C_ds = ( df2dzC_ds[0] - df2dzC_ds[1] - df2dzC_ds[2] + df2dzC_ds[3] )
    l3C_ds = ( df3dzC_ds[0] - df3dzC_ds[1] - df3dzC_ds[2] + df3dzC_ds[3] )
    # representacion Chinnery para dip-slip u123C
    l1C_tf = ( df1dzC_tf[0] - df1dzC_tf[1] - df1dzC_tf[2] + df1dzC_tf[3] )
    l2C_tf = ( df2dzC_tf[0] - df2dzC_tf[1] - df2dzC_tf[2] + df2dzC_tf[3] )
    l3C_tf = ( df3dzC_tf[0] - df3dzC_tf[1] - df3dzC_tf[2] + df3dzC_tf[3] )

    return l1A_ss, l2A_ss, l3A_ss, l1B_ss, l2B_ss, l3B_ss, l1C_ss, l2C_ss, l3C_ss, l1A_ds, l2A_ds, l3A_ds, l1B_ds, l2B_ds, l3B_ds, l1C_ds, l2C_ds, l3C_ds, l1A_tf, l2A_tf, l3A_tf, l1B_tf, l2B_tf, l3B_tf, l1C_tf, l2C_tf, l3C_tf


  ######
  l1A_ss1, l2A_ss1, l3A_ss1, l1B_ss, l2B_ss, l3B_ss, l1C_ss, l2C_ss, l3C_ss, l1A_ds1, l2A_ds1, l3A_ds1, l1B_ds, l2B_ds, l3B_ds, l1C_ds, l2C_ds, l3C_ds, l1A_tf1, l2A_tf1, l3A_tf1, l1B_tf, l2B_tf, l3B_tf, l1C_tf, l2C_tf, l3C_tf = get_df123ABC_from_Chinnerys_representation(x, y, -z, dip, c, L, W, mu, lmbd)
  l1A_ss, l2A_ss, l3A_ss, l1B_ss, l2B_ss, l3B_ss, l1C_ss, l2C_ss, l3C_ss, l1A_ds, l2A_ds, l3A_ds, l1B_ds, l2B_ds, l3B_ds, l1C_ds, l2C_ds, l3C_ds, l1A_tf, l2A_tf, l3A_tf, l1B_tf, l2B_tf, l3B_tf, l1C_tf, l2C_tf, l3C_tf = get_df123ABC_from_Chinnerys_representation(x, y, z, dip, c, L, W, mu, lmbd)


  # dUxyzdz for strike-slip
  dUxdz_ss = (U1/(2*np.pi)) * (  l1A_ss - l1A_ss1 + l1B_ss + z*l1C_ss ) 
  dUydz_ss = (U1/(2*np.pi)) * ( (l2A_ss - l2A_ss1 + l2B_ss + z*l2C_ss)*np.cos(dip_rad) - (l3A_ss - l3A_ss1 + l3B_ss + z*l3C_ss)*np.sin(dip_rad) ) 
  dUzdz_ss = (U1/(2*np.pi)) * ( (l2A_ss - l2A_ss1 + l2B_ss - z*l2C_ss)*np.sin(dip_rad) + (l3A_ss - l3A_ss1 + l3B_ss - z*l3C_ss)*np.cos(dip_rad) ) 
  # dUxyzdz for dip-slip
  dUxdz_ds = (U2/(2*np.pi)) * (  l1A_ds - l1A_ds1 + l1B_ds + z*l1C_ds ) 
  dUydz_ds = (U2/(2*np.pi)) * ( (l2A_ds - l2A_ds1 + l2B_ds + z*l2C_ds)*np.cos(dip_rad) - (l3A_ds - l3A_ds1 + l3B_ds + z*l3C_ds)*np.sin(dip_rad) ) 
  dUzdz_ds = (U2/(2*np.pi)) * ( (l2A_ds - l2A_ds1 + l2B_ds - z*l2C_ds)*np.sin(dip_rad) + (l3A_ds - l3A_ds1 + l3B_ds - z*l3C_ds)*np.cos(dip_rad) ) 
  # dUxyzdz for tensile fault
  dUxdz_tf = (U3/(2*np.pi)) * (  l1A_tf - l1A_tf1 + l1B_tf + z*l1C_tf ) 
  dUydz_tf = (U3/(2*np.pi)) * ( (l2A_tf - l2A_tf1 + l2B_tf + z*l2C_tf)*np.cos(dip_rad) - (l3A_tf - l3A_tf1 + l3B_tf + z*l3C_tf)*np.sin(dip_rad) ) 
  dUzdz_tf = (U3/(2*np.pi)) * ( (l2A_tf - l2A_tf1 + l2B_tf - z*l2C_tf)*np.sin(dip_rad) + (l3A_tf - l3A_tf1 + l3B_tf - z*l3C_tf)*np.cos(dip_rad) ) 

  # soluciones de deformacion dUxyzdx
  dUxdz = dUxdz_ss + dUxdz_ds + dUxdz_tf
  dUydz = dUydz_ss + dUydz_ds + dUydz_tf
  dUzdz = dUzdz_ss + dUzdz_ds + dUzdz_tf


  ###################################################################################
  # print check-list of outputs
  if verbose is True:
    print( "OUTPUTS ->          dUx/dz       dUy/dz       dUz/dz" ) 
    print( "------------------------------------------------------------------------------------------------------------------------------------" )
    print( "Strike-slip   ->  %+.3e   %+.3e   %+.3e   " % (dUxdz_ss, dUydz_ss, dUzdz_ss) )
    print( "Dip-slip      ->  %+.3e   %+.3e   %+.3e   " % (dUxdz_ds, dUydz_ds, dUzdz_ds) )
    print( "Tensile fault ->  %+.3e   %+.3e   %+.3e   " % (dUxdz_tf, dUydz_tf, dUzdz_tf) )
    print( "------------------------------------------------------------------------------------------------------------------------------------")
    print( "                  %+.3e   %+.3e   %+.3e   " % (dUxdz, dUydz, dUzdz) )
    print( "------------------------------------------------------------------------------------------------------------------------------------")
    #print( "EOF\n\n" )

  return dUxdz, dUydz, dUzdz

