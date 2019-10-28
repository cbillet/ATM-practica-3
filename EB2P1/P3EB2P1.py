#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 09:29:28 2019

@author: macbookair
"""
#FUENTE lon 220 lat -28
import numpy as np
from matplotlib import pyplot as plt
import cartopy.feature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.crs as ccrs
import os
import xarray as xr


ruta="/Users/macbookair/Desktop/practicasCG/atmo/Practica 3/EB2P1/" #ruta CARO
os.chdir(ruta)
dir = 'EB2P1.nc'
dS = xr.open_dataset(dir, decode_times=False)
print(dS)       # visualizo la info del .nc
phi=dS['stream'].values
time=dS['time'].values
lon=dS['lon'].values
lat=dS['lat'].values
lons, lats = np.meshgrid(lon, lat)
ucomp=dS['ucomp'].values
vcomp=dS['vcomp'].values
h=dS['h'].values
H=40000/9.8


#%%

aucomp=np.mean(ucomp[49,:,:], axis=1)
bucomp=np.mean(np.mean(ucomp[50:59,:,:],axis=0), axis=1)
lons, aucomp= np.meshgrid(lon,aucomp)
lons, bucomp=np.meshgrid(lon,bucomp)
uz=ucomp[49,:,:] - aucomp + bucomp #correccion de u

ue=np.empty((10,128,256)) #anomalias de u
for i in range (10):
    ue[i,:,:] = ucomp[i+50,:,:] - uz
    
    
avcomp=np.mean(vcomp[49,:,:], axis=1)
bvcomp=np.mean(np.mean(vcomp[50:59,:,:],axis=0), axis=1)
lons, avcomp= np.meshgrid(lon,avcomp)
lons, bvcomp=np.meshgrid(lon,bvcomp)
vz=vcomp[49,:,:] - avcomp + bvcomp

ve=np.empty((10,128,256))
for i in range (10):
    ve[i,:,:] = vcomp[i+50,:,:] - vz

#ENERGIA CINETICA
kemedia=(H/2)*(uz*uz+vz*vz) #energia cinetica media
kepert=(H/2)*(ue*ue+ve*ve) #energia cinetica de las perturbaciones

#%%ENERGIA POTENCIAL DEL FLUJO MEDIO Y LAS PERTRBACIONES

#anomalias de h

ah=np.mean(h[49,:,:], axis=1)
bh=np.mean(np.mean(h[50:59,:,:],axis=0), axis=1)
lons, ah= np.meshgrid(lon,ah)
lons, bh= np.meshgrid(lon,bh)
hz=h[49,:,:] - ah + bh #correccion de h

he=np.empty((10,128,256)) #anomalias de h
for i in range (10):
    he[i,:,:] = h[i+50,:,:] - hz
   

etaz= hz/9.8 - H
etae= he/9.8 - H
    
   
epmedia = 9.8/2*(etaz*etaz + (H)*(H))
eppert = 9.8/2*(etae*etae)



#%% GRAFICOS TODO EL DOMINIO

#ENERGIA CINETICA MEDIA
plt.imshow(kemedia) 
plt.colorbar() #hago un grafico pra visualizar min y max

plt.figure(figsize=(6,4),dpi=200)
cmin = 0
cmax = 25
ncont = 6
clevs = np.linspace(cmin, cmax, ncont)
LONMIN= 0
LONMAX= 359.9
LATMIN = -90
LATMAX = 90
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
crs_latlon = ccrs.PlateCarree()
ax.set_extent([LONMIN, LONMAX, LATMIN, LATMAX], crs=crs_latlon)
im=ax.contourf(lons, lats, kemedia/1e4, clevs, cmap=plt.get_cmap("Reds"), extend='both', transform=crs_latlon)
plt.colorbar(im, fraction=0.052, pad=0.08, shrink=0.8, aspect=20, orientation='horizontal', label='x 10$^{4}$ Kg m$^{2}$ s$^{-2}$')
CS=plt.contour(lons,lats,etae[1,:,:],transform=crs_latlon,linewidths=0.5)
ax.clabel(CS, inline=0, fontsize=5)
ax.add_feature(cartopy.feature.LAND, facecolor='#d9d9d9')
ax.add_feature(cartopy.feature.COASTLINE)
ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
ax.gridlines(crs=crs_latlon, linewidth=0.3, linestyle='-')
ax.set_xticks(np.arange(0, 360, 60), crs=crs_latlon)
ax.set_yticks(np.arange(-90, 90, 30), crs=crs_latlon)
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
plt.title("Energia cinetca media Kz", fontsize=8, y=0.98, loc="center")
plt.savefig('Energia Cinetica Media.jpg')

#POTENCIAL MEDIA
#plt.imshow(epmedia) 
#plt.colorbar() #hago un grafico pra visualizar min y max
cmin = 8
cmax = 8.3
ncont = 13
clevs = np.linspace(cmin, cmax, ncont)
plt.figure(figsize=(6,4),dpi=200) #figsize=(6,4),dpi=200
LONMIN= 0
LONMAX= 359.9
LATMIN = -90
LATMAX = 90
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
crs_latlon = ccrs.PlateCarree()
ax.set_extent([LONMIN, LONMAX, LATMIN, LATMAX], crs=crs_latlon)
im=ax.contourf(lons, lats, epmedia/1e7, clevs, cmap=plt.get_cmap("Reds"), extend='both', transform=crs_latlon)
plt.colorbar(im, fraction=0.052, pad=0.08, shrink=0.8, aspect=20, orientation='horizontal', label='x 10$^{7}$  Kg m$^{2}$ s$^{-2}$')
CS=plt.contour(lons,lats,etae[1,:,:],transform=crs_latlon,linewidths=0.5)
ax.clabel(CS, inline=0, fontsize=5)
ax.add_feature(cartopy.feature.LAND, facecolor='#d9d9d9')
ax.add_feature(cartopy.feature.COASTLINE)
ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
ax.gridlines(crs=crs_latlon, linewidth=0.3, linestyle='-')
ax.set_xticks(np.arange(0, 360, 60), crs=crs_latlon)
ax.set_yticks(np.arange(-90, 90, 30), crs=crs_latlon)
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
plt.title("Energia potencial media Az", fontsize=8, y=0.98, loc="center")
plt.savefig('Energia potencial Media.jpg')

#KE E
plt.imshow(kepert[1,:,:]), plt.colorbar()


fig=plt.figure(figsize=(6,4),dpi=200)
cmin = 1
cmax = 4
ncont = 7
clevs = np.linspace(cmin, cmax, ncont)
LONMIN= 0
LONMAX= 359.9
LATMIN = -90
LATMAX = 90
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
crs_latlon = ccrs.PlateCarree()
ax.set_extent([LONMIN, LONMAX, LATMIN, LATMAX], crs=crs_latlon)
im=ax.contourf(lons, lats, kepert[1,:,:]/1e4, clevs, cmap=plt.get_cmap("Reds"), extend='both', transform=crs_latlon)
plt.colorbar(im, fraction=0.052, pad=0.08, shrink=0.8, aspect=20, orientation='horizontal', label='x 10$^{4}$ Kg m$^{2}$ s$^{-2}$')
CS=plt.contour(lons,lats,etae[1,:,:],transform=crs_latlon,linewidths=0.5)
ax.clabel(CS, inline=0, fontsize=5)
ax.add_feature(cartopy.feature.LAND, facecolor='#d9d9d9')
ax.add_feature(cartopy.feature.COASTLINE)
ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
ax.gridlines(crs=crs_latlon, linewidth=0.3, linestyle='-')
ax.set_xticks(np.arange(0, 360, 60), crs=crs_latlon)
ax.set_yticks(np.arange(-90, 90, 30), crs=crs_latlon)
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
plt.title("Ke", fontsize=8, y=0.98, loc="center")
plt.savefig('Energia Cinetica Media.jpg')

#A e


fig=plt.figure(figsize=(6,4),dpi=200)
cmin = 8
cmax = 8.3
ncont = 13
clevs = np.linspace(cmin, cmax, ncont)
LONMIN= 0
LONMAX= 359.9
LATMIN = -90
LATMAX = 90
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
crs_latlon = ccrs.PlateCarree()
ax.set_extent([LONMIN, LONMAX, LATMIN, LATMAX], crs=crs_latlon)
im=ax.contourf(lons, lats, eppert[1,:,:]/1e7, clevs, cmap=plt.get_cmap("Reds"), extend='both', transform=crs_latlon)
plt.colorbar(im, fraction=0.052, pad=0.08, shrink=0.8, aspect=20, orientation='horizontal', label='x 10$^{7}$ Kg m$^{2}$ s$^{-2}$')
CS=plt.contour(lons,lats,etae[1,:,:],transform=crs_latlon,linewidths=0.5)
ax.clabel(CS, inline=0, fontsize=5)
ax.add_feature(cartopy.feature.LAND, facecolor='#d9d9d9')
ax.add_feature(cartopy.feature.COASTLINE)
ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
ax.gridlines(crs=crs_latlon, linewidth=0.3, linestyle='-')
ax.set_xticks(np.arange(0, 360, 60), crs=crs_latlon)
ax.set_yticks(np.arange(-90, 90, 30), crs=crs_latlon)
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
plt.title("Ae", fontsize=8, y=0.98, loc="center")
plt.savefig('Energia potencial de las perturbaciones.jpg')
