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


ruta="/Users/macbookair/Desktop/practicasCG/atmo/Practica 3/EB1P1/" #ruta CARO
os.chdir(ruta)
dir = 'EB1P1.nc'
dS = xr.open_dataset(dir, decode_times=False)
print(dS)       # visualizo la info del .nc
phi=dS['stream'].values
time=dS['time'].values
lon=dS['lon'].values
lat=dS['lat'].values
ucomp=dS['ucomp'].values
vcomp=dS['vcomp'].values
h=dS['h'].values
H=40000/9.8

lons, lats = np.meshgrid(lon, lat)

#%%ENERGIA CINETICA DEL FLUJO MEDIO Y LAS ERTURBACIONES
#ANOMALIAS U Y V

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
#plt.imshow(kemedia) 
#plt.colorbar() #hago un grafico pra visualizar min y max

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

#%% EJERCICIO 2 Calculos

#Adveccion  TOTAL
from DerX import derivx
from DerY import derivy

dx=(lon[1]-lon[0])*(np.pi/180)*6371e3
dy=(lat[1]-lat[0])*(np.pi/180)*6371e3
dkepertX=derivx(kepert,dx,lat)
dkepertY=derivy(kepert,dy)

Adv= -(uz*dkepertX + vz*dkepertY + ue*dkepertX + ve*dkepertY)

#Termino de Convercion Baroclinica

dueX = derivx(ue,dx,lat)
dveY = derivy(ve,dy)

CBaroc = 9.8*H*etae*(dueX+dveY)

#Dispercion de Energia Cinetica

detauX=derivx(etae*ue,dx,lat)
detavY=derivy(etae*ve,dy)

Disp=-9.8*H*(detauX+detavY)

#Conversion Barotropica


duzX=derivx(np.tile(uz,(1,1,1)),dx,lat)
dvzX=derivx(np.tile(vz,(1,1,1)),dx,lat)
duzY=derivy(np.tile(uz,(1,1,1)),dy)
dvzY=derivy(np.tile(vz,(1,1,1)),dy)

CBarot = -H*(ue*ue*duzX + ue*ve*duzY + ve*ue*dvzX + ve*ve*dvzY)

#Viento ageostrofico

omega = 7.3e-5
f=2*omega*np.sin(lat*np.pi/180)
f=np.transpose(np.tile(f,(256,1)))
dhX=derivx(h,dx,lat)
dhY=derivy(h,dy)

Ug= - (1/f)*dhY #viento geostrofico
Vg= (1/f)*dhX

Uag=ucomp-Ug # a mi viento comun le resto el geostrofico para obtener el ageostrofico
Vag=vcomp-Vg

aUag=np.mean(Uag[49,:,:], axis=1) #ahora lo divido entre media y perturbavicion a Uag
bUag=np.mean(np.mean(Uag[50:59,:,:],axis=0), axis=1)
lons, aUag= np.meshgrid(lon,aUag)
lons, bUag= np.meshgrid(lon,bUag)
Uagz=Uag[49,:,:] - aUag + bUag #correccion a la media de Uag

Uage=np.empty((10,128,256)) #anomalias de Uag
for i in range (10):
    Uage[i,:,:] = Uag[i+50,:,:] - Uagz


aVag=np.mean(Vag[49,:,:], axis=1) #divido entre parte media y perturbada a Vag
bVag=np.mean(np.mean(Vag[50:59,:,:],axis=0), axis=1)
lons, aVag= np.meshgrid(lon,aVag)
lons, bVag= np.meshgrid(lon,bVag)
Vagz=Vag[49,:,:] - aVag + bVag #correccion de h

Vage=np.empty((10,128,256)) #anomalias de h
for i in range (10):
    Vage[i,:,:] = Vag[i+50,:,:] - Vagz




#%% EJERCICIO 2 Graficos
#(90àS–10àN) X (60àE–180àO)

#Adveccion de KePert y Kepert
    
    
#plt.imshow(Adv[0,:,:]) , plt.colorbar() #hago un grafico pra visualizar min y max

AdvF=[Adv[0,:,:],Adv[1,:,:],Adv[2,:,:], Adv[3,:,:]]
kepertF=[kepert[0,:,:],kepert[1,:,:],kepert[2,:,:], kepert[3,:,:]]
cmin = -0.6
cmax = 0.6
ncont = 13
clevs = np.linspace(cmin, cmax, ncont)
ind=['1','2','3','4']
for i in range(0,4):
    fig=plt.figure(figsize=(6,4),dpi=200)
    LONMIN= 100
    LONMAX= 300
    LATMIN = -90
    LATMAX = 10
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    crs_latlon = ccrs.PlateCarree()
    ax.set_extent([LONMIN, LONMAX, LATMIN, LATMAX], crs=crs_latlon)
    im=ax.contourf(lons, lats, AdvF[i], clevs, cmap=plt.get_cmap("RdBu"), extend='both', transform=crs_latlon)
    plt.colorbar(im, fraction=0.052, pad=0.08, shrink=0.8, aspect=20, orientation='horizontal', label='x m$^{2}$ s$^{2}$ día$^{-1}$' )
    CS=plt.contour(lons,lats,kepertF[i],transform=crs_latlon,linewidths=0.5)
    ax.clabel(CS, inline=0, fontsize=5)
    plt.annotate('Fuente',xy=(42,-28)) #xytext=(45,-23)
    plt.plot(40,-28,'.k')
    ax.add_feature(cartopy.feature.LAND, facecolor='#d9d9d9')
    ax.add_feature(cartopy.feature.COASTLINE)
    ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax.gridlines(crs=crs_latlon, linewidth=0.3, linestyle='-')
    ax.set_xticks(np.arange(100, 300, 30), crs=crs_latlon)
    ax.set_yticks(np.arange(-90, 10, 20), crs=crs_latlon)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    plt.title("Adveccion Total Ke dia " + ind[i] , fontsize=8, y=0.98, loc="center")
    plt.savefig('Adveccion Total dia ' + ind[i])


#Conversion Baroclinica y Kepert
plt.imshow(CBaroc[0,:,:]) , plt.colorbar() #hago un grafico pra visualizar min y max

CBarocF=[CBaroc[0,:,:],CBaroc[1,:,:],CBaroc[2,:,:],CBaroc[3,:,:]]

cmin = -150
cmax = 150
ncont = 7
clevs = np.linspace(cmin, cmax, ncont)

ind=['1','2','3','4']
for i in range(0,4):
    fig=plt.figure(figsize=(6,4),dpi=200)
    LONMIN= 100
    LONMAX= 300
    LATMIN = -90
    LATMAX = 10
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    crs_latlon = ccrs.PlateCarree()
    ax.set_extent([LONMIN, LONMAX, LATMIN, LATMAX], crs=crs_latlon)
    im=ax.contourf(lons, lats, CBarocF[i], clevs, cmap=plt.get_cmap("RdBu"), extend='both', transform=crs_latlon)
    plt.colorbar(im, fraction=0.052, pad=0.08, shrink=0.8, aspect=20, orientation='horizontal', label='x m$^{2}$ s$^{2}$ día$^{-1}$')
    CS=plt.contour(lons,lats,kepertF[i],transform=crs_latlon,linewidths=0.5)
    ax.clabel(CS, inline=0, fontsize=5)
    plt.annotate('Fuente',xy=(42,-28)) #xytext=(45,-23)
    plt.plot(40,-28,'.k')
    ax.add_feature(cartopy.feature.LAND, facecolor='#d9d9d9')
    ax.add_feature(cartopy.feature.COASTLINE)
    ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax.gridlines(crs=crs_latlon, linewidth=0.3, linestyle='-')
    ax.set_xticks(np.arange(100, 300, 30), crs=crs_latlon)
    ax.set_yticks(np.arange(-90, 10, 20), crs=crs_latlon)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    plt.title("Termino de conversion baroclina " + ind[i] , fontsize=8, y=0.98, loc="center")
    plt.savefig('Termino de conversion baroclina' + ind[i])


#Conversion Barotropixa y Kepert
    
#plt.imshow(CBarot[3,:,:]) , plt.colorbar() #hago un grafico pra visualizar min y max

CBarotF=[CBarot[0,:,:],CBarot[1,:,:],CBarot[2,:,:],CBarot[3,:,:]]

cmin = -0.2
cmax = 0.2
ncont = 9
clevs = np.linspace(cmin, cmax, ncont)

ind=['1','2','3','4']
for i in range(0,4):
    fig=plt.figure(figsize=(6,4),dpi=200)
    LONMIN= 100
    LONMAX= 300
    LATMIN = -90
    LATMAX = 10
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    crs_latlon = ccrs.PlateCarree()
    ax.set_extent([LONMIN, LONMAX, LATMIN, LATMAX], crs=crs_latlon)
    im=ax.contourf(lons, lats, CBarotF[i], clevs, cmap=plt.get_cmap("RdBu"), extend='both', transform=crs_latlon)
    plt.colorbar(im, fraction=0.052, pad=0.08, shrink=0.8, aspect=20, orientation='horizontal', label='x m$^{2}$ s$^{2}$ día$^{-1}$')
    CS=plt.contour(lons,lats,kepertF[i],transform=crs_latlon,linewidths=0.5)
    ax.clabel(CS, inline=0, fontsize=5)
    plt.annotate('Fuente',xy=(42,-28)) #xytext=(45,-23)
    plt.plot(40,-28,'.k')
    ax.add_feature(cartopy.feature.LAND, facecolor='#d9d9d9')
    ax.add_feature(cartopy.feature.COASTLINE)
    ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax.gridlines(crs=crs_latlon, linewidth=0.3, linestyle='-')
    ax.set_xticks(np.arange(100, 300, 30), crs=crs_latlon)
    ax.set_yticks(np.arange(-90, 10, 20), crs=crs_latlon)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    plt.title("Termino de conversion barotropica" + ind[i] , fontsize=8, y=0.98, loc="center")
    plt.savefig('Termino de conversion barotropica' + ind[i])


#Dispercion de energia Cinetica, flujo ageostrofico y Kepert


#ndia=0
#matriz_etae=etae[i,:,:]
#matriz_uage=Uage[i,:,:]
#matriz_vage=Vage[i,:,:]
#flujou=matriz_etae[np.where(lat<=-5),:]*matriz_vage[np.where(lat<=-5),:]
#flujov=matriz_etae[np.where(lat<=-5),:]*matriz_vage[np.where(lat<=-5),:]
#Q75 = np.percentile(np.sqrt(np.add(np.power(np.squeeze(flujou),2), np.power(np.squeeze(flujov),2))),75)
#M = np.sqrt(np.add(np.power(np.squeeze(flujou),2),np.power(np.squeeze(flujov),2))) < Q75
#px_mask = ma.array(np.squeeze(flujou),mask = M)
#py_mask = ma.array(np.squeeze(flujov),mask = M)
#xii=np.squeeze(lats[np.where(lat<=-5),:])
#yii=np.squeeze(lons[np.where(lat<=-5),:])
#plt.quiver( xii, yii , px_mask,py_mask,color='green',width=1.5e-3,headwidth=3, headlength=3.2)
    

#plt.imshow(Disp[3,:,:]) , plt.colorbar() #hago un grafico pra visualizar min y max

from numpy import ma
DispF=[Disp[0,:,:],Disp[1,:,:],Disp[2,:,:],Disp[3,:,:]]

cmin = -200
cmax = 200
ncont = 9
clevs = np.linspace(cmin, cmax, ncont)
ind=['1','2','3','4']
for i in range(0,4):
    DispF=Disp[i,:,:]
    matriz_etae=etae[i,:,:]
    matriz_uage=Uage[i,:,:]
    matriz_vage=Vage[i,:,:]
    flujou=matriz_etae[np.where(lat<=-5),:]*matriz_vage[np.where(lat<=-5),:]
    flujov=matriz_etae[np.where(lat<=-5),:]*matriz_vage[np.where(lat<=-5),:]
    Q75 = np.percentile(np.sqrt(np.add(np.power(np.squeeze(flujou),2), np.power(np.squeeze(flujov),2))),75)
    M = np.sqrt(np.add(np.power(np.squeeze(flujou),2),np.power(np.squeeze(flujov),2))) < Q75
    px_mask = ma.array(np.squeeze(flujou),mask = M)
    py_mask = ma.array(np.squeeze(flujov),mask = M)
    xii=np.squeeze(lats[np.where(lat<=-5),:])
    yii=np.squeeze(lons[np.where(lat<=-5),:])
    fig=plt.figure(figsize=(6,4),dpi=200)
    LONMIN= 0
    LONMAX= 359.9
    LATMIN = -90
    LATMAX = 90
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    crs_latlon = ccrs.PlateCarree()
    ax.set_extent([LONMIN, LONMAX, LATMIN, LATMAX], crs=crs_latlon)
    im=ax.contourf(lons, lats, Disp[i], clevs, cmap=plt.get_cmap("RdBu"), extend='both', transform=crs_latlon)
    plt.colorbar(im, fraction=0.052, pad=0.08, shrink=0.8, aspect=20, orientation='horizontal', label='x m$^{2}$ s$^{2}$ día$^{-1}$')
    CS=plt.contour(lons,lats,kepertF[i],transform=crs_latlon,linewidths=0.5)
    plt.quiver( xii, yii , px_mask,py_mask,color='green',width=1.5e-3,headwidth=3, headlength=3.2)
    ax.clabel(CS, inline=0, fontsize=5)
    plt.annotate('Fuente',xy=(42,-28)) #xytext=(45,-23)
    plt.plot(40,-28,'.k')
    ax.add_feature(cartopy.feature.LAND, facecolor='#d9d9d9')
    ax.add_feature(cartopy.feature.COASTLINE)
    ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax.gridlines(crs=crs_latlon, linewidth=0.3, linestyle='-')
    ax.set_xticks(np.arange(0, 359.9, 30), crs=crs_latlon)
    ax.set_yticks(np.arange(-90, 90, 20), crs=crs_latlon)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    plt.title("Dispercion de Ke dia " + ind[i] , fontsize=8, y=0.98, loc="center")
    plt.savefig('Dispercion de Ke dia ' + ind[i])
