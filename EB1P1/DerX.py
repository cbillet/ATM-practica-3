""" 
Función para derivar en la dirección x tomando en cuenta la esfericidad de la tierra

INPUTS
array_2D: numpy array de dos dimensiones
dy: paso espacial en dirección x

OUTPUTS
dUx: array de dos dimensiones con las derivadas

""" 

#%%

# Funcion derivada segun x tomando en cuenta la esfericidad de la tierra
def derivx(U,dx,lat):
    # U es el la matrix, dx el paso horizonta
    import math
    import numpy as np
    dUx=U*0
    a=len(U[0,:,0])
    b=len(U[0,0,:])
    for time in range(0,np.size(U,0)):
        i=0
        while i<a:
            j=0
            while j<b:
                if j==0:
                    dUx[time,i,j]=(U[time,i,j+1]-U[time,i,j-1])/(dx*math.cos(lat[i]*math.pi/180))/2
                elif j==b-1:
                    dUx[time,i,j]=(U[time,i,j-j]-U[time,i,j-1])/(dx*math.cos(lat[i]*math.pi/180))/2
                else:
                    dUx[time,i,j]=(U[time,i,j+1]-U[time,i,j-1])/(dx*math.cos(lat[i]*math.pi/180))/2
                j=j+1
            i=i+1
    return dUx
