def derivy(array_2D,dy):
    # array_2D es el la matrix, dy el paso vertical
    import numpy as np
    dUy=array_2D*0
    a=len(array_2D[0,:,0])
    b=len(array_2D[0,0,:])
    for time in range(0,np.size(array_2D,0)):

        i=0
        while i<a:
            j=0
            while j<b:
                if i==0:
                    dUy[time,i,j]=(array_2D[time,i+1,j]-array_2D[time,i,j])/dy
                elif i==a-1:
                    dUy[time,i,j]=(array_2D[time,i,j]-array_2D[time,i-1,j])/dy
                else:
                    dUy[time,i,j]=(array_2D[time,i+1,j]-array_2D[time,i-1,j])/dy/2
                j=j+1
            i=i+1
    return dUy