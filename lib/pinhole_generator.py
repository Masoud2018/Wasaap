import numpy as np

class pinholes:
    def __init__(self,vector):
        self.grid = vector
        

    def RecPinhole(self,D):
        if D:
            x=np.abs(self.grid)
            rec=np.double(x<D/2)
            rec[x==D/2]=1
        else:
            print('Not enogh input')
        return(rec)
    
    def CircPinhole(self,D,x_c = 0,y_c = 0):
        if  D:
            [x,y] = self.grid
            r=np.sqrt((x-x_c)**2+(y-y_c)**2)
            circ=np.double(r<D/2)
            circ[r==D/2]=1
        
        else:
            print('Not enogh input')
        return(circ)