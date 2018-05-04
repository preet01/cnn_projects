import numpy as np


def affine_forward(x, w, b):

    out = np.zeros((x.shape[0],w.shape[1]))

    x1=np.reshape(x,(x.shape[0],-1))   
    out=np.dot(x1,w)+b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):

    x, w, b = cache
    dx, dw, db = None, None, None

    dx=np.dot(dout,w.T)
    dx=np.reshape(dx,x.shape)
    dw=np.dot(np.reshape(x,(x.shape[0],np.prod(x.shape[1:]))).T,dout)
    db=np.sum(dout,axis=0)
    return dx, dw, db


def relu_forward(x):

    out = np.zeros((x.shape))
    out = x.copy()  # Must use copy in numpy to avoid pass by reference.
    out[out < 0] = 0

    cache = x
    return out, cache


def relu_backward(dout, cache):
    dx, x = None, cache
    dx=dout
    dx[x<=0]=0

    return dx

def conv_forward_naive(x, w, b, conv_param):
   
    p=conv_param['pad']
    s=conv_param['stride']

    x_pad=np.pad(x,((0,0),(0,0),(p,p),(p,p)), 'constant')

    (N, C, H, W)=x.shape
    (F, C, hh, ww)=w.shape	
   
    out_r=int((H-hh+2*p)/s)+1
    out_c=int((W-ww+2*p)/s)+1
   
    out=np.zeros((N,F,out_r,out_c))
    
    (N, C, H, W)=x_pad.shape
    #print(x_pad.shape)
    #print(out.shape)
 
    
      
    for ff in range(F):
            
        for ii in range(0,out_c):
              
            for jj in  range(0,out_r):
                   #print(nn,ff,ii,jj)
                    
                   out[0,ff,ii,jj]=np.sum(x_pad[0, :, ii*s:ii*s+hh, jj*s:jj*s+ww]*w[ff,...])+b[ff]  

   

    cache = (x, w, b, conv_param)
    return out, cache



def conv_backward_naive(dout, cache):

    (x, w, b, conv_param)=cache
    p=conv_param['pad']
    s=conv_param['stride']
    
    dx=np.zeros((x.shape))
    dw=np.zeros((w.shape))

    x_pad=np.pad(x,((0,0),(0,0),(p,p),(p,p)), 'constant')
    #print(x_pad.shape)
  
    (N, C, out_r, out_c)=dw.shape   
 
    _,_,hh,ww=w.shape

   
    
    
    for ff in range(C):
            
        for ii in range(0,out_r):
              
            for jj in  range(0,out_c):


                    dw[ff,...]+= dout[0,ff,jj,ii]* x_pad[0, :, ii*s:ii*s+hh, jj*s:jj*s+ww] 
                    

    db=np.sum(dout)
   
    return dx, dw, db

