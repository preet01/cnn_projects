import numpy as np


def affine_forward(x, w, b):
    """
    Computes the foward pass for an affine layer.
    x:input x
    w:weight for affine_forward layer
    b:bias
    out:output of weight,x,bias
    cache:data stored for backward pass
    """
    out = np.zeros((x.shape[0],w.shape[1]))

    x1=np.reshape(x,(x.shape[0],-1))   
    out=np.dot(x1,w)+b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.
    dout:Upstream derivative
    cache:data from affine_forward layer.
    dx:Gradient with respect to x
    dw:Gradient with respect to w
    db:Gradient with respect to b
    """
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
    out[out < 0] = 0#removing all the number less than 0

    cache = x
    return out, cache


def relu_backward(dout, cache):

     """
    Computes the backward pass for an relu_backward layer.
    dout:Upstream derivative
    cache:data from relu_backward layer.
    """
    dx, x = None, cache
    dx=dout
    dx[x<=0]=0

    return dx

def conv_forward_naive(x, w, b, conv_param):

    """
    An implementation of the forward pass for a convolutional layer. 
    x:input 
    w:convolutional filter weights
    b:convolutional filter bias
    conv_param:dictionary containing padding and striding for the filters
  
    """
    p=conv_param['pad']
    s=conv_param['stride']

    x_pad=np.pad(x,((0,0),(0,0),(p,p),(p,p)), 'constant')

    (N, C, H, W)=x.shape
    (F, C, hh, ww)=w.shape	
   
    out_r=int((H-hh+2*p)/s)+1#formula for calcuating the output:    int(input-filter_size+2*padding)+1
    out_c=int((W-ww+2*p)/s)+1
   
    out=np.zeros((N,F,out_r,out_c))
    
    (N, C, H, W)=x_pad.shape
 
    
      
  for nn in range(N):
      
       for ff in range(F):
            
           for ii in range(0,out_c):#selecting the width(rows wise) of x
              
               for jj in  range(0,out_r):#selecting the length(column wise) of x
                    
                    out[nn,ff,ii,jj]=np.sum(x_pad[nn, :, ii*s:ii*s+hh, jj*s:jj*s+ww]*w[ff,...])+b[ff]     
                    #padded x multiplied with weights
   

    cache = (x, w, b, conv_param)
    return out, cache



def conv_backward_naive(dout, cache):


    """
    An implementation of the backward pass for a convolutional layer. 
    dout:upstream derivatives 
    cache:data from forward conv layer which helps to calculate dx,dw,db
    dx:Gradient with respect to x
    dw:Gradient with respect to w
    db:Gradient with respect to b
 
    """
   

    (x, w, b, conv_param)=cache
    p=conv_param['pad']
    s=conv_param['stride']
    
    dx=np.zeros((x.shape))
    dw=np.zeros((w.shape))

    x_pad=np.pad(x,((0,0),(0,0),(p,p),(p,p)), 'constant')
 
   
    N,F,H,W=dout.shape
    _,_,hh,ww=w.shape
   
   
    for nn in range(N):
 
        for ff in range(F):

            for ii in range(H):
 
                for jj in range(W):

                     dw[ff,...]+=x_pad[nn,:,ii*s:ii*s+hh,jj*s:jj*s+ww] * dout[nn,ff,ii,jj]

                     db[ff,]+= dout[nn,ff,ii,jj]
                     
                     dx[nn,:,ii*s:ii*s+hh,jj*s:jj*s+ww]+=w[ff,...]*dout[nn,ff,ii,jj]


    dx=dx[:,:,p:-p,p:-p]#dx calcualated in for loop is for dx_pad,it has to be reduced to x i.e. removing the padding
       
    return dx, dw, db

