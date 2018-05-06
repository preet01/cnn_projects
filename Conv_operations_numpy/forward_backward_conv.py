import numpy as np


def affine_forward(x, weight, bias):
    """
    Computes the foward pass for an affine layer.
    x:input x
    weight:weight for affine_forward layer
    bias:bias
    out:output of weight,x,bias
    cache:data stored for backward pass
    """
    out = np.zeros((x.shape[0],weight.shape[1]))

    x1=np.reshape(x,(x.shape[0],-1))   
    out=np.dot(x1,weight)+bias
    cache = (x, weight, bias)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.
    dout:Upstream derivative
    cache:data from affine_forward layer.
    dx:Gradient with respect to x
    dweight:Gradient with respect to weight
    dbias:Gradient with respect to bias
    """
    x, weight, bias = cache
    dx, dweight, dbias = None, None, None

    dx=np.dot(dout,weight.T)
    dx=np.reshape(dx,x.shape)
    dweight=np.dot(np.reshape(x,(x.shape[0],np.prod(x.shape[1:]))).T,dout)
    dbias=np.sum(dout,axis=0)
    return dx, dweight, dbias


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

def conv_forward_naive(x, weight, bias, conv_param):

    """
    An implementation of the forward pass for a convolutional layer. 
    x:input 
    weight:convolutional filter weights
    bias:convolutional filter bias
    conv_param:dictionary containing padding and striding for the filters
  
    """
    p=conv_param['pad']
    s=conv_param['stride']

    x_pad=np.pad(x,((0,0),(0,0),(p,p),(p,p)), 'constant')

    (N, C, H, W)=x.shape
    (F, C, filter_len, filter_width)=weight.shape	
   
    out_r=int((H-filter_len+2*p)/s)+1#formula for calcuating the output:    int(input-filter_size+2*padding)+1
    out_c=int((W-filter_width+2*p)/s)+1
   
    out=np.zeros((N,F,out_r,out_c))
    
    (N, C, H, W)=x_pad.shape
 
    
      
  for num_batches in range(N):
      
       for num_channels in range(F):
            
           for num_row in range(0,out_c):#selecting the width(rows wise) of x
              
               for num_col in  range(0,out_r):#selecting the length(column wise) of x
                    
                    out[num_batches,num_channels,num_row,num_col] = np.sum(x_pad[num_batches, :, num_row*s:num_row*s+filter_len, num_col*s:num_col*s+filter_width]*weight[num_channels,...]) + bias[num_channels]     
                    #padded x multiplied with weights
   

    cache = (x, weight, bias, conv_param)
    return out, cache



def conv_backward_naive(dout, cache):


    """
    An implementation of the backward pass for a convolutional layer. 
    dout:upstream derivatives 
    cache:data from forward conv layer which helps to calculate dx,dweight,dbias
    dx:Gradient with respect to x
    dweight:Gradient with respect to weight
    dbias:Gradient with respect to bias
 
    """
   

    (x, weight, bias, conv_param)=cache
    p=conv_param['pad']
    s=conv_param['stride']
    
    dx=np.zeros((x.shape))
    dweight=np.zeros((weight.shape))

    x_pad=np.pad(x,((0,0),(0,0),(p,p),(p,p)), 'constant')
 
   
    N,F,H,W=dout.shape
    _,_,filter_len,filter_width=weight.shape
   
   
    for num_batches in range(N):
 
        for num_channels in range(F):

            for num_row in range(H):
 
                for num_col in range(W):

                     dweight[num_channels,...]+ = x_pad[num_batches,:,num_row*s:num_row*s+filter_len,num_col*s:num_col*s+filter_width] * dout[num_batches,num_channels,num_row,num_col]

                     dbias[num_channels,]+ = dout[num_batches,num_channels,num_row,num_col]
                     
                     dx[num_batches,:,num_row*s:num_row*s+filter_len,num_col*s:num_col*s+filter_width]+ = weight[num_channels,...]*                      dout[num_batches,num_channels,num_row,num_col]


    dx=dx[:,:,p:-p,p:-p]#dx calcualated in for loop is for dx_pad,it has to be reduced to x i.e. removing the padding
       
    return dx, dweight, dbias

