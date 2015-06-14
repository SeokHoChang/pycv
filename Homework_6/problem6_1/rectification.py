#2015200920 SeokHochang
from numpy import * 
import sys
import math
from scipy import misc
import itertools as iter
import numpy as np
def rectification(Img1,Img2,K1,K2,E) :
   
 
    width_im1 = shape(Img1)[1];
    height_im1 = shape(Img1)[0];
 
    width_im2 = shape(Img2)[1];
    height_im2 = shape(Img2)[0];
    
    
    if( not ( shape(K1)==(3,3) and shape(K2)==(3,3) and shape(E)==(3,4) )):
        print 'wrong input'
        sys.exit();

    coordinates = list(iter.product(xrange(width_im1) , xrange(height_im1)));
    zeros_=matrix(zeros((3,width_im1*height_im1)));

    zeros_[0:2,:]=matrix(transpose(coordinates));
    zeros_[2,:]=1;
    im1_coord=zeros_.astype(double);
    im2_coord=zeros_.astype(double);
    
    im1_imgvec=np.linalg.inv(K1).dot(im1_coord);
    im2_imgvec=np.linalg.inv(K2).dot(im2_coord);

    R=matrix(E[:,0:3]);
    R=R.astype(double);
    T=matrix(E[:,3]);
    T=T.astype(double);

    e1=K1.dot(T);
    e1=e1/e1[2,0];

    e2=-K2.dot(transpose(R)).dot(T);
    e2=e2/e2[2,0];

    c=transpose(matrix(K1[:,2]));
    
    t_=transpose(T);
    v0=t_;    
    v1=cross(v0,c);
    v2=cross(v1,v0);

    H1=matrix('0. 0. 0.; 0. 0. 0.; 0. 0. 0.');

    H1[:,0]=transpose(v0)/np.linalg.norm(v0);
    H1[:,2]=transpose(v1)/np.linalg.norm(v1);
    H1[:,1]=transpose(v2)/np.linalg.norm(v2);

    H2=H1*transpose(R);
   
# inverse warping 
    coord= matrix('0;0;1');
    Img1_warp = matrix(zeros((720,1280)));
    for i in range(1280*720):
        x = i%1280;
        y = i/1280;
        coord[0,0]=x;
        coord[1,0]=y;
        
        tmp=np.linalg.inv(H1).dot(coord);
        if (x>0 and x<1280 and y>0 and y<720):        
            Img1_warp[y,x] =interp(tmp[0,:],tmp[1,:],Img1);
        
   
    coord= matrix('0;0;1');
    Img2_warp = matrix(zeros((720,1280)));
    for i in range(1280*720):
        x = i%1280;
        y = i/1280;
        coord[0,0]=x;
        coord[1,0]=y;
        
        tmp=np.linalg.inv(H2).dot(coord);
        if (x>0 and x<1280 and y>0 and y<720):        
            Img2_warp[y,x] =interp(tmp[0,:],tmp[1,:],Img2);
         

    return Img1_warp,Img2_warp;

def interp(x,y,Img):
    x1= np.floor(x).astype(int);
    x2= x1+1;
    y1= np.floor(y).astype(int);
    y2= y1+1;
    width = shape(Img)[1];
    height = shape(Img)[0];
 
    if x1>=0 and x1<width-1 and y1>=0 and y1<height-1 :
        M11= Img[y1,x1];
        M12= Img[y1,x2];
        M21= Img[y2,x1];
        M22= Img[y2,x2];

        val=(x2-x)*(y2-y)*M11 + (x-x1)*(y2-y)*M21 + (x2-x)*(y-y1)*M12 + (x-x1)*(y-y1)*M22;
    else : 
        val=0;

    return val; 
def main() :
    print 'main'
    left_img = misc.imread('ImageL_00300.png');
    right_img = misc.imread('ImageR_00300.png');
    left_K = matrix('1684.442145, .0 ,635.175409; .0 , 1687.004180 , 366.459755 ;.0 ,.0, 1.0');
    right_K = matrix('1684.392238 , .0 , 636.240634 ; .0, 1688.713774 , 355.493619 ;.0 ,.0 ,1.0');
    E = matrix('0.999959 , 0.009030 , 0.000236, 201.729621 ; -0.009031, 0.999952 , 0.003845, -2.326901 ; -0.000202 , -0.003847 , 0.999993 , -10.086409');
    
    
    (img1,img2)=rectification(left_img,right_img,left_K,right_K,E);
    misc.imsave("rect1.png",img1);
    misc.imsave("rect2.png",img2);
    
if __name__=='__main__':
    main()
