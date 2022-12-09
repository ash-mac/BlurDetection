#!/usr/bin/env python
# coding: utf-8

# In[27]:


import skimage, skimage.io, skimage.color, skimage.filters, skimage.metrics
import numpy as np
import matplotlib.pyplot as plt


# In[28]:


def convolve(im, kernel):
  padding = int(2 * (kernel.shape[0]//2))
  imPad = np.zeros((im.shape[0] + padding, im.shape[1] + padding))
  imPad[padding//2 : im.shape[0] + padding//2, padding//2 : im.shape[1] + padding//2] = im
  res = np.zeros(im.shape)    
  for i in range(im.shape[0]):
    for j in range(im.shape[1]):
      res[i, j] = np.sum(kernel * imPad[i:i+kernel.shape[0], j:j+kernel.shape[0]])
  return res    


# In[29]:


def myGaussian(sigma = 1):
  sigma = sigma * 1.45
  siz = int(4 * sigma + 1)  
  def h(x, y, u, v, sigma = sigma):
    return np.exp(-((x-u)**2 + (y-v)**2)/(sigma**2))/(2 * np.pi * (sigma**2))
  gauss_kernel = np.zeros((siz, siz), dtype = 'float')
  i_ = siz//2
  j_ = siz//2
  for i in range(gauss_kernel.shape[0]):
    for j in range(gauss_kernel.shape[1]):
      gauss_kernel[i, j] = h(i, j, i_, j_)
  return gauss_kernel/np.sum(gauss_kernel)


# In[30]:


def sobel_filter(im, sigma = 1):
  fx = np.array([[-1, 0, 1],
                 [-2, 0, 2],
                 [-1, 0, 1]])
  fy = np.array([[1, 2, 1],
                 [0, 0, 0],
                 [-1, -2, -1]])
  imGray = im.copy()
  myGauss = myGaussian(sigma = sigma)
  imGray = convolve(imGray, myGauss)
  imX = convolve(imGray, fx)
  imY = convolve(imGray, fy)
  imR = np.sqrt(imX**2 + imY**2)
  return imX, imY, imR, np.arctan(imY/(imX + 1e-9))


# In[31]:


def getDir(theta):
  if(abs(theta)<np.pi/8):
    return 0
  if(abs(theta)<(np.pi/4 + np.pi/8)):
    if(theta > 0):
      return 2
    else:
      return 3
  else:
    return 1


# In[32]:


def nms(imGrad, im):
  removed = 0
  total = 0
  nmsIm = im.copy()
  for i in range(1, im.shape[0] - 1):
    for j in range(1, im.shape[1] - 1):
      direction = getDir(imGrad[i, j])      
      if(direction == 0):
        total+=1
        if(im[i, j] < max(im[i, j-1], im[i, j+1])):
          nmsIm[i, j] = 0.0  
          removed+=1
      elif(direction == 1):
        total+=1
        if(im[i, j] < max(im[i-1, j], im[i+1, j])):
          nmsIm[i, j] = 0.0
          removed+=1
      elif(direction == 2):
        total+=1
        if(im[i, j] < max(im[i-1, j+1], im[i+1, j-1])):
          nmsIm[i, j] = 0.0
          removed+=1
      elif(direction == 3):
        total+=1
        if(im[i, j] < max(im[i-1, j-1], im[i+1, j+1])):
          nmsIm[i, j] = 0.0
          removed+=1
  return nmsIm, removed


# In[33]:


def hysteresis1(im, low = 0.05, high = 0.15):
  imH = im.copy()
  for i in range(im.shape[0]):
    for j in range(im.shape[1]):
      if(imH[i, j] > high):
        imH[i, j] = 1.0
      elif(imH[i, j] < low):
        imH[i, j] = 0        
  return imH


# In[34]:


def hysteresis2(imH1, low = 0.05, high = 0.15):  
  def extend(i, j, imH):    

    if(j-1>=0 and imH[i][j-1]<=high and imH[i][j-1]>=low):
      imH[i][j-1] = 1.0
      extend(i, j-1, imH)
    if(j+1<imH.shape[1] and imH[i][j+1]<=high and imH[i][j+1]>=low)  :
      imH[i][j+1] = 1.0
      extend(i, j+1, imH)
    
    if(i-1>=0 and imH[i-1][j]<=high and imH[i-1][j]>=low):
      imH[i-1][j] = 1.0
      extend(i-1, j, imH)
    if(i+1<imH.shape[0] and imH[i+1][j]<=high and imH[i+1][j]>=low):
      imH[i+1][j] = 1.0
      extend(i+1, j, imH)
    
    if(i-1>=0 and j+1<imH.shape[1] and imH[i-1][j+1]<=high and imH[i-1][j+1]>=low):
      imH[i-1][j+1] = 1.0
      extend(i-1, j+1, imH)
    if(i+1<imH.shape[0] and j-1>=0 and imH[i+1][j-1]<=high and imH[i+1][j-1]>=low):
      imH[i+1][j-1] = 1.0
      extend(i+1, j-1, imH)
    
    if(i+1<imH.shape[0] and j+1<imH.shape[1] and imH[i+1][j+1]<=high and imH[i+1][j+1]>=low):
      imH[i+1][j+1] = 1.0
      extend(i+1, j+1, imH)
    if(i-1>=0 and j-1>=0 and imH[i-1][j-1]<=high and imH[i-1][j-1]>=low):
      imH[i-1][j-1] = 1.0
      extend(i-1, j-1, imH)
    return  
  imH = imH1.copy()
  for i in range(imH.shape[0]):
    for j in range(imH.shape[1]):
      if(imH[i, j] > high):
        extend(i, j, imH)
  for i in range(imH.shape[0]):
    for j in range(imH.shape[1]):
      if(imH[i, j]<=high):                
        imH[i, j] = 0        
  return imH


# In[35]:


def hysteresis(nmsImage, low = 0.05, high = 0.15):
  imH1 = hysteresis1(nmsImage, low = low, high = high)
  imH2 = hysteresis2(imH1, low = low, high = high)
  return imH2


# In[36]:


def myCanny(image, low = 0.05, high = 0.15, sigma = 1):  
  imageX, imageY, imageR, imageG = sobel_filter(image, sigma = sigma)
  imageNms, rem = nms(imageG, imageR)
  imageH = hysteresis(imageNms, low = low, high = high)
  tot = np.sum(imageH == True)
  return imageH, rem, tot


# In[37]:


def BlurOrNot(img_path):
  im = skimage.io.imread(img_path)
  im = np.array(im, dtype = 'float')
  im = im/255
  if(im.ndim == 3):
    im = skimage.color.rgb2gray(im)
  imH, rem, tot = myCanny(im)
  score = (rem/tot)
  prob = 1/(1 + np.exp(-1 * (score - 10)))
  if 0.3<prob<0.6:
    if prob < 0.5:
        prob = 0.5
    prob = min(prob * 1.5, 1)
  return prob, score


# In[41]:


im_path = input('Enter image path:')
im = skimage.io.imread(im_path)
skimage.io.imshow(im)
prob, score = BlurOrNot(im_path)
label = None
if prob > 0.5:
  label = 'Blurred'
else:
  label = 'Sharp'
plt.title(f'{label} with BlurProb = {prob} and Blur Score = {score}')
plt.xlabel(f'{label}')
plt.show()









