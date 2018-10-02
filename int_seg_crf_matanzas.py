
#general
from __future__ import division

import sys, getopt, os
import cv2
import numpy as np
from scipy.misc import imresize, imread,imsave
from scipy.io import savemat

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
from numpy.lib.stride_tricks import as_strided as ast
from skimage.morphology import remove_small_objects
from scipy.stats import mode as md
import random, string
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral, unary_from_labels, unary_from_softmax
from joblib import Parallel, delayed, cpu_count

# =========================================================
def writeout(tmp, cl, labels, outpath, thres):

   l, cnt = md(cl.flatten())
   l = np.squeeze(l)
   if cnt/len(cl.flatten()) > thres:
      outfile = id_generator()+'.jpg'
      outpath = outpath+labels[l]+'/'+outfile
      imsave(outpath, tmp)

# =========================================================
def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
   return ''.join(random.choice(chars) for _ in range(size))

# =========================================================
def norm_shape(shap):
   '''
   Normalize numpy array shapes so they're always expressed as a tuple,
   even for one-dimensional shapes.
   '''
   try:
      i = int(shap)
      return (i,)
   except TypeError:
      # shape was not a number
      pass

   try:
      t = tuple(shap)
      return t
   except TypeError:
      # shape was not iterable
      pass

   raise TypeError('shape must be an int, or a tuple of ints')


# =========================================================
# Return a sliding window over a in any number of dimensions
# version with no memory mapping
def sliding_window(a,ws,ss = None,flatten = True):
    '''
    Return a sliding window over a in any number of dimensions
    '''
    if None is ss:
        # ss was not provided. the windows will not overlap in any direction.
        ss = ws
    ws = norm_shape(ws)
    ss = norm_shape(ss)
    # convert ws, ss, and a.shape to numpy arrays
    ws = np.array(ws)
    ss = np.array(ss)
    shap = np.array(a.shape)
    # ensure that ws, ss, and a.shape all have the same number of dimensions
    ls = [len(shap),len(ws),len(ss)]
    if 1 != len(set(ls)):
        raise ValueError(\
        'a.shape, ws and ss must all have the same length. They were %s' % str(ls))

    # ensure that ws is smaller than a in every dimension
    if np.any(ws > shap):
        raise ValueError(\
        'ws cannot be larger than a in any dimension.\
 a.shape was %s and ws was %s' % (str(a.shape),str(ws)))
    # how many slices will there be in each dimension?
    newshape = norm_shape(((shap - ws) // ss) + 1)
    # the shape of the strided array will be the number of slices in each dimension
    # plus the shape of the window (tuple addition)
    newshape += norm_shape(ws)
    # the strides tuple will be the array's strides multiplied by step size, plus
    # the array's strides (tuple addition)
    newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
    a = ast(a,shape = newshape,strides = newstrides)
    if not flatten:
        return a
    # Collapse strided so that it has one more dimension than the window.  I.e.,
    # the new array is a flat list of slices.
    meat = len(ws) if ws.shape else 0
    firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
    dim = firstdim + (newshape[-meat:])
    # remove any dimensions with size 1
    #dim = filter(lambda i : i != 1,dim)

    return a.reshape(dim), newshape

# =========================================================
def getCRF_justcol(img, Lc, theta, n_iter, label_lines, compat_col=40, scale=5, prob=0.5):

      H = img.shape[0]
      W = img.shape[1]

      d = dcrf.DenseCRF2D(H, W, len(label_lines)+1)
      U = unary_from_labels(Lc.astype('int'), len(label_lines)+1, gt_prob= prob)

      d.setUnaryEnergy(U)

      del U

      # sdims = The scaling factors per dimension.
      # schan = The scaling factors per channel in the image.
      # This creates the color-dependent features and then add them to the CRF
      feats = create_pairwise_bilateral(sdims=(theta, theta), schan=(scale, scale, scale), #11,11,11
                                  img=img, chdim=2)

      del img

      d.addPairwiseEnergy(feats, compat=compat_col,
                    kernel=dcrf.DIAG_KERNEL,
                    normalization=dcrf.NORMALIZE_SYMMETRIC)

      del feats
      Q = d.inference(n_iter)

      preds = np.array(Q, dtype=np.float32).reshape(
        (len(label_lines)+1, H, W)).transpose(1, 2, 0)
      preds = np.expand_dims(preds, 0)
      preds = np.squeeze(preds)

      return np.argmax(Q, axis=0).reshape((H, W)), preds #, p, R, np.abs(d.klDivergence(Q)/ (H*W))



#==============================================================================

# mouse callback function
def anno_draw(event,former_x,former_y,flags,param):
    global current_former_x,current_former_y,drawing, mode

    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        current_former_x,current_former_y=former_x,former_y

    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            if mode==True:
                cv2.line(im,(current_former_x,current_former_y),(former_x,former_y),(0,0,255),10) #5)
                current_former_x = former_x
                current_former_y = former_y
                #print former_x,former_y
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        if mode==True:
            cv2.line(im,(current_former_x,current_former_y),(former_x,former_y),(0,0,255),10) #5)
            current_former_x = former_x
            current_former_y = former_y
    return former_x,former_y


#==============================================================================
#==============================================================================

#==============================================================
if __name__ == '__main__':

   argv = sys.argv[1:]
   try:
      opts, args = getopt.getopt(argv,"hi:w:")
   except getopt.GetoptError:
      print('python int_seg_crf.py -i image')
      sys.exit(2)

   for opt, arg in opts:
      if opt == '-h':
         print('Example usage: python int_seg_crf.py -i J:\Elwha\Elwha_20120927\Elwha_20120927_2500_25000.tif -w 1000')
         sys.exit()
      elif opt in ("-i"):
         image_path = arg
      elif opt in ("-w"):
         win = arg

   #===============================================
   #fct=.25
   ##image_path = "Elwha_20120927/Elwha_20120927_42500_22500.tif"

   win = int(win) ##1000


   ## From Chris: list for Plum Island
   # labels = ['mudpile','vegetation','mudbank','water']
   # cmap = ['#868e96','#FEE893', '#5F7D8E', '#B5674D' ]
   # cmap = colors.ListedColormap(cmap)
   # classes = {'mudpile':'m', 'vegetation':'v','mudbank':'b', 'water':'w'}
  # From Karen - New Short list:
   labels = ['sand','wetland veg','water','dune grass', 'woody vegetation', 'structure','road','surf']
   cmap = ['#FEE893', '#5F7D8E', '#0052A5', '#8DD080', '#076443', '#868e96','gray','b']
   cmap = colors.ListedColormap(cmap)
   classes = {'sand':'d', 'wetland veg':'g', 'water':'y', 'dune grass':'v', 'woody vegetation':'n','structure':'q','road':'r','surf':'s'}


   # From Chris for Zafer - Short list:
   # labels = ['water','vegetation','other']
   # cmap = ['#0052A5', '#076443', '#868e96']
   # cmap = colors.ListedColormap(cmap)
   # classes = { 'water':'y', 'vegetation':'v', 'other':'q'}


   ## From Karen - Full list:
   #labels = ['subtidal rocky','subtidal sand','deeper water','dry sand','intertidal sand','gravel','boulders','wet cobbles','dune vegetation','marsh vegetation', 'woody vegetation','anthropogenic']
   #cmap = ['#004159','#AACEE2','#0052A5','FEE893','#CAB388','#A8A795D','#B5674D','#17806D', '#acd5b6','#72ba3a', '#076443', '#868e96']
   #cmap = colors.ListedColormap(cmap)
   #classes = {'subtidal rocky':'r', 'subtidal sand':'t','deeper water':'y', 'dry sand':'d', 'intertidal sand':'f', 'gravel':'g', 'boulders':'h', 'wet cobbles':'j', 'dune vegetation':'v', 'woody vegetation':'n', 'antro':'q'}


   ##From Daniel:
   #labels = ['terrain','water','veg','shadow','sediment','road','anthro','wood']
   #cmap = ['#FF8C00','b','g','k','w','r','c','#8B4513']
   #cmap = colors.ListedColormap(cmap)
   #classes = {'wood':'l', 'sediment':'c','water': 'w', 'terrain':'t', 'veg':'v', 'road':'r', 'anthro':'a', 'shadow':'n'}

   theta=100
   compat_col=40
   scale=1
   n_iter=20

   thres = .9
   tile = 128
   outpath = 'naip2010_autoclassified'+str(tile)
   #=======================================================
   try:
      os.mkdir(outpath)
   except:
      pass

   for f in classes.keys():
      try:
         os.mkdir(outpath+os.sep+f)
      except:
         pass


   #===============================================

   drawing=False # true if mouse is pressed
   mode=True # if True, draw rectangle. Press 'm' to toggle to curve

   ##im = imresize(cv2.imread(image_path),fct)

   img = cv2.imread(image_path)

   mask=img[:,:,0]<20
   img[mask] = 0

   nxo, nyo, nz = np.shape(img)
   # pad image so it is divisible by N windows with no remainder
   img = np.pad(img, [(0,win-np.mod(nxo,win)), (0,win-np.mod(nyo,win)), (0,0)], mode='constant')
   nx, ny, nz = np.shape(img)
   Z,ind = sliding_window(img, (win, win,3), (win, win,3))

   gridy, gridx = np.meshgrid(np.arange(ny), np.arange(nx))
   Zx,_ = sliding_window(gridx, (win, win), (win, win))
   Zy,_ = sliding_window(gridy, (win, win), (win, win))

   out = np.zeros((nx,ny))

   for ck in range(len(Z)):

      im = Z[ck]
      counter=1
      for label in labels:

         cv2.namedWindow(label)
         cv2.setMouseCallback(label,anno_draw)
         while(1):
            cv2.imshow(label,im)
            k=cv2.waitKey(1)&0xFF
            if k==27:
               im[im[:,:,2]==255] = counter
               counter += 1
               break
         cv2.destroyAllWindows()

      Lc = im[:,:,2]
      Lc[Lc>=counter] = 0

      out[Zx[ck],Zy[ck]] = Lc


   im = imread(os.path.normpath(image_path))
   Lc = out[:nxo,:nyo] ##np.round(imresize(Lc,np.shape(im), interp='nearest'))

   Lcorig = Lc.copy().astype('float')
   Lcorig[Lcorig<1] = np.nan

   print('Generating dense scene from sparse labels ....')
   res,p = getCRF_justcol(im, Lc, theta, n_iter, classes, compat_col, scale)

   # tmp = [i for i, x in enumerate([x.startswith('water') for x in labels]) if x].pop()

   # res[im[:,:,0]==0] = tmp

   savemat(image_path.split('.')[0]+'_mres.mat', {'sparse': Lc.astype('int'), 'class': res.astype('int'), 'preds': p.astype('float16'), 'labels': labels}, do_compression = True)

   #=============================================
   name, ext = os.path.splitext(image_path)
   name = name.split(os.sep)[-1]
   print('Generating plot ....')
   fig = plt.figure()
   fig.subplots_adjust(wspace=0.4)
   ax1 = fig.add_subplot(131)
   ax1.get_xaxis().set_visible(False)
   ax1.get_yaxis().set_visible(False)

   _ = ax1.imshow(im)
   plt.title('a) Orthomosaic', loc='left', fontsize=6)

   ax1 = fig.add_subplot(132)
   ax1.get_xaxis().set_visible(False)
   ax1.get_yaxis().set_visible(False)

   _ = ax1.imshow(im)
   plt.title('b) manual prediction', fontsize=6)
   im2 = ax1.imshow(Lcorig-1, cmap=cmap, alpha=0.5, vmin=0, vmax=len(labels))
   divider = make_axes_locatable(ax1)
   cax = divider.append_axes("right", size="5%")
   cb=plt.colorbar(im2, cax=cax)
   cb.set_ticks(np.arange(len(labels)+1))
   cb.ax.set_yticklabels(labels)
   cb.ax.tick_params(labelsize=4)

   ax1 = fig.add_subplot(133)
   ax1.get_xaxis().set_visible(False)
   ax1.get_yaxis().set_visible(False)

   _ = ax1.imshow(im)
   plt.title('c) CRF prediction', loc='left', fontsize=6)
   im2 = ax1.imshow(res, cmap=cmap, alpha=0.5, vmin=0, vmax=len(labels))
   divider = make_axes_locatable(ax1)
   cax = divider.append_axes("right", size="5%")
   cb=plt.colorbar(im2, cax=cax)
   cb.set_ticks(np.arange(len(labels)+1))
   cb.ax.set_yticklabels(labels)
   cb.ax.tick_params(labelsize=4)
   plt.savefig(name+'_mres.png', dpi=600, bbox_inches='tight')
   del fig; plt.close()

   #==============================

   print('Generating tiles from dense class map ....')
   Z,ind = sliding_window(im, (tile,tile,3), (int(tile/2), int(tile/2),3))

   C,ind = sliding_window(res, (tile,tile), (int(tile/2), int(tile/2)))

   w = Parallel(n_jobs=-1, verbose=0, pre_dispatch='2 * n_jobs', max_nbytes=None)(delayed(writeout)(Z[k], C[k], labels, outpath+os.sep, thres) for k in range(len(Z)))
