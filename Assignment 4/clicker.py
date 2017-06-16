from matplotlib import pyplot as plt
from pylab import *

class DataSetBuilder(object):
  
    def __init__(self, ax, im, pix_err=1):
        self.canvas = ax.get_figure().canvas
        self.cid = None
        self.pt_lst_pos = []
        self.pt_lst_neg = []
        self.lab_pos = []
        self.lab_neg = []
        
        self.pt_plot_pos = ax.plot([], [], marker='o', color='r', linestyle='none', zorder=5)[0]
        self.pt_plot_neg = ax.plot([], [], marker='o', linestyle='none', zorder=5)[0]
        self.pix_err = pix_err
        self.connect_sf()
        
        self.image = im

    def set_visible(self, visible):
        self.pt_plot.set_visible(visible)

    def clear(self):
        self.pt_lst_pos = []
        self.pt_lst_neg = []
        self.redraw()

    def connect_sf(self):
        if self.cid is None:
            self.cid = self.canvas.mpl_connect('button_press_event',
                                               self.click_event)

    def disconnect_sf(self):
        if self.cid is not None:
            self.canvas.mpl_disconnect(self.cid)
            self.cid = None

    def click_event(self, event):
        ''' Extracts locations of samples, left click and right click are different classes'''
        if event.xdata is None or event.ydata is None:
            return
        if event.button == 1:
            self.pt_lst_pos.append((event.xdata, event.ydata))
            self.lab_pos.append(1)
        elif event.button == 3:
            self.pt_lst_neg.append((event.xdata, event.ydata))
            self.lab_neg.append(0)   
        self.redraw()

    def redraw(self):
        if len(self.pt_lst_pos) > 0:
            x, y = zip(*self.pt_lst_pos)
        else:
            x, y = [], []
        self.pt_plot_pos.set_xdata(x)
        self.pt_plot_pos.set_ydata(y)

        if len(self.pt_lst_neg) > 0:
            x, y = zip(*self.pt_lst_neg)
        else:
            x, y = [], []
        self.pt_plot_neg.set_xdata(x)
        self.pt_plot_neg.set_ydata(y)
   
        self.canvas.draw()

    def return_points(self):
        '''Returns the clicked points as [x, y, label] NumPy array'''
        data = np.vstack( (np.vstack(np.floor(self.pt_lst_pos)), np.vstack(np.floor(self.pt_lst_neg))) )
        labels = np.vstack( (np.vstack(np.floor(self.lab_pos)), np.vstack(np.floor(self.lab_neg))) )
        
        return np.hstack( (data, labels))


        
def feature_vector(loc, im, size = 10):
  
  # window size
  w = size
  # a patch of the size +/- w is extracted as a feature vector
  patch = im[loc[1]-w:loc[1]+w, loc[0]-w:loc[0]+w]
  p = np.array(patch).flatten()
  return p 


def main():

  ax = gca()
  im = plt.imread('parking_train.png')
  ax.imshow(im)

  cc = DataSetBuilder(ax, im)
  plt.show()

  X = []
  Y = []
  
  for c in cc.return_points():
       
    X.append( feature_vector(c, im) )
    Y.append(np.array(c[2]))
    
    #ax = gca()
    #ax.imshow(im[c[1]-10:c[1]+10, c[0]-10:c[0]+10])
    #plt.show()

 
  with open('X_trn.np','wb') as f:
    np.save(f, np.array(X))

  with open('Y_trn.np','wb') as f:
    np.save(f, np.array(Y))  

  
#print ('Saved %d samples.') % len(Y)
  
if __name__ == "__main__":
    main()
