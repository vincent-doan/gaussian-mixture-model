# Basic imports
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sp
import matplotlib as mpl

def normalize(x):
    mean = x.mean(axis=0)
    stddev = x.std(axis=0)
    return (x-mean)/stddev

def convert(raw_label):
    convert_dict = dict()
    labels_list = list()
    i = 1
    for label in raw_label:
        if label not in convert_dict:
            convert_dict[label] = i
            labels_list.append(label)
            i += 1
    return convert_dict, labels_list

def compute_purity(clustering, labeling):
    purity = 0
    label_count_dict = dict()
    for cluster, label in zip(clustering, labeling):
        if cluster not in label_count_dict:
            label_count_dict[cluster] = [label]
        else:
            label_count_dict[cluster].append(label)
    for cluster, label_list in label_count_dict.items():
        max_count = max([label_list.count(label) for label in label_list])
        purity += max_count
    purity /= len(clustering)
    return purity

def plot_1D(gmm, x, col, file_path):
  """
  Plot 1D data
  """
  # Plot the actual data distribution
  plt.hist(x, bins=25, density=True)

  x = np.linspace(x.min(), x.max(), 100, endpoint=False)
  ys = np.zeros_like(x)

  j = 0
  for w in gmm.weights:
      # Plot the normal distribution for one mixture component
      y = sp.multivariate_normal.pdf(x, mean=gmm.means[j,:], cov=gmm.covariances[j])*w
      plt.plot(x, y)

      ys += y
      j += 1
    
  # Plot the sum of normal distributions of all mixture components  
  plt.xlabel(col)
  plt.plot(x, ys)
  plt.savefig(file_path)
  plt.show()

def make_ellipses(gmm, ax):
    """
    Create a ecclipse to denote cluster aka mixture component in 2D plot
    """
    colors = ['turquoise', 'orange', 'orchid', 'lightsalmon', 'slateblue',
              'dodgerblue', 'lightpink', 'silver', 'maroon', 'seashell']
    used_colors = colors[:gmm.k]
    for n, color in enumerate(used_colors):
        covariances = gmm.covariances[n]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 3. * np.sqrt(2.) * np.sqrt(v)
        mean=gmm.means[n]
        mean=mean.reshape(2,1)
        ell = mpl.patches.Ellipse(mean, v[0], v[1],
                                  180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
        ax.set_aspect('equal', 'datalim')

def plot_2D(gmm, x, col, label, labels_list, file_path):
    """
    Plot 2D data
    """
    # Plot the ellipses
    h = plt.subplot(111, aspect='equal')
    make_ellipses(gmm, h)

    # Plot the scatterplot
    scatter = plt.scatter(x[:,0],x[:,1],c=label,marker='x')
    plt.xlim(-3, 4)
    plt.ylim(-3, 4)
    plt.xlabel(col[0])
    plt.ylabel(col[1])

    # Create legend
    plt.legend(scatter.legend_elements()[0], labels_list, title="Legend")
    plt.savefig(file_path)
    plt.show()