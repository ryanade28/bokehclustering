import numpy as np
from sklearn import cluster, datasets
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Select, Slider
from bokeh.palettes import Spectral6
from bokeh.plotting import figure

np.random.seed(0)



# Mendefinisikan algoritma algoritma library clustering
def clustering(X, algorithm, n_clusters):
    # Melakukan normalisasi pada dataset acak yang akan dibuat
    X = StandardScaler().fit_transform(X)

    # Estimasi bandwith untuk algoritma mean shift
    bandwidth = cluster.estimate_bandwidth(X, quantile=0.3)

    # Mendefinisikan korelasi matrix untuk struktur ward
    connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)

    # Membuat hubungan simetrik
    connectivity = 0.5 * (connectivity + connectivity.T)

    # Mengaplikasikan algoritma algoritma library cluster
    if algorithm=='MiniBatchKMeans':
        model = cluster.MiniBatchKMeans(n_clusters=n_clusters)

    elif algorithm=='Birch':
        model = cluster.Birch(n_clusters=n_clusters)

    elif algorithm=='DBSCAN':
        model = cluster.DBSCAN(eps=.2)

    elif algorithm=='AffinityPropagation':
        model = cluster.AffinityPropagation(damping=.9,
                                            preference=-200)

    elif algorithm=='MeanShift':
        model = cluster.MeanShift(bandwidth=bandwidth,
                                  bin_seeding=True)

    elif algorithm=='SpectralClustering':
        model = cluster.SpectralClustering(n_clusters=n_clusters,
                                           eigen_solver='arpack',
                                           affinity="nearest_neighbors")

    elif algorithm=='Ward':
        model = cluster.AgglomerativeClustering(n_clusters=n_clusters,
                                                linkage='ward',
                                                connectivity=connectivity)

    elif algorithm=='AgglomerativeClustering':
        model = cluster.AgglomerativeClustering(linkage="average",
                                                affinity="cityblock",
                                                n_clusters=n_clusters,
                                                connectivity=connectivity)

    elif algorithm=='KMeans':
        model = cluster.KMeans(n_clusters=n_clusters,
                               random_state=0)

    # Mengaplikasikan algoritma yang dipilih untuk dataset x
    model.fit(X)
    
    if hasattr(model, 'labels_'):
            y_pred = model.labels_.astype(int)
    else:
            y_pred = model.predict(X)

    return X, y_pred
# Fungsi untuk menentukan dataset yang akan digunakan
def get_dataset(dataset, n_samples):
    if dataset == 'Noisy Circles':
        return datasets.make_circles(n_samples=n_samples,
                                    factor=0.5,
                                    noise=0.05)

    elif dataset == 'Noisy Moons':
        return datasets.make_moons(n_samples=n_samples,
                                   noise=0.05)

    elif dataset == 'Blobs':
        return datasets.make_blobs(n_samples=n_samples,
                                   random_state=8)

    elif dataset == 'Swiss Roll':
        return datasets.make_swiss_roll(n_samples=n_samples,
                                        random_state=8)
    
    elif dataset == 'S Curve':
        return datasets.make_s_curve(n_samples=n_samples,
                                        random_state=8)

    elif dataset == "No Structure":
        return np.random.rand(n_samples, 2), None

# setting parameter default
n_samples = 1500
n_clusters = 2
algorithm = 'MiniBatchKMeans'
dataset = 'Noisy Circles'

X, y = get_dataset(dataset, n_samples)
X, y_pred = clustering(X, algorithm, n_clusters)
spectral = np.hstack([Spectral6] * 20)
colors = [spectral[i] for i in y]

# setting plot yang akan di gunakan
plot = figure(toolbar_location="right", title=algorithm)
source = ColumnDataSource(data=dict(x=X[:, 0], y=X[:, 1], colors=colors))
plot.circle('x', 'y', fill_color='colors', line_color=None, source=source)

# setting fitur bokeh yang akan digunakan (select)
clustering_algorithms= [
    'MiniBatchKMeans',
    'AffinityPropagation',
    'MeanShift',
    'SpectralClustering',
    'Ward',
    'AgglomerativeClustering',
    'DBSCAN',
    'Birch',
    'KMeans'
]

datasets_names = [
    'Noisy Circles',
    'Noisy Moons',
    'Blobs',
    'Swiss Roll',
    'S Curve',
    'No Structure'
]

algorithm_select = Select(value='MiniBatchKMeans',
                          title='Select algorithm:',
                          width=200,
                          options=clustering_algorithms)

dataset_select = Select(value='Noisy Circles',
                        title='Select dataset:',
                        width=200,
                        options=datasets_names)

samples_slider = Slider(title="Number of samples",
                        value=1500.0,
                        start=1000.0,
                        end=3000.0,
                        step=100,
                        width=400)

clusters_slider = Slider(title="Number of clusters",
                         value=2.0,
                         start=2.0,
                         end=10.0,
                         step=1,
                         width=400)

# prosedur update apabila algoritma diganti
def update_algorithm_or_clusters(attrname, old, new):
    global X

    algorithm = algorithm_select.value
    n_clusters = int(clusters_slider.value)

    X, y_pred = clustering(X, algorithm, n_clusters)
    colors = [spectral[i] for i in y_pred]

    source.data = dict(colors=colors, x=X[:, 0], y=X[:, 1])

    plot.title.text = algorithm

# prosedur update apabila jumlah dan ukuran dataset diganti    
def update_samples_or_dataset(attrname, old, new):
    global X, y

    dataset = dataset_select.value
    algorithm = algorithm_select.value
    n_clusters = int(clusters_slider.value)
    n_samples = int(samples_slider.value)

    X, y = get_dataset(dataset, n_samples)
    X, y_pred = clustering(X, algorithm, n_clusters)
    colors = [spectral[i] for i in y_pred]

    source.data = dict(colors=colors, x=X[:, 0], y=X[:, 1])

algorithm_select.on_change('value', update_algorithm_or_clusters)
clusters_slider.on_change('value_throttled', update_algorithm_or_clusters)

dataset_select.on_change('value', update_samples_or_dataset)
samples_slider.on_change('value_throttled', update_samples_or_dataset)

# setting layout
selects = row(dataset_select, algorithm_select, width=420)
inputs = column(selects, samples_slider, clusters_slider)

# deploy
curdoc().add_root(row(inputs, plot))
curdoc().title = "Clustering"
