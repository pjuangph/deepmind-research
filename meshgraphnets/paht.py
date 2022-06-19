import tensorflow as tf
import json, os, functools
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np

# Directed |E| <= n(n-1)
# Undirected |E| <= n(n-1)/2

def triangles_to_edges(faces):
    """Computes mesh edges from triangles."""
    # collect edges from triangles
    # This makes sense if you draw the triangles. Triangle 1 is in center with Triangles 2,3, and 4 surrounding it
    # Basically this is a representation of all the possible connections
    #   0:2 this is 1 to 2
    #   1:3 this is 2 to 4
    #   2 and 0 is from 3 to 1
    edges = tf.concat([faces[:, 0:2],
                        faces[:, 1:3],
                        tf.stack([faces[:, 2], faces[:, 0]], axis=1)], axis=0)
    # those edges are sometimes duplicated (within the mesh) and sometimes
    # single (at the mesh boundary).
    # sort & pack edges as single tf.int64
    receivers = tf.reduce_min(edges, axis=1) # Edges shape = 1803,2,3. Axis = 1 you keep 1803,3
    senders = tf.reduce_max(edges, axis=1)
    packed_edges = tf.bitcast(tf.stack([senders, receivers], axis=1), tf.int32)
    # remove duplicates and unpack
    unique_edges = tf.bitcast(tf.unique(packed_edges)[0], tf.int32)
    senders, receivers = tf.unstack(unique_edges, axis=1)
    # create two-way connectivity
    return (tf.concat([senders, receivers], axis=0),
            tf.concat([receivers, senders], axis=0))

            
def _parse(proto, meta):
    """Parses a trajectory from tf.Example."""
    feature_lists = {k: tf.io.VarLenFeature(tf.string)
                    for k in meta['field_names']}
    features = tf.io.parse_single_example(proto, feature_lists)
    out = {}
    for key, field in meta['features'].items():
        data = tf.io.decode_raw(features[key].values, getattr(tf, field['dtype']))
        data = tf.reshape(data, field['shape'])
        if field['type'] == 'static':
            data = tf.tile(data, [meta['trajectory_length'], 1, 1])
        elif field['type'] == 'dynamic_varlen':
            length = tf.io.decode_raw(features['length_'+key].values, tf.int32)
            length = tf.reshape(length, [-1])
            data = tf.RaggedTensor.from_row_lengths(data, row_lengths=length)
        elif field['type'] != 'dynamic':
            raise ValueError('invalid data format')
        out[key] = data
    return out

def load_dataset(path, split):
    """Load dataset."""
    with open(os.path.join(path, 'meta.json'), 'r') as fp:
        meta = json.loads(fp.read())
    ds = tf.data.TFRecordDataset(os.path.join(path, split+'.tfrecord'))
    ds = ds.map(functools.partial(_parse, meta=meta), num_parallel_calls=8)
    ds = ds.prefetch(1)
    return ds

test_ds = load_dataset('datasets/airfoil','test')
for data in test_ds:
    # These are the x and y coordinates
    # Number of vertices 601x5233
    x=data['mesh_pos'][:,:,0].numpy() # The shape is 601 x 5233 x 2 
    y=data['mesh_pos'][:,:,1].numpy() # Last index is the x,y position
                                # The 601 x 5233 is (i,j)
    # fig = plt.figure(clear=True)
    # plt.plot(x.flatten(),y.flatten(),'.',color='black') # Plots all the points 
    
    # Lets get the edges that connect these coordinates
    cells = data['cells']    # Shape is 601, 10216, 3
    edges = triangles_to_edges(cells) # Information travels from cell-to-cell not vertex-to-vertex
                                    # Each cell can have at most 3 connections
                                    # Total Number of cells 601*10216
                                    # Not possible to tell which vertex is in what cell or how to draw line connecting vertices
    print(data)
print('loaded')