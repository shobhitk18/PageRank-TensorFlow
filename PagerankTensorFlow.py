import tensorflow as tf
import numpy as np
import sys

'''
def get_transition_matrix(data, nrows, ncols):

	# We must build a square matrix as this will be used to create 
	# transition matrix M later. Inititialize with all 0's
	adj_matrix = np.zeros([nrows, ncols])

	# Create a reversed adjacency matrix where each row depicts the incoming edges
	# rather than outgoing edges generally done for adjacency matrix formulation
	# This is done to make things easier while we calculate the final 
	for i in range(data.shape[0]):
		edge = data[i]
		adj_matrix[edge[1]][edge[0]] = 1

	# Get out-degree for each node by summing over cols of adj_matrix
	outlink_arr = np.sum(adj_matrix, axis=0)

	# move reference of adj_matrix to t_mat
	trans_mat = adj_matrix 

	for col in range(ncols):
		num_outlink = trans_mat[:,col].sum()
		if num_outlink != 0:
			trans_mat[:,col] = trans_mat[:,col]/num_outlink

	return trans_mat

'''

def print_output(final_rank):

	rank_idx = np.argsort(final_rank[:,0], axis=0)

	# reverse the array to get descending order
	rank_idx_asc = rank_idx[::-1]
	print("1. Printing top 20 node ids with their ranks")
	print("S No. \t Node Id \t Rank")
	for i in range(20):
		print(i+1, "\t" , rank_idx_asc[i], "\t" , final_rank[rank_idx_asc[i]][0])

	rank_idx_desc  = rank_idx
	print("\n2. Printing bottom 20 node ids with their ranks")
	print("S No. \t Node Id \t Rank")
	for i in range(20):
		print(i+1, "\t" , rank_idx_desc[i], "\t" , final_rank[rank_idx_desc[i]][0])


def calculate_pagerank(sparse_mat, R, num_nodes):
	#threshold for breaking out of the convergence loop
	min_err = 1.0e-3
	#probability for teleportation, assuming to be 0.15
	teleport_prob = 0.15
	# beta value is the probability of making a non-teleport transition = 0.85
	beta = (1-teleport_prob)

	# v should contain the last pagerank vector used for calculating err_norm
	v = tf.placeholder(tf.float32, shape=[num_nodes,1])
	adj_matrix = tf.sparse.placeholder(tf.float32, shape=[num_nodes,num_nodes])
	
	# To get the transition matrix, we normalise the adj matrix by dividing each
	# col(from) value by the sum of values in that column(out-degree)
	dout = tf.sparse.reduce_sum(adj_matrix, axis=0)
	M = adj_matrix/dout
	
	e = np.ones([num_nodes, 1])*teleport_prob/num_nodes

	# calculating pagerank using the formula : v = (bM)v + (1-b)e/N 
	# where bM = M , (1-b)e/N = e
	pagerank = tf.add(tf.sparse.sparse_dense_matmul(M, v)*beta , e)

	diff_in_rank = tf.linalg.norm(tf.subtract(pagerank ,v), ord=1)

	init = tf.global_variables_initializer()

	# Main pagerank algorithm on tensor flow, run until converges
	with tf.Session() as sess:
		sess.run(init)
		while(True):
			new_pagerank = sess.run(pagerank, feed_dict={adj_matrix: sparse_mat,   v: R}) #updated R
			err_norm = sess.run(diff_in_rank, feed_dict={pagerank: new_pagerank, v: R})
			R = new_pagerank
			if(err_norm < min_err):
				break
			#print("err_norm=", err_norm)
			#print("new_rank=", R)
	return new_pagerank


def get_pagerank(filepath):

	# read the link information in the numpy 2-d array i.e [from , to]
	data = np.loadtxt(filepath, dtype=np.int64, skiprows=4)
	max_nodeid = np.amax(data, axis=0)

	# nrows = ncols as transition matrix is always a square matrix
	# nrows = ncols = max_nodes + 1(Assuming nodes starting from zero)
	nrows = ncols = maxnodes = max(max_nodeid[0],max_nodeid[1]) + 1
	
	# We should interchange the columns to have [to, from] relation
	row_idx = data[:,1]
	col_idx = data[:,0]
	index = np.column_stack((row_idx, col_idx))

	# initialize the data value to be used for sparse adj matrix entries as 1
	vals = np.ones(data.shape[0])
	# Shape of the sparse tensor
	tshape = np.asarray([nrows, ncols], dtype=np.int64)
	# Create a sparse matrix in tensor flow
	sparse_mat = tf.SparseTensorValue(indices=index, values=vals, dense_shape=tshape)

	# Populate the initial probability matrix, assuming uniform distribution
	initial_prob = 1.0/maxnodes
	R = np.ones(maxnodes).reshape(maxnodes,1) 	# have it as a column matrix
	R = np.multiply(initial_prob, R)
	
	# Now we will calculate the pagerank using the formula:
	# v' = bMv + (1-b)e/N
	final_rank = calculate_pagerank(sparse_mat, R, maxnodes)

	# we get our final rank vector now.. let's print it
	print_output(final_rank)

if __name__ == '__main__':
	# Get the filename containing the link data
	filepath = sys.argv[1]
	get_pagerank(filepath)