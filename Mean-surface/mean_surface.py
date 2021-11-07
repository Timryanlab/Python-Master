# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 11:33:15 2021

@author: ajnel
"""

import numba
from numba import cuda
import cupy as cp
import numpy as np

knn_gpu_search = cp.RawKernel(r'''
extern "C" 
__global__ void knn_gpu_search(double* x1,
                               double* x2,
                               double* index_out,
                               double* distance_out, 
                               int k,
                               int m, 
                               int n){
    //A K nearest neighbor search will
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    int row = tid*k;
    int data_row = tid;
    float d = 0;
    if(tid < m){
    // Initialize array to -1
    // and high distance
    index_out[row] = data_row;
    distance_out[row] = 0;
    for(int i=0; i<k; i++){
      index_out[row+i] = -1;
      distance_out[row+i] = 100000;
      };

    for(int i =0; i<n; i++){
      //looping over x2 entries
      d = 0;
      for(int l=0; l<3;l++){
        d += (x1[tid*3 + l] - x2[i*3 + l])*(x1[tid*3 + l] - x2[i*3 + l]);
        }
      for(int l=0; l<k; l++){
        if(distance_out[row + l] > d){
          if(d > -1){
            for(int j =k-1; j>l; j--){
              distance_out[row + j] = distance_out[row + j-1];
              index_out[row + j] = index_out[row + j-1];}
            distance_out[row + l] = d;
            index_out[row + l] = i;
            break;
          }
        }
      }
    }     
    }}
                          ''','knn_gpu_search')
                          
                          
knn_covar = cp.RawKernel(r'''
extern "C" 
__global__ void knn_covar(double* x1,
                          double* covout,
                          double* knn_index,
                          int k,
                          int m){
    //determine the covariance matrix of the knn-neighbors
    
    int tid = blockDim.x*blockIdx.x + threadIdx.x; // get thread index    

    if(tid < m){ // avoid over calculation
      int knn_row = tid*k; // knn is mxk 
      int data_row = tid*3; // x1 is mx3
      // compute mean
      double mean[3] = {0, 0, 0};
      for(int i=0; i<3; i++){ // loop over 3 axes
        for(int j=0; j<k; j++){ // loop over k neighbors
         mean[i] += x1[(int)knn_index[knn_row + j]*3 + i];
        }
        mean[i] = mean[i]/k;
      }
      
      // determine covariance components
      int count = 0; // counting variable for covariance matrix components
      for(int i=0; i<3; i++){
        for(int j=i; j<3; j++){
          for(int l = 0; l<k; l++){
            covout[tid*6 + count] +=((x1[(int)knn_index[knn_row + l]*3 + i] - mean[i])*(x1[(int)knn_index[knn_row + l]*3 + j] - mean[j]))/(k-1); 
          }
          count += 1;
        }
      }
    }
    
    }
                          ''','knn_covar')
                          

                          
mean_surface_spread = cp.RawKernel(r'''
extern "C" 
__global__ void mean_surface_spread(double* data_in,
                               double* data_out,
                               double* knn_index,
                               double* knn_distance,
                               double step, // a scalar for the repulsive force
                               int k, // k specifies what degree we incorporate
                               int m, // size of data set
                               int n){
    //This algorithm will walk through a knn-search
    // then caluclate a small kick in the average direction
    // of the k-nn localizations
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    int row = tid*k;
    int data_row = tid*3;
    
    // *** KNN-SEARCH***
    float d = 0;
    if(tid < m){
    // Initialize array to -1
    // and high distance
    for(int i=0; i<k; i++){
      knn_index[row+i] = -1;
      knn_distance[row+i] = 100000;
      };

    for(int i =0; i<n; i++){
      //looping over x2 entries
      d = 0;
      for(int l=0; l<3;l++){
        d += (data_in[data_row + l] - data_in[i*3 + l])*(data_in[data_row + l] - data_in[i*3 + l]);
        }
      for(int l=0; l<k; l++){
        if(knn_distance[row + l] > d){
          if(d > -1){
            for(int j =k-1; j>l; j--){
              knn_distance[row + j] = knn_distance[row + j-1];
              knn_index[row + j] = knn_index[row + j-1];}
            knn_distance[row + l] = d;
            knn_index[row + l] = i;
            break;
          }
        }
      }
      // *** K-nn are found, calculate covariance matrix over neighbors
      // knn_index gives the k-nn

      double kick[3] = {0,0,0}; // summed kick for any molecule
      double covar[3][3] = {
          {0 ,0, 0},
          {0 ,0, 0},
          {0 ,0, 0}
          };
      double means[3] = {0.0, 0.0, 0.0};

      
      // We want to know what the mean positions are, from there we can calculate
      // the covariance matrix for pca
      for(int j =0; j<3;j++){ // loop over each axis
        for(int i=1; i<k; i++){ // loop over k-nearest neighbors
          means[j] += (data_in[(int)knn_index[row + i]*3 + j])/((double)k-1);
        }
      } 
      
      for(int j =0; j<3;j++){ // loop over each axis
        for(int l =0; l<3; l++){
          for(int i=1; i<k; i++){ // loop over k-nearest neighbors
            covar[j][l] += ((data_in[(int)knn_index[row + i]*3 + j] - means[j])*(data_in[(int)knn_index[row + i]*3 + l] - means[l]))/(k-1);
          }
        }
      } 
      
      data_out[data_row + 0] = covar[0][0];
      data_out[data_row + 1] = covar[1][0];
      data_out[data_row + 2] = covar[2][2];
    }}}
                          ''','mean_surface_spread')


def open_csv(file):
    with open(file) as f:
        text = f.read()
        lines = text.split('\n')
        data = np.zeros((len(lines),3))
        try:
            for i in range(len(lines)):
                split_line = lines[i].split(',')
                data[i,:] = np.array([float(split_line[0]),float(split_line[1]),float(split_line[2])])
        except:
            pass
        return data

if __name__ == '__main__':
    file = 'quick_data.csv'
    data = open_csv(file)
    
    # data = np.array([[1.2, 0,0],
    #                   [1.3, 0.1,0],
    #                   [1.4, 0.2, 0],
    #                   [1.5,0,0],
    #                   [1.6,0,0],
    #                   [0.9,0,0],
    #                   [0.8,0,0],
    #                   [0.7,0,0]])
    
    data2 = data*0 + data 
    k = 4
    nn_array = np.zeros((data.shape[0],k))
    # This code will search the nearest neighbors between data sets
    # There is an index point for every entry in data, which will be the
    # set of probe points into the data2 input.
    # c_nn_arr will be a (data.shape[0],k) array listing the k nearest neighbors from
    # data2 to probe point from data1
    covout = np.zeros((data.shape[0],6))
    
    c_data = cp.asarray(data)
    c_data2 = cp.asarray(data2)
    c_nn_ar = cp.asarray(nn_array)
    c_nn_dist = cp.asarray(nn_array*0)
    c_covout = cp.asarray(covout)
    #mean_surface_spread((1024,),( data.shape[0]//1024 + 1,),(c_data,c_data2,c_nn_ar,c_nn_dist,0.1,k, data.shape[0],data2.shape[0]))
    knn_gpu_search((1024,),( data.shape[0]//1024 + 1,),(c_data,c_data2,c_nn_ar,c_nn_dist,k, data.shape[0],data2.shape[0]))
    knn_covar((1024,),( data.shape[0]//1024 + 1,),(c_data,c_covout,c_nn_ar,k, data.shape[0]))
    nn_array = np.round(cp.asnumpy(c_nn_ar)).astype(np.int)
    nn_dist = np.power(cp.asnumpy(c_nn_dist),0.5)
    data2 = cp.asnumpy(c_data2)
    covout = cp.asnumpy(c_covout)
    
    # At this point we have Knn index and distance
    # As well as a supposed covariance matrix
    # We can test these values here to verify the proper mathematics
    # Verified w/ matlab that the nearest neighbor algorithm was working correctly
    # now using the indecies to calculate the covariance matrices 
    # index = 30
    # this_mat = np.cov(data[nn_array[index,:],:].transpose())
    # comp = covout[index,:]
    # summ = 0
    # lol_what = { 0: [0,0],
    #              1: [0,1],
    #              2: [0,2],
    #              3: [1,1],
    #              4: [1,2],
    #              5: [2,2], }
    # for ii in range(6):
    #     [i,j] = lol_what[ii]
    #     summ = summ + comp[ii] - this_mat[i,j]
    # print(summ)  # gives 1.3552527156068805e-20
        
