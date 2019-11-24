import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from sklearn.utils.extmath import randomized_svd
from numpy.linalg import multi_dot
from math import sqrt
import time

class CUR:
    def __init__(self,matrix):
        np.random.seed(11)
        self.matrix = matrix
        self.C = []
        self.R = []
        self.W = []
        self.U = []
        self.row_probability = []
        self.column_probability = []
    
    def init_probabilities(self,matrix):
        frob = np.sum(np.square(matrix))
        # print(frob) # got whole matrix's frobenious Norm
        # print(matrix.shape)

        for row_index in tqdm(range(matrix.shape[0])):
            self.row_probability.append( np.sum( np.square(matrix[row_index]) ) / frob )
        
        matrix_t = np.transpose(matrix) #done since column sum is too slow
        for row_index in tqdm(range(matrix_t.shape[0])):
            self.column_probability.append( np.sum( np.square(matrix_t[row_index] ) ) / frob)

        # print(self.row_probability)
        # print(self.column_probability)
        # print(np.sum(self.row_probability))
        # print(np.sum(self.column_probability))

    def create_CR(self,r):
        C_columns = np.random.choice(self.matrix.shape[1], r, p=self.column_probability)
        R_rows = np.random.choice(self.matrix.shape[0], r, p=self.row_probability)
#choosing r columns to C and r Rows to R from their dimension length and probabilities
        # print(C_columns)
        # print(R_rows)



        # # Delete this part after wards just debugging
        # C_columns = [1,3]
        # R_rows = [5,3]
        # r=2





        # print(np.transpose(self.matrix)[C_columns]) # because normal column access is slow
        # print(self.matrix[R_rows])
        # for i in C_columns: # normal method to print
            # print(self.matrix[ : ,i])
        
        for i in range(len(C_columns)):
            # print(self.column_probability[C_columns[i]])
            # print(sqrt(r*self.column_probability[C_columns[i]]))
            self.C.append(np.transpose(self.matrix)[C_columns[i]]/sqrt(r*self.column_probability[C_columns[i]]))
        self.C = np.transpose(self.C)
        # print(self.C.shape)
        # print(self.C)


        # print(self.matrix[R_rows])
        # self.R = self.matrix[R_rows]
        # print(self.R.shape)

        for i in range(len(R_rows)):
            self.R.append(self.matrix[R_rows[i]]/sqrt(r*self.row_probability[R_rows[i]]))
            # self.R.append(self.matrix[R_rows[i]])
        # print(self.R)
        self.R = np.array(self.R)
        # print(self.R.shape)
        # print(self.R)

        for i in R_rows:
            self.W.append(self.matrix[i][C_columns])
    
        self.W = np.array(self.W)
        # print(self.W.shape)
        # print(self.W)

    def create_U(self,r):
        X, Sigma, YT = randomized_svd(self.W, 
                                n_components=2, #number of singlar values in SVD of W
                                n_iter=1,
                                random_state=None)
        # print(X.shape)
        # print(Sigma.shape)
        # print(YT.shape)
        # print(X)
        # print(YT)
        # print(Sigma)
        # exit(1)
        # print("\n")
        # print(Sigma.shape)
        # print(np.matmul(X,Sigma,YT))
        # print("\n")
        # print(self.W)
        # print("\n")
        # print(Sigma)
        SigmaInv_values = np.array([1/i if i!=0 else 0 for i in Sigma ])
        print(SigmaInv_values)
        # print("\n\n")
        # print(X)
        SigmaInv = np.zeros((X.shape[1],X.shape[1]),float)
        # print(SigmaInv)
        np.fill_diagonal(SigmaInv,SigmaInv_values)
        # print(SigmaInv)
        # exit(1)
        # print("\n")
        # print(YT)
        # print("\n")
        # print("\n")
        # print(SigmaInv.shape)
        # self.U = np.matmul(np.matmul(np.transpose(YT),np.square(SigmaInv))),np.transpose(X)))
        self.U = multi_dot([np.transpose(YT),np.square(SigmaInv),np.transpose(X)])
        # print(self.U.shape)
        # print(self.U)
    
    def MSE(self):
        reConsMatrix = multi_dot([self.C,self.U,self.R])
        # print(reConsMatrix.shape)
        # print(self.matrix.shape)
        # print(reConsMatrix)
        # exit(1)
        row,column = self.matrix.shape
        # print(row)
        # print(column)
        print((np.sum(np.square(reConsMatrix-self.matrix))))
        print(sqrt((np.sum(np.square(reConsMatrix-self.matrix)))/(row*column)))



        
        
    


if __name__=="__main__" :
    # print("In main")
    data_matrix = pd.read_pickle("ratingsMatrix_noZeros.pickle")
    # data_matrix = np.array([[1,1,1,0,0],[3,3,3,0,0],[4,4,4,0,0],[5,5,5,0,0],[0,0,0,4,4],[0,0,0,5,5],[0,0,0,2,2]])
    # print(type(data_matrix))
    # exit(1)
    start = time.time()
    test = CUR(data_matrix)
    # exit(1)
    test.init_probabilities(data_matrix)
    # exit(1)
    r = 1000
    test.create_CR(r)
    # exit(1)
    test.create_U(r)
    # exit(1)
    test.MSE()
    end = time.time()
    print(end-start)
