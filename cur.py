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

        for row_index in tqdm(range(matrix.shape[0])):
            self.row_probability.append( np.sum( np.square(matrix[row_index]) ) / frob )
        
        matrix_t = np.transpose(matrix) #done since column sum is too slow
        for row_index in tqdm(range(matrix_t.shape[0])):
            self.column_probability.append( np.sum( np.square(matrix_t[row_index] ) ) / frob)


    def create_CR(self,r):
        if r>self.matrix.shape[1]:
            raise Exception("R is too large")

        C_columns = np.random.choice(self.matrix.shape[1], r, p=self.column_probability)
        R_rows = np.random.choice(self.matrix.shape[0], r, p=self.row_probability)
        
        for i in range(len(C_columns)):
            self.C.append(np.transpose(self.matrix)[C_columns[i]]/sqrt(r*self.column_probability[C_columns[i]]))
        self.C = np.transpose(self.C)
    
        for i in range(len(R_rows)):
            self.R.append(self.matrix[R_rows[i]]/sqrt(r*self.row_probability[R_rows[i]]))
        self.R = np.array(self.R)

        for i in R_rows:
            self.W.append(self.matrix[i][C_columns])
        self.W = np.array(self.W)

    def create_U(self,r):
        X, Sigma, YT = randomized_svd(self.W, 
                                n_components=r, #number of non zero singlar values in SVD of W
                                n_iter=1,
                                random_state=None)
        # print(Sigma.shape)
        # print(X.shape)
        # print(YT.shape)
        # exit(1)
        total_energy = np.sum(np.square(Sigma))
        energy_removed = 0
        check_energy = 0
        for i,e in reversed(list(enumerate(Sigma))):
            energy_removed += e**2
            if ((total_energy-energy_removed)/total_energy)<0.9:
                print(i+1)
                break
        Sigma = Sigma[:i+1]
        for j in range(i+1):
            check_energy += Sigma[j]**2
        # print(check_energy/total_energy)
        # print(np.sum(np.square(Sigma))/total_energy)
        # print(Sigma.shape)
        XT = np.transpose(X)
        X = np.transpose(XT[:i+1])
        YT = YT[:i+1]
        # print(X.shape)
        # print(YT.shape)
        SigmaInv_values = np.array([1/i if i!=0 else 0 for i in Sigma ])
        SigmaInv = np.zeros((X.shape[1],X.shape[1]),float)
        np.fill_diagonal(SigmaInv,SigmaInv_values)
        self.U = multi_dot([np.transpose(YT),np.square(SigmaInv),np.transpose(X)])
    
    def RMSE(self):
        reConsMatrix = multi_dot([self.C,self.U,self.R])
        row,column = self.matrix.shape
        print("RMSE "+str(sqrt((np.sum(np.square(reConsMatrix-self.matrix)))/(row*column))))
        print("MAE "+str(np.sum(np.abs(reConsMatrix -self.matrix))/(row*column)))
        return (sqrt((np.sum(np.square(reConsMatrix-self.matrix)))/(row*column)),(np.sum(np.abs(reConsMatrix -self.matrix))/(row*column)))




if __name__=="__main__" :
    data_matrix = pd.read_pickle("ratingsMatrix_noZeros.pickle")
    # data_matrix = np.array([[1,1,1,0,0],[3,3,3,0,0],[4,4,4,0,0],[5,5,5,0,0],[0,0,0,4,4],[0,0,0,5,5],[0,0,0,2,2]])
    total_time = 0
    RMSE_err = 0
    MAE_err = 0
    r_list = [100,500,1000,1500,2000,2500,3500]
    for r in r_list:
        print("r: "+str(r))
        start = time.time()
        test = CUR(data_matrix)
        test.init_probabilities(data_matrix)
        test.create_CR(r)
        test.create_U(r)
        a,b = test.RMSE()
        RMSE_err += a
        MAE_err += b
        end = time.time()
        print("Time for this Iteration "+str(end-start))
        total_time += end - start
        print("\n")
        
    print("Average RMSE error "+str(RMSE_err/len(r_list)))
    print("Average MAE error "+str(MAE_err/len(r_list)))
    print("Time Taken is " +str(total_time/len(r_list)))
