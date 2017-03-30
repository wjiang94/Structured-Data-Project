import scipy.io
import numpy as np
from numpy import linalg as LA
import time

##########################################################################################################################


def okl(K,Y,lam_list):
    
    """
    OKL with block coordinate descent
    
    model = okl(K,Y,lam_list)
    -------------------------------------------   
    Inputs:
        K: input kernel matrix: (l x l) matrix,
        Y: training outputs: (l x m) matrix.
        lam_list: list of regularization parameters.
        
    --------------------------------------------
    Output:
        model : output model containing the following information:
        L : output kernel
        C : coefficient matrix
        lambda : regularization parameter
        J : value of the objective functional
        time : execution time          
    """

    #constants
    l = K.shape[0]
    m = Y.shape[1]
    N = len(lam_list)
    MAX_ITER = 1000
    TOL = 0.001
    delta = TOL * LA.norm(Y)

    # initialization 
    J = np.zeros((MAX_ITER,1))
    L = np.eye(m)
    C = np.zeros((l,m))

    # eigendecomposition of the input kernel matrix
    DX, UX = LA.eigh(K)
    '''
    For Symetric Real matrix :
    LA.eig -> complex space (due to computation issue)
    LA.eigh -> real space
    '''
    DX_ = DX.reshape((len(DX),1)) 
    dx = abs(DX_)
    '''
    The eigenvalues may be negative -> abs
    '''
    DX = np.diag(DX)
    Ytilde = np.dot(UX.T,Y)

    # MAIN loop
    model={}
    for k in range(N):
        start_time = time.time()
        lam = lam_list[k]
        print("lambda = "+str(lam))
        nit = 0
        res = LA.norm(Y)
    
        while(res > delta):
            # Sub-problem w.r.t. C.
            # Solve the Sylvester equation KCL+lambda*C = Y using eigendecomposition of K and L. 
            DY, UY = LA.eigh(L)
            DY_ = DY.reshape((len(DY),1))
            dy = abs(DY_)
            DY = np.diag(DY)
            Q = np.dot(Ytilde,UY)
            V = Q / (np.dot(dx,dy.T)+lam) #element wise division
            C = np.dot(np.dot(UX,V),UY.T)
        
            # Sub-problem w.r.t. L
            F = np.dot(V,UY.T)
            E = np.dot(DX,F)
            R = np.dot(E.T,E)
            DE, UE = LA.eigh(R)        
            DE_ = DE.reshape((len(DE),1))
            dep = abs(DE_)+lam
            DE = np.diag(DE)
        
            Lp = L
            temp = np.dot(R,L) + np.dot(L.T,R.T) + lam*np.dot(E.T,F)
            P = np.dot(np.dot(UE.T,temp),UE)
            temp = np.dot(dep,np.ones((1,m))) + np.dot(np.ones((m,1)),dep.T)
            L =np.dot(np.dot(UE, P/temp), UE.T)
        
            # Compute the value of the objective functional
            temp = F / 4 - Ytilde / (2*lam)
            J[nit] = LA.norm(Y)**2 / (2*lam) + np.trace(np.dot(np.dot(temp.T,E),L))
            #Compute the variation of L
            res = LA.norm(L-Lp)
        
            #Check whether the maximum number of iterations has been reached
            if nit >= MAX_ITER:
                print('Reached maximum number of iterations')
                break
            
            nit += 1
        modelk={}
        modelk['L']=L
        modelk['C']=C
        modelk['nit']=nit
        modelk['lambda']=lam
        modelk['J']=J[:nit]
        modelk['time']=time.time() - start_time
        model[k]=modelk
    return model


##########################################################################################################################


def lrokl(K,Y,lam_list,p):
    
    """
    Low rank OKL with block coordinate descent
    
    model = lrokl(K,Y,lam_list,p)
    -------------------------------------------   
    Inputs:
        K: input kernel matrix: (l x l) matrix,
        Y: training outputs: (l x m) matrix.
        lam_list: list of regularization parameters.
        p : rank parameter
    --------------------------------------------
    Output:
        model : output model containing the following information:
        B : linear operators
        A : coefficient matrix
        lambda : regularization parameter
        J : value of the objective functional
        time : execution time          
    """

    #constants
    l = K.shape[0]
    m = Y.shape[1]
    N = len(lam_list)
    MAX_ITER = 1000
    TOL = 0.001
    delta = TOL * LA.norm(Y)

    # initialization 
    J = np.zeros((MAX_ITER,1))
    B = np.eye(m,p)

    # eigendecomposition of the input kernel matrix
    DX, UX = LA.eigh(K)
    '''
    For Symetric Real matrix :
    LA.eig -> complex space (due to computation issue)
    LA.eigh -> real space
    '''
    DX_ = DX.reshape((len(DX),1)) 
    dx = abs(DX_)
    '''
    The eigenvalues may be negative -> abs
    '''
    DX = np.diag(DX)
    Ytilde = np.dot(UX.T,Y)

    # MAIN loop
    model={}
    for k in range(N):
        start_time = time.time()
        lam = lam_list[k]
        #print("lambda = "+str(lam))
        nit = 0
        res = LA.norm(Y)
    
        while(res > delta):
        # Sub-problem w.r.t. A.
            # Solve the Sylvester equation KAB'B+lambda*A = YB 
            # via eigendecomposition of K and B'B
            DY, UY = LA.eigh(np.dot(B.T,B))
            DY_ = DY.reshape((len(DY),1))
            dy = abs(DY_)
            DY = np.diag(DY)
            Q = np.dot(np.dot(Ytilde,B),UY)
            V = Q / (np.dot(dx,dy.T)+lam) #element wise division
        
            # Sub-problem w.r.t. B
            Bp = B
            E = np.dot(DX,V)
            temp = LA.inv(np.dot(E.T,E)+lam*np.eye(p))
            P = np.dot(np.dot(Ytilde.T,E),temp) 
            B = np.dot(P,UY.T)

            # Compute the value of the objective functional
            if l > m:
                temp =np.dot((Ytilde - np.dot(E,P.T)).T,Ytilde)/lam
            else:
                temp =(np.dot(Ytilde,(Ytilde - np.dot(E,P.T)).T))/lam
            J[nit] = (np.trace(temp) + np.trace(np.dot(V.T,E)))/2
            
            # Compute the increment of B
            res = LA.norm(B-Bp)
        
            #Check whether the maximum number of iterations has been reached
            if nit >= MAX_ITER:
                print('Reached maximum number of iterations')
                break
            nit += 1
        A = np.dot(np.dot(UX,V),UY.T)
        modelk={}
        modelk['A']=A
        modelk['B']=B
        modelk['nit']=nit
        modelk['lambda']=lam
        modelk['J']=J[:nit]
        modelk['time']=time.time() - start_time
        model[k]=modelk
    return model


##########################################################################################################################


def mse(Y,Y_pred):
    return np.mean((Y-Y_pred)**2)

def predict(K,C,L):
    return np.dot(np.dot(K,C),L)