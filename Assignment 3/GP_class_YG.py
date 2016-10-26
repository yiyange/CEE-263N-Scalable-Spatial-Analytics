class GP_RainFall:

    def __init__(self,trn_data,tst_data,h): 
        # h is distance between x and x' (bandwidth)
        self.X_trn_raw, self.Y_trn_raw = trn_data[:,:-1], trn_data[:,-1:]
        self.X_tst_raw = tst_data
        self.h = h
        self.K_tsttst = self.covariance(self.ConvertUnit(self.X_tst_raw),self.ConvertUnit(self.X_tst_raw))
        self.K_trntst = self.covariance(self.ConvertUnit(self.X_trn_raw),self.ConvertUnit(self.X_tst_raw))
        self.K_tsttrn = self.covariance(self.ConvertUnit(self.X_tst_raw),self.ConvertUnit(self.X_trn_raw))
        self.K_trntrn = self.covariance(self.ConvertUnit(self.X_trn_raw),self.ConvertUnit(self.X_trn_raw))
        self.I = np.identity(self.K_trntrn.shape[0])
        

    def ConvertUnit(self,array):
        array_list = []
        for n in array:
            meters = utm.from_latlon(n[0],n[1])
            array_list.append(np.array([meters[0], meters[1]]))
        converted_array = np.vstack(array_list)
        return converted_array
        
    #Computing Gaussian covariance:
    def covariance(self, X, Z):
        d = spatial.distance_matrix(X,Z) 
        K = np.exp(-(d**2) / (2*self.h*self.h)) 
        return K

    # Make Predictions
    # predicted mean: m(f)=K(xtest,x)[K+σ^2I−1]^(-1)y
    # cov(f)=K(xtest,xtest)−K(xtest,x)[K+σ2I]^(−1)K(x,xtest)

    def predict(self, sigma, X_trn_raw, X_tst_raw, Y_trn_pre):
        X_tst = self.ConvertUnit(X_tst_raw)
        X_trn = self.ConvertUnit(X_trn_raw)
        
        K_tsttrn = self.covariance(X_tst,X_trn)
        K_trntrn = self.covariance(X_trn,X_trn)
        
        I = np.identity(K_trntrn.shape[0])
        
        mean = np.mean(Y_trn_pre)
        Y_trn = Y_trn_pre - mean*np.ones(Y_trn_pre.shape)
        
        pred_mean = np.dot(np.dot(K_tsttrn,inv(K_trntrn + sigma**2*I)),Y_trn)
        pred_Y = pred_mean + mean*np.ones(pred_mean.shape) # Re-add the mu to get the accurate value
        
        return pred_Y
    
    def predict_cv(self, k, sigma):
        kf = KFold(len(data), n_folds = k, shuffle=True)
        RMSE = []
        for train_index, test_index in kf:
            X_trn_cv, X_tst_cv = self.X_trn_raw[train_index], self.X_trn_raw[test_index]
            Y_trn_cv, Y_tst_cv = self.Y_trn_raw[train_index], self.Y_trn_raw[test_index]
            Y_pred = self.predict(sigma, X_trn_cv, X_tst_cv, Y_trn_cv)
            Y_true = Y_tst_cv
            error = sqrt(mean_squared_error(Y_pred, Y_true))
            RMSE.append(error)
        
        self.RMSE = np.mean(RMSE)
        return self.RMSE
    
    def simulation(self, sigma):
        m_f = self.predict(sigma, self.X_trn_raw, self.X_tst_raw, self.Y_trn_raw) # Get m(f), K_tsttrn, K_trntrn and I

        cov = self.K_tsttst - np.dot(np.dot(self.K_tsttrn,
                            inv(self.K_trntrn + sigma**2*self.I)), self.K_trntst)
        L = np.linalg.cholesky(cov + 0.001*np.eye(cov.shape[0])) #gamma~0.001
        u = np.random.normal(0,1,cov.shape[0])
        f_sim = m_f.reshape(-1,) + np.dot(L,u)
        return f_sim, cov
