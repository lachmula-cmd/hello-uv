import numpy as np

class EmbeddingModel:
    def __init__(self, name):
        self.name = name

    def embed(self, X):
        try:
            raise NotImplementedError("Not implemented")

        except:
            raise

class IdentityModel(EmbeddingModel):
    def __init__(self):
        super().__init__("identity")
        

    def embed(self, X):
        return X

class MeanCentreModel(EmbeddingModel):
    def __init__(self):
            super().__init__("mean-centre")

    def embed(self, X):
        x_mean = np.mean(X, axis=0)
        print(X.shape)
        return X - x_mean

class RandomProjectionModel(EmbeddingModel):
    def __init__(self, dim=0, seed=0):
            super().__init__(f"random-{dim}")
            self.dim = dim
            self.seed = seed

    def embed(self, X):
        W = np.random.default_rng(self.seed).normal(size=(3,self.dim))
        return X @ W

def embed_all (models,X):
     model_dict = {}
     for model in models:
         r = model.embed(X)
         model_dict[model.name] = r.shape
     return model_dict
    
          
         

     

        

