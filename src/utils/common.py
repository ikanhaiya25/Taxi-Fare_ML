import pickle

def save_object(path,obj):
    with open(path,"wb") as file:
        pickle.dump(obj,file)


def load_object(path):
    with open(path,"rb") as file:
        return pickle.load(file)