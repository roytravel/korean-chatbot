from matplotlib import pyplot as plt

class Visualizer:
    def __init__(self) -> None:
        pass
    
    def visualize(self, train_data, test_data) -> plt:
        """ NSMC 데이터셋 """
        print('max length of training sequence :', max(len(l) for l in train_data['document']))
        print('average length of training sequence :', sum(map(len, train_data['document']))/len(train_data['document']))
        plt.hist([len(s) for s in train_data['document']], bins=50)
        plt.xlabel('length of data')
        plt.ylabel('number of data')
        plt.show()