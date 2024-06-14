import os
import pickle
from preprocessing import preprocess_train
from optimization import get_optimal_vector
from inference import tag_all_test
import time
from sklearn.model_selection import train_test_split


# TODO  don't forget to run this code on machine! !!!!!

def main():
    threshold = 1
    lam = 0.3

    #split_file('data/train2.wtag')

    #return

    train_path = "data/train1.wtag"
    test_path = "data1/test1"

    weights_path = 'weights.pkl'
    predictions_path = 'Mydata/predictions_our_comp2.wtag' # todo Notice path

    # Start time
    start_time = time.time()
    statistics, feature2id = preprocess_train(train_path, threshold)
    get_optimal_vector(statistics=statistics, feature2id=feature2id, weights_path=weights_path, lam=lam)

    # End time
    end_time = time.time()
   #
    with open(weights_path, 'rb') as f:
        optimal_params, feature2id = pickle.load(f)
    pre_trained_weights = optimal_params[0]
# saving the pre_trained for train 1:
    #with open('pre_trained_weights.pkl', 'wb') as f:
     #  pickle.dump(pre_trained_weights, f)

    #Load pre_trained_weights from the file
   # with open('pre_trained_weights.pkl', 'rb') as f:
    #    pre_trained_weights = pickle.load(f)

    print(pre_trained_weights)
    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print("Training Time of Model 1:", elapsed_time, "seconds")

    tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path)

"""
In this function we split the train2 into test and train.
def split_file(file_path):
    data = []
    with open(file_path) as file:
            for line in file:
                if line[-1:] == "\n":
                    line = line[:-1]
                data.append(line)

    train, test = train_test_split(data, test_size=0.25, random_state=3)
    # Create the directory if it does not exist
    if not os.path.exists("Mydata"):
        os.makedirs("Mydata")

    f = open("Mydata/train.wtag","w")
    for line in train:
        f.write(line + "\n")
    f.close()

    f = open("Mydata/test.wtag","w")
    for line in test:
        f.write(line + "\n")
    f.close()
    """

if __name__ == '__main__':
    main()
