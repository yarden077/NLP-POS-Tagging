from preprocessing import read_test, represent_input_with_features
from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def memm_viterbi(sentence, pre_trained_weights, feature2id):
    """
    Write your MEMM Viterbi implementation below
    You can implement Beam Search to improve runtime
    Implement q efficiently (refer to conditional probability definition in MEMM slides)
    @param pre_trained_weights: numpy array (vector) of weights.
    @param feature2id:contains the tags and the features and the counts and more.

    """
    sentence.append("~")
    n = len(sentence)
    num_tags = len(feature2id.feature_statistics.tags)
    tags_list = sorted(list(feature2id.feature_statistics.tags))
    pi_prev = np.zeros([num_tags, num_tags])
    pi_prev[tags_list.index('*')][tags_list.index('*')] = 1
    back_pointers_list = np.zeros((n, num_tags, num_tags), dtype=int)
    beam_width = 2
    relevant_tags = [[tags_list.index('*'), tags_list.index('*')]]
    for k in range(2, n - 1):
        visited_u = []
        pi_current = np.zeros([num_tags, num_tags])
        for i in range(len(relevant_tags)):
            u = relevant_tags[i][1]
            if u in visited_u:
                continue
            visited_u.append(u)
            for v in range(num_tags):
                max_value = 0
                max_index = None
                for j in range(len(relevant_tags)):
                    if relevant_tags[j][1] == u:
                        t = relevant_tags[j][0]
                        current_value = (pi_prev[t][u] * np.exp(np.sum(pre_trained_weights[
                                                                         represent_input_with_features(
                                                                             (
                                                                                 sentence[k], tags_list[v],
                                                                                 sentence[k - 1],
                                                                                 tags_list[u], sentence[k - 2],
                                                                                 tags_list[t], sentence[k + 1]),
                                                                             feature2id.feature_to_idx
                                                                         )])) /
                                 np.sum(
                                           np.exp(np.sum(pre_trained_weights[
                                                             represent_input_with_features(
                                                                 (sentence[k], tags_list[y], sentence[k - 1],
                                                                  tags_list[u], sentence[k - 2], tags_list[t],
                                                                  sentence[k + 1]),
                                                                 feature2id.feature_to_idx)]
                                                         )) for y in range(num_tags)
                                       ))
                        if current_value >= max_value:
                            max_value = current_value
                            max_index = t

                # Store the maximum value and its index in back_pointers_list
                back_pointers_list[k - 2, u, v] = max_index
                pi_current[u][v] = max_value

        # Flatten the matrix and get the indices of the top Beam_Width elements
        flat_indices = np.argsort(pi_current.flatten())[-beam_width:]

        # Updating relevant_u to include only the top possible part of speech tags for u.
        relevant_t, relevant_u = np.unravel_index(flat_indices, pi_current.shape)
        relevant_tags = np.column_stack((relevant_t, relevant_u))

        pi_prev = pi_current.copy()
    # Find the index of the maximum element along axis 0 (column-wise)
    col_index = np.argmax(pi_current, axis=1)[0]

    # Find the index of the maximum element along axis 1 (row-wise)
    row_index = np.argmax(pi_current, axis=0)[0]
    list_to_return = []

    # Building the list of tags the function returns
    for i in range(n - 2, 1, -1):
        list_to_return.append(tags_list[row_index])
        max_row_index = back_pointers_list[i - 2, int(row_index), int(col_index)]
        col_index = row_index
        row_index = max_row_index
    list_to_return.append(tags_list[col_index])
    list_to_return.append(tags_list[row_index])
    list_to_return.reverse()
    return list_to_return[2:]


def tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path):
    tagged = "test" in test_path
    test = read_test(test_path, tagged=tagged)

    output_file = open(predictions_path, "a+")
    # Calculating Accuracy
    total_correct = 0
    total = 0
    tags_list = sorted(list(feature2id.feature_statistics.tags))
    tags_dict = {tag: 0 for tag in tags_list}
    tags_dict['$'] = 0
    confusion_matrix = np.zeros((len(tags_list), len(tags_list)))

    for k, sen in tqdm(enumerate(test), total=len(test)):
        sentence = sen[0]
        pred = memm_viterbi(sentence, pre_trained_weights, feature2id)[1:]
        sentence = sentence[2:]

        # Calculating the accuracy of model and top 10 wrong tags
        #temp_correct, temp_total, confusion_matrix = calculate_accuracy(pred, sen[1][2:], tags_dict, tags_list, confusion_matrix)
        #total_correct += temp_correct
        #total += temp_total

        for i in range(len(pred)):
            if i > 0:
                output_file.write(" ")
            output_file.write(f"{sentence[i]}_{pred[i]}")
        output_file.write("\n")

     #Printing accuracy of model
   # print(f"Accuracy: {(total_correct / total) * 100}\n")

    # Sorting the dictionary by its values (number of wrong predictions)
    #sorted_tags_dict = dict(sorted(tags_dict.items(), key=lambda item: item[1], reverse=True))
    #top_10_wrong_tags = list(sorted_tags_dict.keys())[:10]
    #print(f"Top 10 tags that the model is wrong about:\n {top_10_wrong_tags}")
    #display_confusion_matrix(confusion_matrix, top_10_wrong_tags, tags_list)

    output_file.close()


def calculate_accuracy(pred_list, true_tags_list, tags_dict, tags_list, confusion_matrix):
    """
    Calculates the accuracy and counting how many times the prediction was wrong for every tag
    @param confusion_matrix: This matrix counts how many times model got wrong for tuple of tags
    @param tags_list: a list that contains all possible tags
    @param pred_list: prediction tags from Viterbi
    @param tags_dict: a dict that contains tags as keys and number of times the model got wrong on the specific key as values.

    """
    correct = 0
    for pred, true in zip(pred_list, true_tags_list):
        if pred != true:
            tags_dict[true] += 1
            confusion_matrix[tags_list.index(true)][tags_list.index(pred)] += 1
        else:
            correct += 1
    total = len(pred_list)

    return correct, total, confusion_matrix


def display_confusion_matrix(confusion_matrix, top_10_wrong_tags, tags_list):
    """
    This function displays the confusion matrix.

    """
    # Create DataFrame
    df = pd.DataFrame(confusion_matrix, columns=tags_list, index=tags_list)
    filtered_df = df.loc[top_10_wrong_tags]

    # Keeping only tags that their corresponding column is not all zeros
    non_zero_cols = filtered_df.columns[(filtered_df != 0).any()]
    filtered_df = filtered_df[non_zero_cols]

    # Convert DataFrame to integers
    filtered_df = filtered_df.astype(int)
    # Create heatmap
    plt.figure(figsize=(13, 10))
    sns.heatmap(filtered_df, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xticks(rotation=0)  # Rotate x-axis labels by 90 degrees
    plt.yticks(rotation=0)  # Keep y-axis labels horizontal
    plt.xlabel('Predicted Labels', fontweight='bold')
    plt.ylabel('True Labels', fontweight='bold')
    plt.title('Confusion Matrix', fontsize=30, fontweight='bold', color='Blue')
    plt.show()

