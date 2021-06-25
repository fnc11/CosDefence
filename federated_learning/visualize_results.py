import json
import os
from sklearn import metrics

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

json_folder = os.path.join(base_path, 'json_files/')
file_name = json_folder + 'results_2021-04-27 10:20:03.txt'

with open(file_name) as result_file:
    result_data = json.load(result_file)
    print(result_data['model'])
    print(result_data['config'])
    print(result_data['training_losses'])
    print(result_data['testing_losses'])
    print(result_data['total_accuracies'])
    print(result_data['class_accuracies'])
    y_true = result_data['final_test_data']['ground_truths']
    y_pred = result_data['final_test_data']['predictions']
    # Print the confusion matrix
    print(metrics.confusion_matrix(y_true, y_pred))
    # Print the precision and recall, among other metrics
    print(metrics.classification_report(y_true, y_pred, digits=3))
