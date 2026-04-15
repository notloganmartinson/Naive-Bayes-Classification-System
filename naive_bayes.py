import csv
import math
import random
from copy import deepcopy

class NaiveBayesClassifier:
    def __init__(self):
        # structure to hold all counts needed to compute probabilities
        self.class_counts = {}
        self.feature_counts = {}
        self.total_instances = 0
        self.attributes = []
        self.class_values = []
        self.is_trained = False

    def load_metadata(self, meta_file):
        # reads metadata file lisitng attr and their possible values, ending with classification
        self.attributes = []
        try:
            with open(meta_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(':')
                    if len(parts) == 2:
                        # extract list seperated by commas for each attribute
                        values = [v.strip() for v in parts[1].split(',')]
                        self.attributes.append(values)
            if self.attributes: #final item in metadata file is the classification
                self.class_values = self.attributes.pop()
            return True
        except Exception as e:
            print(f"Error reading metadata: {e}")
            return False

    def train(self, meta_file, train_file):
        if not self.load_metadata(meta_file): # load metadata to understand structure of data
            return False

        self.class_counts = {cv: 0 for cv in self.class_values} # init counts to zero before processing training data
        self.feature_counts = {cv: [{val: 0 for val in attr} for attr in self.attributes] for cv in self.class_values}
        self.total_instances = 0

        try:
            with open(train_file, 'r') as f: #system trained based on training data
                reader = csv.reader(f)
                for row in reader:
                    if not row:
                        continue
                    label = row[-1].strip() # final item on each line is the classification label
                    features = [v.strip() for v in row[:-1]] # the rest are the comma seperated attr
                    if label in self.class_counts:
                        self.class_counts[label] += 1 # inc count for observed class
                        self.total_instances += 1 # inc count for each observed feature value given the class
                        for i, feat_val in enumerate(features):
                            if feat_val in self.feature_counts[label][i]:
                                self.feature_counts[label][i][feat_val] += 1
            self.is_trained = True
            print("Training complete.")
            return True
        except Exception as e:
            print(f"Error reading training data: {e}")
            return False

    def predict(self, features):
        best_class = None
        max_prob = -float('inf')

        for cv in self.class_values: # laplacian smoothing
            prob = math.log(self.class_counts[cv] + 1) - math.log(self.total_instances + len(self.class_values))
            
            for i, feat_val in enumerate(features):
                if feat_val in self.feature_counts[cv][i]:
                    count = self.feature_counts[cv][i][feat_val]
                else:
                    count = 0
                
                num_attr_values = len(self.attributes[i])
                #calc conditional prob
                prob += math.log(count + 1) - math.log(self.class_counts[cv] + num_attr_values)
            
            if prob > max_prob: #track class with highest probability
                max_prob = prob
                best_class = cv

        return best_class

    def classify_file(self, input_file, output_file): # system read set of data and provide classifications
        if not self.is_trained:
            print("Model must be trained first.")
            return

        try:
            with open(input_file, 'r') as fin, open(output_file, 'w', newline='') as fout:
                reader = csv.reader(fin)
                writer = csv.writer(fout)
                for row in reader:
                    if not row:
                        continue
                    
                    if row[-1].strip() in self.class_values:
                        features = [v.strip() for v in row[:-1]]
                    else:
                        features = [v.strip() for v in row]

                    predicted = self.predict(features)
                    features.append(predicted)
                    writer.writerow(features)
            print(f"Classification complete. Output saved to {output_file}")
        except Exception as e:
            print(f"Error processing file: {e}")

    def evaluate(self, test_file):
        if not self.is_trained:
            print("Model must be trained first.")
            return

        correct = 0
        total = 0
        confusion_matrix = {actual: {pred: 0 for pred in self.class_values} for actual in self.class_values} #dictionary for confusion matrix

        try:
            with open(test_file, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row:
                        continue
                    actual_label = row[-1].strip()
                    features = [v.strip() for v in row[:-1]]
                    
                    if actual_label in self.class_values:
                        predicted = self.predict(features)
                        confusion_matrix[actual_label][predicted] += 1
                        total += 1
                        if predicted == actual_label:
                            correct += 1

            accuracy = correct / total if total > 0 else 0
            print(f"\nAccuracy: {accuracy:.4f} ({correct}/{total})")
            
            print("\nConfusion Matrix (Row=Actual, Col=Predicted):")
            header = "Actual \\ Pred\t" + "\t".join(self.class_values)
            print(header)
            for actual in self.class_values:
                row_str = f"{actual}\t\t"
                for pred in self.class_values:
                    row_str += f"{confusion_matrix[actual][pred]}\t"
                print(row_str)

            print("\nPerformance Metrics:") # calc and print precision, recal, F-measure for test data
            for cv in self.class_values:
                tp = confusion_matrix[cv][cv]
                fp = sum(confusion_matrix[actual][cv] for actual in self.class_values if actual != cv)
                fn = sum(confusion_matrix[cv][pred] for pred in self.class_values if pred != cv)

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f_measure = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                print(f"Class: {cv}")
                print(f"  Precision: {precision:.4f}")
                print(f"  Recall:    {recall:.4f}")
                print(f"  F-measure: {f_measure:.4f}")

        except Exception as e:
            print(f"Error testing data: {e}")

def cross_validation(meta_file, data_file, k):
    classifier = NaiveBayesClassifier() # completelet independent for user
    if not classifier.load_metadata(meta_file):
        return

    data = []
    try:
        with open(data_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    data.append(row)
    except Exception as e:
        print(f"Error reading data file: {e}")
        return

    random.shuffle(data)
    fold_size = len(data) // k
    accuracies = []

    for i in range(k):
        start_idx = i * fold_size
        end_idx = start_idx + fold_size if i < k - 1 else len(data)
        
        test_fold = data[start_idx:end_idx]
        train_fold = data[:start_idx] + data[end_idx:]

        classifier.class_counts = {cv: 0 for cv in classifier.class_values} # reset coutns for the new fold training
        classifier.feature_counts = {cv: [{val: 0 for val in attr} for attr in classifier.attributes] for cv in classifier.class_values}
        classifier.total_instances = 0

        for row in train_fold: # train on k-1 folds
            label = row[-1].strip()
            features = [v.strip() for v in row[:-1]]
            if label in classifier.class_counts:
                classifier.class_counts[label] += 1
                classifier.total_instances += 1
                for j, feat_val in enumerate(features):
                    if feat_val in classifier.feature_counts[label][j]:
                        classifier.feature_counts[label][j][feat_val] += 1
        
        correct = 0
        for row in test_fold: # test on the remaining 1 fold
            label = row[-1].strip()
            features = [v.strip() for v in row[:-1]]
            pred = classifier.predict(features)
            if pred == label:
                correct += 1
        
        acc = correct / len(test_fold) if len(test_fold) > 0 else 0
        accuracies.append(acc)
        print(f"Fold {i+1} Accuracy: {acc:.4f}")

    avg_acc = sum(accuracies) / k
    print(f"\nAverage Accuracy for {k}-fold Cross-Validation: {avg_acc:.4f}")

def main():
    nb = NaiveBayesClassifier()
    
    while True:
        print("\n--- Naive Bayes Classifier Menu ---")
        print("1. Train the model")
        print("2. Classify a dataset")
        print("3. Test model accuracy & print confusion matrix")
        print("4. K-fold cross-validation")
        print("5. Quit")
        
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == '1':
            meta = input("Enter metadata filename (e.g., car.meta): ").strip()
            train_file = input("Enter training filename (e.g., car.train): ").strip()
            nb.train(meta, train_file)
            
        elif choice == '2':
            if not nb.is_trained:
                print("Please train the model first (Option 1).")
                continue
            in_file = input("Enter input data filename: ").strip()
            out_file = input("Enter output filename: ").strip()
            nb.classify_file(in_file, out_file)
            
        elif choice == '3':
            if not nb.is_trained:
                print("Please train the model first (Option 1).")
                continue
            test_file = input("Enter test filename (e.g., car.test): ").strip()
            nb.evaluate(test_file)
            
        elif choice == '4':
            meta = input("Enter metadata filename (e.g., car.meta): ").strip()
            data = input("Enter data filename for CV: ").strip()
            try:
                k = int(input("Enter number of folds (k): ").strip())
                cross_validation(meta, data, k)
            except ValueError:
                print("Invalid input for k. Must be an integer.")
                
        elif choice == '5':
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
