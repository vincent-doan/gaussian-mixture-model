import pandas as pd
from gmm import *
import util
import os

def iris_gmm(features_list, no_components):
    """
    Takes a list of features (string) and clusters the data using GMM
    Plots available when length of features_list is 1 or 2
    """
    # Obtain file path to dataset
    parent_directory = os.path.abspath(os.path.join(os.getcwd(), ".."))
    file_path = os.path.join(parent_directory, "data/iris-dataset/iris-data.csv")

    data = pd.read_csv(file_path)

    # For one feature
    if len(features_list) == 1:
        # Normalize data
        X = util.normalize(np.array(data[features_list]))

        # Filepath to save plot
        file_path = os.path.join(parent_directory, "results/iris-dataset/" + "1D-" + features_list[0] + "-" + str(no_components))

        # Instantiate GMM
        gmm = GaussianMixtureModel(X=X, k=no_components)
        gmm.fit()
        util.plot_1D(gmm, X, features_list, file_path)
    
    # For two features
    elif len(features_list) == 2:
        # Normalize data
        X = util.normalize(np.array(data[features_list]))
        
        # Convert class column into list of integers for color-coding scatterplot
        raw_label = list(data["label"])
        convert_dict, labels_list = util.convert(raw_label)
        label = [convert_dict[label] for label in raw_label]
        
        # Filepath to save plot
        file_path = os.path.join(parent_directory, "results/iris-dataset/" + "2D-" + features_list[0] + "-" + features_list[1] + "-" + str(no_components))

        # Instantiate GMM
        gmm = GaussianMixtureModel(X=X, k=no_components)
        gmm.fit()
        util.plot_2D(gmm, X, features_list, label, labels_list, file_path)
    
    else:
        # Normalize data
        X = util.normalize(np.array(data[features_list]))

        # Convert class column into list of integers for purity calculation
        raw_label = list(data["label"])
        convert_dict, _ = util.convert(raw_label)
        labeling = [convert_dict[label] for label in raw_label]

        # Instantiate GMM
        gmm = GaussianMixtureModel(X=X, k=no_components)
        gmm.fit()
        clustering = list(np.argmax(gmm.R, axis=1).A1)

        # Compute purity
        purity = util.compute_purity(clustering, labeling)
        print("Purity of clustering:", round(purity,4))

        # Filepath to save plot
        file_path = os.path.join(parent_directory, "results/iris-dataset/" + str(len(features_list)) + "D-" + str(features_list) + "-" + str(no_components) + ".txt")
        with open(file_path, "w") as fp:
            fp.write(str(round(purity, 4)))        

def main():
    iris_gmm(["sepal_width", "petal_length", "petal_width"], 3)

if __name__ == "__main__":
    main()