
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class Cloudy:
    def __init__(self, folderName):
        self.images = load_image(folderName, range(1, 301), "cloudy")
        self.training_set = get_set(self.images, range(0, 150))#50% of images
        self.validation_set = get_set(self.images, range(150, 225))#25# of rest
        self.test_set = get_set(self.images, range(225, 300))#25# of rest
        self.hist_features_training = hist_features(self.training_set)
        self.hist_features_validation = hist_features(self.validation_set)
        self.hist_features_testing = hist_features(self.test_set)
        self.mystery_features_training = mystery_features(self.training_set)
        self.mystery_features_validation = mystery_features(self.validation_set)
        self.mystery_features_testing = mystery_features(self.test_set)
        print(bcolors.OKGREEN + "Cloudy initialized." + bcolors.ENDC)
        self.save_features()


    def save_features(self):
        print(bcolors.OKCYAN + "Saving features..." + bcolors.ENDC)
        # Ensure the directory exists
        save_dir = "CloudySet"
        os.makedirs(save_dir, exist_ok=True)
        try:
            np.save("CloudySet/hist_features_training_cloudy.npy", self.hist_features_training)
            np.save("CloudySet/hist_features_validation_cloudy.npy", self.hist_features_validation)
            np.save("CloudySet/hist_features_testing_cloudy.npy", self.hist_features_testing)
            np.save("CloudySet/mystery_features_training_cloudy.npy", self.mystery_features_training)
            np.save("CloudySet/mystery_features_validation_cloudy.npy", self.mystery_features_validation)
            np.save("CloudySet/mystery_features_testing_cloudy.npy", self.mystery_features_testing)
            print(bcolors.OKGREEN + "Saved features" + bcolors.ENDC)
        except FileNotFoundError:
            print(bcolors.WARNING + "Saving error..." + bcolors.ENDC)

class Shine:
    def __init__(self, folderName):
        self.images = load_image(folderName, range(1, 253), "shine")
        self.training_set = get_set(self.images, range(0, 126))#50% of images
        self.validation_set = get_set(self.images, range(126, 189))#25# of rest
        self.test_set = get_set(self.images, range(189, 251))#25# of rest
        self.hist_features_training = hist_features(self.training_set)
        self.hist_features_validation = hist_features(self.validation_set)
        self.hist_features_testing = hist_features(self.test_set)
        self.mystery_features_training = mystery_features(self.training_set)
        self.mystery_features_validation = mystery_features(self.validation_set)
        self.mystery_features_testing = mystery_features(self.test_set)
        print(bcolors.OKGREEN + "Shine initialized." + bcolors.ENDC)
        self.save_features()

    def save_features(self):
        print(bcolors.OKCYAN + "Saving features..." + bcolors.ENDC)
        # Ensure the directory exists
        save_dir = "ShineSet"
        os.makedirs(save_dir, exist_ok=True)
        try:
            np.save("ShineSet/hist_features_training_shine.npy", self.hist_features_training)
            np.save("ShineSet/hist_features_validation_shine.npy", self.hist_features_validation)
            np.save("ShineSet/hist_features_testing_shine.npy", self.hist_features_testing)
            np.save("ShineSet/mystery_features_training_shine.npy", self.mystery_features_training)
            np.save("ShineSet/mystery_features_validation_shine.npy", self.mystery_features_validation)
            np.save("ShineSet/mystery_features_testing_shine.npy", self.mystery_features_testing)
            print(bcolors.OKGREEN + "Saved features" + bcolors.ENDC)
        except FileNotFoundError:
            print(bcolors.WARNING + "Saving error..." + bcolors.ENDC)

class Sunrise:
    def __init__(self, folderName):
        self.images = load_image(folderName, range(1,357), "sunrise")
        self.training_set = get_set(self.images, range(0, 178))#50% of images
        self.validation_set = get_set(self.images, range(178, 267))#25# of rest
        self.test_set = get_set(self.images, range(267, 356))#25# of rest
        self.hist_features_training = hist_features(self.training_set)
        self.hist_features_validation = hist_features(self.validation_set)
        self.hist_features_testing = hist_features(self.test_set)
        self.mystery_features_training = mystery_features(self.training_set)
        self.mystery_features_validation = mystery_features(self.validation_set)
        self.mystery_features_testing = mystery_features(self.test_set)
        print(bcolors.OKGREEN + "Sunrise initialized." + bcolors.ENDC)
        self.save_features()

    def save_features(self):
        print(bcolors.OKCYAN + "Saving features..." + bcolors.ENDC)
        # Ensure the directory exists
        save_dir = "SunriseSet"
        os.makedirs(save_dir, exist_ok=True)
        try:
            np.save("SunriseSet/hist_features_training_sunrise.npy", self.hist_features_training)
            np.save("SunriseSet/hist_features_validation_sunrise.npy", self.hist_features_validation)
            np.save("SunriseSet/hist_features_testing_sunrise.npy", self.hist_features_testing)
            np.save("SunriseSet/mystery_features_training_sunrise.npy", self.mystery_features_training)
            np.save("SunriseSet/mystery_features_validation_sunrise.npy", self.mystery_features_validation)
            np.save("SunriseSet/mystery_features_testing_sunrise.npy", self.mystery_features_testing)
            print(bcolors.OKGREEN + "Saved features" + bcolors.ENDC)
        except FileNotFoundError:
            print(bcolors.WARNING + "Saving error..." + bcolors.ENDC)

def testing(best_k, training_features, training_labels, testing_features, testing_label):

    knn = KNeighborsClassifier(n_neighbors=best_k, metric='manhattan')
    knn.fit(training_features, training_labels)
    predictions = knn.predict(testing_features)
    test_accuracy = accuracy_score(testing_label, predictions)

    print(f"Test Accuracy with K = {best_k}: {test_accuracy:.4f}")
    return test_accuracy

def training(training_features, training_labels, validation_features, validation_label):

    k_vals = [1, 3, 5, 7]
    accuracies = []

    for k in k_vals:
        #L1 manhattan distance
        knn = KNeighborsClassifier(n_neighbors=k, metric="manhattan")
        knn.fit(training_features, training_labels)
        predictions = knn.predict(validation_features)
        acc = accuracy_score(validation_label, predictions)
        accuracies.append(acc)
        print(f"K = {k}, Validation Accuracy = {acc: 4f}")

    plt.figure()
    plt.plot(k_vals, accuracies, marker = 'o')
    plt.title("KNN Validation Accuracy for Different K")
    plt.xlabel("K Value")
    plt.ylabel("Validation Accuracy")
    plt.grid(True)
    plt.show()

    #Determine best K
    best_k_index = np.argmax(accuracies)
    best_k = k_vals[best_k_index]
    print(f"Best K: {best_k} with accuracy: {accuracies[best_k_index]:.4f}")

    return best_k

def mystery_features(images, max_len=1000):
    features = []

    for img in images:
        edges = cv2.Canny(img, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        f = []
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area != 0 else 0
            f.extend([area, perimeter, solidity])

        # Pad or truncate
        if len(f) < max_len:
            f.extend([0] * (max_len - len(f)))  # pad with zeros
        else:
            f = f[:max_len]  # truncate to fixed size

        features.append(f)

    return np.array(features)


def hist_features(images):
    hsv_images = [cv2.cvtColor(img, cv2.COLOR_BGR2HSV) for img in images]
    features = np.zeros((len(hsv_images), 32))  # m x n array

    for i, img in enumerate(hsv_images):
        v = img[:, :, 2]  # Take the V channel (brightness)
        hist, _ = np.histogram(v, bins=32, range=(0, 256), density=True)
        features[i, :] = hist

    return features


def get_set(images, rng):
    set = []
    for i in rng:
        set.append(images[i])

    return set

def load_image(folderName, rng, imgName):
    s = []
    for i in rng:
        img = cv2.imread(folderName + imgName + str(i) + ".jpg")
        if img is None:
            img = cv2.imread(folderName + imgName + str(i) + ".jpeg")
            if img is None:
                print("Image: " + imgName + str(i) + ".jpg" +" couldn't be loaded")
            else:
                print(bcolors.OKGREEN + "Image: " + imgName + str(i) + ".jpeg" + " settled and loaded" + bcolors.ENDC)
                s.append(img)
        else:
            s.append(img)  # Add shine1,shine2... in array

    return s



if __name__ == "__main__":
    #Load all images and extract features
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #If the images are not in the same folder, WRITE THE FOLDER NAME
    folderName = ''
    cloudy = Cloudy(folderName)
    shine = Shine(folderName)
    sunrise = Sunrise(folderName)

    print(bcolors.OKCYAN + "\n--- TRAINING PHASE (Histogram Features) ---" + bcolors.ENDC)

    #Combine histogram features from all three classes for training
    X_train = np.vstack([
        cloudy.hist_features_training,
        shine.hist_features_training,
        sunrise.hist_features_training
    ])
    #Create corresponding labels: 0 for cloudy, 1 for shine, 2 for sunrise
    y_train = (
        [0] * len(cloudy.hist_features_training) +
        [1] * len(shine.hist_features_training) +
        [2] * len(sunrise.hist_features_training)
    )
    #Combine validation histogram features and labels
    X_val = np.vstack([
        cloudy.hist_features_validation,
        shine.hist_features_validation,
        sunrise.hist_features_validation
    ])
    y_val = (
        [0] * len(cloudy.hist_features_validation) +
        [1] * len(shine.hist_features_validation) +
        [2] * len(sunrise.hist_features_validation)
    )

    #Train KNN classifier with various K values and select best
    best_k_hist = training(X_train, y_train, X_val, y_val)

    #Prepare test features and labels
    X_test = np.vstack([
        cloudy.hist_features_testing,
        shine.hist_features_testing,
        sunrise.hist_features_testing
    ])
    y_test = (
        [0] * len(cloudy.hist_features_testing) +
        [1] * len(shine.hist_features_testing) +
        [2] * len(sunrise.hist_features_testing)
    )

    #Evaluate classifier on the test set using best K
    print(bcolors.OKCYAN + "\n--- TESTING PHASE (Histogram Features) ---" + bcolors.ENDC)
    test_accuracy_hist = testing(best_k_hist, X_train, y_train, X_test, y_test)
    # ---------------------------------------------------------------
    print(bcolors.OKCYAN + "\n--- TRAINING PHASE (Mystery Features) ---" + bcolors.ENDC)
    #Combine mystery features from all three classes for training
    X_train_myst = np.vstack([
        cloudy.mystery_features_training,
        shine.mystery_features_training,
        sunrise.mystery_features_training
    ])
    y_train_myst = y_train  # Same label logic

    #Combine mystery validation features and labels
    X_val_myst = np.vstack([
        cloudy.mystery_features_validation,
        shine.mystery_features_validation,
        sunrise.mystery_features_validation
    ])
    y_val_myst = y_val

    #Train KNN classifier on mystery features and find best K
    best_k_myst = training(X_train_myst, y_train_myst, X_val_myst, y_val_myst)

    #Prepare test features and labels for mystery features
    X_test_myst = np.vstack([
        cloudy.mystery_features_testing,
        shine.mystery_features_testing,
        sunrise.mystery_features_testing
    ])
    y_test_myst = y_test  # Labels are reused

    #Evaluate mystery-feature-based classifier on test set
    print(bcolors.OKCYAN + "\n--- TESTING PHASE (Mystery Features) ---" + bcolors.ENDC)
    test_accuracy_myst = testing(best_k_myst, X_train_myst, y_train_myst, X_test_myst, y_test_myst)
