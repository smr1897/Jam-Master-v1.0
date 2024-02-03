import librosa
import os
import numpy as np
import json
from tensorflow.keras import layers, models
# Dataset = "Test/"
# max_duration = 0.0


# def check_duration(dataset_path):
#     global max_duration  # Use the global variable
#
#     for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
#         for f in filenames:
#             file_path = os.path.join(dirpath, f)
#             signal, sample_r = librosa.load(file_path, sr=None)
#
#             # Specify a default sample rate if it is not provided
#             if sample_r is None:
#                 sample_r = 22050  # You can change this to an appropriate default
#
#             duration = librosa.get_duration(y=signal, sr=sample_r)
#             if duration > max_duration:
#                 max_duration = duration
#
#     return max_duration
#
# if __name__ == "__main__":
#     x = check_duration(Dataset)
#     print(x)

# import json
# import numpy as np
# import librosa
# import os
#
# Dataset = "Train/"
# max_duration = 0.0
#
#
# def check_duration(dataset_path):
#     for i , (dirpath,dirnames,filenames) in enumerate(os.walk(dataset_path)):
#         for f in filenames:
#             file_path = os.path.join(dirpath, f)
#             signal,sample_r = librosa.load(file_path,sr=None)
#             duration = librosa.get_duration(y=signal,sr=None)
#             if(duration>max_duration):
#                 max_duration = duration
#
#     return max_duration
#
# if __name__ == "__main__":
#     x=check_duration(Dataset)
#     print(x)
#
#

Dataset = "train_dataset.json"
def load_data(datast_path):
    with open(datast_path,"r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"],dtype=object)
    X = X[...,np.newaxis]
    #Y = np.array(data["labels"],dtype=object)

    #X = sum([sublist for sublist in data["mfcc"]])

    return X

if __name__ == "__main__":
    X = load_data(Dataset)
    print(X.shape)
    print(X.shape[1])
    print(X.shape[2])
    print(X.shape[3])
