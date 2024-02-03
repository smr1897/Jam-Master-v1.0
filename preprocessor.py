import json
import os
import librosa
import math

DATASET_PATH = "PaddedTest/"
JSON_PATH = "test_dataset.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 15
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

def save_mfcc(dataset_path,json_path,num_mfcc=13,n_fft=2048,hop_length=512,num_segments=5):
    data = {
        "mapping":[],
        "labels": [],
        "mfcc":[]

    }

    samples_per_segment = int(SAMPLES_PER_TRACK/num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment/hop_length)

    for i , (dirpath,dirnames,filenames) in enumerate(os.walk(dataset_path)):

            if dirpath != dataset_path:
                semantic_label = dirpath.split("/")[-1]
                data["mapping"].append(semantic_label)
                print("\nProcessing: {}".format(semantic_label))

                for f in filenames:
                    file_path = os.path.join(dirpath,f)
                    signal,sample_rate = librosa.load(file_path,sr=SAMPLE_RATE)
                    #duration = librosa.get_duration(y=signal,sr=sample_rate)
                    #samples_per_track = (sample_rate * duration)
                    #num_samples_per_segment = int(samples_per_track/num_segments)
                    #expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)
                    #print(samples_per_track,num_samples_per_segment,expected_num_mfcc_vectors_per_segment)

                    for s in range(num_segments):
                        start_sample = samples_per_segment * s
                        finish_sample = start_sample + samples_per_segment

                        mfcc = librosa.feature.mfcc(y=signal[start_sample:finish_sample],
                                                    sr=sample_rate,
                                                    n_fft=n_fft,
                                                    n_mfcc=num_mfcc,
                                                    hop_length=hop_length)
                        mfcc = mfcc.T

                        if len(mfcc) == num_mfcc_vectors_per_segment:
                            data["mfcc"].append(mfcc.tolist())
                            data["labels"].append(i-1)
                            print("{},segment:{}".format(file_path,s+1))
    with open(json_path,"w") as fp:
        json.dump(data,fp,indent=4)

if __name__ == "__main__":
    save_mfcc(DATASET_PATH,JSON_PATH,num_segments=10)



