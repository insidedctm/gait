# Gait Analysis
Recognise people by their gait and analyse their gait

### Installation
* clone this repository
* ensure the tf-pose-estimation application is installed in the root directory
* pip install opencv-python pandas scikit-learn

### Construct Gait Energy Images (GEI) from video
A GEI (see the Hoffman paper in 'papers') is a compression of a binary mask silhouette into a single binary image:

![GEI example](https://github.com/insidedctm/gait/tree/master/images/normal_id001_1.avi_1.png)

Given a video a GEI can be constructed as follows, storing in the directory `data/gei/GaitRecogniser` to be used to train a 
Gait Recogniser model.

```
python make_gei_from_raw.py /path/to/video.mov data/gei/GaitRecogniser 120 310
```

### Using the GaitRecogniser
The GaitRecogniser class implements a model trained on a database of GEI images. The database, by convention, 
is stored in data/gei/GaitRecognier. Each GEI image should have the following filename `<label>_<id>.png` where the `label`
identifies a partiular person and `id` uniquely identifies different GEI for the same person.

The recogniser can be used as follows:
``` python
from GaitRecogniser import GaitRecogniser

# Initialise and train 
recogniser = GaitRecogniser()
recogniser.train()

# Run prediction on a previously unseen GEI image
prediction = recogniser.predict_from_file('test_gei.png')
print(prediction)
```

### Gait Analysis
We can use the pose estimation framework (tf-pose-estimation) to analyse the gait of people, extracting metrics such as 
how long a stride they take. Capturing this information over time allows us to identify whether changes in gait have
occurred that suggest that some therapeutic action should be taken.

In a video of a person walking each sequence of frames where a foot leaves the ground and then subsequently touches the 
ground again is called a 'gait cycle'. The largest angle between the right and left leg over the course of gait cycle is a 
measure of the length of a person's stride (stride angle). Given a video sequence of a person walking the following 
command extracts the stride angle for each gait cycle.

```
python gait_metrics.py /path/to/video.mov metrics.npy 100 190
```
The extracted stride angles are saved to `metrics.npy` for further analysis. The 3rd and 4th parameters are the start and end 
frame to be analysed.

The stride angles extracted will tend to be very noisy. The `dft_analysis.py` module provides smoothing and gait cycle 
identification using the Discrete Fourier Transform (DFT). The following command takes the metrics output from the previous 
step, smooths the data and extracts the median stride angle, appending to a CSV file.

```
python dft_analysis.py metrics.npy --output_append_path=stride_history.csv
```

Running gait_metrics and dft_analysis on a sequence of videos captured over time allows a time-series of stride angles to be
accumulated. The time series can be used for identication of changes in the gait.
