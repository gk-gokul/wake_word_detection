# WakeWordDetection

Wake Word Detection (also known as Hot word detection) is a technique mainly used in ChatBots to **wake them**. 'Okay Google', 'Siri' and 'Alexa' are the wake words
used by Google assistant, Apple and Amazon's Alexa respectively.

### Why to use a Wake word?
The main function of any chatbot is to take a audio input by user and answer what it is designed for! Let's take an example of Google Assistant, as soon as it listen
the wake words it starts to take the audio input by user (and sends it to cloud for speech synthesis) and then gives the answer. That's why the Wake word prevents us 
to send every thing on clound for speech synthesis. As speech synthesis takes too much computational power, wake word save this by only sending the audio fragment which user wants it to synthesize.
In short, Wake word is a point in time from when we have to start doing speech synthesis.

### How you can use it?
In this project, we have created a Wake word detection using TensorFlow. Which has an awesome accuracy of 98% on the test data. You can even train this Deep learning Model on
whatever Wake Word you like. So, let us see how you can do it...

### Step 1: `Data Preparation`:
<br>
First, go to the fileData Preaparation.py. It contains two function record_audio_and_save() and record_background_sound().


`record_audio_and_save()`:- It records a audio of 2 seconds of you saying the Wake Word. It takes two parameters namely `n_times` and `save_path`. n_times is 'How many times it
                            should record you saying the Wake Word?'. Default is set to 100. In save_path you have to provide the path to the directory where it can store generated
                            `.wav` files.
                            
`record_background_sound()`:- It records a audio of 2 seconds of the background sounds. It takes two parameters namely `n_times` and `save_path`. n_times is 'How many times it
                              should record the backgound sounds?'. Default is set to 50. In save_path you have to provide the path to the directory where it can store generated
                              `.wav` files.
 
 Note:- Don't provide the same directory to both the functions or it will overwrite the previous!
 
 ### Step 2: `Data Preprocessing`:
 
In this file you don't have to do anything it will take all your audio files, pre-process it, make a pandas dataframe and save it as csv from where we will load it for
training.

Note:- If you have changed the directory of audio files from default `audio_data/` and `background_sound/` to some other. Don't forget it to change in this file too.

 ### Step 3: `Model Training`:
 
 Sit and relax! Let the model train...
 
 Okay... What is does it takes the csv file and convert the data to numpy array.
 
The Model architecture which is as follows:

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 256)               10496
_____________  ____________________________________________________
activation (Activation)      (None, 256)               0
_________________________________________________________________
dropout (Dropout)            (None, 256)               0
_________________________________________________________________
dense_1 (Dense)              (None, 256)               65792
_________________________________________________________________
activation_1 (Activation)    (None, 256)               0
_________________________________________________________________
dropout_1 (Dropout)          (None, 256)               0
_________________________________________________________________
dense_2 (Dense)              (None, 2)                 514
=================================================================
Total params: 76,802
Trainable params: 76,802
Non-trainable params: 0
_________________________________________________________________
```

After that it will save the model for future use...


### Step 4: `prediction.py`:

It will load the saved model and run the prediction every time it prints `Say Now: ` it records the sound live and give it to the model for prediction which it then decides
whether it has wake word or not!



