<h2>This is a python emotion classifier of images using Support Vector Machine (SVM).</h2>
<br>
<br>
Referenced from: https://www.youtube.com/watch?v=0rjlviOQlbc
<br>
<br>
To open a new Virtual Environment:<br>
~ py -m venv emotionClassifierVenv
<br>
<br>
To activate the Virtual Environment:<br>
~ cd emotionClassifierVenv/scripts<br>
~ activate<br>
<br>
<br>
Libraries to install:<br>
~ pip install opencv-python <br>
~ pip install matplotlib <br>
~ pip install scikit-learn<br>
~ pip install PyWavelets<br>
<br>
<br>
To Run the App:<br>
~ py classifier.py<br>
OR<br>
~ python classifier.py<br>
<br>
<br>
To deactivate the Virtual Environment:<br>
~ deactivate<br>
<br>
<br>
The classifier.py file should be run in the following order:<br><br>
If data1.pickle has not been generated yet, Run the command:<br>
~ py extract_img_to_pickle.py OR ~ python extract_img_to_pickle.py
<br>
<br>
If data1.pickle has already been generated, but model.sav has not been generated yet, run the command (this will take around 20 minutes or longer depending on your device's processing power):<br>
~ py train_classifier.py OR ~ python train_classifier.py
<br>
<br>
If both data1.pickle and model.sav have already been generated, run the command (This will also take around 20 minutes or longer):<br>
~ py classifier.py OR ~ python classifier.py
<br>
<br>
the performance results of the SVM model will be displayed when classifier.py has completed its execution.<br>
<br>
<br>
<b>NOTE*: Remember to check line 11 of extract_img_to_pickle.py file to ensure that the "dir" variable is set to the correct path of the training images folder on your device.</b>