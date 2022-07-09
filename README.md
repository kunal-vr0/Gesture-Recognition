# Gesture-Recognition
The idea of this project is to train the program by recording different gestures and be able to recognise them.
I am using openCV for computer vision, meidapipe for hand detection and pickle to store data.
For each hand gesture it records 144 data points and compare them with the recorded gesture and reports positively within some level of tolerance.

Instructions for usage:

#1. You can provide a data file to use or record gestures from scratch. If you wish to use some data file it asks for the file name or a default file can be used.

#2. Press 'n' to add a new entry.

#3. Write after you press 'n', the program will ask for a name for the gesture in the terminal. It records the name until you press enter key.

#4. Then show the gesture in the camera and press 't' when you are ready to record the gesture.

#5. Once you press 't' the gesture is added to the database and the program will start recognising it. Press 'n' for any new entry and repeat the process.

#6. The program cannot be exited while recording the gesture.

#7. To view the currently added gesture names and number of gestures present in the database press 'p'. Cannot be used while recording the gesture.

#8. To view the commands press 'i', this will print the purpose of each key.
