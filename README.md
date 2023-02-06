# M2TD 

M2TD is a web app that allows a user to create music through body movements. Specific motions are mapped to a distinct sounds ensuring that the user is always in rhythm. 

# Runing  
1. Download .wav files to /sounds folder 
2. Configure /configs/sound_mappings.yaml parts_tracked: with format
```$BODY_PART: vector: [ $X,$Y ] sound_file: $sound_file.wav```
3. Run ```sudo python main.py```
   1. Move body part specified in the config file in the motion of the configured vector to trigger the playing of the wav file
   2. While you can press r to map a body part movement to a sound file
      1. Follow prompts to enter body part name and the wav file then a recording will start. Press the space key to end the recording and the movement is saved to the sound dictionary. 
   
# Body Part Options 
```'''
'LEFT_ANKLE', 'LEFT_EAR', 'LEFT_ELBOW', 'LEFT_EYE', 'LEFT_EYE_INNER', 'LEFT_EYE_OUTER', 
'LEFT_FOOT_INDEX', 'LEFT_HEEL', 'LEFT_HIP', 'LEFT_INDEX', 'LEFT_KNEE', 'LEFT_PINKY', 'LEFT_SHOULDER', 'LEFT_THUMB',
 'LEFT_WRIST', 'MOUTH_LEFT', 'MOUTH_RIGHT', 'NOSE', 'RIGHT_ANKLE', 'RIGHT_EAR', 'RIGHT_ELBOW', 'RIGHT_EYE', 'RIGHT_EYE_INNER', 'RIGHT_EYE_OUTER', 
'RIGHT_FOOT_INDEX', 'RIGHT_HEEL', 'RIGHT_HIP', 'RIGHT_INDEX', 'RIGHT_KNEE', 'RIGHT_PINKY', 'RIGHT_SHOULDER', 'RIGHT_THUMB', 'RIGHT_WRIST'