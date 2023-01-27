# M2TD 

M2TD is a web app that allows a user to create music through body movements. Specific motions are mapped to a distinct sounds ensuring that the user is always in rhythm. 

# Runing  
1. Download .wav files to /sounds folder 
2. Configure /configs/sound_mappings.yaml parts_tracked: with format
```$BODY_PART: vector: [ $X,$Y ] sound_file: $sound_file.wav```
3. Run ```sudo python main.py```
   1. Move body part specified in the config file in the motion of the configured vector to trigger the playing of the wav file