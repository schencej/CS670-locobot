# CS670-locobot
Step1: Setup environment
run "pip install -r requirements.txt"

Step2: Detect the object 
run "python3 detect.py [arg]" will do object detection and store coordinates of [arg] in a file "coor/coordiantes"

Step3: Grab or move to the object
run "python3 arm_test.py" will read coordinates from "coor/coordinates" and grab the object.
run "python3 move.py" will read coordinates from "coor/coordinates" and move to the object.
