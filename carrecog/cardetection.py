import cv2 
import numpy as py 

carcascade=cv2.CascadeClassifier("car.xml")
# the car.xml is a classifier as well 

video=cv2.VideoCapture("traffic.avi")
# video capture is a means to read the video 
 
# because there is no specific duration, while true means the loop will run as long as there are frames in the video
while True:
    status, frame = video.read() 
    # using video read function, we get 2 pieces of data, status is checking if there IS a frame, frame is holding the frame data
    greyframe=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # converting the frames to grey for the detectMultiscale, because it only accepts with grey images
    car=carcascade.detectMultiScale(greyframe,)
    # the function that looks for certain "landmarks" that define the shape you're looking for: eg. looking for a face? 
    # the function will look for the lips/mouth/nose, based off the face cascade you're using (here we're using car)
    # paramaters are: the actual frame, scale factor: how big the object can be, 
    # minimum neighbours: how many more of the detected items there can be

    for (x,y,w,h) in car:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
    # the shape that will appear over detected objects, we're defining the points needed to generate a 
    # rectangle, first the place we put the rectangle on (frame), then the object we place the rectangle over's
    # dimensions, x,y us the left top corner, (x+w,y+h) gives us the bottom right corner, and the rectangle
    # completes itself, with the colour, and thickness 

    if status==False:
        break 
    # ends video when the frames run out 
    cv2.imshow("Car Detection",frame)
    # showing the video, the video name, and the frames to give the illusion of the video playing
    key=cv2.waitKey(10)
    # setting a key to break the video when you want to end early
    if key==27:
        break 
    # 27 is the escape key, defining that hitting 27 ends the video 

video.release()
# "releases" the video when we are done using it from our memory so as to not clutter and take up unneccessary memory
# and so we can access the video again outside of code. this is also to alleviate RAM  memory.

cv2.destroyAllWindows()
# destroys the window holding the video so as to not keep the RAM busy.





