### MOTION DETECTION SYSTEM

import cv2 # open CV / To read the image/video

import imutils #To resize

camera=cv2.VideoCapture(0) # cam ID

firstFrame = None # First/ Initial Place/frame

area = 500 # Threshold Value

while True:
    
    _,image = camera.read() #read from camera
    
    text = "Normal" # To show text in the output window if Motion is Not Detected

    image = imutils.resize(image, width=1000) # resize
    
    grayImage=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Color Ima to Gray Ima
    
    gaussianImage=cv2.GaussianBlur(grayImage,(21,21),0) #smoothening

    if firstFrame is None:
        
        firstFrame=gaussianImage #capturing the first Frame
        continue

    imgDiff = cv2.absdiff(firstFrame, gaussianImage) # Absolute Difference

    threshImage = cv2.threshold(imgDiff, 25, 255, cv2.THRESH_BINARY)[1] #color ima -> Gray scale Ima -> Black & white Ima

    threshImage = cv2.dilate(threshImage, None, iterations=2) # To remove left overs - erotion

    cnts=cv2.findContours(threshImage.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #Applying Contour

    cnts=imutils.grab_contours(cnts) # To capture/hold/grab all the contour

    for c in cnts:
        if cv2.contourArea(c)< area: #if contour area is less than the initially given threshold value /area it will continue to the next line and omit for loop
            continue

        (x,y,w,h)=cv2.boundingRect(c) # Bounding rectangle on the observed Contour

        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2) # Syntax : cv2.rectangle(sourcename, starting_value, ending_value, colour, thickness)

        text="Motion Detected" # To show text in the output window if Motion is Detected

    print(text) 

    cv2.putText(image,text,(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2) # To put text on the image/video

    cv2.imshow("Camera Feed",image)

    key=cv2.waitKey(10)#OR --> key=cv2.waitKey(1) & 0xFF    # 10 frames per seconds
    print(key)

    if key == ord("s"): #OR--> if key == 115..... when we press s it will break ... we can give any alphabets or ASCII value number (ord("s"))--> it will take ASCII value- 115 
        break

camera.release()    # will release the camera once we press the key alphabet or number

cv2.destroyAllWindows() # will destroy all windows

    

