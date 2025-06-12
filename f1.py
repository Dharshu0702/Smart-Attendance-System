import cv2
import imutils
alg = "haarcascade_frontalcatface.xml"
haar = cv2.CascadeClassifier(alg)
cam = cv2.VideoCapture(2)
while True:
    _, img = cam.read()
    gray = haar.detectMultiScale(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY), scaleFactor=1.1,minNeighbors=5, minSize=(30, 30))
    face = haar.detectMultiScale(gray,1.3,4)
    for(x,y,w,h)in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,225,0),2)
    cv2.imshow("face detected",img)
    key = cv2.waitKey(10)
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()
