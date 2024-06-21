import pygame
import cv2
import numpy as np

video=cv2.VideoCapture(0)

facedetect=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

pygame.init()
 
window=pygame.display.set_mode((1200,700))

pygame.display.set_caption("Face Detection App")

img=pygame.image.load("bac1.jpg").convert()

Start=True

while Start:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            Start=False
            pygame.quit()
    ret,frame=video.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=facedetect.detectMultiScale(frame, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255))
    imgRGB=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgRGB=np.rot90(imgRGB)
    imgRGB=pygame.surfarray.make_surface(imgRGB).convert()

    font=pygame.font.Font("Astonpoliz.ttf", 40)
    text=font.render("{} Face Detected".format(len(faces)), True, (0,0,0))
    
    
    window.blit(img, (0,0))
    window.blit(imgRGB, (280,95))
    pygame.draw.rect(window, (255,255,255), (280,50,640,70), border_top_left_radius=10, border_top_right_radius=10)
    window.blit(text, (480,70))
    

  
    pygame.display.update()