#importing necessary libraries
import cv2
import os

#reading test image
img = cv2.imread("test2.jpeg") #image name

#reading label name from obj.names file
with open(os.path.join("project_files",'obj.names'), 'r') as f:
    classes = f.read().splitlines()

#importing model weights and config file
net = cv2.dnn.readNet('project_files/yolov5_tiny.weights', 'project_files/yolov5_tiny.cfg')
model = cv2.dnn_DetectionModel(net)
model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)
classIds, scores, boxes = model.detect(img, confThreshold=0.6, nmsThreshold=0.4)

#detection 
for (classId, score, box) in zip(classIds, scores, boxes):
    cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
                  color=(0, 255, 0), thickness=2)
    
# Resize the output image to a smaller window size
new_width = 800  # New width in pixels
new_height = 600  # New height in pixels
resized_img = cv2.resize(img, (new_width, new_height))

cv2.imshow("pothole", resized_img)
cv2.imwrite("result1" + ".jpg", resized_img)  # Result name
cv2.waitKey(0)
cv2.destroyAllWindows()    
 
# cv2.imshow("pothole",img)
# cv2.imwrite("result1"+".jpg",img) #result name
# cv2.waitKey(0)
# cv2.destroyAllWindows()
