from sklearn.model_selection import train_test_split
import os
import cv2
import numpy as np

path = 'Dataset'
class_names = os.listdir(path)
images = []
labels = []

haarcascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def face_detection(img, sf, mn):
    return haarcascade.detectMultiScale(img, scaleFactor=sf, minNeighbors=mn)

# load
for class_name in class_names:
  class_path = path + '/' + class_name
  for img in os.listdir(class_path):
    img_path = class_path + '/' + img
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    faces = face_detection(image, 1.2, 5)

    for face in faces:
      x, y, w, h = face
      face_image = image[y:y+h, x:x+w]
      # preprocess
      face_image = cv2.resize(face_image, (180, 180))
      face_image = cv2.GaussianBlur(face_image, (11, 11), 0)
      images.append(face_image)
      labels.append(class_names.index(class_name))

# split
# menggunakan stratify sesuai label agar jumlah setiap gambar pada setiap class untuk train 75% dan test 25%
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.25, random_state=42, stratify=labels)

lbph = cv2.face.LBPHFaceRecognizer_create()

def train():
    lbph.train(X_train, np.array(y_train))

def detect():
    true = 0
    all = 0
    for img, label in zip(X_test, y_test):
        pred, confidence = lbph.predict(img)
        all+=1
        if(pred == label):
            true+=1
    print(f'Average Accuracy : {np.round(true/all, 1)*100}%')
    lbph.save('lbph.yml')        

def predict():
    # (DONE) Load the saved model
    # (DONE IN MENU) If the model is not found, then print an error message and redirect back to the menu
    lbph.read('lbph.yml')

    # Ask to input the image’s absolute path to be predicted
    path = input('Input image absolute path : ')
    image = cv2.imread(path)
    imagegray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        

    # Predict the input image based on the saved recognizer to produce the prediction result
    # The prediction results consist of the predicted names, detected face location of the person, and accuracy which will be drawn to the input image
    faces = face_detection(imagegray, 1.2, 5)
    for face in faces:
        x, y, w, h = face
        face_image = imagegray[y:y+h, x:x+w]
        # preprocess
        face_image = cv2.resize(face_image, (180, 180))
        face_image = cv2.GaussianBlur(face_image, (11, 11), 0)
        pred, confidence = lbph.predict(face_image)
        
        cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), thickness=3)
        cv2.putText(image, f'{class_names[pred]} : {((1 - (confidence / 300)) * 100)}%', (20,20), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255,0,0), thickness=3)
        
    cv2.imshow('hi', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

choice = 0
flag = 0 # indicate model not found

while choice != 3:
    print('================================')
    print('Football Player Face Recognition')
    print('1. Train and Test Model')
    print('2. Predict')
    print('3. Exit')
    choice = int(input('>> '))

    if choice == 1:
        print('Training and Testing')
        train()
        detect()
        print('Training and Testing Finished')
        input('Press enter to continue ...')
        flag = 1 # indicate model found, udah prnh di train
    elif choice == 2:
        if flag == 0:
            print('ERROR ! Model not found')
        else:
            predict()
            
        input('Press enter to continue ...')
    elif choice == 3:
        print('goodbye')
        print('')
    else :
        print('you input the wrong choice')
        input('Press enter to continue ...')