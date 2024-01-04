from flask import Flask, render_template, request, session,Response
import os
import shutil
from werkzeug.utils import secure_filename
import subprocess
import sys
import cv2
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
sys.path.append('yolov5/')
# from detect import all_detected_labels
#*** Backend operation

# WSGI Application
# Defining upload folder path
UPLOAD_FOLDER = os.path.join('staticFiles', 'uploads')
# # Define allowed files
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif','mp4','MPEG-4'}

# Provide template folder name
# The default folder name should be "templates" else need to mention custom folder name for template path
# The default folder name for static files should be "static" else need to mention custom folder for static path
app = Flask(__name__, template_folder='templateFiles', static_folder='staticFiles')
# Configure upload folder for Flask application
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
 
# Define secret key to enable session
app.secret_key = 'This is your secret key to utilize session in Flask'

def load_model(model_type='yolov5s', model_path='finalsafety.pt'):
    # Load the pre-trained PyTorch model
    return  torch.hub.load('ultralytics/yolov5', model_type, model_path)


def annotate_image(df, img):
    # create bouding box around detected objects
    for i in range(len(df)):
        cv2.rectangle(img, (int(df['xmin'][i]), int(df['ymin'][i])), (int(df['xmax'][i]), int(df['ymax'][i])), (0, 255, 0), 2)
        cv2.putText(img, str(df['name'][i]), (int(df['xmin'][i]), int(df['ymin'][i])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

    cv2.imwrite('staticFiles/output/output.jpg', img)


def generate_frames():
    cap = cv2.VideoCapture(0)  # Use webcam (change to '1' for an external camera)

    while True:
        success, frame = cap.read()

        if not success:
            break
        else:
            # Perform YOLOv5 object detection on the frame
            results = model(frame)

            # Draw bounding boxes on the frame
            for _, box in enumerate(results.xyxy[0]):
                xmin, ymin, xmax, ymax, conf, cls = box.tolist()
                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
                cv2.putText(frame, f'{model.names[int(cls)]}: {conf:.2f}', (int(xmin), int(ymin - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Convert the frame to JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame to the Flask app
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    # return render_template('live.html')
 

model = load_model("custom", "./finalsafety.pt")
print(model.names)
 
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/',  methods=("POST", "GET"))
def uploadFile():
    global labels
    if request.method == 'POST':
        # Upload file flask
        uploaded_img = request.files['uploaded-file']
        # Extracting uploaded data file name
        img_filename = secure_filename(uploaded_img.filename)#x.jpg
        # Upload file to database (defined uploaded folder in static path)
        uploaded_img.save(os.path.join(app.config['UPLOAD_FOLDER'], img_filename))
       
        # print(img_filename)
        extension = os.path.splitext(img_filename)[1][1:]
        
        # Storing uploaded file path in flask session
        # session['uploaded_img_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)

        # if(extension=="mp4"):
        #     cmd2=r"python yolov5/detect.py --weights finalsafety.pt --conf 0.075 --source "+"staticFiles/uploads/"+img_filename
        # else:
        #      cmd2=r"python yolov5/detect.py --weights finalsafety.pt --conf 0.8 --source "+"staticFiles/uploads/"+img_filename

        # print(cmd2)


        # read image and pass it to the model
        img = cv2.imread("staticFiles/uploads/"+img_filename)
        results = model(img)

        # convert results to dataframe
        df = results.pandas().xyxy[0]

        print(df)
        
        annotate_image(df, img)

        # get number of detected objects and group them by class
        labels = df['name'].value_counts().to_dict()
        print(labels)

        # save results to txt file
        # with open('templateFiles/label.txt', 'w') as file:
        #     file.write(str(labels))
        
        
        
        # subprocess.call(cmd2,shell=True)
        # # change image directory
        # origin = 'yolov5/runs/detect/exp/'
        # target = 'staticFiles/output/'

        # # Fetching the list of all the files
        # files = os.listdir(origin)

        # # Fetching all the files to directory
        # for file_name in files:
        #     shutil.copy(origin+file_name, target+file_name)
        #     print("Files are copied successfully")
        
        # # code to delete directory
        # shutil.rmtree('yolov5/runs/detect/exp')
        # #rename
        # source = 'staticFiles/output/'+img_filename
  
        # # destination file path
        # dest = 'staticFiles/output/'+'output.jpg'
        # os.rename(source,dest)
        return render_template('index1.html')
 
@app.route('/show_image')
def displayImage():
        global labels
    # Retrieving uploaded file path from session
    # img_file_path = session.get('staticFiles/output/output.jpg', None)
    
    # Display image in Flask application web page
    # my_list=all_detected_labels
        your_class_labels={"with_helmet":" T intersection ahead","with_vest":" Cross road","gapInDivider":"Go slow, gap in divider ahead","leftTurn":"Take left turn","metro":" Metro station near by","noOvertake":"Over taking prohibited","noParking":" No parking","parking":"Parking allowed here","pedestrianCrossing":"Pedestrian crossing ahead, wait","school":" School nearby","speedBreaker":"Slow down, speed breaker ahead","speedLimit35":"Keep speed limit to 35kmph","speedLimit40":"Keep speed limit to 35kmph","stayLeft":" Stay to the left","red":"Stop!!!","Pedestrian":"Be careful, there is a pedestrian","green":"Go","yellow":"Go slow","red":"Stop","Animal":"Be careful, animal nearby"}

        with open('templateFiles/label.txt', 'r') as file:
            file_contents = file.read()
        list1=[]
        list1=file_contents.split()
        str1=""
        j=1
        # for i in list1:
        #     x=""
        #     x=str(j)+your_class_labels[i]+"<br /> "
        #     str1+=x
        #     j+=1
        
        # print(str1)  

        withoutv=0
        withouth=0
        withv=0
        withh=0

        #mylist = list(dict.fromkeys(list1))
        myset = set(list1)
        s = list(myset)



   


        #mylist = list(dict.fromkeys(list1))

        for i in list1:
                    if i=='without_helmet':
                        withouth+=1
                    if i=='without_vest':
                        withoutv+=1
                    if i=='with_helmet':
                        withh+=1
                    if i=='with_vest':
                        withv+=1
        text1='without_helmet ='+str(withouth)
        text2='without_vest ='+str(withoutv)
        text3='with_helmet ='+str(withh)
        text4='with_vest ='+str(withv)

        s.append(text1)
        s.append(text2)
        s.append(text3)
        s.append(text4)
        print('\n')
        print('\n')
        print('\n')
        print('\n')
        for i in s:
             print(i,'\n')
        print('\n')
        print('\n')
        print('\n')
        print('\n')

        # s.append(labels)

        # convert dictionary to list of strings in the format "key = value"
        rez = [str(key) + ' = ' + str(value) for key, value in labels.items()]

        return render_template('show_image.html', user_image = 'staticFiles/output/output.jpg',
                                    file_contents=rez)


        #return render_template('show_image.html', user_image = 'staticFiles/output/output.jpg',file_contents=mylist,withoutv=withoutv,withouth=withouth)

        #return render_template('show_image.html', user_image = 'staticFiles/output/output.jpg',file_contents=list1)

@app.route('/newsite')
def indexpage():
    return render_template('indexpage.html')

@app.route('/livedet')
def livedet():
    return render_template('live.html')
    #return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/reset')
def reset():
    myfile = "staticFiles/output/output.jpg"
    # If file exists, delete it.
    if os.path.isfile(myfile):
        os.remove(myfile)
    return render_template('index.html')


if __name__=='__main__':
    app.run(debug = True)








import cv2
import torch
from torchvision import transforms
from PIL import Image

# Load the pre-trained PyTorch model







def detect_objects():
 
    # Load pre-trained cascade classifier for face detection (or any other classifier)
 model = torch.load('model.pt')
 model.eval()

# Define the transformation to preprocess the frames before feeding them into the model
 preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# Open camera
 video_capture = cv2.VideoCapture(0)

 while True:
    ret, frame = video_capture.read()

    if not ret:
        break

    # Convert OpenCV frame to PIL image
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Preprocess the frame
    input_tensor = preprocess(pil_image)
    input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension

    # Perform inference
    with torch.no_grad():
        output = model(input_batch)

    # Process the output (adjust this according to your model's output)
    # For example, if it's a classification model, you might extract the predicted class
    predicted_class = torch.argmax(output, dim=1).item()
    # Do something with the predicted_class...

    # Display the frame
    cv2.imshow('Frame', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

 video_capture.release()
 cv2.destroyAllWindows()

@app.route('/video_feed')
def video_feed():
    return Response(detect_objects(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
