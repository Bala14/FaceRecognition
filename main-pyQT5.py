import typing
from PyQt5 import QtCore
import face_recognition
import cv2
import numpy as np
import sys
from fbs_runtime.application_context.PyQt5 import ApplicationContext
from PyQt5.QtWidgets import QApplication, QMainWindow, \
    QPushButton, QSizePolicy, QStyle, QVBoxLayout, QWidget
from PyQt5.QtWidgets import  QWidget, QLabel, QApplication
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QFont, QImage, QPixmap, QTextBlock
from qt_material import apply_stylesheet
from pyzbar import pyzbar


mapData = {'Karthik' : {id:"MRN-1","fullName": "Karthik SR", "vaccinated": "Yes", "DoB":"02-Aug-1975", "picToEncode":"K-3.jpg", "pic":"K-3.jpg"},
            'Obama' : {id:"MRN-2","fullName": "Obama", "vaccinated": "Yes", "DoB":"24-Sep-1994", "picToEncode":"obama.jpg", "pic":"obama.jpg"},
            'Amar' : {id:"MRN-3","fullName": "Amarendra Bahubali", "vaccinated": "No", "DoB":"01-Apr-1982", "pic":"A-1.jpg"},
            'Unknown': {"fullName": "Unable to recognize", "vaccinated": "", "DoB":""}}

class Thread(QThread):
    print ("Initializing....")
    changePixmap = pyqtSignal(QImage)
    capturedImage = pyqtSignal(QImage)
    patientData = pyqtSignal(dict)
    readBarcode = False
    process_this_frame = True
    video_capture = cv2.VideoCapture(0)
    
    def run(self):
        # Create arrays of known face encodings and their names
        known_face_names= []
        known_face_encodings = []
        print ("About to encode faces....")
        # Load a sample picture and learn how to recognize it.
        for name, patientData in mapData.items() :
            if "picToEncode" in patientData: 
                image = face_recognition.load_image_file(patientData["picToEncode"])
                image_encoding = face_recognition.face_encodings(image)[0]
                known_face_encodings.append(image_encoding)
                known_face_names.append(name)
        
        print ("Faces encoding complete....")
        patient = ""

        while True:
            # Grab a single frame of video
            ret, frame = self.video_capture.read()

            # Only process every other frame of video to save time
            if ret and self.process_this_frame:
                if self.readBarcode: 
                    patient = self.read_barcodes(frame)
                else :
                    patient = self.recognize_face(frame, known_face_encodings, known_face_names)

            if patient and patient != "Unknown":
                self.process_this_frame = False
                

            # Display the resulting image
            rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgbImage.shape
            bytesPerLine = ch * w
            convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
            p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
            self.changePixmap.emit(p)            
            if (patient in mapData):
                self.patientData.emit(mapData[patient])
            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release handle to the webcam
        self.video_capture.release()    


    def recognize_face(self, frame, known_face_encodings, known_face_names) :
        face_locations = []
        face_encodings = []
        # Find all the faces and face encodings in the current frame of video
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                
            face_names.append(name)
            
        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            #cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            #font = cv2.FONT_HERSHEY_DUPLEX
            #cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        return face_names[0] if len(face_names)>0  else ""


    def switchToBarcode(self):
        self.readBarcode = not self.readBarcode
        self.process_this_frame = True       


    def capture(self):
        ret, frame = self.video_capture.read()
        if ret:
                # https://stackoverflow.com/a/55468544/6622587
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.capturedImage.emit(p)
        
    def read_barcodes(self, frame) -> typing.Any:
        barcodes = pyzbar.decode(frame)
        barcode_info = "Unknown"
        for barcode in barcodes:
            x, y , w, h = barcode.rect
            #1
            barcode_info = barcode.data.decode('utf-8')
            cv2.rectangle(frame, (x, y),(x+w, y+h), (0, 255, 0), 2)
            
            #2
            #cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            #font = cv2.FONT_HERSHEY_DUPLEX
            #cv2.putText(frame, barcode_info, (x + 6, y - 6), font, 2.0, (255, 255, 255), 1)
            #3
            # with open("barcode_result.txt", mode ='w') as file:
            #     file.write("Recognized Barcode:" + barcode_info)
        
        return barcode_info


class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'PyQT5 - Sample Detection App'
        self.left = 100
        self.top = 100
        self.width = 640
        self.height = 480
        self.initUI()

    @pyqtSlot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))

    @pyqtSlot(QImage)
    def setPatientImage(self, image):
        self.capturedImage.setPixmap(QPixmap.fromImage(image))
    
    @pyqtSlot(map)
    def setPatientData(self, patientData):
        self.patientDetails.setText("Name: Mr."+ patientData["fullName"])
        self.patientDob.setText("DoB: "+ patientData["DoB"])
        self.vaccinationStatus.setText("Vaccinated: "+ patientData["vaccinated"])
        if "pic" in patientData:
            pixmap = QPixmap(patientData["pic"])
            pixmap4 = pixmap.scaled(256, 256, QtCore.Qt.KeepAspectRatio)
            self.patientPhoto.setPixmap(pixmap4)        
            self.patientPhoto.resize(pixmap4.width(),
                          pixmap4.height())
        if (patientData["fullName"] != "Unable to recognize"):
            self.notMe.setVisible(True)
        else:
            self.notMe.setVisible(False)

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.resize(1800, 1200)

        # create a label
        self.header = QLabel(self)
        self.header.move(100, 0)
        self.header.setFont(QFont('Roboto', 24))
        self.header.setText("Welcome To Face Detection App. \n This is an automated check-in App which will perform a facial recognition or barcode scanner to identify the patient.")
        #self.header.setProperty('class', 'success')
        self.header.setAlignment(QtCore.Qt.AlignCenter)

        # create a label
        self.switchToBarcode = QPushButton('Switch Barcode / Face Recognition', self)
        self.switchToBarcode.move(1000, 200)
        self.switchToBarcode.setProperty('class', 'success')
        self.switchToBarcode.resize(350, 100)
        
        self.label = QLabel(self) 
        self.label.move(280, 200)
        self.label.setText("Waiting for encodings to complete and video stream to start.")
        self.label.setProperty('class', 'success')
        self.label.resize(640, 480)

        self.capturedImage = QLabel(self) 
        self.capturedImage.move(280, 600)
        self.capturedImage.resize(640, 480)

        self.patientDetailsText = QLabel(self) 
        self.patientDetailsText.setText("Patient Identification:")
        self.patientDetailsText.move(1000, 250)
        font = QFont('Roboto', 18)
        font.setUnderline (True)
        font.setBold(True)
        self.patientDetailsText.setFont(font)
        #self.patientDetailsText.resize(300, 20)

        self.patientDetails = QLabel(self) 
        self.patientDetails.move(1000, 275)
        self.patientDetails.setFont(QFont('Roboto', 16))
        self.patientDetails.resize(400, 100)
        self.patientDob = QLabel(self) 
        self.patientDob.move(1000, 300)
        self.patientDob.setFont(QFont('Roboto', 16))
        self.patientDob.resize(400, 100)
        self.vaccinationStatus = QLabel(self) 
        self.vaccinationStatus.move(1000, 325)
        self.vaccinationStatus.setFont(QFont('Roboto', 16))
        self.vaccinationStatus.resize(400, 100)

        self.patientPhoto = QLabel(self) 
        self.patientPhoto.move(1300, 250)
        self.patientPhoto.setFont(QFont('Roboto', 16))
        self.patientPhoto.resize(640, 480)


        self.notMe = QPushButton('!! Not Me !!', self)
        self.notMe.setVisible(False)
        self.notMe.move(1000, 550)
        self.notMe.setProperty('class', 'danger')        
        self.notMe.resize(350, 100)
        th = Thread(self)
        
        th.changePixmap.connect(self.setImage)
        th.patientData.connect(self.setPatientData)
        #th.capturedImage.connect(self.setCapturedImage)
        self.switchToBarcode.clicked.connect(th.switchToBarcode)
        th.start()        
        self.show()

if __name__ == '__main__':
    appctxt = ApplicationContext()       # 1. Instantiate ApplicationContext
    ex = App()    
    apply_stylesheet(appctxt.app, theme='dark_cyan.xml')
    exit_code = appctxt.app.exec_()      # 2. Invoke appctxt.app.exec_()
    sys.exit(exit_code)
