import face_recognition
import cv2
import numpy as np
import time

class FaceRecognize:

    video_capture = None
    known_face_encodings = []
    known_face_names = []
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    start_time = 0
    end_time = 0
    unknown_count = 0
    max_unknown_count = 10
    max_time_face_recognize = 30
    patient_found = False

    def __del__(self):
        print('Face Recognition Object Destroyed')

    def initialize(self):
        image_loaded = face_recognition.load_image_file("M:/Bala.jpg")
        image_face_encoding = face_recognition.face_encodings(image_loaded)[0]
        self.known_face_encodings = [image_face_encoding]
        self.known_face_names = ['Bala']

    def start(self):
        self.start_time = int(time.time())
        self.end_time = int(time.time()) + self.max_time_face_recognize
        self.video_capture = cv2.VideoCapture(0)
        while self.timeout() and not self.patient_found:
            ret, frame = self.video_capture.read()
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]
            if self.process_this_frame:
                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)
                self.face_names = []
                for face_encoding in self.face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = 'Unknown'
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                    self.face_names.append(name)
            self.process_this_frame = not self.process_this_frame
            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
            cv2.imshow('Image', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if len(self.face_names) > 0:
                if self.face_names[0] != 'Unknown':
                    self.face_names
                    self.patient_found = True
                elif self.face_names[0] == 'Unknown':
                    self.unknown_count += 1
                    if self.unknown_count == self.max_unknown_count:
                        self.face_names
                        self.patient_found = True
            self.start_time = int(time.time())
        self.close()
        return self.face_names

    def close(self):
        self.video_capture.release()
        cv2.destroyAllWindows()

    def timeout(self):
        if self.start_time <= self.end_time:
            return True
        return False