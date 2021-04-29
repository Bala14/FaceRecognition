import cv2
import numpy as np
from pyzbar.pyzbar import decode

class CheckInQRCodeReader:

    capture_image = None
    qrcode_data = []
    qrcode_captured = False

    def __del__(self):
        print('Check In QR Code Object Destroyed')

    def start(self):
        self.capture_image = cv2.VideoCapture(0)
        while not self.qrcode_captured:
            ret, image = self.capture_image.read()
            gray_img = cv2.cvtColor(image, 0)
            qrcode = decode(gray_img)
            for obj in qrcode:
                points = obj.polygon
                (x, y, w, h) = obj.rect
                pts = np.array(points, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(image, [pts], True, (0, 255, 0), 3)
                qrcode_text = obj.data.decode("utf-8")
                self.qrcode_data = qrcode_text.split(',')
                self.showCapturedImage(image)
                self.qrcode_captured = True
            self.showCapturedImage(image)
        self.close()
        return self.qrcode_data

    def showCapturedImage(self, image):
        cv2.imshow('Image', image)
        cv2.waitKey(10)

    def close(self):
        self.capture_image.release()
        cv2.destroyAllWindows()