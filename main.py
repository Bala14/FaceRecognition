from face_recognize import FaceRecognize
from checkin_qrcode_reader import CheckInQRCodeReader

def qrCodeReader():
    print('Start QR Code Reader')
    checkInQRCodeReader = CheckInQRCodeReader()
    qrCodeData = checkInQRCodeReader.start()
    del checkInQRCodeReader
    print('Patient Detail : ', qrCodeData)

def faceRecognize():
    print('Start Face Recognition')
    faceRecognize = FaceRecognize()
    faceRecognize.initialize()
    faceRecognizedName = faceRecognize.start()
    del faceRecognize
    if len(faceRecognizedName) > 0 and faceRecognizedName[0] != 'Unknown':
        print('Patient Detail : ', faceRecognizedName[0])
    else:
        print('Face Recognition Failed')
        qrCodeReader()

def main():
    faceRecognize()

main()