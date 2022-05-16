from mtcnn import MTCNN
import cv2

def get_face(img):
    '''Crops image to only include face plus a border'''
    height, width, _ = img.shape
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    face_box = face_cascade.detectMultiScale(img)
    # Get dimensions of bounding box
    x, y, w, h = tuple(map(tuple, face_box))[0]
    # Calculate padding as segmentation is too tight.eE
    pad_w = int(w/2.5)
    pad_h = int(h/2.5)
    # Get co-ordinates of crop
    x1 = max(0, x-pad_w)
    y1 = max(0, y-pad_h)
    x2 = min(width, x+w+pad_w)
    y2 = min(height, y+h+pad_h)
    # draw
    draw = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Crop image
    cropped = img[y1:y2,x1:x2]
    return cropped
    # print(type(img))
    img1 = cv2.imread(img)
    # print(img)
    detector = MTCNN()
    faces = detector.detect_faces(img1)
    for face in faces:
        x, y, w, h = face['box']
        cropped = img1[y:y+h, x:x+w]
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.imshow('image', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # break
        return cropped
    return None

if __name__ == '__main__':
    img = '/home/whoisltd/works/other/face-comparison/ben2.jpg'
    cropped = get_face(img)
    cv2.imshow('image', cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()