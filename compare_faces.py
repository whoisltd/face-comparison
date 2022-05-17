import cv2
import argparse
import numpy as np
from pathlib import Path
from face_detection import get_face
# from face_compare.images import get_face
# from face_compare import model
from facenet import loadModel
import tensorflow
# from face_compare.model import facenet_model, img_to_encoding

# load model
# model1 = model.facenet_model(input_shape=(3, 96, 96))

def img_to_encoding(image, model):
    # Resize for model
    factor_0 = 160 / image.shape[0]
    factor_1 = 160 / image.shape[1]
    factor = min(factor_0, factor_1)

    dsize = (int(image.shape[1] * factor), int(image.shape[0] * factor))
    resized = cv2.resize(image, dsize)

    diff_0 = 160 - resized.shape[0]
    diff_1 = 160 - resized.shape[1]
    img = np.pad(resized, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)), 'constant')

    # if img.shape[0:2] != (160, 160):
    #     img = cv2.resize(img, (160, 160))

    img_pixels = tensorflow.keras.preprocessing.image.img_to_array(img) #what this line doing? must?
    
    img_pixels = np.expand_dims(img_pixels, axis = 0)
    
    # img_pixels /= 255
    # img_pixels *= 255
    img_pixels /= 127.5
    img_pixels -= 1
    
    embedding = model.predict(img_pixels)[0].tolist()
    # resized = cv2.resize(image, (160, 160))
    # Swap channel dimensions
    # input_img = resized[...,::-1]
    # # Switch to channels first and round to specific precision.
    # input_img = np.around(np.transpose(input_img, (2,0,1))/255.0, decimals=12)
    # x_train = np.array([input_img])
    # embedding = model.predict_on_batch(x_train)
    return embedding
def findEuclideanDistance(source_representation, test_representation):
    if type(source_representation) == list:
        source_representation = np.array(source_representation)

    if type(test_representation) == list:
        test_representation = np.array(test_representation)

    euclidean_distance = source_representation - test_representation
    # print("1:", euclidean_distance)
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    # print("2:", euclidean_distance)
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

def l2_normalize(x):
    return x / np.sqrt(np.sum(np.multiply(x, x)))

def run(image_one, image_two, save_dest=None):
    # Load images
    image1 = cv2.imread(str(image_one))
    image2 = cv2.imread(str(image_two))
    face_one = get_face(image1)
    face_two = get_face(image2)

    # Optionally save cropped images
    if save_dest is not None:
        print(f'Saving cropped images in {save_dest}.')
        cv2.imwrite(str(save_dest.joinpath('face_one.png')), face_one)
        cv2.imwrite(str(save_dest.joinpath('face_two.png')), face_two)

    # Calculate embedding vectors
    # from face_compare import model
    model1 = loadModel()

    embedding_one = img_to_encoding(face_one, model1)
    embedding_two = img_to_encoding(face_two, model1)
    print('okee:', np.array(embedding_one).shape)
    # Convert list to array
    embedding_one = np.array(embedding_one)
    embedding_two = np.array(embedding_two)
    print(embedding_one.shape)
    # Calculate distance
    dist = findEuclideanDistance(l2_normalize(embedding_one), l2_normalize(embedding_two))

    # dist = np.linalg.norm(embedding_one - embedding_two)
    print(f'Distance between two images is {dist}')
    if dist > 0.7:
        print('These images are of two different people!')
    else:
        print('These images are of the same person!')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Face Comparison Tool')

    ap.add_argument('--image-one', dest='image_one', type=Path, required=True, help='Input Image One')
    ap.add_argument('--image-two', dest='image_two', type=Path, required=True, help='Input Image Two')
    ap.add_argument('-s', '--save-to', dest='save_dest', type=Path, help='Optionally save the cropped faces on disk. Input directory to save them to')
    args = ap.parse_args()

    run(args.image_one, args.image_two, args.save_dest)
