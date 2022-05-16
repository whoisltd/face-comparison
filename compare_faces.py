import cv2
import argparse
import numpy as np
from pathlib import Path
from face_detection import get_face
# from face_compare.images import get_face
from face_compare import model
# from face_compare.model import facenet_model, img_to_encoding

# load model
model1 = model.facenet_model(input_shape=(3, 96, 96))

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
    embedding_one = model.img_to_encoding(face_one, model1)
    embedding_two = model.img_to_encoding(face_two, model1)

    dist = np.linalg.norm(embedding_one - embedding_two)
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
