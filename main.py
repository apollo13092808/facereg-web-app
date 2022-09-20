import argparse
from encode_faces import encode_face
from facereg_image import reg_image
from facereg_webcam import reg_cam
from generate_datasets_mediapipe import generate_dataset

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--count", type=int,
                help="Number of face images need to be captured")
ap.add_argument("-i", "--input", type=str,
                help="Camera/Wecam or Folder of faces captured or Test image")
ap.add_argument("-m", "--model", type=str,
                help="Face detection model to use: either `hog` or `cnn`")
ap.add_argument("-e", "--encodings", type=str,
                help="Path to serialized db of facial encodings")
args = vars(ap.parse_args())


def command_1():
    if args["input"] == "0":
        type = 0
        generate_dataset(type, args["count"])
    else:
        generate_dataset(args["device"], args["count"])


def command_2():
    encode_face(args["input"], args["model"], args["encodings"])


def command_3():
    reg_image(args["input"], args["model"], args["encodings"])


def command_4():
    if args["input"] == "0":
        type = 0
        reg_cam(type, args["model"], args["encodings"])
    else:
        reg_cam(args["input"], args["model"], args["encodings"])


def main():
    """Đang tìm cách chạy terminal
    """
    # Tạo dataset bằng cách chụp ảnh bằng camera
    # Chạy: py .\main.py -i 0 -c 20
    # command_1() # py .\main.py -i 0 -c 20
    
    # Tạo file pickle chứa encodings của khuôn mặt và label (họ tên người đó) tương ứng
    # Chạy: py .\main.py -i <path to datasets folder> -m cnn -e encodings.pickle
    # command_2() # py .\main.py -i <path to datasets folder> -m cnn -e encodings.pickle
    
    # Nhận diện trên ảnh
    # Chạy: py .\main.py -i <path to image> -m hog -e encodings.pickle
    # command_3() 
    
    # Nhận diện trên camera
    # Chạy: py .\main.py -i 0 -m hog -e encodings.pickle
    command_4() 

if __name__ == '__main__':
    main()
