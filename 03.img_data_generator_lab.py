import os
import random
import pathlib
import shutil
import glob
import cv2
import numpy as np

def load_name_images(image_path_pattern) :
    name_images = []
    # 지정한 path pattern과 일치하는 파일을 찾기
    image_paths = glob.glob(image_path_pattern)
    # 파일 읽기
    for image_path in image_paths :
        path = pathlib.Path(image_path)
        # File Path
        fullpath = str(path.resolve())
        print(f"이미지 파일 (절대경로) : {fullpath}")
        # 파일명
        filename = path.name
        print("이미지 파일(이름) : {filename}")
        # 이미지 읽기
        image = cv2.imread(fullpath)
        if image is None :
            print(f"이미지 파일[{fullpath}]을 읽을 수 없습니다.")
            continue
        name_images.append((filename, image))
    return name_images

def scratch_image(image, use_flip=True, use_threshold=True, use_filter=True) :
    # 어떤 방법을 적용할 것인가. (flip, blur)
    methods = [use_flip, use_threshold, use_filter]
    # 증강에 사용할 필터 작성
    # filter1 = np.ones((3, 3))
    # 오리지널 이미지를 배열에 저장
    images = [image]
    # 증강 방법의 함수
    scratch = np.array([
        # Flip
        lambda x : cv2.flip(x,1),
        # thresholding
        lambda x : cv2.threshold(x, 100, 255, cv2.THRESH_TOZERO)[1],
        # Blur
        lambda x: cv2.GaussianBlur(x, (5,5), 0)
    ])
    
    # 이미지 증가 row로 붙이기
    doubling_images = lambda f, img : np.r_[img, [f(i) for i in img]]
    for func in scratch[methods] :
        images = doubling_images(func, images)
    return images

def delete_dir(dir_path, is_delete_top_dir=True) :
    for root, dirs, files in os.walk(dir_path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    if is_delete_top_dir:
        os.rmdir(dir_path)

RETURN_SUCCESS = 0
RETURN_FAILURE = -1
# Test Image Directory
TEST_IMAGE_PATH = './test_image'
# Face Image Directory
IMAGE_PATH_PATTERN = "./face_image/*"
# Output Directory
OUTPUT_IMAGE_DIR = "./face_scratch_image"

def main() :
    print("=" * 10)
    print("이미지 증강, OPENCV")
    print("지정 이미지를 증강합니다. (Flip, Blur)")
    print("=" * 10)
    
    # 디렉토리 작성
    if not os.path.isdir(OUTPUT_IMAGE_DIR) :
        os.mkdir(OUTPUT_IMAGE_DIR)
    # 디렉토리 내 삭제
    delete_dir(OUTPUT_IMAGE_DIR, False)
    
    # 대상 이미지 중 20%를 테스트용으로 분류
    image_files = glob.glob(IMAGE_PATH_PATTERN)
    random.shuffle(image_files)
    for i in range(len(image_files)//5) :
        shutil.move(str(image_files[i]), TEST_IMAGE_PATH)


    # 이미지 파일 읽기
    name_images = load_name_images(IMAGE_PATH_PATTERN)
    
    # 이미지 증강
    for name_image in name_images :
        filename, extension = os.path.splitext(name_image[0])
        image = name_image[1]
        # 이미지 증강
        scratch_face_images = scratch_image(image)
        # 이미지 보존
        for idx, image in enumerate(scratch_face_images) :
            output_path = os.path.join(OUTPUT_IMAGE_DIR, f"{filename}_{str(idx)}{extension}")
            print(f"출력 파일 (절대경로) : {output_path}")
            cv2.imwrite(output_path, image)
        
    return RETURN_SUCCESS
        
if __name__ == "__main__" :
    main()