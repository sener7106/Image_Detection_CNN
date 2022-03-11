import os 
import pathlib
import glob
import settings
import cv2

def load_name_images(image_path_pattern) :
    name_images=[]
    # 지정한 Path Pattern과 일치하는 파일 찾기
    # glob 사용자 조건에 맞는 문자열을 리스트 형식으로 반환
    image_paths = glob.glob(image_path_pattern)
    
    # 파일 Read
    for image_path in image_paths :
        path = pathlib.Path(image_path)
        # FilePath
        fullpath = str(path.resolve())
        print(f"이미지 파일(절대 경로): {fullpath}")
        # 파일명
        filename = path.name
        print(f"이미지 파일 (이름) : {filename}")
        # 이미지 읽기
        image = cv2.imread(fullpath)
        if image is None :
            print(f"이미지 파일[{fullpath}]를 읽을 수 없습니다.")
            continue
        # 이미지파일을 tuple 형식으로 저장
        name_images.append((filename, image))
    return name_images
        

def detect_image_face(file_path, image, cascade_filepath) :
    # 이미지 파일을 회색조로 변경
    image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Cascade File 읽기
    cascade = cv2.CascadeClassifier(cascade_filepath)
    # 얼굴인식
    faces = cascade.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=15, minSize=(64, 64))
    if len(faces) == 0 :
        print("얼굴 인식 실패")
        return
    # 1개 이상의 얼굴을 인식한 경우
    face_count = 1
    for (xpos, ypos, width, height) in faces:
        face_image = image[ypos:ypos+height, xpos:xpos+width]
        if face_image.shape[0] > 64:
            face_image = cv2.resize(face_image, (64,64))
        print(face_image.shape)
        # 저장
        path = pathlib.Path(file_path)
        directory = str(path.parent.resolve())
        filename = path.stem
        extension = path.suffix
        output_path = os.path.join(directory, f"{filename}_{face_count:03}{extension}")
        print(f"출력 파일 (절대경로) : {output_path}")
        cv2.imwrite(output_path, face_image)
        face_count = face_count + 1

def delete_dir(dir_path, is_delete_top_dir=True):
    for root, dirs, files in os.walk(dir_path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    if is_delete_top_dir:
        os.rmdir(dir_path)

RETURN_SUCCESS = 0
RETURN_FAILURE = -1
# Origin Image Pattern
IMAGE_PATH_PATTERN = "./origin_image/*"
# Output Directory
OUTPUT_IMAGE_DIR = "./face_image"

def main() :
    print("=" * 10)
    print("이미지 얼굴 인식 Opencv")
    print("지정한 이미지 파일의 얼굴을 인식하여 추출하고, 사이즈를 변환합니다.")
    print("=" * 10)
    
    # 디렉토리 작성
    if not os.path.isdir(OUTPUT_IMAGE_DIR) :
        os.mkdir(OUTPUT_IMAGE_DIR)
    # 디렉토리 내 파일 삭제
    delete_dir(OUTPUT_IMAGE_DIR, False)
    
    # 이미지 파일 읽기
    name_images = load_name_images(IMAGE_PATH_PATTERN)
    
    # 이미지 얼굴 인식
    for name_image in name_images:
        file_path = os.path.join(OUTPUT_IMAGE_DIR, f"{name_image[0]}")
        image = name_image[1]
        cascade_filepath = settings.CASCADE_FILE_PATH
        detect_image_face(file_path, image, cascade_filepath)
        
    return RETURN_SUCCESS

if __name__ == "__main__" :
    main()
        
    
    