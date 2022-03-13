import sys
import os
import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt
import settings

def detect_face(model, cascade_filepath, image):
    # 이미지를 BGR형식에서 RGB형식으로 변환
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # plt.imshow(image)
    # plt.show()
    # print(image.shape)
    
    # 그레이스케일로 이미지로 변환
    image_gs = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # 얼굴인식 실행   
    cascade = cv2.CascadeClassifier(cascade_filepath)
    faces = cascade.detectMultiScale(
        image_gs, scaleFactor=1.1, minNeighbors=2, minSize=(64, 64))
    
    # 얼굴이 1개 이상을 검출된 경우
    if len(faces) > 0 :
        print(f"인식한 얼굴의 수 : {len(faces)}")
        for (xpos, ypos, width, height) in faces:
            # 1. 인식된 얼굴 자르기
            face_image = image[ypos:ypos+height, xpos:xpos+width]
            print(f"인식한 얼굴의 사이즈 : {face_image.shape}")
            # 2. 인식한 얼굴 사이즈 축소
            if face_image.shape[0] < 64 or face_image.shape[1] < 64 :
                print("인식한 얼굴의 사이즈가 너무 작습니다.")
                continue
            face_image = cv2.resize(face_image, (64,64))
            # 3. 인식한 얼굴 주변에 빨간색 사각형을 표시
            # image, x, y, width, height, color, thickness
            cv2.rectangle(image, (xpos,ypos), (xpos+width, ypos+height), (255, 0, 0), thickness=2)
            # dimension 
            face_image = np.expand_dims(face_image, axis=0)

            # 4. 인식한 얼굴의 이름 가져오기
            name = detect_who(model, face_image)
            # 5. 인식한 얼굴에 이름 표시
            cv2.putText(image, name, (xpos, ypos+height+20), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
            
    else :
        print("얼굴을 인식할 수 없습니다.")
            
            
def detect_who(model, face_image):
    # 예측
    name = ""
    result = model.predict(face_image)
    print(f"예측결과: {result}")
    # 전지현과 송혜교일 가능성을 출력
    print(f"전지현일 가능성:{result[0][0]*100:.3f}%")
    print(f"송혜교일 가능성:{result[0][1]*100:.3f}%")
    # 이름 반환
    name_number_label = np.argmax(result)
    if name_number_label == 0:
        name = "Jihyun"
    elif name_number_label == 1:
        name = "Hyegyo"

    return name

RETURN_SUCCESS = 0
RETURN_FAILURE = -1
# Inpou Model Directory
INPUT_MODEL_PATH = "./model/model.h5"

def main():
    print("=" * 10)
    print("Keras를 이용한 얼굴인식")
    print("학습 모델과 지정한 이미지 파일로 연예인 구분하기")
    print("=" * 10)

    # 인수 체크
    argvs = sys.argv
    if len(argvs) != 2 or not os.path.exists(argvs[1]):
        print(
            "이미지 파일을 지정해 주세요. Uasge) python 05.img_face_judgement_lab [이미지 경로]")
        return RETURN_FAILURE
    image_file_path = argvs[1]

    # 이미지 파일 읽기
    image = cv2.imread(image_file_path)
    if image is None:
        print(f"이미지 파일을 읽을 수 없습니다. {image_file_path}")
        return RETURN_FAILURE

    # 모델 파일 읽기
    if not os.path.exists(INPUT_MODEL_PATH):
        print("이미지를 검출하기 위한 모델 파일이 없습니다.")
        return RETURN_FAILURE
    model = keras.models.load_model(INPUT_MODEL_PATH)

    # 얼굴인식
    cascade_filepath = settings.CASCADE_FILE_PATH
    result_image = detect_face(model, cascade_filepath, image)
    plt.imshow(result_image)
    plt.show()

    return RETURN_SUCCESS


if __name__ == "__main__":
    main()
