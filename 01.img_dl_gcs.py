
import os
import sys
from urllib.parse import quote
import settings
import requests
import re

# 지정 키워드로 검색한 URL 얻기


def get_image_urls(keyword, total_num):
    image_urls = []
    i = 0
    while i < total_num:
        # Query
        query = CUSTOM_SEARCH_URL + "?key=" + settings.API_KEY + \
            "&cx=" + settings.CUSTOM_SEARCH_ENGINE + "&num=" + \
            str(10 if(total_num-i) > 10 else (total_num-i)) + "&start=" + \
            str(i+1) + "&q=" + quote(keyword) + "&searchType=image"
        print(query)

        # Get requests
        response = requests.get(query)
        # JSON 데이터 얻기
        json = response.json()
        # 10개씩 URL을 저장
        for j in range(len(json["items"])):
            image_urls.append(json["items"][j]["link"])
        i = i + 10
    return image_urls

# 이미지의 URL을 기준으로 이미지를 저장


def get_image_files(dir_path, keyword_count, image_urls):
    # 이미지 Url 순회
    for (idx, image_url) in enumerate(image_urls):
        try:
            # 이미지 다운로드
            print(image_url)
            image = download_image(image_url)
            # 파일명 작성
            # filename_extension_pair = os.path.splitext(image_url)
            # extension = filename_extension_pair[1]
            # extension = extension if len(extension) <= 4 else extension[0:4]
            extension = ".jpg"
            filename = os.path.join(dir_path, f"{keyword_count:02}_{idx+1:03}{extension}")
            print(filename)
            # 영상을 파일로서 지정
            save_image(filename, image)
        except RuntimeError as ex:
            print(f"type:{type(ex)}")
            print(f"args:{ex.args}")
            print(f"{ex}")
            continue
        except BaseException as ex:
            print(f"type:{type(ex)}")
            print(f"args:{ex.args}")
            print(f"{ex}")
            continue


# 이미지 다운로드
def download_image(url):
    response = requests.get(url, timeout=100)
    if response.status_code != 200:
        raise RuntimeError("이미지를 다운로드할 수 없습니다.")
    content_type = response.headers["content-type"]
    if 'image' not in content_type:
        raise RuntimeError("이미지가 아닙니다.")
    return response.content

# 이미지파일을저장


def save_image(filename, image):
    with open(filename, "wb") as file:
        file.write(image)

# 디렉토리 또는 디렉토리 내 파일 삭제


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
# CUSTOM_SEARCH_URL
# Download Directory Path
ORIGIN_IMAGE_DIR = "./origin_image"
CUSTOM_SEARCH_URL = "https://www.googleapis.com/customsearch/v1/"


def main():
    print("=" * 10)
    print("이미지 다운로더 Google Custom Search API")
    print("지정한 키워드로 검색한 이미지를 다운로드합니다.")
    print("=" * 10)

    # 인수체크
    argvs = sys.argv
    if len(argvs) != 2 or not argvs[1]:
        print("키워드를 지정해주세요.   (콤마로 구분 가능)")
        return RETURN_FAILURE

    # 키워드 얻기
    keywords = [x.strip() for x in argvs[1].split(',')]

    # 디렉토리 작성
    if not os.path.isdir(ORIGIN_IMAGE_DIR):
        os.mkdir(ORIGIN_IMAGE_DIR)
    delete_dir(ORIGIN_IMAGE_DIR, False)

    # 각 키워드에 해당하는 이미지 파일 얻기
    keyword_count = 1
    for keyword in keywords:
        # 키워드 표시
        print(f"키워드 [{keyword}] 로 검색한 이미지를 다운로드합니다..")
        # 이미지 URL
        image_urls = get_image_urls(keyword, 100)
        # 이미지 파일 다운로드
        get_image_files(ORIGIN_IMAGE_DIR, keyword_count, image_urls)

        # 카운트
        keyword_count = keyword_count + 1

    return RETURN_SUCCESS


if __name__ == "__main__":
    main()
