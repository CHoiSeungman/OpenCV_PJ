from flask import Flask, render_template, Response
import cv2
import numpy as np
import random
import datetime

app = Flask(__name__)

# 이미지 경로
img_path = 'C:\\opencv\\project_230915\\images\\face1.png'

# 이미지 로드 및 리사이징
img = cv2.imread(img_path)
img = cv2.resize(img, dsize=(500, 700), interpolation=cv2.INTER_LINEAR)
#filtered_image=None

# 이미지 필터 함수
def apply_filter(filter_type):
    global filtered_image#+++
    if filter_type=='nomal':
        filtered_image=nomal_filter()
        return filtered_image
    elif filter_type == 'canny':
        filtered_image = canny_filter()
        return filtered_image
    elif filter_type == 'sobel':
        filtered_image = sobel_filter()
        return filtered_image
    elif filter_type == 'random':
        filtered_image = random_filter()
        return filtered_image
    elif filter_type == 'sobel_line':
        filtered_image = sobel_line_filter()
        return filtered_image
    elif filter_type == 'save_file':
        return save_file(filtered_image)  # 이미지 저장 함수에 filtered_image를 전달합니다. ++++++
    
## 원본 필터 함수
def nomal_filter():
    size_image = np.shape(img)
    blank_image = np.zeros((size_image[0], size_image[1], 3), np.uint8)
    blank_image.fill(255)
    
    for i in range(90, 660, 30):
        cv2.line(blank_image, (50, i), (blank_image.shape[1] - 50, i), (0, 0, 0))
        
    filtered_image = np.hstack((img, blank_image))
    return filtered_image
        
# 캔니 필터 함수
def canny_filter():
    global filtered_image##+++
    img1=cv2.GaussianBlur(img,(3,3),0)
    edge=cv2.Canny(img1,50,100)
    #이미지 병합
    size_image=np.shape(edge)
    blank_image=np.zeros((size_image[0], size_image[1]),np.uint8)
    
    #엽서 직선라인 긋기
    for i in range(90,660,30):
        cv2.line(blank_image,(50,i),(size_image[1]-50,i),(255,255,255))
        
    filtered_image = np.hstack((edge, blank_image))
    return filtered_image

# 소벨 필터 함수
def sobel_filter():
    global filtered_image#+++
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)

    mag = cv2.magnitude(dx, dy)
    mag = np.clip(mag, 30, 255).astype(np.uint8)

    size_image=np.shape(mag)
    blank_image=np.zeros((size_image[0], size_image[1]),np.uint8)

    for i in range(90,660,30):
        cv2.line(blank_image,(50,i),(size_image[1]-50,i),(255,255,255))

    filtered_image = np.hstack((mag, blank_image))
    return filtered_image

# 채널 분리 랜덤 필터 함수
def random_filter():
    global filtered_image##+++
    b, g, r = cv2.split(img)
    h, w, _ = img.shape
    zero = np.zeros((h, w, 1), dtype=np.uint8)

    channels = [b, g, r, zero]## 4개의 채널 추출

    # 중복 허용 3개까지
    channels_select = 3
    selected_channels = random.choices(channels, k=channels_select)
    #if num_channels_to_select>=4:
        #print("채널수가 많습니다.")

    # 선택된 채널을 병합
    merged_image = cv2.merge(selected_channels)

    blank_image = np.zeros((h,w,3),dtype=np.uint8)
    blank_image.fill(255)
    for i in range(90, 660, 30):
        cv2.line(blank_image, (50, i), (blank_image.shape[1] - 50, i), (0, 0, 0))

    # 가로연결
    filtered_image = np.hstack((merged_image, blank_image))
    return filtered_image
        

# 소벨 라인 랜덤 필터 함수
def sobel_line_filter():
    global filtered_image##++++++
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#<- 이 부분 중요
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)

    # 그래디언트 크기 계산 및 범위 제한
    mag = cv2.magnitude(dx, dy)
    mag = np.clip(mag, 30, 255).astype(np.uint8)
    
    # size_image에 mag와 같은 크기 입력
    size_image = mag.shape

    # 3채널 이미지 생성
    dst = np.zeros((size_image[0], size_image[1], 3), dtype=np.uint8)

    # 랜덤 RGB 컬러 생성
    random_color = [random.randint(0, 255) for _ in range(3)]
    #print(f"Random Color: {random_color}")

    # 하얀색 픽셀에 랜덤 컬러 적용
    for white_pixels in range(100, 256):
        want_pixels = np.where((mag == white_pixels))
        dst[want_pixels[0], want_pixels[1], :] = random_color

    # mag 이미지를 3채널로 변환
    mag_color = cv2.cvtColor(mag, cv2.COLOR_GRAY2BGR)
    
    # 이미지 가중치 활용
    addweight_img = cv2.addWeighted(mag_color, 0.6, dst, 0.4, 0, 0)

    blank_image = np.zeros((size_image[0], size_image[1], 3), dtype=np.uint8)
    blank_image.fill(0)
    
    #선 긋기
    for i in range(90, 660, 30):
        cv2.line(blank_image, (50, i), (size_image[1] - 50, i), (255, 255, 255))

    # 가로 연결
    filtered_image = np.hstack((addweight_img, blank_image)) 
    return filtered_image

## 원하는 경로로 저장
def save_file(filtered_image):
    now = datetime.datetime.now()
    Name = now.strftime("%y%m%d%M%S")
    output_path='C:\\opencv\\project_230915\\save_image\\'+Name+'.png'
    cv2.imwrite(output_path,filtered_image)
    return filtered_image##++++++

@app.route('/')
def index():
    return render_template('index.html', image_url='')

@app.route('/filter/<filter_type>')
def apply_image_filter(filter_type):
    filtered_img = apply_filter(filter_type)
    _, img_encoded = cv2.imencode('.jpg', filtered_img)
    img_bytes = img_encoded.tobytes()
    
    return Response(img_bytes, mimetype='image/jpeg')

if __name__ == '__main__':
    app.debug=True
    app.run(host="0.0.0.0")
    