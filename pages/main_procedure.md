# 計算機環境
[OS]: Ubuntu 16.04.5 LTS

[Python]: 3.5.2 / 2.7.15

# 流程
## 1.擷取視頻串流成圖像
> Input: rtsp位址
> Output: 圖像(001.png, 002.png, ...)

**串流位址：(點此查閱表格 / 或聯繫@朝雅)**

```bash
avconv -rtsp_transport tcp -i [rtsp地址] -r [每秒幾禎] -f image2 %04d.png
# [Example]
# (直接在當前目錄下生成圖像)
# avconv -rtsp_transport tcp -i rtsp://admin:unicom0593@175.42.64.76:554/Streaming/Channels/301 -r 30 -f image2 %04d.png
```

## 2.行人偵測模型，偵測人數
* **務必先完成detectron環境安裝及相關源代碼改動**(見[detectron安裝文件](object_detection.md))
> Input: (圖像 or 放置圖像的目錄) + 模型權重檔 + 設定檔
> Output: 標記圖像(於輸出目錄下) + pickle檔案(紀錄人數信息)

```python
# (先進入detectron目錄)
python2 tools/infer_simple.py \
--cfg [設定檔位置] \
--output-dir [輸出目錄] \
--image-ext [圖像來源格式] \
--output-ext [圖像輸出格式] \
--wts [模型權重檔] \
[圖像來源目錄]

# [Example] (須先進入detectron目錄)
# python2 tools/infer_simple.py \
# --cfg configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \
# --output-dir /tmp/detectron-visualizations \
# --image-ext png \
# --output-ext png \
# --wts https://s3-us-west-2.amazonaws.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.#SgT4y1cO/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl \
# ../template/
```

## 3.平滑化數據
> Input: pickle檔案(紀錄人數信息)
> Output: 圖像

```python
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np

# 平滑化函式
def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

with open('8_stats.pkl', 'rb') as f:
    person_stat = pkl.load(f)
    res = moving_average(person_stat, n=10) # 每10個frame
    plt.plot(res)
    plt.show()

```

# 其他
## 將視頻轉為圖像
```python
import cv2
import sys
import os

vidcap = cv2.VideoCapture(sys.argv[1])
success, image = vidcap.read()
count = 0
success = True

while success:
  path = os.path.join(sys.argv[2], str(count) + ".png")
  # save frame as PNG file
  cv2.imwrite(path, image)
  success, image = vidcap.read()
  count += 1

```

## 將圖像轉為視頻
```python
import cv2
import argparse
import os
import re
from tqdm import tqdm

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-ext", "--extension", required=False, default='png', help="extension name. default is 'png'.")
ap.add_argument("-o", "--output", required=False, default='output.mp4', help="output video file")
args = vars(ap.parse_args())

# Arguments
dir_path = '/tmp/detectron-visualizations' # 更改圖像來源處
ext = args['extension']
output = args['output']

images = []
for f in os.listdir(dir_path):
    if f.endswith(ext): images.append(f)

images=sorted(images, key=lambda x: int(re.sub("\D", "", x)))

# Determine the width and height from the first image
image_path = os.path.join(dir_path, images[0])
frame = cv2.imread(image_path)
height, width, channels = frame.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
out = cv2.VideoWriter(output, fourcc, 20.0, (width, height))

for image in tqdm(images):
    image_path = os.path.join(dir_path, image)
    frame = cv2.imread(image_path)
    out.write(frame) # Write out frame to video

# Release everything if job is finished
out.release()
cv2.destroyAllWindows()

print("The output video is {}".format(output))
```

## 廣播視頻(視頻 -> rtsp串流)
```bash
git clone https://github.com/revmischa/rtsp-server.git
cd rtsp-server
sudo ./rtsp-server.pl
cd
ffmpeg -re -i [視頻位置] -f rtsp -muxdelay 0.1 -strict -2 [服務器地址及端口]

# [Example]
# ffmpeg -re -i ./test_live_stream.mp4 -f rtsp -muxdelay 0.1 -strict -2 rtsp://112.112.112.112:5545/abc
# 觀看串流直接在網址列輸入rtsp://112.112.112.112/abc(須先安裝VLC播放套件)
```
