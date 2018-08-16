# 一、計算機環境
[OS]: Ubuntu 16.04.5 LTS  
[Python]: 2.7.15  
[GPU]: GeForce GTX 1080Ti(11GB)兩顆

# 二、安裝所需套件
> Reference：(https://medium.com/@xinh3ng/install-cuda-9-1-and-cudnn-7-for-tensorflow-1-5-0-cda36239bc68) 

* 下載套件(百度網盤地址)  
> tensorflow_gpu_1.5.0-cp27.whl: https://pan.baidu.com/s/1uVSpxtKGSH9QHQo11PWlIg  
> nccl-cuda9.0.deb: https://pan.baidu.com/s/1pT30ppQN7ByuWv_iamemzw  
> libcudnn7.deb: https://pan.baidu.com/s/1DBswFyWRTfdy5CZixuC2hw  
> libcudnn7-dev.deb: https://pan.baidu.com/s/1hick70me6ZngyiuLbeKmsw  
> libcudnn7-doc.deb: https://pan.baidu.com/s/1AixJucHHqloGWfxaVVgtNg  
> cuda-repo-9-0.deb: https://pan.baidu.com/s/12aPiHUfj_K6xJbIXkx9PCA  
> eigen3.zip: https://pan.baidu.com/s/1HSFXgJFaj4wCby8oong8yw / https://download.csdn.net/download/luojie140/10385556    

## 1. 安裝CUDA, CUDNN..等GPU運算套件
```bash
# 安裝cuda-toolkit
$ sudo dpkg -i cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64.deb
$ sudo apt-key add /var/cuda-repo-9-0-local/7fa2af80.pub
$ sudo apt-get update
$ sudo apt-get install cuda-9-0

# 設置路徑
$ export PATH=/usr/local/cuda/bin:${PATH}
$ export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
$ export LD_LIBRARY_PATH=”/usr/local/cuda/extras/CUPTI/lib64":${LD_LIBRARY_PATH}
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"/usr/lib/nvidia-375"
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"/usr/lib/nvidia-396"

# 安裝cudnn的三個套件(須自行下載)
$ sudo dpkg -i libcudnn7_7.0.5.15-1+cuda9.0_amd64.deb
$ sudo dpkg -i libcudnn7-dev_7.0.5.15-1+cuda9.0_amd64.deb
$ sudo dpkg -i libcudnn7-doc_7.0.5.15-1+cuda9.0_amd64.deb

# 安裝nccl套件(需自行下載)
$ sudo dpkg -i nccl-repo-ubuntu1604-2.2.12-ga-cuda9.0_1-1_amd64.deb
$ sudo apt install libnccl2 libnccl-dev

# 安裝NVIDIA CUDA Profile Tools介面
$ sudo apt-get install libcupti-dev
```
## 2. 安裝TENSORFLOW(如不需要tensorflow則跳過此步驟)
```bash
$ sudo apt-get install cuda-command-line-tools
$ export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}/usr/local/cuda/extras/CUPTI/lib64

# 安裝Python2.7版本的tensorlfow(gpu)
$ wget https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.5.0-cp27-none-linux_x86_64.whl 
$ pip2 install tensorflow_gpu-1.5.0-cp27-none-linux_x86_64.whl —user
# 若上述指令失敗，可嘗試以下指令
# pip2 install tensorflow-gpu==1.5 --upgrade --ignore-installed —user

# 安裝Python3.7版本的tensorlfow(gpu)

$ wget https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.9.0-cp35-cp35m-linux_x86_64.whl
$ pip3 install tensorflow_gpu-1.9.0-cp35-cp35m-linux_x86_64.whl —user
# 若上述指令失敗，可嘗試以下指令
# pip3 install tensorflow-gpu==1.5 --ignore-installed —user
```

## 3. 安裝caffe2環境
> [重要!]CAFFE2環境似乎不支援virtualenv，需特別注意

* **請先按[caffe2官方文件](https://caffe2.ai/docs/getting-started.html?platform=ubuntu&configuration=compile)安裝**  

```bash
# 更新eigen版本
$ unzip eigen3.zip

# 如果已存在eigen3套件，則移除
$ sudo rm -rf /usr/include/eigen3/
# [Example] sudo mv ./eigen3 /usr/include/
$ sudo mv [新eigen3目錄位置] /usr/include/

# 補安裝future套件
$ pip2 install future
$ pip3 install future

# 將caffe2位置export進PYTHONPATH變數
# [Example]
# export PYTHONPATH=$PYTHONPATH:/home/r06725053/pytorch/build
$ export PYTHONPATH=$PYTHONPATH:[pytorch套件位置]

# 如果沒有找到hypothesis套件
$ pip2 install hypothesis
```

## 4. 安裝COCO API套件
```bash
# [Example]
# export COCOAPI=/home/r06725053/cocoapi
$ export COCOAPI=[cocoapi套件位置]
$ git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
$ cd $COCOAPI/PythonAPI
$ make install
```

## 5. 安裝DETECTRON套件
```bash
# [Example]
# DETECTRON=/home/r06725053/detectron
$ DETECTRON=[detectron套件位置]
$ git clone https://github.com/facebookresearch/detectron $DETECTRON
$ cd $DETECTRON
$ python setup.py develop --user
```

# 三、修改detectron源代碼
## 1. detectron/detectron/utils/vis.py

```diff
@@ -251,7 +251,7 @@ 輸出圖像預設類型改為png
def vis_one_image(
         im, im_name, output_dir, boxes, segms=None, keypoints=None, thresh=0.9,
         kp_thresh=2, dpi=200, box_alpha=0.0, dataset=None, show_class=False,
-        ext='pdf', out_when_no_box=False):
+        ext='png', out_when_no_box=False):

@@ -288,12 +288,17 @@僅標註行人類別
@@ def vis_one_image(line, color=colors[len(kp_lines) + 1], linewidth=1.0, alpha=0.7)

mask_color_id = 0
for i in sorted_inds:
    bbox = boxes[i, :4]
	score = boxes[i, -1]
    if score < thresh:
		continue
    
+   # filter class except for 'person'
+   if classes[i] != 1:
+		continue

@@ -387,6 +391,6 @@ 修改輸出圖像名稱
@@ def vis_one_image(line, color=colors[len(kp_lines) + 1], linewidth=1.0, alpha=0.7)
...
-   output_name = os.path.basename(im_name) + '.' + ext
+   output_name = 'processed_' + os.path.basename(im_name)
```

## 2. detectron/tools/infer_simple.py
```diff
@@ -45,6 +45,8 @@ 導入regex, pickle兩個套件
@@ import detectron.core.test_engine as infer_engine
 import detectron.datasets.dummy_datasets as dummy_datasets
 import detectron.utils.c2 as c2_utils
 import detectron.utils.vis as vis_utils
+import re
+import pickle as pkl

@@ -125,19 +127,34 @@ 修改輸出圖像名稱、尺寸(增加模型判斷速度)、統計通過threshold(0.7)的行人數量
@@ def main(args):
     im_list = glob.iglob(args.im_or_folder + '/*.' + args.image_ext)
   else:
     im_list = [args.im_or_folder]

+    im_list = list(im_list)
+    im_list = sorted(im_list, key=lambda nm: re.search(r'(\d+)',nm).group(0))
+
+    count_stat = []
     for i, im_name in enumerate(im_list):
         out_name = os.path.join(
-            args.output_dir, '{}'.format(os.path.basename(im_name) + '.' + args.output_ext)
+            args.output_dir, '{}_{}'.format('processed_', os.path.basename(im_name))
         )
         logger.info('Processing {} -> {}'.format(im_name, out_name))
         im = cv2.imread(im_name)
+        scale_ratio = 600.0 / min(im.shape[:2])
+        h, w = int(im.shape[0] * scale_ratio), int(im.shape[1] * scale_ratio)
+        resized_im = cv2.resize(im, (w, h))
+
         timers = defaultdict(Timer)
         t = time.time()
         with c2_utils.NamedCudaScope(0):
             cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
-                model, im, None, timers=timers
+                model, resized_im, None, timers=timers
             )
+            over_threshold = [c for c in cls_boxes[1] if c[4] > 0.7]
+            count_stat.append(len(over_threshold))
         logger.info('Inference time: {:.3f}s'.format(time.time() - t))
         for k, v in timers.items():
             logger.info(' | {}: {:.3f}s'.format(k, v.average_time))

@@ -148,7 +165,7 @@ 修改輸出圖像名稱、尺寸(增加模型判斷速度)、統計通過threshold(0.7)的行人數量
@@ def main(args):
         vis_utils.vis_one_image(
-            im[:, :, ::-1],  # BGR -> RGB for visualization
+            resized_im[:, :, ::-1],  # BGR -> RGB for visualization
             im_name,
             args.output_dir,
             cls_boxes,

@@ -163,6 +180,9 @@ 修改輸出圖像名稱、尺寸(增加模型判斷速度)、統計通過threshold(0.7)的行人數量
@@ def main(args):
             out_when_no_box=args.out_when_no_box
         )

+    with open('../_stats.pkl','wb') as co:
+        pkl.dump(count_stat, co)
+
```
