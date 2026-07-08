# Hi3559AV100 多模态视频融合与触摸交互系统

## 1. 项目简介

本项目基于 **Hi3559AV100** 嵌入式平台，面向可见光、红外、紫外等多源视频输入场景，构建板端实时视频采集、融合显示、触摸交互与网络推流系统。

系统底层基于海思 **MPP** 视频处理框架，完成 VI/VPSS/IVE/VO/HDMI 等模块配置，实现多路视频采集、缩放、融合与 HDMI 实时输出；前端基于 **Qt + HIFB** 实现触摸屏交互界面，支持实时预览、融合模式切换、拍照录像、文件管理等功能入口。

在融合算法方面，项目支持固定权重融合、基于 mask 的区域自适应融合，并引入轻量级神经网络 **TinyFusionMaskNet**，用于区分目标特征区域与背景区域，使系统能够对特征区域进行重点融合，减少背景区域干扰。

---

## 2. 项目目标

本项目最终目标是实现一套完整的板端多模态视频融合系统：

* 支持可见光、红外、紫外多路视频输入；
* 支持 HDMI 实时显示融合结果；
* 支持触摸屏 Qt 界面操作；
* 支持点击进入可见光、红外光、紫外光融合模式；
* 支持拍照、录像、文件管理等基础功能；
* 支持 RTSP 网络推流；
* 支持神经网络辅助判断目标特征区域，仅对目标特征部分进行增强融合；
* 支持板端开机自启动、WiFi/4G 模块初始化和外设联调。

---

## 3. 系统架构

系统采用“后台视频处理进程 + 前台 Qt 触摸界面”的分层架构。

```text
产品入口 ./HDMI
    ├── 后台视频融合进程
    │       ├── VI      多路视频采集
    │       ├── VPSS    视频缩放 / 格式处理
    │       ├── IVE     图像融合加速
    │       ├── VO      视频输出
    │       ├── HDMI    实时显示
    │       └── RTSP    网络推流
    │
    └── 前台 Qt UI 进程
            ├── HIFB 图层叠加
            ├── 触摸菜单
            ├── 模式切换
            ├── 拍照录像
            └── 文件管理
```

采用双进程结构的原因是：MPP 视频输出层和 Qt/HIFB 图形层在板端初始化顺序较敏感，前后台分离后可以保证视频层稳定输出，同时 Qt 界面作为透明图层叠加在视频画面上，系统稳定性更高。

---

## 4. 视频链路设计

### 4.1 当前通道映射

当前调试版本中，通道映射如下：

```text
Chn2 / VpssGrp2：可见光 VIS
Chn3 / VpssGrp3：红外 IR
Chn5 / VpssGrp5：紫外 UV（预留 / 后续接入）
```

### 4.2 后台视频处理流程

```text
摄像头输入
    ↓
VI 采集
    ↓
VPSS 缩放与格式处理
    ↓
IVE 加权融合 / CPU mask 融合
    ↓
VO 输出
    ↓
HDMI 显示
```

当前已实现可见光与红外实时融合：

```text
DstY = 0.60 × VIS + 0.40 × IR
```

其中 Y 平面参与亮度融合，UV 平面保留可见光色度，避免红外色度导致画面偏色。

后续紫外通道接入后，可扩展为三模态融合：

```text
DstY = 0.50 × VIS + 0.40 × IR + 0.10 × UV
```

---

## 5. 神经网络融合方案

项目训练并验证了轻量级融合区域预测网络 **TinyFusionMaskNet**。

### 5.1 网络作用

TinyFusionMaskNet 不直接生成融合图像，而是预测融合区域 mask：

```text
三路输入图像
    ↓
TinyFusionMaskNet
    ↓
目标特征区域 mask
    ↓
区域自适应融合
```

mask 用于判断哪些区域属于目标特征区域，哪些区域属于背景区域。

### 5.2 区域自适应融合策略

背景区域主要保留可见光纹理信息：

```text
Base = 0.94 × VIS + 0.04 × IR + 0.02 × UV
```

目标区域增强红外和紫外特征：

```text
Target = 0.50 × VIS + 0.40 × IR + 0.10 × UV
```

最终融合结果：

```text
Fused = Base × (1 - Mask) + Target × Mask
```

### 5.3 板端部署状态

当前已完成 TinyFusionMaskNet 的板端功能性部署验证：

* PyTorch 权重导出为二进制参数文件；
* ARM 端 C 程序实现网络前向推理；
* 板端输出 mask 与离线 PyTorch 输出结果完成一致性验证；
* 支持后续接入后台实时视频融合进程。

由于当前 ARM CPU 浮点推理速度较慢，实时系统中建议采用低频 mask 更新方式：

```text
视频融合：每帧实时执行
网络推理：每隔数秒更新一次 mask
```

后续可进一步通过 NNIE / HiSVP / 模型量化等方式提升神经网络推理速度。

---

## 6. Qt 触摸界面

前端界面基于 Qt 开发，通过 HIFB 图形层叠加在 HDMI 视频画面上。

主要功能包括：

* 实时预览；
* 三路融合入口；
* 模式切换；
* 拍照；
* 录像；
* 文件管理；
* 系统状态显示；
* 时间显示；
* 返回桌面；
* 后续扩展 WiFi/4G、存储、电量、传感器信息显示。

Qt 界面不直接处理视频帧，视频处理由后台 MPP 进程完成。Qt 只负责交互控制和图层显示，从而降低界面层与视频链路之间的耦合。

---

## 7. RTSP 推流

项目实现了融合视频的 RTSP 推流功能，可将板端处理后的实时视频流通过网络输出，用于远程预览、上位机接入或多终端显示。

RTSP 推流链路：

```text
融合视频帧
    ↓
编码模块
    ↓
RTSP Server
    ↓
客户端拉流播放
```

该功能依赖网络初始化，支持有线网络、WiFi 或 4G 模块接入。

---

## 8. 板端初始化与外设支持

项目涉及板端系统初始化与外设移植调试，主要包括：

* Hi3559AV100 SDK 移植；
* MPP 运行环境配置；
* HDMI 显示初始化；
* Qt 运行环境配置；
* HIFB 图层调试；
* WiFi 模块移植；
* 4G 模块移植；
* NFS 调试；
* 开机启动脚本配置；
* 程序自启动配置；
* 摄像头通道映射与调试；
* 串口、GPIO、存储等外设扩展预留。

---

## 9. 目录结构

项目主要目录结构如下：

```text
hisi3559-gdc-fusion/
├── board/
│   ├── mm5_static_fusion.c          # 板端静态三模态融合验证
│   ├── tinyfusion_cpu_infer.c       # TinyFusionMaskNet ARM CPU 推理
│   └── ...
│
├── train/
│   ├── train_tiny_fusion_mask.py    # TinyFusionMaskNet 训练
│   ├── predict_tiny_fusion_mask.py  # mask 预测
│   └── ...
│
├── tools/
│   ├── prepare_mm5_dataset.py       # 数据集预处理
│   ├── export_tinyfusion_board_package.py
│   ├── check_board_fusion_consistency.py
│   └── ...
│
├── qt_project/
│   └── hichuqt/                     # Qt 触摸界面工程
│
├── scripts/
│   ├── build_gdc_pmf.sh             # GDC/PMF 相关编译脚本
│   └── ...
│
├── docs/
│   └── experiments/                 # 实验结果与文档
│
└── README.md
```

---

## 10. 编译说明

### 10.1 板端 C 程序编译

示例：

```bash
cd /home/xia/hisi3559/hisi3559-gdc-fusion/board

aarch64-himix100-linux-gcc -O2 -Wall \
  mm5_static_fusion.c \
  -o /home/xia/nfs/mm5_static_fusion \
  -lrt
```

### 10.2 后台视频融合程序编译

进入对应 sample 目录：

```bash
cd /home/xia/Hi3559AV100_SDK/Hi3559AV100_SDK_V2.0.3.1/mpp_arm/sample/vio_7K_1080P60_4K30_1080P30_7A_1080P30x2_720x576
```

编译：

```bash
make OSTYPE=linux clean || true
make OSTYPE=linux -j4
```

拷贝到 NFS：

```bash
cp sample_vio_7K_1080P60_4K30_1080P30_7A_1080P30x2_720x576 \
   /home/xia/nfs/sample_vio_vis_ir_fusion

chmod +x /home/xia/nfs/sample_vio_vis_ir_fusion
```

### 10.3 Qt 界面编译

进入 Qt 工程目录：

```bash
cd ~/qt_project/hichuqt
```

使用板端 Qt 交叉编译工具：

```bash
rm -f Makefile .qmake.stash
rm -f *.o moc_*.cpp moc_*.o HDMI_ui

/home/xia/tools/qt/qt5.12.7_hi3559av100_release/bin/qmake hichuqt.pro

make clean || true
make -j4
```

拷贝到 NFS：

```bash
cp HDMI_ui /home/xia/nfs/HDMI_ui
chmod +x /home/xia/nfs/HDMI_ui
```

---

## 11. 启动方式

板端运行：

```bash
cd /get
chmod +x HDMI
chmod +x HDMI_ui
chmod +x sample_vio_vis_ir_fusion

./HDMI
```

启动脚本 `HDMI` 会依次完成：

```text
1. 清理旧进程
2. 启动后台视频融合进程
3. 等待 /dev/fb0 就绪
4. 启动 Qt/HIFB 前台界面
5. Qt 退出后关闭后台视频进程
```

示例启动脚本核心逻辑：

```bash
BACKEND=/get/sample_vio_vis_ir_fusion
UI=/get/HDMI_ui

$BACKEND &
BACKEND_PID=$!

while [ ! -e /dev/fb0 ]; do
    sleep 0.2
done

export TZ=CST-8
export QT_QPA_PLATFORM=linuxfb:fb=/dev/fb0:size=1920x1080
export QT_QPA_FB_DRM=0

$UI

kill -9 $BACKEND_PID
```

---

## 12. 当前已完成内容

* Hi3559AV100 SDK 移植与板端环境配置；
* HDMI 视频输出链路调试；
* Qt/HIFB 触摸界面叠加显示；
* 前后台分离式产品运行架构；
* 可见光与红外双路实时融合；
* IVE 加权融合输出；
* 静态三模态 mask 融合验证；
* TinyFusionMaskNet 训练与板端功能性推理验证；
* RTSP 推流功能；
* WiFi/4G 模块移植与初始化；
* 开机自启动脚本配置；
* 摄像头通道映射、NFS 程序同步、显示异常等问题排查。

---

## 13. 后续计划

* 接入正常红外相机并稳定验证 VIS + IR 实时融合；
* 接入紫外通道，实现 VIS + IR + UV 三路融合；
* 将静态 mask 融合逻辑迁移到实时视频融合线程；
* 将 TinyFusionMaskNet 作为低频 mask 更新模块接入后台视频融合进程；
* 优化网络推理速度，评估 NNIE / HiSVP 加速方案；
* 完善 Qt 菜单控制逻辑，实现融合模式切换、拍照录像和文件管理闭环；
* 完善 RTSP 推流与本地录像联动；
* 完善系统开机自启动和异常恢复机制。

---

## 14. 技术关键词

```text
Hi3559AV100
嵌入式 Linux
Hisilicon MPP
VI / VPSS / IVE / VO
HDMI
HIFB
Qt
RTSP
WiFi / 4G
TinyFusionMaskNet
多模态图像融合
可见光 / 红外 / 紫外
ARM 端推理
开机自启动
```
