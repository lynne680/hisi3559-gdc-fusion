技术路线

1.早期方案采用 IVE 平均融合，即：

Dst = Cam2 × 0.5 + Cam3 × 0.5

该方法速度较快，但没有处理两摄像头之间的视角差、旋转差和透视差，因此画面中存在明显重影和边缘双线。

后续方案引入棋盘格标定和 GDC PMF 透视变换。先通过棋盘格计算 Cam3 到 Cam2 的单应性矩阵，再转换为 GDC PMF 所需的反向映射系数，使 Cam3 画面映射到 Cam2 坐标系下。

当前 PMF 系数为：

scale = 524288
output(Cam2) -> input(Cam3)

529011, 46064, 22108332
-42495, 519698, 10987334
3, 0, 524288
2. 当前融合策略

当前采用 Cam2 主导、Cam3 辅助的弱融合策略。Y 亮度平面进行低权重融合，C 色度平面保持 Cam2 为主，以降低 Cam3 边缘线和色彩错位。

当前权重为：

#define CAM2_WEIGHT_Q16     58982
#define CAM3_WEIGHT_Q16      6554

约等于：

Cam2 90% + Cam3 10%
3. 代码流程

程序主要流程如下：

1. 初始化 VI、VPSS、VO 等底层视频模块
2. 从 VpssGrp 2 获取 Cam2 当前帧
3. 从 VpssGrp 3 获取 Cam3 当前帧
4. 调用 GDC PMF 对 Cam3 进行透视校正
5. 调用 IVE Add 对 Cam2 与 Cam3_warp 进行加权融合
6. 将融合结果通过 HI_MPI_VO_SendFrame 输出到 HDMI
7. 释放输入帧，进入下一轮循环

核心函数包括：

BuildPmfAttr_1080P_Cam3ToCam2()
RunGdcPmfOnce()
RunIveAddPlane()
RunIveFusionOnce()
Gdc_Pmf_Ive_Fusion_Thread()
SAMPLE_VIO_LVDS()
4. 实际运行效果

当前运行日志显示：

[GDC+IVE Fusion OK] frames=60 fps=27.18 total=6.56ms gdc=3.88ms ive=2.66ms send=0.02ms
[GDC+IVE Fusion OK] frames=120 fps=26.54 total=6.57ms gdc=3.88ms ive=2.66ms send=0.02ms
[GDC+IVE Fusion OK] frames=180 fps=26.54 total=6.56ms gdc=3.88ms ive=2.65ms send=0.02ms

说明当前 1080P 下融合帧率约为 26–27fps。相比 OpenCV CPU 方案约 2.3fps，GDC + IVE 硬件方案实时性明显提升。

5. 当前局限

GDC PMF 本质上是单应性透视变换，只能较好对齐某一个主要平面或某一个主要距离范围。由于两个摄像头之间存在物理间距，不同深度物体存在不同视差，因此单个 PMF 矩阵无法让全场景所有深度同时完全重合。

当前工程上采用的折中策略是：

Cam2 作为主画面
Cam3 经过 GDC PMF 粗校正
Cam3 以低权重参与融合
必要时进一步降低 Cam3 权重
6. 文件说明
src/gdc.c                       GDC PMF + IVE 融合主程序
scripts/build_gdc_pmf.sh        编译脚本
logs/                           运行日志
docs/                           技术过程文档
images/                         效果图和标定图


