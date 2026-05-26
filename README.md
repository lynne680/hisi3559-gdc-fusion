# Hi3559AV100 GDC PMF + IVE 双摄视频融合项目

## 1. 项目目标

本项目基于 Hi3559AV100 开发板和 Hi3559AV100_SDK_V2.0.3.1，实现两路可见光摄像头视频融合，并通过 HDMI 实时输出融合画面。

当前使用的两路摄像头为：

- Cam2：VpssGrp 2，对应 ViPipe 4
- Cam3：VpssGrp 3，对应 ViPipe 5

最终技术路线为：

```text
Cam2 原始帧获取
Cam3 原始帧获取
Cam3 经过 GDC PMF 做透视校正
Cam2 与校正后的 Cam3 通过 IVE Add 做加权融合
融合结果通过 HDMI 输出

编译方式

在 SDK 环境下使用脚本编译：

cd /home/xia/hisi3559/hisi3559-gdc-fusion
./scripts/build_gdc_pmf.sh

实际编译时，需要确认脚本中的 SDK 路径与本机路径一致。
