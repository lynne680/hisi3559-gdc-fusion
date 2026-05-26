#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif /* End of #ifdef __cplusplus */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/ioctl.h>

#include "sample_comm.h"

/******************************************************************************
* function : to process abnormal case
******************************************************************************/
void SAMPLE_VIO_HandleSig(HI_S32 signo)
{
    if (SIGINT == signo || SIGTERM == signo)
    {
        SAMPLE_COMM_All_ISP_Stop();
        SAMPLE_COMM_SYS_Exit();
        printf("\033[0;31mprogram termination abnormally!\033[0;39m\n");
    }
    exit(-1);
}

HI_S32 SAMPLE_VIO_LVDS(VO_INTF_TYPE_E enVoIntfType)
{
    HI_S32                  s32Ret              = HI_SUCCESS;
    HI_S32                  i                   = 0;
    VI_DEV                  ViDev[6]            = {0, 2, 4, 5, 6, 7};
    VI_PIPE                 ViPipe[6]           = {0, 2, 4, 5, 6, 7};
    VI_CHN                  ViPhyChn            = VI_CHN0;  
    HI_S32                  s32ViDevCnt         = 6;  
    HI_S32                  s32VpssDevGrpCnt    = 6;  
    VPSS_GRP                VpssDevGrp[6]       = {0, 1, 2, 3, 4 ,5};  
    VPSS_CHN                VpssChn             = VPSS_CHN0;
    VPSS_GRP_ATTR_S         stVpssGrpAttr       = {0};
    VPSS_CHN_ATTR_S         stVpssChnAttr[VPSS_MAX_PHY_CHN_NUM];
    HI_BOOL                 abChnEnable[VPSS_MAX_PHY_CHN_NUM] = {0};
    VO_DEV                  VoDev               = SAMPLE_VO_DEV_DHD0;
    VO_CHN                  VoChn[6]            = {0, 1, 2, 3, 4, 5}; 
    VO_INTF_SYNC_E          g_enIntfSync        = VO_OUTPUT_1080P60; 
    HI_U32                  g_u32DisBufLen      = 8;
    PIC_SIZE_E              enPicSize           = PIC_1080P; 
    WDR_MODE_E              enWDRMode           = WDR_MODE_NONE;
    DYNAMIC_RANGE_E         enDynamicRange      = DYNAMIC_RANGE_SDR8;
    PIXEL_FORMAT_E          enPixFormat         = PIXEL_FORMAT_YVU_SEMIPLANAR_420;
    VIDEO_FORMAT_E          enVideoFormat       = VIDEO_FORMAT_LINEAR;
    COMPRESS_MODE_E         enCompressMode      = COMPRESS_MODE_NONE;
    VI_VPSS_MODE_E          enMastPipeMode      = VI_OFFLINE_VPSS_OFFLINE;
    SIZE_S                  stSize[6]           = {{1920, 1080}, {3840, 2160}, {1920, 1080}, {1920, 1080}, {720, 576}, {1920, 1080}};
    HI_U32                  u32BlkSize;
    VB_CONFIG_S             stVbConf;
    SAMPLE_VI_CONFIG_S      stViConfig;
    SAMPLE_VO_CONFIG_S      stVoConfig;
    VI_VPSS_MODE_S          stVIVPSSMode;

    /************************************************
    step 1:  Get all sensors information
    *************************************************/
    SAMPLE_COMM_VI_GetSensorInfo(&stViConfig);
    stViConfig.astViInfo[0].stSnsInfo.enSnsType = XLM0_LVDS_1080P_60FPS_16BIT;
    stViConfig.astViInfo[1].stSnsInfo.enSnsType = XLM0_LVDS_4K_30FPS_16BIT;
    stViConfig.astViInfo[2].stSnsInfo.enSnsType = XLM0_LVDS_1080P_30FPS_16BIT;
    stViConfig.astViInfo[3].stSnsInfo.enSnsType = XLM0_LVDS_1080P_30FPS_16BIT;
    stViConfig.astViInfo[4].stSnsInfo.enSnsType = XLM0_LVDS_720x576_50FPS_16BIT;
    stViConfig.astViInfo[5].stSnsInfo.enSnsType = XLM0_LVDS_1080P_30FPS_16BIT;
    
    stViConfig.s32WorkingViNum = s32ViDevCnt;

    for( i = 0; i < s32ViDevCnt; i++) {
        stViConfig.as32WorkingViId[i]                        = i;
        stViConfig.astViInfo[i].stSnsInfo.MipiDev            = SAMPLE_COMM_VI_GetComboDevBySensor(stViConfig.astViInfo[i].stSnsInfo.enSnsType, i);
        stViConfig.astViInfo[i].stSnsInfo.s32BusId           = i;
        stViConfig.astViInfo[i].stDevInfo.ViDev              = ViDev[i];
        stViConfig.astViInfo[i].stDevInfo.enWDRMode          = enWDRMode;
        stViConfig.astViInfo[i].stPipeInfo.enMastPipeMode    = enMastPipeMode;
        stViConfig.astViInfo[i].stPipeInfo.aPipe[0]          = ViPipe[i];
        stViConfig.astViInfo[i].stPipeInfo.aPipe[1]          = -1;
        stViConfig.astViInfo[i].stPipeInfo.aPipe[2]          = -1;
        stViConfig.astViInfo[i].stPipeInfo.aPipe[3]          = -1;
        stViConfig.astViInfo[i].stChnInfo.ViChn              = ViPhyChn;
        stViConfig.astViInfo[i].stChnInfo.enPixFormat        = enPixFormat;
        stViConfig.astViInfo[i].stChnInfo.enDynamicRange     = enDynamicRange;
        stViConfig.astViInfo[i].stChnInfo.enVideoFormat      = enVideoFormat;
        stViConfig.astViInfo[i].stChnInfo.enCompressMode     = enCompressMode;
    }

    /************************************************
    step2:  Init SYS and common VB
    *************************************************/
    hi_memset(&stVbConf, sizeof(VB_CONFIG_S), 0, sizeof(VB_CONFIG_S));
    stVbConf.u32MaxPoolCnt              = 128;
    u32BlkSize = COMMON_GetPicBufferSize(stSize[1].u32Width, stSize[1].u32Height, PIXEL_FORMAT_YVU_SEMIPLANAR_422, DATA_BITWIDTH_8, COMPRESS_MODE_NONE, DEFAULT_ALIGN);
    stVbConf.astCommPool[0].u64BlkSize  = u32BlkSize;
    stVbConf.astCommPool[0].u32BlkCnt   = 60;

    s32Ret = SAMPLE_COMM_SYS_Init(&stVbConf);
    if (HI_SUCCESS != s32Ret) goto EXIT;

    for(i = 0;i < VI_MAX_PIPE_NUM;i++) stVIVPSSMode.aenMode[i] = VI_OFFLINE_VPSS_OFFLINE;

    s32Ret = HI_MPI_SYS_SetVIVPSSMode(&stVIVPSSMode);
    if (HI_SUCCESS != s32Ret) goto EXIT;

    /************************************************
    step 4: start VI
    *************************************************/
    s32Ret = SAMPLE_COMM_VI_StartVi(&stViConfig);
    if (HI_SUCCESS != s32Ret) goto EXIT4;

    /************************************************
    step 5: start VPSS
    *************************************************/
    stVpssGrpAttr.u32MaxW                        = 1920; 
    stVpssGrpAttr.u32MaxH                        = 1080; 
    stVpssGrpAttr.enPixelFormat                  = PIXEL_FORMAT_YVU_SEMIPLANAR_420;
    stVpssGrpAttr.enDynamicRange                 = enDynamicRange;
    stVpssGrpAttr.stFrameRate.s32SrcFrameRate    = -1;
    stVpssGrpAttr.stFrameRate.s32DstFrameRate    = -1;

    abChnEnable[0]                               = HI_TRUE;

    stVpssChnAttr[0].enChnMode                   = VPSS_CHN_MODE_USER;
    stVpssChnAttr[0].enCompressMode              = enCompressMode;
    stVpssChnAttr[0].enDynamicRange              = enDynamicRange;
    stVpssChnAttr[0].enVideoFormat               = enVideoFormat;
    stVpssChnAttr[0].enPixelFormat               = PIXEL_FORMAT_YVU_SEMIPLANAR_420;
    stVpssChnAttr[0].stFrameRate.s32SrcFrameRate = -1;
    stVpssChnAttr[0].stFrameRate.s32DstFrameRate = -1;
    stVpssChnAttr[0].u32Depth                    = 2;
    stVpssChnAttr[0].bMirror                     = HI_FALSE;
    stVpssChnAttr[0].bFlip                       = HI_FALSE;
    stVpssChnAttr[0].stAspectRatio.enMode        = ASPECT_RATIO_NONE;

    for(int grp=0; grp<6; grp++) {
        stVpssChnAttr[0].u32Width  = (grp==4) ? 720 : 1920;
        stVpssChnAttr[0].u32Height = (grp==4) ? 576 : 1080;
        s32Ret = SAMPLE_COMM_VPSS_Start(VpssDevGrp[grp], &abChnEnable[0], &stVpssGrpAttr, &stVpssChnAttr[0]);
        if (HI_SUCCESS != s32Ret) goto EXIT4;
    }

    /************************************************
    step 6:  VI bind VPSS
    *************************************************/
    for(i = 0; i < s32VpssDevGrpCnt; i++)
    {
        s32Ret = SAMPLE_COMM_VI_Bind_VPSS(ViPipe[i], ViPhyChn, VpssDevGrp[i]);
        if (HI_SUCCESS != s32Ret) goto EXIT3;
    }

    /************************************************
    step 7:  start V0
    *************************************************/
    SAMPLE_COMM_VO_GetDefConfig(&stVoConfig);
    stVoConfig.VoDev                                    = VoDev;
    stVoConfig.enVoIntfType                             = enVoIntfType;
    stVoConfig.enIntfSync                               = g_enIntfSync;
    stVoConfig.enPicSize                                = enPicSize;
    stVoConfig.u32DisBufLen                             = g_u32DisBufLen;
    stVoConfig.enDstDynamicRange                        = enDynamicRange;
    stVoConfig.enVoMode                                 = VO_MODE_9MUX;

    s32Ret = SAMPLE_COMM_VO_StartVO(&stVoConfig);
    if (HI_SUCCESS != s32Ret) goto EXIT2;

    /************************************************
    step 8:  VPSS bind VO
    *************************************************/
    for(i = 0; i < s32VpssDevGrpCnt; i++)
    {
        s32Ret = SAMPLE_COMM_VPSS_Bind_VO(VpssDevGrp[i], VpssChn, stVoConfig.VoDev, VoChn[i]);
        if (HI_SUCCESS != s32Ret) goto EXIT2;
    }

    // ====================================================================
    // 🔥 【核心阵地】：手动硬指认真实摄像头！ 🔥
    // ====================================================================
    VO_LAYER VoLayer = stVoConfig.VoDev; 
    
    // 排查绝招：先把下面这三个数字写成相同的，例如 {0, 0, 0}，然后编译运行。
    // 如果屏幕全是彩条，说明 0 号口没插摄像头。
    // 然后改成 {2, 2, 2} 重新运行，直到找出出画面的那 3 个通道号，最后填回来！
    // 候选的通道号只有：0, 1, 2, 3, 4, 5。
    // （提示：一般 1080P 的摄像头集中在 0, 2, 3, 5 这几个通道上）
    
    int real_cameras[3] = {3, 3, 3};  // <--- 排查时，请修改这里！
    
    // 1. 无情地关掉所有 16 个通道，保证屏幕没有任何多余的彩条！
    for (int chn = 0; chn < 16; chn++) {
        HI_MPI_VO_DisableChn(VoLayer, chn);
    }

    // 2. 依次开启真正有摄像头的通道，并全屏拉伸叠加
    for (int k = 0; k < 3; k++) {
        int chn = real_cameras[k];
        VO_CHN_ATTR_S stChnAttr;
        
        if (HI_MPI_VO_GetChnAttr(VoLayer, chn, &stChnAttr) == HI_SUCCESS) {
            stChnAttr.stRect.s32X = 0;
            stChnAttr.stRect.s32Y = 0;
            stChnAttr.stRect.u32Width = 1920;
            stChnAttr.stRect.u32Height = 1080;
            
            // 赋予 Z 轴优先级。k=0 在最底，k 越往后在越顶层。
            stChnAttr.u32Priority = k; 
            stChnAttr.bDeflicker = HI_FALSE;

            HI_MPI_VO_SetChnAttr(VoLayer, chn, &stChnAttr);
            HI_MPI_VO_EnableChn(VoLayer, chn);
        }
    }
    // ====================================================================

    printf("\n>>> 底层视频配置完毕，进入后台静默运行模式 <<<\n");
    while (1) sleep(1); 

    for (i = 0; i < s32VpssDevGrpCnt; i++) SAMPLE_COMM_VPSS_UnBind_VO(VpssDevGrp[i], VpssChn, stVoConfig.VoDev, VoChn[i]);

EXIT2:
    SAMPLE_COMM_VO_StopVO(&stVoConfig);
EXIT3:
    for (i = 0; i < s32VpssDevGrpCnt; i++) SAMPLE_COMM_VPSS_Stop(VpssDevGrp[i], abChnEnable);

EXIT4:
    SAMPLE_COMM_VI_StopVi(&stViConfig);
EXIT:
    SAMPLE_COMM_SYS_Exit();

    return s32Ret;
}

#ifdef __HuaweiLite__
int app_main(int argc, char *argv[])
#else
int main(int argc, char* argv[])
#endif
{
    HI_S32 s32Ret = HI_FAILURE;
    VO_INTF_TYPE_E enVoIntfType = VO_INTF_HDMI;

#ifndef __HuaweiLite__
    signal(SIGINT, SAMPLE_VIO_HandleSig);
    signal(SIGTERM, SAMPLE_VIO_HandleSig);
#endif

    s32Ret = SAMPLE_VIO_LVDS(enVoIntfType);
    if (HI_SUCCESS == s32Ret) SAMPLE_PRT("sample_vio exit success!\n");
    else SAMPLE_PRT("sample_vio exit abnormally!\n");
    
    return s32Ret;
}

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* End of #ifdef __cplusplus */
