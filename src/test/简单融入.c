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

#include "hi_comm_ive.h"
#include "mpi_ive.h"
#include "mpi_sys.h"
#include "mpi_vb.h"

static volatile HI_BOOL g_bFusionRun = HI_TRUE;

void SAMPLE_VIO_HandleSig(HI_S32 signo)
{
    if (SIGINT == signo || SIGTERM == signo)
    {
        g_bFusionRun = HI_FALSE;
        usleep(200000);
        SAMPLE_COMM_All_ISP_Stop();
        SAMPLE_COMM_SYS_Exit();
        printf("\033[0;31mprogram termination abnormally!\033[0;39m\n");
    }

    exit(-1);
}

/*
 * IVE 双通道融合线程：
 * 1. 从 VpssGrp 2 和 VpssGrp 3 分别取帧；
 * 2. 使用 IVE 对 Y 平面做 0.5 + 0.5 平均融合；
 * 3. 使用 IVE 对 UV/VU 平面做 0.5 + 0.5 平均融合；
 * 4. 输出帧使用 VB Block，送 VO/HDMI 显示。
 */
void* Video_Fusion_Thread(void* arg)
{
    HI_S32 s32Ret = HI_SUCCESS;
    VIDEO_FRAME_INFO_S stFrameCam2;
    VIDEO_FRAME_INFO_S stFrameCam3;
    VIDEO_FRAME_INFO_S stDstFrame;

    VPSS_GRP VpssGrp_Cam2 = 2;
    VPSS_GRP VpssGrp_Cam3 = 3;
    VPSS_CHN VpssChn = 0;

    VO_LAYER VoLayer = SAMPLE_VO_DEV_DHD0;
    VO_CHN   VoChn_Out = 0;

    HI_BOOL bAllocated = HI_FALSE;

    VB_BLK hDstBlk[8] = {
        VB_INVALID_HANDLE, VB_INVALID_HANDLE, VB_INVALID_HANDLE, VB_INVALID_HANDLE,
        VB_INVALID_HANDLE, VB_INVALID_HANDLE, VB_INVALID_HANDLE, VB_INVALID_HANDLE
    };

    HI_U64  u64DstPhyAddr[8] = {0};
    HI_VOID *pDstVirAddr[8] = {HI_NULL};
    HI_U32  u32DstPoolId[8] = {0};

    HI_U32 u32DstSize = 0;
    HI_U32 u32YSize = 0;
    HI_U32 u32BufIndex = 0;

    int success_frames = 0;
    int cam2_fail_cnt = 0;
    int cam3_fail_cnt = 0;
    int vo_fail_cnt = 0;

    HI_BOOL bFirstFramePrint = HI_TRUE;

    printf("\n=======================================================\n");
    printf(">>> [IVE融合线程] VB输出帧 + IVE平均融合 + HDMI输出 已启动 <<<\n");
    printf(">>> 融合公式：Dst = Cam2 * 0.5 + Cam3 * 0.5 <<<\n");
    printf("=======================================================\n");

    while (g_bFusionRun)
    {
        memset(&stFrameCam2, 0, sizeof(VIDEO_FRAME_INFO_S));
        memset(&stFrameCam3, 0, sizeof(VIDEO_FRAME_INFO_S));
        memset(&stDstFrame, 0, sizeof(VIDEO_FRAME_INFO_S));

        s32Ret = HI_MPI_VPSS_GetChnFrame(VpssGrp_Cam2, VpssChn, &stFrameCam2, 100);
        if (s32Ret != HI_SUCCESS)
        {
            cam2_fail_cnt++;
            if (cam2_fail_cnt % 30 == 0)
            {
                printf("GetChnFrame Cam2 failed, grp=%d chn=%d ret=0x%x cnt=%d\n",
                       VpssGrp_Cam2, VpssChn, s32Ret, cam2_fail_cnt);
            }
            continue;
        }

        s32Ret = HI_MPI_VPSS_GetChnFrame(VpssGrp_Cam3, VpssChn, &stFrameCam3, 100);
        if (s32Ret != HI_SUCCESS)
        {
            cam3_fail_cnt++;
            if (cam3_fail_cnt % 30 == 0)
            {
                printf("GetChnFrame Cam3 failed, grp=%d chn=%d ret=0x%x cnt=%d\n",
                       VpssGrp_Cam3, VpssChn, s32Ret, cam3_fail_cnt);
            }

            HI_MPI_VPSS_ReleaseChnFrame(VpssGrp_Cam2, VpssChn, &stFrameCam2);
            continue;
        }

        if (bFirstFramePrint)
        {
            printf("Cam2 frame: width=%u height=%u strideY=%u strideC=%u pixfmt=%d phyY=0x%llx phyC=0x%llx pool=%u\n",
                   stFrameCam2.stVFrame.u32Width,
                   stFrameCam2.stVFrame.u32Height,
                   stFrameCam2.stVFrame.u32Stride[0],
                   stFrameCam2.stVFrame.u32Stride[1],
                   stFrameCam2.stVFrame.enPixelFormat,
                   (unsigned long long)stFrameCam2.stVFrame.u64PhyAddr[0],
                   (unsigned long long)stFrameCam2.stVFrame.u64PhyAddr[1],
                   stFrameCam2.u32PoolId);

            printf("Cam3 frame: width=%u height=%u strideY=%u strideC=%u pixfmt=%d phyY=0x%llx phyC=0x%llx pool=%u\n",
                   stFrameCam3.stVFrame.u32Width,
                   stFrameCam3.stVFrame.u32Height,
                   stFrameCam3.stVFrame.u32Stride[0],
                   stFrameCam3.stVFrame.u32Stride[1],
                   stFrameCam3.stVFrame.enPixelFormat,
                   (unsigned long long)stFrameCam3.stVFrame.u64PhyAddr[0],
                   (unsigned long long)stFrameCam3.stVFrame.u64PhyAddr[1],
                   stFrameCam3.u32PoolId);

            bFirstFramePrint = HI_FALSE;
        }

        if (stFrameCam2.stVFrame.u32Width != stFrameCam3.stVFrame.u32Width ||
            stFrameCam2.stVFrame.u32Height != stFrameCam3.stVFrame.u32Height ||
            stFrameCam2.stVFrame.u32Stride[0] != stFrameCam3.stVFrame.u32Stride[0] ||
            stFrameCam2.stVFrame.u32Stride[1] != stFrameCam3.stVFrame.u32Stride[1] ||
            stFrameCam2.stVFrame.enPixelFormat != stFrameCam3.stVFrame.enPixelFormat)
        {
            printf("Cam2/Cam3 frame format mismatch, skip this frame\n");

            HI_MPI_VPSS_ReleaseChnFrame(VpssGrp_Cam2, VpssChn, &stFrameCam2);
            HI_MPI_VPSS_ReleaseChnFrame(VpssGrp_Cam3, VpssChn, &stFrameCam3);
            continue;
        }

        if (!bAllocated)
        {
            HI_U32 i;
            HI_U32 j;

            u32YSize = stFrameCam2.stVFrame.u32Stride[0] * stFrameCam2.stVFrame.u32Height;
            u32DstSize = u32YSize + stFrameCam2.stVFrame.u32Stride[1] * stFrameCam2.stVFrame.u32Height / 2;

            for (i = 0; i < 8; i++)
            {
                hDstBlk[i] = HI_MPI_VB_GetBlock(VB_INVALID_POOLID, u32DstSize, HI_NULL);
                if (hDstBlk[i] == VB_INVALID_HANDLE)
                {
                    printf("HI_MPI_VB_GetBlock failed, index=%u size=%u\n", i, u32DstSize);

                    for (j = 0; j < i; j++)
                    {
                        if (pDstVirAddr[j] != HI_NULL)
                        {
                            HI_MPI_SYS_Munmap(pDstVirAddr[j], u32DstSize);
                            pDstVirAddr[j] = HI_NULL;
                        }

                        if (hDstBlk[j] != VB_INVALID_HANDLE)
                        {
                            HI_MPI_VB_ReleaseBlock(hDstBlk[j]);
                            hDstBlk[j] = VB_INVALID_HANDLE;
                        }
                    }

                    HI_MPI_VPSS_ReleaseChnFrame(VpssGrp_Cam2, VpssChn, &stFrameCam2);
                    HI_MPI_VPSS_ReleaseChnFrame(VpssGrp_Cam3, VpssChn, &stFrameCam3);

                    g_bFusionRun = HI_FALSE;
                    break;
                }

                u64DstPhyAddr[i] = HI_MPI_VB_Handle2PhysAddr(hDstBlk[i]);
                u32DstPoolId[i] = HI_MPI_VB_Handle2PoolId(hDstBlk[i]);

                pDstVirAddr[i] = HI_MPI_SYS_Mmap(u64DstPhyAddr[i], u32DstSize);
                if (pDstVirAddr[i] == HI_NULL)
                {
                    printf("HI_MPI_SYS_Mmap failed, index=%u phy=0x%llx size=%u\n",
                           i, (unsigned long long)u64DstPhyAddr[i], u32DstSize);

                    HI_MPI_VB_ReleaseBlock(hDstBlk[i]);
                    hDstBlk[i] = VB_INVALID_HANDLE;

                    for (j = 0; j < i; j++)
                    {
                        if (pDstVirAddr[j] != HI_NULL)
                        {
                            HI_MPI_SYS_Munmap(pDstVirAddr[j], u32DstSize);
                            pDstVirAddr[j] = HI_NULL;
                        }

                        if (hDstBlk[j] != VB_INVALID_HANDLE)
                        {
                            HI_MPI_VB_ReleaseBlock(hDstBlk[j]);
                            hDstBlk[j] = VB_INVALID_HANDLE;
                        }
                    }

                    HI_MPI_VPSS_ReleaseChnFrame(VpssGrp_Cam2, VpssChn, &stFrameCam2);
                    HI_MPI_VPSS_ReleaseChnFrame(VpssGrp_Cam3, VpssChn, &stFrameCam3);

                    g_bFusionRun = HI_FALSE;
                    break;
                }

                memset(pDstVirAddr[i], 0, u32DstSize);

                printf("Fusion VB buffer[%u]: phy=0x%llx vir=%p pool=%u size=%u\n",
                       i,
                       (unsigned long long)u64DstPhyAddr[i],
                       pDstVirAddr[i],
                       u32DstPoolId[i],
                       u32DstSize);
            }

            if (!g_bFusionRun)
            {
                break;
            }

            u32BufIndex = 0;
            bAllocated = HI_TRUE;

            printf("Fusion dst VB buffer allocated: size=%u, ySize=%u, bufNum=8\n",
                   u32DstSize, u32YSize);
        }

        IVE_SRC_IMAGE_S stSrc1;
        IVE_SRC_IMAGE_S stSrc2;
        IVE_DST_IMAGE_S stDst;
        IVE_HANDLE hIveHandle;
        IVE_ADD_CTRL_S stAddCtrl;
        HI_BOOL bFinish = HI_FALSE;

        memset(&stSrc1, 0, sizeof(stSrc1));
        memset(&stSrc2, 0, sizeof(stSrc2));
        memset(&stDst, 0, sizeof(stDst));
        memset(&stAddCtrl, 0, sizeof(stAddCtrl));

        stAddCtrl.u0q16X = 32768;
        stAddCtrl.u0q16Y = 32768;

        /*
         * 第一轮：融合 Y 平面。
         */
        stSrc1.enType = IVE_IMAGE_TYPE_U8C1;
        stSrc1.u32Width = stFrameCam2.stVFrame.u32Width;
        stSrc1.u32Height = stFrameCam2.stVFrame.u32Height;
        stSrc1.au32Stride[0] = stFrameCam2.stVFrame.u32Stride[0];
        stSrc1.au64PhyAddr[0] = stFrameCam2.stVFrame.u64PhyAddr[0];

        stSrc2.enType = IVE_IMAGE_TYPE_U8C1;
        stSrc2.u32Width = stFrameCam3.stVFrame.u32Width;
        stSrc2.u32Height = stFrameCam3.stVFrame.u32Height;
        stSrc2.au32Stride[0] = stFrameCam3.stVFrame.u32Stride[0];
        stSrc2.au64PhyAddr[0] = stFrameCam3.stVFrame.u64PhyAddr[0];

        stDst.enType = IVE_IMAGE_TYPE_U8C1;
        stDst.u32Width = stSrc1.u32Width;
        stDst.u32Height = stSrc1.u32Height;
        stDst.au32Stride[0] = stSrc1.au32Stride[0];
        stDst.au64PhyAddr[0] = u64DstPhyAddr[u32BufIndex];

        s32Ret = HI_MPI_IVE_Add(&hIveHandle, &stSrc1, &stSrc2, &stDst, &stAddCtrl, HI_TRUE);
        if (s32Ret != HI_SUCCESS)
        {
            printf("IVE Add Y failed, ret=0x%x\n", s32Ret);

            HI_MPI_VPSS_ReleaseChnFrame(VpssGrp_Cam2, VpssChn, &stFrameCam2);
            HI_MPI_VPSS_ReleaseChnFrame(VpssGrp_Cam3, VpssChn, &stFrameCam3);
            continue;
        }

        s32Ret = HI_MPI_IVE_Query(hIveHandle, &bFinish, HI_TRUE);
        if (s32Ret != HI_SUCCESS)
        {
            printf("IVE Query Y failed, ret=0x%x\n", s32Ret);

            HI_MPI_VPSS_ReleaseChnFrame(VpssGrp_Cam2, VpssChn, &stFrameCam2);
            HI_MPI_VPSS_ReleaseChnFrame(VpssGrp_Cam3, VpssChn, &stFrameCam3);
            continue;
        }

        /*
         * 第二轮：融合 UV/VU 半平面。
         */
        memset(&stSrc1, 0, sizeof(stSrc1));
        memset(&stSrc2, 0, sizeof(stSrc2));
        memset(&stDst, 0, sizeof(stDst));

        stSrc1.enType = IVE_IMAGE_TYPE_U8C1;
        stSrc1.u32Width = stFrameCam2.stVFrame.u32Width;
        stSrc1.u32Height = stFrameCam2.stVFrame.u32Height / 2;
        stSrc1.au32Stride[0] = stFrameCam2.stVFrame.u32Stride[1];
        stSrc1.au64PhyAddr[0] = stFrameCam2.stVFrame.u64PhyAddr[1];

        stSrc2.enType = IVE_IMAGE_TYPE_U8C1;
        stSrc2.u32Width = stFrameCam3.stVFrame.u32Width;
        stSrc2.u32Height = stFrameCam3.stVFrame.u32Height / 2;
        stSrc2.au32Stride[0] = stFrameCam3.stVFrame.u32Stride[1];
        stSrc2.au64PhyAddr[0] = stFrameCam3.stVFrame.u64PhyAddr[1];

        stDst.enType = IVE_IMAGE_TYPE_U8C1;
        stDst.u32Width = stSrc1.u32Width;
        stDst.u32Height = stSrc1.u32Height;
        stDst.au32Stride[0] = stSrc1.au32Stride[0];
        stDst.au64PhyAddr[0] = u64DstPhyAddr[u32BufIndex] + u32YSize;

        s32Ret = HI_MPI_IVE_Add(&hIveHandle, &stSrc1, &stSrc2, &stDst, &stAddCtrl, HI_TRUE);
        if (s32Ret != HI_SUCCESS)
        {
            printf("IVE Add UV failed, ret=0x%x\n", s32Ret);

            HI_MPI_VPSS_ReleaseChnFrame(VpssGrp_Cam2, VpssChn, &stFrameCam2);
            HI_MPI_VPSS_ReleaseChnFrame(VpssGrp_Cam3, VpssChn, &stFrameCam3);
            continue;
        }

        s32Ret = HI_MPI_IVE_Query(hIveHandle, &bFinish, HI_TRUE);
        if (s32Ret != HI_SUCCESS)
        {
            printf("IVE Query UV failed, ret=0x%x\n", s32Ret);

            HI_MPI_VPSS_ReleaseChnFrame(VpssGrp_Cam2, VpssChn, &stFrameCam2);
            HI_MPI_VPSS_ReleaseChnFrame(VpssGrp_Cam3, VpssChn, &stFrameCam3);
            continue;
        }

        /*
         * 组装融合输出帧。
         * 保留 Cam2 的尺寸、stride、像素格式、动态范围等元信息。
         * 替换为融合输出 VB block 的物理地址、虚拟地址和 pool id。
         */
        memcpy(&stDstFrame, &stFrameCam2, sizeof(VIDEO_FRAME_INFO_S));

        stDstFrame.u32PoolId = u32DstPoolId[u32BufIndex];

        stDstFrame.stVFrame.u64PhyAddr[0] = u64DstPhyAddr[u32BufIndex];
        stDstFrame.stVFrame.u64PhyAddr[1] = u64DstPhyAddr[u32BufIndex] + u32YSize;

        stDstFrame.stVFrame.u64VirAddr[0] = (HI_U64)(HI_UL)pDstVirAddr[u32BufIndex];
        stDstFrame.stVFrame.u64VirAddr[1] = (HI_U64)(HI_UL)((HI_U8*)pDstVirAddr[u32BufIndex] + u32YSize);

        s32Ret = HI_MPI_VO_SendFrame(VoLayer, VoChn_Out, &stDstFrame, -1);
        if (s32Ret == HI_SUCCESS)
        {
            success_frames++;
            if (success_frames % 60 == 0)
            {
                printf("\033[0;32m>>> [IVE融合成功] 融合画面送 HDMI 成功 60 帧 <<<\033[0;39m\n");
            }

            u32BufIndex++;
            if (u32BufIndex >= 8)
            {
                u32BufIndex = 0;
            }
        }
        else
        {
            vo_fail_cnt++;
            if (vo_fail_cnt % 30 == 0)
            {
                printf("VO SendFrame fusion failed, layer=%d chn=%d ret=0x%x cnt=%d pool=%u phy=0x%llx\n",
                       VoLayer,
                       VoChn_Out,
                       s32Ret,
                       vo_fail_cnt,
                       stDstFrame.u32PoolId,
                       (unsigned long long)stDstFrame.stVFrame.u64PhyAddr[0]);
            }
        }

        HI_MPI_VPSS_ReleaseChnFrame(VpssGrp_Cam2, VpssChn, &stFrameCam2);
        HI_MPI_VPSS_ReleaseChnFrame(VpssGrp_Cam3, VpssChn, &stFrameCam3);
    }

    if (bAllocated)
    {
        HI_U32 i;

        for (i = 0; i < 8; i++)
        {
            if (pDstVirAddr[i] != HI_NULL)
            {
                HI_MPI_SYS_Munmap(pDstVirAddr[i], u32DstSize);
                pDstVirAddr[i] = HI_NULL;
            }

            if (hDstBlk[i] != VB_INVALID_HANDLE)
            {
                HI_MPI_VB_ReleaseBlock(hDstBlk[i]);
                hDstBlk[i] = VB_INVALID_HANDLE;
            }
        }
    }

    return NULL;
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
    VPSS_GRP                VpssDevGrp[6]       = {0, 1, 2, 3, 4, 5};
    VPSS_GRP_ATTR_S         stVpssGrpAttr       = {0};
    VPSS_CHN_ATTR_S         stVpssChnAttr[VPSS_MAX_PHY_CHN_NUM];
    HI_BOOL                 abChnEnable[VPSS_MAX_PHY_CHN_NUM] = {0};
    VO_DEV                  VoDev               = SAMPLE_VO_DEV_DHD0;
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
    pthread_t               fusion_thread_id;

    SAMPLE_COMM_VI_GetSensorInfo(&stViConfig);

    stViConfig.astViInfo[0].stSnsInfo.enSnsType = XLM0_LVDS_1080P_60FPS_16BIT;
    stViConfig.astViInfo[1].stSnsInfo.enSnsType = XLM0_LVDS_4K_30FPS_16BIT;
    stViConfig.astViInfo[2].stSnsInfo.enSnsType = XLM0_LVDS_1080P_30FPS_16BIT;
    stViConfig.astViInfo[3].stSnsInfo.enSnsType = XLM0_LVDS_1080P_30FPS_16BIT;
    stViConfig.astViInfo[4].stSnsInfo.enSnsType = XLM0_LVDS_720x576_50FPS_16BIT;
    stViConfig.astViInfo[5].stSnsInfo.enSnsType = XLM0_LVDS_1080P_30FPS_16BIT;

    stViConfig.s32WorkingViNum = s32ViDevCnt;

    for (i = 0; i < s32ViDevCnt; i++)
    {
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

    hi_memset(&stVbConf, sizeof(VB_CONFIG_S), 0, sizeof(VB_CONFIG_S));
    stVbConf.u32MaxPoolCnt = 128;

    u32BlkSize = COMMON_GetPicBufferSize(stSize[1].u32Width,
                                         stSize[1].u32Height,
                                         PIXEL_FORMAT_YVU_SEMIPLANAR_422,
                                         DATA_BITWIDTH_8,
                                         COMPRESS_MODE_NONE,
                                         DEFAULT_ALIGN);

    stVbConf.astCommPool[0].u64BlkSize = u32BlkSize;
    stVbConf.astCommPool[0].u32BlkCnt  = 60;

    s32Ret = SAMPLE_COMM_SYS_Init(&stVbConf);
    if (HI_SUCCESS != s32Ret)
    {
        printf("SAMPLE_COMM_SYS_Init failed, ret=0x%x\n", s32Ret);
        goto EXIT;
    }

    for (i = 0; i < VI_MAX_PIPE_NUM; i++)
    {
        stVIVPSSMode.aenMode[i] = VI_OFFLINE_VPSS_OFFLINE;
    }

    s32Ret = HI_MPI_SYS_SetVIVPSSMode(&stVIVPSSMode);
    if (HI_SUCCESS != s32Ret)
    {
        printf("HI_MPI_SYS_SetVIVPSSMode failed, ret=0x%x\n", s32Ret);
        goto EXIT;
    }

    s32Ret = SAMPLE_COMM_VI_StartVi(&stViConfig);
    if (HI_SUCCESS != s32Ret)
    {
        printf("SAMPLE_COMM_VI_StartVi failed, ret=0x%x\n", s32Ret);
        goto EXIT4;
    }

    stVpssGrpAttr.u32MaxW                     = 1920;
    stVpssGrpAttr.u32MaxH                     = 1080;
    stVpssGrpAttr.enPixelFormat               = PIXEL_FORMAT_YVU_SEMIPLANAR_420;
    stVpssGrpAttr.enDynamicRange              = enDynamicRange;
    stVpssGrpAttr.stFrameRate.s32SrcFrameRate = -1;
    stVpssGrpAttr.stFrameRate.s32DstFrameRate = -1;

    memset(stVpssChnAttr, 0, sizeof(stVpssChnAttr));
    memset(abChnEnable, 0, sizeof(abChnEnable));

    abChnEnable[0] = HI_TRUE;

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

    for (i = 0; i < s32VpssDevGrpCnt; i++)
    {
        stVpssChnAttr[0].u32Width  = (i == 4) ? 720 : 1920;
        stVpssChnAttr[0].u32Height = (i == 4) ? 576 : 1080;

        s32Ret = SAMPLE_COMM_VPSS_Start(VpssDevGrp[i],
                                        abChnEnable,
                                        &stVpssGrpAttr,
                                        stVpssChnAttr);
        if (HI_SUCCESS != s32Ret)
        {
            printf("SAMPLE_COMM_VPSS_Start failed, grp=%d ret=0x%x\n", VpssDevGrp[i], s32Ret);
            goto EXIT4;
        }
    }

    for (i = 0; i < s32VpssDevGrpCnt; i++)
    {
        s32Ret = SAMPLE_COMM_VI_Bind_VPSS(ViPipe[i], ViPhyChn, VpssDevGrp[i]);
        if (HI_SUCCESS != s32Ret)
        {
            printf("SAMPLE_COMM_VI_Bind_VPSS failed, pipe=%d chn=%d grp=%d ret=0x%x\n",
                   ViPipe[i], ViPhyChn, VpssDevGrp[i], s32Ret);
            goto EXIT3;
        }
    }

    SAMPLE_COMM_VO_GetDefConfig(&stVoConfig);

    stVoConfig.VoDev             = VoDev;
    stVoConfig.enVoIntfType      = enVoIntfType;
    stVoConfig.enIntfSync        = g_enIntfSync;
    stVoConfig.enPicSize         = enPicSize;
    stVoConfig.u32DisBufLen      = g_u32DisBufLen;
    stVoConfig.enDstDynamicRange = enDynamicRange;
    stVoConfig.enVoMode          = VO_MODE_1MUX;

    s32Ret = SAMPLE_COMM_VO_StartVO(&stVoConfig);
    if (HI_SUCCESS != s32Ret)
    {
        printf("SAMPLE_COMM_VO_StartVO failed, ret=0x%x\n", s32Ret);
        goto EXIT2;
    }

    s32Ret = pthread_create(&fusion_thread_id, NULL, Video_Fusion_Thread, NULL);
    if (s32Ret != 0)
    {
        printf("pthread_create failed, ret=%d\n", s32Ret);
        g_bFusionRun = HI_FALSE;
        goto EXIT2;
    }

    printf("\n>>> 底层视频配置完毕，进入后台运行模式 <<<\n");
    printf(">>> 当前目标：HDMI 显示 VpssGrp 2 与 VpssGrp 3 的 IVE 平均融合画面 <<<\n");

    while (g_bFusionRun)
    {
        sleep(1);
    }

    pthread_join(fusion_thread_id, NULL);

EXIT2:
    SAMPLE_COMM_VO_StopVO(&stVoConfig);

EXIT3:
    for (i = 0; i < s32VpssDevGrpCnt; i++)
    {
        SAMPLE_COMM_VPSS_Stop(VpssDevGrp[i], abChnEnable);
    }

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

    if (HI_SUCCESS == s32Ret)
    {
        SAMPLE_PRT("sample_vio exit success!\n");
    }
    else
    {
        SAMPLE_PRT("sample_vio exit abnormally!\n");
    }

    return s32Ret;
}

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif
