#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <signal.h>
#include <sys/select.h>
#include <sys/time.h>
#include <termios.h>
#include <stdint.h>

#include "sample_comm.h"
#include "mpi_sys.h"
#include "mpi_vb.h"
#include "mpi_gdc.h"
#include "mpi_ive.h"

static volatile HI_BOOL g_bRun = HI_TRUE;
static struct termios g_stOldTermios;
static HI_BOOL g_bTermiosSaved = HI_FALSE;

#define GDC_OUT_BUF_NUM     4
#define FUSION_OUT_BUF_NUM  4

#define CAM2_WEIGHT_Q16     58982
#define CAM3_WEIGHT_Q16      6554

typedef struct hiFRAME_BUF_S
{
    VIDEO_FRAME_INFO_S stFrame;
    VB_BLK hBlk;
    HI_VOID *pVirAddr;
    HI_U32 u32FrameSize;
    HI_U32 u32YSize;
} FRAME_BUF_S;

static double GetTimeMs(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}

static HI_VOID RestoreKeyboardMode(HI_VOID)
{
    if (g_bTermiosSaved)
    {
        tcsetattr(STDIN_FILENO, TCSANOW, &g_stOldTermios);
        g_bTermiosSaved = HI_FALSE;
    }
}

static HI_S32 SetKeyboardRawMode(HI_VOID)
{
    struct termios stNewTermios;

    if (tcgetattr(STDIN_FILENO, &g_stOldTermios) != 0)
    {
        printf("tcgetattr failed\n");
        return HI_FAILURE;
    }

    g_bTermiosSaved = HI_TRUE;
    stNewTermios = g_stOldTermios;
    stNewTermios.c_lflag &= ~(ICANON | ECHO);
    stNewTermios.c_cc[VMIN] = 0;
    stNewTermios.c_cc[VTIME] = 0;

    if (tcsetattr(STDIN_FILENO, TCSANOW, &stNewTermios) != 0)
    {
        printf("tcsetattr failed\n");
        return HI_FAILURE;
    }

    return HI_SUCCESS;
}

void SAMPLE_VIO_HandleSig(HI_S32 signo)
{
    if (SIGINT == signo || SIGTERM == signo || SIGSEGV == signo)
    {
        g_bRun = HI_FALSE;
        RestoreKeyboardMode();
        usleep(200000);
        SAMPLE_COMM_All_ISP_Stop();
        SAMPLE_COMM_SYS_Exit();
        printf("\033[0;31mprogram termination abnormally!\033[0;39m\n");
    }

    exit(-1);
}

static HI_VOID CleanGdcTaskFrame(VIDEO_FRAME_INFO_S *pstFrame)
{
    pstFrame->stVFrame.u64VirAddr[0] = 0;
    pstFrame->stVFrame.u64VirAddr[1] = 0;
    pstFrame->stVFrame.u64VirAddr[2] = 0;
    pstFrame->stVFrame.u64PhyAddr[2] = 0;
    pstFrame->stVFrame.u32Stride[2]  = 0;
    pstFrame->stVFrame.enField = VIDEO_FIELD_FRAME;
    pstFrame->stVFrame.enCompressMode = COMPRESS_MODE_NONE;
}

static HI_VOID BuildPmfAttr_1080P_Cam3ToCam2(GDC_PMF_ATTR_S *pstPmfAttr)
{
    HI_S64 *c = HI_NULL;
    static HI_BOOL bPrinted = HI_FALSE;

    memset(pstPmfAttr, 0, sizeof(GDC_PMF_ATTR_S));
    c = pstPmfAttr->as64PMFCoef;

    c[0] = 529011;
    c[1] = 46064;
    c[2] = 22108332;
    c[3] = -42495;
    c[4] = 519698;
    c[5] = 10987334;
    c[6] = 3;
    c[7] = 0;
    c[8] = 524288;

    if (!bPrinted)
    {
        printf("\n[PMF COEF] new calibration, scale=524288, output(Cam2) -> input(Cam3)\n");
        printf("  %lld, %lld, %lld\n", (long long)c[0], (long long)c[1], (long long)c[2]);
        printf("  %lld, %lld, %lld\n", (long long)c[3], (long long)c[4], (long long)c[5]);
        printf("  %lld, %lld, %lld\n", (long long)c[6], (long long)c[7], (long long)c[8]);
        bPrinted = HI_TRUE;
    }
}

static HI_S32 AllocFrameLikeInput(const VIDEO_FRAME_INFO_S *pstInFrame,
                                  FRAME_BUF_S *pstBuf,
                                  const HI_CHAR *pszName)
{
    HI_U32 width;
    HI_U32 height;
    HI_U32 strideY;
    HI_U32 strideC;
    HI_U32 ySize;
    HI_U32 cSize;
    HI_U32 frameSize;
    HI_U64 phyAddr;
    HI_VOID *pVirAddr;

    if (pstInFrame == HI_NULL || pstBuf == HI_NULL)
    {
        return HI_FAILURE;
    }

    memset(pstBuf, 0, sizeof(FRAME_BUF_S));
    pstBuf->hBlk = VB_INVALID_HANDLE;

    width   = pstInFrame->stVFrame.u32Width;
    height  = pstInFrame->stVFrame.u32Height;
    strideY = pstInFrame->stVFrame.u32Stride[0];
    strideC = pstInFrame->stVFrame.u32Stride[1];

    ySize = strideY * height;
    cSize = strideC * height / 2;
    frameSize = ySize + cSize;

    pstBuf->hBlk = HI_MPI_VB_GetBlock(VB_INVALID_POOLID, frameSize, HI_NULL);
    if (pstBuf->hBlk == VB_INVALID_HANDLE)
    {
        printf("%s: HI_MPI_VB_GetBlock failed, frameSize=%u\n", pszName, frameSize);
        return HI_FAILURE;
    }

    phyAddr = HI_MPI_VB_Handle2PhysAddr(pstBuf->hBlk);
    pVirAddr = HI_MPI_SYS_Mmap(phyAddr, frameSize);
    if (pVirAddr == HI_NULL)
    {
        printf("%s: HI_MPI_SYS_Mmap failed, phy=0x%llx size=%u\n",
               pszName,
               (unsigned long long)phyAddr,
               frameSize);
        HI_MPI_VB_ReleaseBlock(pstBuf->hBlk);
        pstBuf->hBlk = VB_INVALID_HANDLE;
        return HI_FAILURE;
    }

    memset(pVirAddr, 0, frameSize);

    pstBuf->pVirAddr = pVirAddr;
    pstBuf->u32FrameSize = frameSize;
    pstBuf->u32YSize = ySize;

    memset(&pstBuf->stFrame, 0, sizeof(VIDEO_FRAME_INFO_S));
    pstBuf->stFrame.u32PoolId = HI_MPI_VB_Handle2PoolId(pstBuf->hBlk);
    pstBuf->stFrame.stVFrame.u32Width       = width;
    pstBuf->stFrame.stVFrame.u32Height      = height;
    pstBuf->stFrame.stVFrame.u32Stride[0]   = strideY;
    pstBuf->stFrame.stVFrame.u32Stride[1]   = strideC;
    pstBuf->stFrame.stVFrame.u32Stride[2]   = 0;
    pstBuf->stFrame.stVFrame.enField        = VIDEO_FIELD_FRAME;
    pstBuf->stFrame.stVFrame.enPixelFormat  = pstInFrame->stVFrame.enPixelFormat;
    pstBuf->stFrame.stVFrame.enVideoFormat  = pstInFrame->stVFrame.enVideoFormat;
    pstBuf->stFrame.stVFrame.enCompressMode = COMPRESS_MODE_NONE;
    pstBuf->stFrame.stVFrame.enDynamicRange = pstInFrame->stVFrame.enDynamicRange;
    pstBuf->stFrame.stVFrame.enColorGamut   = pstInFrame->stVFrame.enColorGamut;
    pstBuf->stFrame.stVFrame.u64PhyAddr[0]  = phyAddr;
    pstBuf->stFrame.stVFrame.u64PhyAddr[1]  = phyAddr + ySize;
    pstBuf->stFrame.stVFrame.u64PhyAddr[2]  = 0;
    pstBuf->stFrame.stVFrame.u64VirAddr[0]  = (HI_U64)(uintptr_t)pVirAddr;
    pstBuf->stFrame.stVFrame.u64VirAddr[1]  = (HI_U64)(uintptr_t)((HI_U8 *)pVirAddr + ySize);
    pstBuf->stFrame.stVFrame.u64VirAddr[2]  = 0;

    printf("%s buffer allocated: phy=0x%llx vir=%p pool=%u size=%u ySize=%u\n",
           pszName,
           (unsigned long long)phyAddr,
           pVirAddr,
           pstBuf->stFrame.u32PoolId,
           frameSize,
           ySize);

    return HI_SUCCESS;
}

static HI_VOID FreeFrameBuf(FRAME_BUF_S *pstBuf)
{
    if (pstBuf == HI_NULL)
    {
        return;
    }

    if (pstBuf->pVirAddr != HI_NULL)
    {
        HI_MPI_SYS_Munmap(pstBuf->pVirAddr, pstBuf->u32FrameSize);
    }

    if (pstBuf->hBlk != VB_INVALID_HANDLE)
    {
        HI_MPI_VB_ReleaseBlock(pstBuf->hBlk);
    }

    memset(pstBuf, 0, sizeof(FRAME_BUF_S));
    pstBuf->hBlk = VB_INVALID_HANDLE;
}

static HI_S32 RunGdcPmfOnce(const VIDEO_FRAME_INFO_S *pstInput,
                            VIDEO_FRAME_INFO_S *pstOutput)
{
    HI_S32 s32Ret;
    GDC_HANDLE hHandle = -1;
    GDC_TASK_ATTR_S stTask;
    GDC_PMF_ATTR_S stPmfAttr;
    static int s_dbg_cnt = 0;

    memset(&stTask, 0, sizeof(stTask));
    memset(&stPmfAttr, 0, sizeof(stPmfAttr));

    stTask.stImgIn = *pstInput;
    stTask.stImgOut = *pstOutput;

    CleanGdcTaskFrame(&stTask.stImgIn);
    CleanGdcTaskFrame(&stTask.stImgOut);
    BuildPmfAttr_1080P_Cam3ToCam2(&stPmfAttr);

    s32Ret = HI_MPI_GDC_BeginJob(&hHandle);
    if (s32Ret != HI_SUCCESS)
    {
        printf("PMF: HI_MPI_GDC_BeginJob failed, ret=0x%x\n", s32Ret);
        return s32Ret;
    }

    s32Ret = HI_MPI_GDC_AddPMFTask(hHandle, &stTask, &stPmfAttr);
    if (s32Ret != HI_SUCCESS)
    {
        printf("PMF: HI_MPI_GDC_AddPMFTask failed, ret=0x%x\n", s32Ret);
        HI_MPI_GDC_CancelJob(hHandle);
        return s32Ret;
    }

    s32Ret = HI_MPI_GDC_EndJob(hHandle);
    if (s32Ret != HI_SUCCESS)
    {
        printf("PMF: HI_MPI_GDC_EndJob failed, ret=0x%x\n", s32Ret);
        HI_MPI_GDC_CancelJob(hHandle);
        return s32Ret;
    }

    if (s_dbg_cnt < 3)
    {
        printf("[PMF DEBUG] Begin/Add/End OK, handle=%d\n", hHandle);
        s_dbg_cnt++;
    }

    return HI_SUCCESS;
}

static HI_S32 RunIveAddPlane(HI_U64 u64Src1Phy,
                             HI_U64 u64Src2Phy,
                             HI_U64 u64DstPhy,
                             HI_U32 u32Width,
                             HI_U32 u32Height,
                             HI_U32 u32Src1Stride,
                             HI_U32 u32Src2Stride,
                             HI_U32 u32DstStride,
                             HI_U16 u16WeightSrc1,
                             HI_U16 u16WeightSrc2,
                             const HI_CHAR *pszPlaneName)
{
    HI_S32 s32Ret;
    IVE_HANDLE hIveHandle;
    IVE_SRC_IMAGE_S stSrc1;
    IVE_SRC_IMAGE_S stSrc2;
    IVE_DST_IMAGE_S stDst;
    IVE_ADD_CTRL_S stAddCtrl;
    HI_BOOL bFinish = HI_FALSE;

    memset(&stSrc1, 0, sizeof(stSrc1));
    memset(&stSrc2, 0, sizeof(stSrc2));
    memset(&stDst, 0, sizeof(stDst));
    memset(&stAddCtrl, 0, sizeof(stAddCtrl));

    stAddCtrl.u0q16X = u16WeightSrc1;
    stAddCtrl.u0q16Y = u16WeightSrc2;

    stSrc1.enType = IVE_IMAGE_TYPE_U8C1;
    stSrc1.u32Width = u32Width;
    stSrc1.u32Height = u32Height;
    stSrc1.au32Stride[0] = u32Src1Stride;
    stSrc1.au64PhyAddr[0] = u64Src1Phy;

    stSrc2.enType = IVE_IMAGE_TYPE_U8C1;
    stSrc2.u32Width = u32Width;
    stSrc2.u32Height = u32Height;
    stSrc2.au32Stride[0] = u32Src2Stride;
    stSrc2.au64PhyAddr[0] = u64Src2Phy;

    stDst.enType = IVE_IMAGE_TYPE_U8C1;
    stDst.u32Width = u32Width;
    stDst.u32Height = u32Height;
    stDst.au32Stride[0] = u32DstStride;
    stDst.au64PhyAddr[0] = u64DstPhy;

    s32Ret = HI_MPI_IVE_Add(&hIveHandle, &stSrc1, &stSrc2, &stDst, &stAddCtrl, HI_TRUE);
    if (s32Ret != HI_SUCCESS)
    {
        printf("IVE Add %s failed, ret=0x%x\n", pszPlaneName, s32Ret);
        return s32Ret;
    }

    s32Ret = HI_MPI_IVE_Query(hIveHandle, &bFinish, HI_TRUE);
    if (s32Ret != HI_SUCCESS)
    {
        printf("IVE Query %s failed, ret=0x%x\n", pszPlaneName, s32Ret);
        return s32Ret;
    }

    return HI_SUCCESS;
}

static HI_S32 RunIveFusionOnce(const VIDEO_FRAME_INFO_S *pstCam2,
                               const VIDEO_FRAME_INFO_S *pstCam3Warp,
                               VIDEO_FRAME_INFO_S *pstDst)
{
    HI_S32 s32Ret;
    HI_U32 width;
    HI_U32 height;

    width = pstCam2->stVFrame.u32Width;
    height = pstCam2->stVFrame.u32Height;

    s32Ret = RunIveAddPlane(pstCam2->stVFrame.u64PhyAddr[0],
                            pstCam3Warp->stVFrame.u64PhyAddr[0],
                            pstDst->stVFrame.u64PhyAddr[0],
                            width,
                            height,
                            pstCam2->stVFrame.u32Stride[0],
                            pstCam3Warp->stVFrame.u32Stride[0],
                            pstDst->stVFrame.u32Stride[0],
                            CAM2_WEIGHT_Q16,
                            CAM3_WEIGHT_Q16,
                            "Y");
    if (s32Ret != HI_SUCCESS)
    {
        return s32Ret;
    }

    s32Ret = RunIveAddPlane(pstCam2->stVFrame.u64PhyAddr[1],
                            pstCam3Warp->stVFrame.u64PhyAddr[1],
                            pstDst->stVFrame.u64PhyAddr[1],
                            width,
                            height / 2,
                            pstCam2->stVFrame.u32Stride[1],
                            pstCam3Warp->stVFrame.u32Stride[1],
                            pstDst->stVFrame.u32Stride[1],
                            65535,
                            0,
                            "C_COPY_CAM2");
    if (s32Ret != HI_SUCCESS)
    {
        return s32Ret;
    }

    return HI_SUCCESS;
}

void* Keyboard_Control_Thread(void* arg)
{
    fd_set readfds;
    struct timeval timeout;
    char ch;
    int ret;

    if (SetKeyboardRawMode() != HI_SUCCESS)
    {
        printf("SetKeyboardRawMode failed. You may need to press Enter after key input.\n");
    }

    printf("\n=======================================================\n");
    printf(">>> GDC PMF + IVE Fusion keyboard control <<<\n");
    printf(">>> q: quit <<<\n");
    printf("=======================================================\n\n");

    while (g_bRun)
    {
        FD_ZERO(&readfds);
        FD_SET(STDIN_FILENO, &readfds);
        timeout.tv_sec = 0;
        timeout.tv_usec = 100000;

        ret = select(STDIN_FILENO + 1, &readfds, HI_NULL, HI_NULL, &timeout);
        if (ret > 0 && FD_ISSET(STDIN_FILENO, &readfds))
        {
            if (read(STDIN_FILENO, &ch, 1) == 1)
            {
                if (ch == 'q' || ch == 'Q')
                {
                    printf("\n[KEY] Quit request received.\n");
                    g_bRun = HI_FALSE;
                    break;
                }
            }
        }
    }

    RestoreKeyboardMode();
    return NULL;
}

void* Gdc_Pmf_Ive_Fusion_Thread(void* arg)
{
    HI_S32 s32Ret = HI_SUCCESS;
    VIDEO_FRAME_INFO_S stFrameCam2;
    VIDEO_FRAME_INFO_S stFrameCam3;
    FRAME_BUF_S stGdcOutBuf[GDC_OUT_BUF_NUM];
    FRAME_BUF_S stFusionBuf[FUSION_OUT_BUF_NUM];
    VPSS_GRP VpssGrp_Cam2 = 2;
    VPSS_GRP VpssGrp_Cam3 = 3;
    VPSS_CHN VpssChn = 0;
    VO_LAYER VoLayer = SAMPLE_VO_DEV_DHD0;
    VO_CHN VoChn_Out = 0;
    HI_BOOL bAllocated = HI_FALSE;
    HI_BOOL bFirstFramePrint = HI_TRUE;
    HI_U32 u32BufIndex = 0;
    int success_frames = 0;
    int get_fail_cnt = 0;
    int gdc_fail_cnt = 0;
    int ive_fail_cnt = 0;
    int vo_fail_cnt = 0;
    int fps_count = 0;
    double fps_start_ms = GetTimeMs();
    double last_gdc_ms = 0.0;
    double last_ive_ms = 0.0;
    double last_send_ms = 0.0;
    double last_total_ms = 0.0;
    HI_U32 i;

    for (i = 0; i < GDC_OUT_BUF_NUM; i++)
    {
        memset(&stGdcOutBuf[i], 0, sizeof(FRAME_BUF_S));
        stGdcOutBuf[i].hBlk = VB_INVALID_HANDLE;
    }

    for (i = 0; i < FUSION_OUT_BUF_NUM; i++)
    {
        memset(&stFusionBuf[i], 0, sizeof(FRAME_BUF_S));
        stFusionBuf[i].hBlk = VB_INVALID_HANDLE;
    }

    printf("\n=======================================================\n");
    printf(">>> [GDC PMF + IVE Fusion] Cam2 + warped Cam3 -> HDMI <<<\n");
    printf(">>> Cam2: VpssGrp 2, Cam3: VpssGrp 3 <<<\n");
    printf(">>> Fusion weight: Cam2=%u/65536, Cam3=%u/65536 <<<\n",
           CAM2_WEIGHT_Q16, CAM3_WEIGHT_Q16);
    printf("=======================================================\n");

    {
        HI_S32 warmRet2 = HI_FAILURE;
        HI_S32 warmRet3 = HI_FAILURE;
        VIDEO_FRAME_INFO_S warmCam2;
        VIDEO_FRAME_INFO_S warmCam3;
        HI_S32 warmCnt;

        printf(">>> warming up VpssGrp 2 and VpssGrp 3...\n");

        for (warmCnt = 0; warmCnt < 100 && g_bRun; warmCnt++)
        {
            memset(&warmCam2, 0, sizeof(VIDEO_FRAME_INFO_S));
            memset(&warmCam3, 0, sizeof(VIDEO_FRAME_INFO_S));

            warmRet2 = HI_MPI_VPSS_GetChnFrame(VpssGrp_Cam2, VpssChn, &warmCam2, 200);
            if (warmRet2 == HI_SUCCESS)
            {
                HI_MPI_VPSS_ReleaseChnFrame(VpssGrp_Cam2, VpssChn, &warmCam2);
            }

            warmRet3 = HI_MPI_VPSS_GetChnFrame(VpssGrp_Cam3, VpssChn, &warmCam3, 200);
            if (warmRet3 == HI_SUCCESS)
            {
                HI_MPI_VPSS_ReleaseChnFrame(VpssGrp_Cam3, VpssChn, &warmCam3);
            }

            if (warmRet2 == HI_SUCCESS && warmRet3 == HI_SUCCESS)
            {
                printf(">>> VPSS warmup OK: Cam2 grp=%d, Cam3 grp=%d <<<\n",
                       VpssGrp_Cam2,
                       VpssGrp_Cam3);
                break;
            }

            if (warmCnt % 10 == 0)
            {
                printf("VPSS warmup waiting... cnt=%d Cam2 ret=0x%x Cam3 ret=0x%x\n",
                       warmCnt,
                       warmRet2,
                       warmRet3);
            }

            usleep(100000);
        }

        if (!(warmRet2 == HI_SUCCESS && warmRet3 == HI_SUCCESS))
        {
            printf("VPSS warmup failed: Cam2 ret=0x%x Cam3 ret=0x%x\n",
                   warmRet2,
                   warmRet3);
            printf("Please reboot board or run old IVE sample to confirm VpssGrp 2/3 are both alive.\n");
        }
    }

    while (g_bRun)
    {
        double t_total0;
        double t_gdc0;
        double t_ive0;
        double t_send0;

        memset(&stFrameCam2, 0, sizeof(VIDEO_FRAME_INFO_S));
        memset(&stFrameCam3, 0, sizeof(VIDEO_FRAME_INFO_S));

        s32Ret = HI_MPI_VPSS_GetChnFrame(VpssGrp_Cam2, VpssChn, &stFrameCam2, 100);
        if (s32Ret != HI_SUCCESS)
        {
            get_fail_cnt++;
            if (get_fail_cnt % 30 == 0)
            {
                printf("HI_MPI_VPSS_GetChnFrame Cam2 failed, ret=0x%x cnt=%d\n",
                       s32Ret,
                       get_fail_cnt);
            }
            continue;
        }

        s32Ret = HI_MPI_VPSS_GetChnFrame(VpssGrp_Cam3, VpssChn, &stFrameCam3, 100);
        if (s32Ret != HI_SUCCESS)
        {
            get_fail_cnt++;
            if (get_fail_cnt % 30 == 0)
            {
                printf("HI_MPI_VPSS_GetChnFrame Cam3 failed, ret=0x%x cnt=%d\n",
                       s32Ret,
                       get_fail_cnt);
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
        }

        if (!bAllocated)
        {
            for (i = 0; i < GDC_OUT_BUF_NUM; i++)
            {
                s32Ret = AllocFrameLikeInput(&stFrameCam3, &stGdcOutBuf[i], "GDC warped Cam3");
                if (s32Ret != HI_SUCCESS)
                {
                    printf("Alloc GDC output buffer failed, index=%u ret=0x%x\n", i, s32Ret);
                    g_bRun = HI_FALSE;
                    break;
                }
            }

            if (!g_bRun)
            {
                HI_MPI_VPSS_ReleaseChnFrame(VpssGrp_Cam2, VpssChn, &stFrameCam2);
                HI_MPI_VPSS_ReleaseChnFrame(VpssGrp_Cam3, VpssChn, &stFrameCam3);
                break;
            }

            for (i = 0; i < FUSION_OUT_BUF_NUM; i++)
            {
                s32Ret = AllocFrameLikeInput(&stFrameCam2, &stFusionBuf[i], "IVE fusion dst");
                if (s32Ret != HI_SUCCESS)
                {
                    printf("Alloc fusion output buffer failed, index=%u ret=0x%x\n", i, s32Ret);
                    g_bRun = HI_FALSE;
                    break;
                }
            }

            if (!g_bRun)
            {
                HI_MPI_VPSS_ReleaseChnFrame(VpssGrp_Cam2, VpssChn, &stFrameCam2);
                HI_MPI_VPSS_ReleaseChnFrame(VpssGrp_Cam3, VpssChn, &stFrameCam3);
                break;
            }

            bAllocated = HI_TRUE;
            bFirstFramePrint = HI_FALSE;
        }

        t_total0 = GetTimeMs();
        t_gdc0 = GetTimeMs();

        s32Ret = RunGdcPmfOnce(&stFrameCam3, &stGdcOutBuf[u32BufIndex].stFrame);
        last_gdc_ms = GetTimeMs() - t_gdc0;

        if (s32Ret != HI_SUCCESS)
        {
            gdc_fail_cnt++;
            if (gdc_fail_cnt % 10 == 0)
            {
                printf("RunGdcPmfOnce failed, ret=0x%x cnt=%d\n",
                       s32Ret,
                       gdc_fail_cnt);
            }
            HI_MPI_VPSS_ReleaseChnFrame(VpssGrp_Cam2, VpssChn, &stFrameCam2);
            HI_MPI_VPSS_ReleaseChnFrame(VpssGrp_Cam3, VpssChn, &stFrameCam3);
            continue;
        }

        t_ive0 = GetTimeMs();

        s32Ret = RunIveFusionOnce(&stFrameCam2,
                                  &stGdcOutBuf[u32BufIndex].stFrame,
                                  &stFusionBuf[u32BufIndex].stFrame);
        last_ive_ms = GetTimeMs() - t_ive0;

        if (s32Ret != HI_SUCCESS)
        {
            ive_fail_cnt++;
            if (ive_fail_cnt % 10 == 0)
            {
                printf("RunIveFusionOnce failed, ret=0x%x cnt=%d\n",
                       s32Ret,
                       ive_fail_cnt);
            }
            HI_MPI_VPSS_ReleaseChnFrame(VpssGrp_Cam2, VpssChn, &stFrameCam2);
            HI_MPI_VPSS_ReleaseChnFrame(VpssGrp_Cam3, VpssChn, &stFrameCam3);
            continue;
        }

        t_send0 = GetTimeMs();

        s32Ret = HI_MPI_VO_SendFrame(VoLayer,
                                     VoChn_Out,
                                     &stFusionBuf[u32BufIndex].stFrame,
                                     -1);

        last_send_ms = GetTimeMs() - t_send0;
        last_total_ms = GetTimeMs() - t_total0;

        if (s32Ret == HI_SUCCESS)
        {
            success_frames++;
            fps_count++;

            if (fps_count >= 60)
            {
                double now_ms = GetTimeMs();
                double fps = 60000.0 / (now_ms - fps_start_ms);

                printf("\033[0;32m>>> [GDC+IVE Fusion OK] frames=%d fps=%.2f total=%.2fms gdc=%.2fms ive=%.2fms send=%.2fms <<<\033[0;39m\n",
                       success_frames,
                       fps,
                       last_total_ms,
                       last_gdc_ms,
                       last_ive_ms,
                       last_send_ms);

                fps_count = 0;
                fps_start_ms = now_ms;
            }

            u32BufIndex++;
            if (u32BufIndex >= GDC_OUT_BUF_NUM)
            {
                u32BufIndex = 0;
            }
        }
        else
        {
            vo_fail_cnt++;
            if (vo_fail_cnt % 30 == 0)
            {
                printf("HI_MPI_VO_SendFrame fusion failed, ret=0x%x cnt=%d\n",
                       s32Ret,
                       vo_fail_cnt);
            }
        }

        HI_MPI_VPSS_ReleaseChnFrame(VpssGrp_Cam2, VpssChn, &stFrameCam2);
        HI_MPI_VPSS_ReleaseChnFrame(VpssGrp_Cam3, VpssChn, &stFrameCam3);
    }

    if (bAllocated)
    {
        for (i = 0; i < GDC_OUT_BUF_NUM; i++)
        {
            FreeFrameBuf(&stGdcOutBuf[i]);
        }

        for (i = 0; i < FUSION_OUT_BUF_NUM; i++)
        {
            FreeFrameBuf(&stFusionBuf[i]);
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
    HI_BOOL                 abChnEnable[VPSS_MAX_PHY_CHN_NUM] = {HI_FALSE};
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
    SIZE_S                  stSize[6]           = {
        {1920, 1080},
        {3840, 2160},
        {1920, 1080},
        {1920, 1080},
        {720, 576},
        {1920, 1080}
    };
    HI_U32                  u32BlkSize;
    VB_CONFIG_S             stVbConf;
    SAMPLE_VI_CONFIG_S      stViConfig;
    SAMPLE_VO_CONFIG_S      stVoConfig;
    VI_VPSS_MODE_S          stVIVPSSMode;
    pthread_t               fusion_thread_id;
    pthread_t               keyboard_thread_id;

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
        stViConfig.as32WorkingViId[i]                     = i;
        stViConfig.astViInfo[i].stSnsInfo.MipiDev         =
            SAMPLE_COMM_VI_GetComboDevBySensor(stViConfig.astViInfo[i].stSnsInfo.enSnsType, i);
        stViConfig.astViInfo[i].stSnsInfo.s32BusId        = i;
        stViConfig.astViInfo[i].stDevInfo.ViDev           = ViDev[i];
        stViConfig.astViInfo[i].stDevInfo.enWDRMode       = enWDRMode;
        stViConfig.astViInfo[i].stPipeInfo.enMastPipeMode = enMastPipeMode;
        stViConfig.astViInfo[i].stPipeInfo.aPipe[0]       = ViPipe[i];
        stViConfig.astViInfo[i].stPipeInfo.aPipe[1]       = -1;
        stViConfig.astViInfo[i].stPipeInfo.aPipe[2]       = -1;
        stViConfig.astViInfo[i].stPipeInfo.aPipe[3]       = -1;
        stViConfig.astViInfo[i].stChnInfo.ViChn           = ViPhyChn;
        stViConfig.astViInfo[i].stChnInfo.enPixFormat     = enPixFormat;
        stViConfig.astViInfo[i].stChnInfo.enDynamicRange  = enDynamicRange;
        stViConfig.astViInfo[i].stChnInfo.enVideoFormat   = enVideoFormat;
        stViConfig.astViInfo[i].stChnInfo.enCompressMode  = enCompressMode;
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
    stVbConf.astCommPool[0].u32BlkCnt  = 80;

    SAMPLE_COMM_SYS_Exit();
    usleep(200000);

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

    stVpssGrpAttr.u32MaxW                     = 3840;
    stVpssGrpAttr.u32MaxH                     = 2160;
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
            printf("SAMPLE_COMM_VPSS_Start failed, grp=%d ret=0x%x\n",
                   VpssDevGrp[i],
                   s32Ret);
            goto EXIT4;
        }
    }

    for (i = 0; i < s32VpssDevGrpCnt; i++)
    {
        s32Ret = SAMPLE_COMM_VI_Bind_VPSS(ViPipe[i], ViPhyChn, VpssDevGrp[i]);
        if (HI_SUCCESS != s32Ret)
        {
            printf("SAMPLE_COMM_VI_Bind_VPSS failed, pipe=%d chn=%d grp=%d ret=0x%x\n",
                   ViPipe[i],
                   ViPhyChn,
                   VpssDevGrp[i],
                   s32Ret);
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

    printf(">>> waiting VI/VPSS stable before starting fusion thread...\n");
    sleep(3);

    s32Ret = pthread_create(&fusion_thread_id, NULL, Gdc_Pmf_Ive_Fusion_Thread, NULL);
    if (s32Ret != 0)
    {
        printf("pthread_create fusion thread failed, ret=%d\n", s32Ret);
        g_bRun = HI_FALSE;
        goto EXIT2;
    }

    s32Ret = pthread_create(&keyboard_thread_id, NULL, Keyboard_Control_Thread, NULL);
    if (s32Ret != 0)
    {
        printf("pthread_create keyboard thread failed, ret=%d\n", s32Ret);
        g_bRun = HI_FALSE;
        pthread_join(fusion_thread_id, NULL);
        goto EXIT2;
    }

    printf("\n>>> 底层视频配置完毕，进入 GDC PMF + IVE 融合模式 <<<\n");
    printf(">>> 当前目标：Cam2 + GDC_PMF(Cam3) -> IVE Add -> HDMI。按 q 退出。<<<\n");

    while (g_bRun)
    {
        sleep(1);
    }

    pthread_join(fusion_thread_id, NULL);
    pthread_join(keyboard_thread_id, NULL);

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
    RestoreKeyboardMode();
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
    signal(SIGSEGV, SAMPLE_VIO_HandleSig);
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
