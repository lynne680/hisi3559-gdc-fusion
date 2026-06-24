#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#define W 224
#define H 224
#define N (W * H)

static uint8_t g_rgb[N];
static uint8_t g_thermal[N];
static uint8_t g_uv[N];
static uint8_t g_mask[N];
static uint8_t g_soft_mask[N];
static uint8_t g_fused[N];

/* Integral image size: (H+1) x (W+1) */
static uint32_t g_integral[(H + 1) * (W + 1)];

static double get_time_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static int read_raw(const char *path, uint8_t *buf, int size)
{
    FILE *fp = fopen(path, "rb");
    if (!fp)
    {
        printf("open failed: %s\n", path);
        return -1;
    }

    int n = fread(buf, 1, size, fp);
    fclose(fp);

    if (n != size)
    {
        printf("read size error: %s, got %d, expect %d\n", path, n, size);
        return -1;
    }

    return 0;
}

static int write_raw(const char *path, uint8_t *buf, int size)
{
    FILE *fp = fopen(path, "wb");
    if (!fp)
    {
        printf("write raw failed: %s\n", path);
        return -1;
    }

    fwrite(buf, 1, size, fp);
    fclose(fp);
    return 0;
}

static int write_pgm(const char *path, uint8_t *buf)
{
    FILE *fp = fopen(path, "wb");
    if (!fp)
    {
        printf("write pgm failed: %s\n", path);
        return -1;
    }

    fprintf(fp, "P5\n%d %d\n255\n", W, H);
    fwrite(buf, 1, N, fp);
    fclose(fp);
    return 0;
}

/*
 * Optimized 11x11 soft mask by integral image.
 * It keeps the same boundary rule as the original implementation:
 * window is clipped at image borders.
 */
static void make_soft_mask_11x11_integral(const uint8_t *mask, uint8_t *soft)
{
    int x, y;

    memset(g_integral, 0, sizeof(g_integral));

    for (y = 1; y <= H; y++)
    {
        uint32_t row_sum = 0;
        for (x = 1; x <= W; x++)
        {
            row_sum += mask[(y - 1) * W + (x - 1)];
            g_integral[y * (W + 1) + x] =
                g_integral[(y - 1) * (W + 1) + x] + row_sum;
        }
    }

    for (y = 0; y < H; y++)
    {
        int y1 = y - 5;
        int y2 = y + 5;
        if (y1 < 0) y1 = 0;
        if (y2 >= H) y2 = H - 1;

        for (x = 0; x < W; x++)
        {
            int x1 = x - 5;
            int x2 = x + 5;
            if (x1 < 0) x1 = 0;
            if (x2 >= W) x2 = W - 1;

            uint32_t A = g_integral[y1 * (W + 1) + x1];
            uint32_t B = g_integral[y1 * (W + 1) + (x2 + 1)];
            uint32_t C = g_integral[(y2 + 1) * (W + 1) + x1];
            uint32_t D = g_integral[(y2 + 1) * (W + 1) + (x2 + 1)];

            uint32_t sum = D - B - C + A;
            uint32_t count = (uint32_t)(x2 - x1 + 1) * (uint32_t)(y2 - y1 + 1);

            soft[y * W + x] = (uint8_t)(sum / count);
        }
    }
}

static void fuse_image_only(void)
{
    int i;

    for (i = 0; i < N; i++)
    {
        int rgb = g_rgb[i];
        int thermal = g_thermal[i];
        int uv = g_uv[i];

        /*
         * Final weights:
         * base   = 0.94 RGB + 0.04 Thermal + 0.02 UV
         * target = 0.50 RGB + 0.40 Thermal + 0.10 UV
         */
        int base = (94 * rgb + 4 * thermal + 2 * uv + 50) / 100;
        int target = (50 * rgb + 40 * thermal + 10 * uv + 50) / 100;

        int m = g_soft_mask[i];
        int out = (base * (255 - m) + target * m + 127) / 255;

        if (out < 0) out = 0;
        if (out > 255) out = 255;

        g_fused[i] = (uint8_t)out;
    }
}

int main(int argc, char *argv[])
{
    char dir[512];
    char sid[128];

    char path_rgb[1024];
    char path_thermal[1024];
    char path_uv[1024];
    char path_mask[1024];
    char path_out_y[1024];
    char path_out_pgm[1024];

    double t0, t1, t2, t3, t4, t5;
    double read_ms, mask_ms, fuse_ms, write_ms, total_ms;

    if (argc < 3)
    {
        printf("usage: %s <data_dir> <sample_id>\n", argv[0]);
        printf("example: %s /get/mm5_board_test 000000\n", argv[0]);
        return -1;
    }

    snprintf(dir, sizeof(dir), "%s", argv[1]);
    snprintf(sid, sizeof(sid), "%s", argv[2]);

    snprintf(path_rgb, sizeof(path_rgb), "%s/rgb_%s.y", dir, sid);
    snprintf(path_thermal, sizeof(path_thermal), "%s/thermal_%s.y", dir, sid);
    snprintf(path_uv, sizeof(path_uv), "%s/uv_%s.y", dir, sid);
    snprintf(path_mask, sizeof(path_mask), "%s/mask_%s.y", dir, sid);

    snprintf(path_out_y, sizeof(path_out_y), "%s/fused_%s.y", dir, sid);
    snprintf(path_out_pgm, sizeof(path_out_pgm), "%s/fused_%s.pgm", dir, sid);

    printf("MM5 static fusion test with timing, integral mask version\n");
    printf("dir: %s\n", dir);
    printf("sample: %s\n", sid);

    t0 = get_time_ms();

    if (read_raw(path_rgb, g_rgb, N) != 0) return -1;
    if (read_raw(path_thermal, g_thermal, N) != 0) return -1;
    if (read_raw(path_uv, g_uv, N) != 0) return -1;
    if (read_raw(path_mask, g_mask, N) != 0) return -1;

    t1 = get_time_ms();

    make_soft_mask_11x11_integral(g_mask, g_soft_mask);

    t2 = get_time_ms();

    fuse_image_only();

    t3 = get_time_ms();

    if (write_raw(path_out_y, g_fused, N) != 0) return -1;
    if (write_pgm(path_out_pgm, g_fused) != 0) return -1;

    t4 = get_time_ms();
    t5 = t4;

    read_ms = t1 - t0;
    mask_ms = t2 - t1;
    fuse_ms = t3 - t2;
    write_ms = t4 - t3;
    total_ms = t5 - t0;

    printf("fusion done\n");
    printf("out y:   %s\n", path_out_y);
    printf("out pgm: %s\n", path_out_pgm);

    printf("\n[TIME]\n");
    printf("read_ms  = %.3f\n", read_ms);
    printf("mask_ms  = %.3f\n", mask_ms);
    printf("fuse_ms  = %.3f\n", fuse_ms);
    printf("write_ms = %.3f\n", write_ms);
    printf("total_ms = %.3f\n", total_ms);
    printf("fps_est  = %.2f\n", 1000.0 / total_ms);

    return 0;
}
