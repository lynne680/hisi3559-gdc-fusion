#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define W 224
#define H 224
#define N (W * H)

static uint8_t g_rgb[N];
static uint8_t g_thermal[N];
static uint8_t g_uv[N];
static uint8_t g_mask[N];
static uint8_t g_soft_mask[N];
static uint8_t g_fused[N];

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

static void make_soft_mask_11x11(const uint8_t *mask, uint8_t *soft)
{
    int x, y, dx, dy;

    for (y = 0; y < H; y++)
    {
        for (x = 0; x < W; x++)
        {
            int sum = 0;
            int count = 0;

            for (dy = -5; dy <= 5; dy++)
            {
                int yy = y + dy;
                if (yy < 0 || yy >= H) continue;

                for (dx = -5; dx <= 5; dx++)
                {
                    int xx = x + dx;
                    if (xx < 0 || xx >= W) continue;

                    sum += mask[yy * W + xx];
                    count++;
                }
            }

            soft[y * W + x] = (uint8_t)(sum / count);
        }
    }
}

static void fuse_image(void)
{
    int i;

    make_soft_mask_11x11(g_mask, g_soft_mask);

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

    printf("MM5 static fusion test\n");
    printf("dir: %s\n", dir);
    printf("sample: %s\n", sid);

    if (read_raw(path_rgb, g_rgb, N) != 0) return -1;
    if (read_raw(path_thermal, g_thermal, N) != 0) return -1;
    if (read_raw(path_uv, g_uv, N) != 0) return -1;
    if (read_raw(path_mask, g_mask, N) != 0) return -1;

    fuse_image();

    if (write_raw(path_out_y, g_fused, N) != 0) return -1;
    if (write_pgm(path_out_pgm, g_fused) != 0) return -1;

    printf("fusion done\n");
    printf("out y:   %s\n", path_out_y);
    printf("out pgm: %s\n", path_out_pgm);

    return 0;
}
