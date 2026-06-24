#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define W 224
#define H 224
#define N (W * H)

#define TOTAL_WEIGHTS 71154

#define OFF_ENC10_W 0
#define OFF_ENC10_B 432
#define OFF_ENC12_W 448
#define OFF_ENC12_B 2752
#define OFF_ENC20_W 2768
#define OFF_ENC20_B 7376
#define OFF_ENC22_W 7408
#define OFF_ENC22_B 16624
#define OFF_MID0_W 16656
#define OFF_MID0_B 30480
#define OFF_MID2_W 30528
#define OFF_MID2_B 51264
#define OFF_UP2_W 51312
#define OFF_UP2_B 57456
#define OFF_DEC20_W 57488
#define OFF_DEC20_B 66704
#define OFF_UP1_W 66736
#define OFF_UP1_B 68784
#define OFF_DEC10_W 68800
#define OFF_DEC10_B 71104
#define OFF_OUT_W 71120
#define OFF_OUT_B 71152

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static void *xmalloc(size_t n)
{
    void *p = malloc(n);
    if (!p)
    {
        printf("malloc failed: %zu bytes\n", n);
        exit(1);
    }
    memset(p, 0, n);
    return p;
}

static int read_file(const char *path, void *buf, size_t size)
{
    FILE *fp = fopen(path, "rb");
    if (!fp)
    {
        printf("open failed: %s\n", path);
        return -1;
    }

    size_t n = fread(buf, 1, size, fp);
    fclose(fp);

    if (n != size)
    {
        printf("read size error: %s, got %zu, expect %zu\n", path, n, size);
        return -1;
    }

    return 0;
}

static int write_file(const char *path, const void *buf, size_t size)
{
    FILE *fp = fopen(path, "wb");
    if (!fp)
    {
        printf("write failed: %s\n", path);
        return -1;
    }

    fwrite(buf, 1, size, fp);
    fclose(fp);
    return 0;
}

static int write_pgm(const char *path, const uint8_t *buf)
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

static inline int idx3(int c, int y, int x, int height, int width)
{
    return c * height * width + y * width + x;
}

static void conv2d_same_3x3(
    const float *input,
    float *output,
    const float *weight,
    const float *bias,
    int in_c,
    int out_c,
    int height,
    int width,
    int relu
)
{
    int oc, ic, y, x, ky, kx;

    for (oc = 0; oc < out_c; oc++)
    {
        for (y = 0; y < height; y++)
        {
            for (x = 0; x < width; x++)
            {
                float sum = bias[oc];

                for (ic = 0; ic < in_c; ic++)
                {
                    for (ky = 0; ky < 3; ky++)
                    {
                        int yy = y + ky - 1;
                        if (yy < 0 || yy >= height) continue;

                        for (kx = 0; kx < 3; kx++)
                        {
                            int xx = x + kx - 1;
                            if (xx < 0 || xx >= width) continue;

                            float v = input[idx3(ic, yy, xx, height, width)];
                            float w = weight[(((oc * in_c + ic) * 3 + ky) * 3 + kx)];
                            sum += v * w;
                        }
                    }
                }

                if (relu && sum < 0.0f) sum = 0.0f;
                output[idx3(oc, y, x, height, width)] = sum;
            }
        }
    }
}

static void conv1x1(
    const float *input,
    float *output,
    const float *weight,
    const float *bias,
    int in_c,
    int out_c,
    int height,
    int width
)
{
    int oc, ic, y, x;

    for (oc = 0; oc < out_c; oc++)
    {
        for (y = 0; y < height; y++)
        {
            for (x = 0; x < width; x++)
            {
                float sum = bias[oc];

                for (ic = 0; ic < in_c; ic++)
                {
                    float v = input[idx3(ic, y, x, height, width)];
                    float w = weight[oc * in_c + ic];
                    sum += v * w;
                }

                output[idx3(oc, y, x, height, width)] = sum;
            }
        }
    }
}

static void maxpool2x2(
    const float *input,
    float *output,
    int channels,
    int in_h,
    int in_w
)
{
    int c, y, x;
    int out_h = in_h / 2;
    int out_w = in_w / 2;

    for (c = 0; c < channels; c++)
    {
        for (y = 0; y < out_h; y++)
        {
            for (x = 0; x < out_w; x++)
            {
                float a = input[idx3(c, y * 2,     x * 2,     in_h, in_w)];
                float b = input[idx3(c, y * 2,     x * 2 + 1, in_h, in_w)];
                float d = input[idx3(c, y * 2 + 1, x * 2,     in_h, in_w)];
                float e = input[idx3(c, y * 2 + 1, x * 2 + 1, in_h, in_w)];

                float m = a;
                if (b > m) m = b;
                if (d > m) m = d;
                if (e > m) m = e;

                output[idx3(c, y, x, out_h, out_w)] = m;
            }
        }
    }
}

/*
 * PyTorch ConvTranspose2d weight layout:
 * weight shape = (in_channels, out_channels, 2, 2)
 */
static void deconv2x2_stride2(
    const float *input,
    float *output,
    const float *weight,
    const float *bias,
    int in_c,
    int out_c,
    int in_h,
    int in_w
)
{
    int ic, oc, y, x, ky, kx;
    int out_h = in_h * 2;
    int out_w = in_w * 2;
    size_t out_size = (size_t)out_c * out_h * out_w;

    memset(output, 0, out_size * sizeof(float));

    for (ic = 0; ic < in_c; ic++)
    {
        for (y = 0; y < in_h; y++)
        {
            for (x = 0; x < in_w; x++)
            {
                float v = input[idx3(ic, y, x, in_h, in_w)];

                for (oc = 0; oc < out_c; oc++)
                {
                    for (ky = 0; ky < 2; ky++)
                    {
                        for (kx = 0; kx < 2; kx++)
                        {
                            float w = weight[(((ic * out_c + oc) * 2 + ky) * 2 + kx)];
                            int yy = y * 2 + ky;
                            int xx = x * 2 + kx;
                            output[idx3(oc, yy, xx, out_h, out_w)] += v * w;
                        }
                    }
                }
            }
        }
    }

    for (oc = 0; oc < out_c; oc++)
    {
        for (y = 0; y < out_h; y++)
        {
            for (x = 0; x < out_w; x++)
            {
                output[idx3(oc, y, x, out_h, out_w)] += bias[oc];
            }
        }
    }
}

static void make_prob_and_mask(
    const float *logits,
    float *prob,
    uint8_t *mask,
    float threshold
)
{
    int y, x;

    for (y = 0; y < H; y++)
    {
        for (x = 0; x < W; x++)
        {
            float l0 = logits[idx3(0, y, x, H, W)];
            float l1 = logits[idx3(1, y, x, H, W)];
            float m = l0 > l1 ? l0 : l1;
            float e0 = expf(l0 - m);
            float e1 = expf(l1 - m);
            float p1 = e1 / (e0 + e1);

            prob[y * W + x] = p1;
            mask[y * W + x] = (p1 >= threshold) ? 255 : 0;
        }
    }
}

int main(int argc, char *argv[])
{
    char dir[512];
    char sid[128];

    char path_weights[1024];
    char path_input[1024];
    char path_prob[1024];
    char path_mask_y[1024];
    char path_mask_pgm[1024];

    double t0, t1, t2, t3, t4;

    if (argc < 3)
    {
        printf("usage: %s <data_dir> <sample_id>\n", argv[0]);
        printf("example: %s /get/tinyfusion_board_test 000000\n", argv[0]);
        return -1;
    }

    snprintf(dir, sizeof(dir), "%s", argv[1]);
    snprintf(sid, sizeof(sid), "%s", argv[2]);

    snprintf(path_weights, sizeof(path_weights), "%s/tinyfusion_weights.bin", dir);
    snprintf(path_input, sizeof(path_input), "%s/input_%s_chw_float.bin", dir, sid);
    snprintf(path_prob, sizeof(path_prob), "%s/board_prob_%s.bin", dir, sid);
    snprintf(path_mask_y, sizeof(path_mask_y), "%s/board_mask_%s.y", dir, sid);
    snprintf(path_mask_pgm, sizeof(path_mask_pgm), "%s/board_mask_%s.pgm", dir, sid);

    printf("TinyFusionMaskNet ARM CPU inference\n");
    printf("dir: %s\n", dir);
    printf("sample: %s\n", sid);

    float *weights = (float *)xmalloc(TOTAL_WEIGHTS * sizeof(float));
    float *input = (float *)xmalloc(3 * H * W * sizeof(float));

    float *enc1a = (float *)xmalloc(16 * 224 * 224 * sizeof(float));
    float *enc1b = (float *)xmalloc(16 * 224 * 224 * sizeof(float));
    float *pool1 = (float *)xmalloc(16 * 112 * 112 * sizeof(float));

    float *enc2a = (float *)xmalloc(32 * 112 * 112 * sizeof(float));
    float *enc2b = (float *)xmalloc(32 * 112 * 112 * sizeof(float));
    float *pool2 = (float *)xmalloc(32 * 56 * 56 * sizeof(float));

    float *mid1 = (float *)xmalloc(48 * 56 * 56 * sizeof(float));
    float *mid2 = (float *)xmalloc(48 * 56 * 56 * sizeof(float));

    float *up2 = (float *)xmalloc(32 * 112 * 112 * sizeof(float));
    float *dec2 = (float *)xmalloc(32 * 112 * 112 * sizeof(float));

    float *up1 = (float *)xmalloc(16 * 224 * 224 * sizeof(float));
    float *dec1 = (float *)xmalloc(16 * 224 * 224 * sizeof(float));

    float *logits = (float *)xmalloc(2 * 224 * 224 * sizeof(float));
    float *prob = (float *)xmalloc(224 * 224 * sizeof(float));
    uint8_t *mask = (uint8_t *)xmalloc(224 * 224 * sizeof(uint8_t));

    t0 = now_ms();

    if (read_file(path_weights, weights, TOTAL_WEIGHTS * sizeof(float)) != 0) return -1;
    if (read_file(path_input, input, 3 * H * W * sizeof(float)) != 0) return -1;

    t1 = now_ms();

    conv2d_same_3x3(input, enc1a, weights + OFF_ENC10_W, weights + OFF_ENC10_B, 3, 16, 224, 224, 1);
    conv2d_same_3x3(enc1a, enc1b, weights + OFF_ENC12_W, weights + OFF_ENC12_B, 16, 16, 224, 224, 1);
    maxpool2x2(enc1b, pool1, 16, 224, 224);

    conv2d_same_3x3(pool1, enc2a, weights + OFF_ENC20_W, weights + OFF_ENC20_B, 16, 32, 112, 112, 1);
    conv2d_same_3x3(enc2a, enc2b, weights + OFF_ENC22_W, weights + OFF_ENC22_B, 32, 32, 112, 112, 1);
    maxpool2x2(enc2b, pool2, 32, 112, 112);

    conv2d_same_3x3(pool2, mid1, weights + OFF_MID0_W, weights + OFF_MID0_B, 32, 48, 56, 56, 1);
    conv2d_same_3x3(mid1, mid2, weights + OFF_MID2_W, weights + OFF_MID2_B, 48, 48, 56, 56, 1);

    deconv2x2_stride2(mid2, up2, weights + OFF_UP2_W, weights + OFF_UP2_B, 48, 32, 56, 56);
    conv2d_same_3x3(up2, dec2, weights + OFF_DEC20_W, weights + OFF_DEC20_B, 32, 32, 112, 112, 1);

    deconv2x2_stride2(dec2, up1, weights + OFF_UP1_W, weights + OFF_UP1_B, 32, 16, 112, 112);
    conv2d_same_3x3(up1, dec1, weights + OFF_DEC10_W, weights + OFF_DEC10_B, 16, 16, 224, 224, 1);

    conv1x1(dec1, logits, weights + OFF_OUT_W, weights + OFF_OUT_B, 16, 2, 224, 224);

    t2 = now_ms();

    make_prob_and_mask(logits, prob, mask, 0.70f);

    t3 = now_ms();

    if (write_file(path_prob, prob, N * sizeof(float)) != 0) return -1;
    if (write_file(path_mask_y, mask, N * sizeof(uint8_t)) != 0) return -1;
    if (write_pgm(path_mask_pgm, mask) != 0) return -1;

    t4 = now_ms();

    printf("inference done\n");
    printf("out prob: %s\n", path_prob);
    printf("out mask: %s\n", path_mask_y);
    printf("out pgm : %s\n", path_mask_pgm);

    printf("\n[TIME]\n");
    printf("read_ms      = %.3f\n", t1 - t0);
    printf("net_ms       = %.3f\n", t2 - t1);
    printf("post_ms      = %.3f\n", t3 - t2);
    printf("write_ms     = %.3f\n", t4 - t3);
    printf("total_ms     = %.3f\n", t4 - t0);

    free(weights);
    free(input);
    free(enc1a);
    free(enc1b);
    free(pool1);
    free(enc2a);
    free(enc2b);
    free(pool2);
    free(mid1);
    free(mid2);
    free(up2);
    free(dec2);
    free(up1);
    free(dec1);
    free(logits);
    free(prob);
    free(mask);

    return 0;
}
