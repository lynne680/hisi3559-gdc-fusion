#!/bin/bash
set -e

CUR_DIR=$(pwd)

rm -rf build_gdc_pmf
mkdir -p build_gdc_pmf

export CC=/opt/hisi-linux/x86-arm/aarch64-himix100-linux/bin/aarch64-himix100-linux-gcc
export READELF=/opt/hisi-linux/x86-arm/aarch64-himix100-linux/bin/aarch64-himix100-linux-readelf

export MPP_INC=${CUR_DIR}/../../out/linux/multi-core/include
export MPP_LIB=${CUR_DIR}/../../out/linux/multi-core/lib
export COMMON_DIR=${CUR_DIR}/../common

export SENSOR_DEFS="\
-DSENSOR0_TYPE=XLM0_LVDS_1080P_60FPS_16BIT \
-DSENSOR1_TYPE=XLM0_LVDS_4K_30FPS_16BIT \
-DSENSOR2_TYPE=XLM0_LVDS_1080P_30FPS_16BIT \
-DSENSOR3_TYPE=XLM0_LVDS_1080P_30FPS_16BIT \
-DSENSOR4_TYPE=XLM0_LVDS_720x576_50FPS_16BIT \
-DSENSOR5_TYPE=XLM0_LVDS_1080P_30FPS_16BIT \
-DSENSOR6_TYPE=XLM0_LVDS_1080P_30FPS_16BIT \
-DSENSOR7_TYPE=XLM0_LVDS_1080P_30FPS_16BIT"

export INC_FLAGS="-I${MPP_INC} -I${COMMON_DIR}"
export CFLAGS="-O2 -Wall -D_GNU_SOURCE ${SENSOR_DEFS} ${INC_FLAGS}"

echo "========== check paths =========="
echo "CUR_DIR=${CUR_DIR}"
echo "MPP_INC=${MPP_INC}"
echo "MPP_LIB=${MPP_LIB}"
echo "COMMON_DIR=${COMMON_DIR}"

echo "========== compile gdc.c =========="
${CC} ${CFLAGS} -std=gnu99 -c ${CUR_DIR}/gdc.c -o build_gdc_pmf/gdc.o

echo "========== compile sample common =========="
${CC} ${CFLAGS} -std=gnu99 -c ${COMMON_DIR}/sample_comm_sys.c  -o build_gdc_pmf/sample_comm_sys.o
${CC} ${CFLAGS} -std=gnu99 -c ${COMMON_DIR}/sample_comm_vi.c   -o build_gdc_pmf/sample_comm_vi.o
${CC} ${CFLAGS} -std=gnu99 -c ${COMMON_DIR}/sample_comm_isp.c  -o build_gdc_pmf/sample_comm_isp.o
${CC} ${CFLAGS} -std=gnu99 -c ${COMMON_DIR}/sample_comm_vpss.c -o build_gdc_pmf/sample_comm_vpss.o
${CC} ${CFLAGS} -std=gnu99 -c ${COMMON_DIR}/sample_comm_vo.c   -o build_gdc_pmf/sample_comm_vo.o

echo "========== prepare hisi libs =========="
HISI_A_LIBS=$(find ${MPP_LIB} -maxdepth 1 -name '*.a' | tr '\n' ' ')

HISI_SO_LIBS=""
for f in ${MPP_LIB}/*.so; do
    if [ -f "$f" ]; then
        HISI_SO_LIBS="${HISI_SO_LIBS} -l:$(basename "$f")"
    fi
done

echo "========== link =========="
${CC} \
  build_gdc_pmf/gdc.o \
  build_gdc_pmf/sample_comm_sys.o \
  build_gdc_pmf/sample_comm_vi.o \
  build_gdc_pmf/sample_comm_isp.o \
  build_gdc_pmf/sample_comm_vpss.o \
  build_gdc_pmf/sample_comm_vo.o \
  -o gdc_pmf_ive_fusion \
  -L${MPP_LIB} \
  -Wl,-rpath-link=${MPP_LIB} \
  -Wl,--no-as-needed \
  -Wl,--start-group \
  ${HISI_A_LIBS} \
  ${HISI_SO_LIBS} \
  -Wl,--end-group \
  -lstdc++ \
  -lpthread \
  -ldl \
  -lrt \
  -lm

echo "========== check output =========="
file gdc_pmf_ive_fusion
${READELF} -d gdc_pmf_ive_fusion | grep NEEDED || true

echo "========== build success =========="
