# Hisilicon Hi35xx sample Makefile

include ../Makefile.param
# 直接使用物理文件的绝对路径，让编译器闭嘴！
MPI_LIBS += /home/xia/Hi3559AV100_SDK/Hi3559AV100_SDK_V2.0.3.1/mpp_arm/out/linux/multi-core/lib/libive.a

SRCS := $(wildcard *.c)
TARGET := $(SRCS:%.c=%)

# compile linux or HuaweiLite
include $(PWD)/../$(OSTYPE).mak
