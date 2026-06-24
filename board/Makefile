.RECIPEPREFIX := >

TARGET := mm5_static_fusion
SRCS := mm5_static_fusion.c
OUT_DIR := /home/xia/nfs/mm5_board_test

CC := aarch64-himix100-linux-gcc
CFLAGS := -O2 -Wall
LDLIBS := -lrt

all: $(TARGET)

$(TARGET): $(SRCS)
>$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

install: $(TARGET)
>mkdir -p $(OUT_DIR)
>cp $(TARGET) $(OUT_DIR)/
>chmod +x $(OUT_DIR)/$(TARGET)
>@echo "installed to $(OUT_DIR)/$(TARGET)"

clean:
>rm -f $(TARGET) *.o

.PHONY: all install clean
