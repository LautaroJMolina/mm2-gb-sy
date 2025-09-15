GPU_CC := icpx

#DPCT2001:53: You can link with more libraries by adding them here.
LIB := 

GPU_FLAGS := -fsycl $(CFLAGS)

ifeq ($(shell which $(GPU_CC)),)
    $(error ERROR - $(GPU_CC) compiler not found)
endif

ROOT_DIR     := $(shell dirname $(shell which $(GPU_CC)))
INCLUDE_DEBUG:= ../
INCLUDE_SYCL := $(ROOT_DIR)/../include
INCLUDE_CL   := $(ROOT_DIR)/../include/sycl

TARGET_0_SRC_0 = ./gpu/planalyze.dp.cpp
TARGET_0_OBJ_0 = ./gpu/planalyze.dp.o
TARGET_0_FLAG_0 = -I $(INCLUDE_SYCL) -I $(INCLUDE_CL) ${FLAGS}

TARGET_0_SRC_1 = ./gpu/plchain.dp.cpp
TARGET_0_OBJ_1 = ./gpu/plchain.dp.o
TARGET_0_FLAG_1 = -I $(INCLUDE_SYCL) -I $(INCLUDE_CL) ${FLAGS}

TARGET_0_SRC_2 = ./gpu/plmem.dp.cpp
TARGET_0_OBJ_2 = ./gpu/plmem.dp.o
TARGET_0_FLAG_2 = -I $(INCLUDE_SYCL) -I $(INCLUDE_CL) ${FLAGS}

TARGET_0_SRC_3 = ./gpu/plrange.dp.cpp
TARGET_0_OBJ_3 = ./gpu/plrange.dp.o
TARGET_0_FLAG_3 = -I $(INCLUDE_SYCL) -I $(INCLUDE_CL) ${FLAGS}

TARGET_0_SRC_4 = ./gpu/plscore.dp.cpp
TARGET_0_OBJ_4 = ./gpu/plscore.dp.o
TARGET_0_FLAG_4 = -I $(INCLUDE_SYCL) -I $(INCLUDE_CL) ${FLAGS}

.PHONY:all clean
OBJS_GPU :=  ${TARGET_0_OBJ_0} ${TARGET_0_OBJ_1} ${TARGET_0_OBJ_2} ${TARGET_0_OBJ_3} ${TARGET_0_OBJ_4}

gpu: $(OBJS_GPU)

$(TARGET_0_OBJ_0):$(TARGET_0_SRC_0)
	$(GPU_CC) ${GPU_FLAGS} -c ${TARGET_0_SRC_0} -o ${TARGET_0_OBJ_0} $(TARGET_0_FLAG_4)

$(TARGET_0_OBJ_1):$(TARGET_0_SRC_1)
	$(GPU_CC) ${GPU_FLAGS} -c ${TARGET_0_SRC_1} -o ${TARGET_0_OBJ_1} $(TARGET_0_FLAG_4)

$(TARGET_0_OBJ_2):$(TARGET_0_SRC_2)
	$(GPU_CC) ${GPU_FLAGS} -c ${TARGET_0_SRC_2} -o ${TARGET_0_OBJ_2} $(TARGET_0_FLAG_4)

$(TARGET_0_OBJ_3):$(TARGET_0_SRC_3)
	$(GPU_CC) ${GPU_FLAGS} -c ${TARGET_0_SRC_3} -o ${TARGET_0_OBJ_3} $(TARGET_0_FLAG_4)

$(TARGET_0_OBJ_4):$(TARGET_0_SRC_4)
	$(GPU_CC) ${GPU_FLAGS} -c ${TARGET_0_SRC_4} -o ${TARGET_0_OBJ_4} $(TARGET_0_FLAG_4)

gpu_clean:
	rm -f  ${OBJS_GPU}
