GPU_CC := acpp

LIB := 

GPU_FLAGS := $(CFLAGS)

ifeq ($(shell which $(GPU_CC)),)
    $(error ERROR - $(GPU_CC) compiler not found)
endif

ROOT_DIR     := $(shell dirname $(shell which $(GPU_CC)))

TARGET_0_SRC_0 = ./gpu/planalyze.cpp
TARGET_0_OBJ_0 = ./gpu/planalyze.o

TARGET_0_SRC_1 = ./gpu/plchain.cpp
TARGET_0_OBJ_1 = ./gpu/plchain.o

TARGET_0_SRC_2 = ./gpu/plmem.cpp
TARGET_0_OBJ_2 = ./gpu/plmem.o

TARGET_0_SRC_3 = ./gpu/plrange.cpp
TARGET_0_OBJ_3 = ./gpu/plrange.o

TARGET_0_SRC_4 = ./gpu/plscore.cpp
TARGET_0_OBJ_4 = ./gpu/plscore.o
TARGET_0_FLAG = ${FLAGS}

.PHONY:all clean
OBJS_GPU :=  ${TARGET_0_OBJ_0} ${TARGET_0_OBJ_1} ${TARGET_0_OBJ_2} ${TARGET_0_OBJ_3} ${TARGET_0_OBJ_4}

gpu: $(OBJS_GPU)

$(TARGET_0_OBJ_0):$(TARGET_0_SRC_0)
	$(GPU_CC) ${GPU_FLAGS} -c ${TARGET_0_SRC_0} -o ${TARGET_0_OBJ_0} $(TARGET_0_FLAG)

$(TARGET_0_OBJ_1):$(TARGET_0_SRC_1)
	$(GPU_CC) ${GPU_FLAGS} -c ${TARGET_0_SRC_1} -o ${TARGET_0_OBJ_1} $(TARGET_0_FLAG)

$(TARGET_0_OBJ_2):$(TARGET_0_SRC_2)
	$(GPU_CC) ${GPU_FLAGS} -c ${TARGET_0_SRC_2} -o ${TARGET_0_OBJ_2} $(TARGET_0_FLAG)

$(TARGET_0_OBJ_3):$(TARGET_0_SRC_3)
	$(GPU_CC) ${GPU_FLAGS} -c ${TARGET_0_SRC_3} -o ${TARGET_0_OBJ_3} $(TARGET_0_FLAG)

$(TARGET_0_OBJ_4):$(TARGET_0_SRC_4)
	$(GPU_CC) ${GPU_FLAGS} -c ${TARGET_0_SRC_4} -o ${TARGET_0_OBJ_4} $(TARGET_0_FLAG)

gpu_clean:
	rm -f  ${OBJS_GPU}
