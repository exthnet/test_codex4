CC := gcc
CFLAGS := -std=gnu11 -O3 -Wall -Wextra -Wpedantic
LDFLAGS := -lm

BUILD_DIR := build
BIN := $(BUILD_DIR)/gemm_bf16
SRC := src/gemm_bf16.c

.PHONY: all clean

all: $(BIN)

$(BIN): $(SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)
