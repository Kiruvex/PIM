# PIM —— 构建与自检（许可证：AGPL-3.0-only，见根 LICENSE）
CC ?= cc
CFLAGS ?= -O2
CTW = scripts/ctw

.PHONY: all ctw smoke clean

all: ctw

ctw:
	$(CC) $(CFLAGS) -o $(CTW) scripts/ctw.c -lm

smoke: ctw
	@echo "--- CTW 自检：alice 前 1MB 深度 16 ---"
	$(CTW) data/alice.txt 16

clean:
	rm -f $(CTW)
