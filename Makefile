# Otrio Game Tree Explorer - Makefile

CC = gcc
CFLAGS = -O3 -march=native -Wall -Wextra
LDFLAGS = -lpthread

# Default target
all: otrio_explore extract_strategy

otrio_explore: otrio_explore.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

extract_strategy: extract_strategy.c
	$(CC) $(CFLAGS) -o $@ $<

# Debug build
debug: CFLAGS = -g -O0 -Wall -Wextra -DDEBUG
debug: otrio_explore extract_strategy

clean:
	rm -f otrio_explore otrio_explore.exe extract_strategy extract_strategy.exe

# Test with depth 4
test: otrio_explore
	./otrio_explore --depth 4 --parallel 4 -o test_c.json
	@echo "Checking output..."
	@python -c "import json; d=json.load(open('test_c.json')); print('Stats:', d['stats'])"
	@rm -f test_c.json

.PHONY: all clean test debug
