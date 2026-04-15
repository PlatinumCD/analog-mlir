SUBDIRS := $(filter-out lib,$(patsubst %/,%,$(dir $(wildcard */Makefile))))

.PHONY: all build run debug clean

all: build

build: $(addprefix build-,$(SUBDIRS))

run: $(addprefix run-,$(SUBDIRS))

debug: $(addprefix debug-,$(SUBDIRS))

clean: $(addprefix clean-,$(SUBDIRS))

build-%:
	@$(MAKE) -C $* build

run-%:
	@$(MAKE) -C $* run

debug-%:
	@$(MAKE) -C $* debug

clean-%:
	@$(MAKE) -C $* clean
