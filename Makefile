# Builds the external engines this project drives.
#
#   make linux     whisper.cpp for CPU, CUDA and Vulkan, built natively
#   make windows   whisper.cpp for CPU, CUDA and Vulkan, for binaries/windows
#   make models    download the ggml models
#   make check     run every installed engine and report the backends it sees
#
# Engines are subprocesses, never linked into the Go binary, so each backend is
# just another executable in binaries/<os>/ and the app picks one at runtime.

SHELL := /bin/bash
.DEFAULT_GOAL := help

# Pin the engine so source builds and downloaded Windows binaries match.
WHISPER_TAG   ?= b4938
WHISPER_REPO  ?= https://github.com/ggml-org/whisper.cpp
WHISPER_DL    := $(WHISPER_REPO)/releases/download/$(WHISPER_TAG)

# Ada (4060) is 89. The default list covers Turing through Hopper so a shipped
# build is not tied to one card; override for a faster local build.
CUDA_ARCHS    ?= 75;80;86;89;90
JOBS          ?= $(shell nproc 2>/dev/null || echo 4)

BUILD         := .build
SRC           := $(BUILD)/whisper.cpp
LINUX_BIN     := binaries/linux
WIN_BIN       := binaries/windows
MODELS        := models

# BUILD_SHARED_LIBS=OFF matters: with shared ggml the installed binary keeps an
# rpath into the build tree and breaks the moment it is cleaned.
CMAKE_COMMON  := -DCMAKE_BUILD_TYPE=Release \
                 -DBUILD_SHARED_LIBS=OFF \
                 -DWHISPER_BUILD_TESTS=OFF \
                 -DWHISPER_BUILD_EXAMPLES=ON \
                 -DWHISPER_BUILD_SERVER=OFF

# GGML_NATIVE=OFF keeps -march=native out of a shipped binary, which would fault
# on any CPU older than the build machine.
PORTABLE      := -DGGML_NATIVE=OFF

MODEL_BASE    := https://huggingface.co/ggerganov/whisper.cpp/resolve/main
VAD_BASE      := https://huggingface.co/ggml-org/whisper-vad/resolve/main
ASR_MODEL     ?= ggml-large-v3-turbo-q5_0.bin
VAD_MODEL     ?= ggml-silero-v5.1.2.bin

REL_LINUX := release_linux
REL_WIN   := release_windows

# Models are ~550 MB. Set INCLUDE_MODELS=0 for a release the user downloads
# models into on first run.
INCLUDE_MODELS ?= 1

.PHONY: help patch linux linux-cpu linux-cuda linux-vulkan \
        windows windows-cpu windows-cuda windows-vulkan windows-ffmpeg \
        release-linux release-windows \
        models model-hf model-turkish model-buzz check clean distclean src

help:
	@echo "Targets:"
	@echo "  make linux           CPU + CUDA + Vulkan into $(LINUX_BIN)/"
	@echo "  make linux-cpu       CPU only"
	@echo "  make linux-cuda      needs the CUDA toolkit (nvcc)"
	@echo "  make linux-vulkan    needs vulkan-headers, vulkan-icd-loader, shaderc"
	@echo
	@echo "  make windows         CPU + Vulkan into $(WIN_BIN)/  (CUDA is separate: 640 MB)"
	@echo "  make windows-cuda    downloads the upstream cuBLAS build, ~640 MB"
	@echo "  make windows-ffmpeg  re-vendor the LGPL ffmpeg/ffprobe build"
	@echo
	@echo "  make release-linux   assemble $(REL_LINUX)/   (part of 'make linux')"
	@echo "  make release-windows assemble $(REL_WIN)/  (part of 'make windows')"
	@echo
	@echo "  make models          download $(ASR_MODEL) and $(VAD_MODEL)"
	@echo "  make model-turkish   TurkMedSTT Turkish fine-tune (needs python torch)"
	@echo "  make model-buzz      BuzzASR Turkish fine-tune"
	@echo "  make model-hf HF_REPO=owner/name HF_NAME=short   any whisper fine-tune"
	@echo "  make check           run each installed engine, print detected devices"
	@echo "  make clean           remove $(BUILD)"
	@echo
	@echo "Variables: WHISPER_TAG=$(WHISPER_TAG)  CUDA_ARCHS=$(CUDA_ARCHS)  JOBS=$(JOBS)"

# --- source ----------------------------------------------------------------

src: $(SRC)/CMakeLists.txt patch

$(SRC)/CMakeLists.txt:
	@mkdir -p $(BUILD)
	git clone --depth 1 --branch $(WHISPER_TAG) $(WHISPER_REPO) $(SRC)

# whisper-cli's JSON writer reports token times on the VAD-compressed timeline
# while segment times are mapped back to the real recording, so with --vad the
# two disagree by however much silence was cut -- which is what put review clips
# minutes away from the word. The library already exposes mapped accessors; the
# patch makes the writer use them. Idempotent.
patch: $(SRC)/CMakeLists.txt
	@python3 patches/vad-token-times.py $(SRC)/examples/cli/cli.cpp

# --- linux -----------------------------------------------------------------

linux: linux-cpu linux-cuda linux-vulkan release-linux
	@echo
	@$(MAKE) --no-print-directory check

linux-cpu: src
	cmake -B $(BUILD)/linux-cpu -S $(SRC) $(CMAKE_COMMON) $(PORTABLE)
	cmake --build $(BUILD)/linux-cpu -j$(JOBS) --target whisper-cli
	@mkdir -p $(LINUX_BIN)
	install -m755 $(BUILD)/linux-cpu/bin/whisper-cli $(LINUX_BIN)/whisper-cli
	@echo "installed $(LINUX_BIN)/whisper-cli"

linux-cuda: src
	@command -v nvcc >/dev/null || { echo "nvcc not found - skipping CUDA build"; exit 0; }; \
	cmake -B $(BUILD)/linux-cuda -S $(SRC) $(CMAKE_COMMON) $(PORTABLE) \
	      -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$(CUDA_ARCHS)" && \
	cmake --build $(BUILD)/linux-cuda -j$(JOBS) --target whisper-cli && \
	mkdir -p $(LINUX_BIN) && \
	install -m755 $(BUILD)/linux-cuda/bin/whisper-cli $(LINUX_BIN)/whisper-cli-cuda && \
	echo "installed $(LINUX_BIN)/whisper-cli-cuda"

linux-vulkan: src
	@test -f /usr/include/vulkan/vulkan.h || { \
	  echo "vulkan headers not found - install vulkan-headers vulkan-icd-loader shaderc"; exit 1; }
	cmake -B $(BUILD)/linux-vulkan -S $(SRC) $(CMAKE_COMMON) $(PORTABLE) -DGGML_VULKAN=ON
	cmake --build $(BUILD)/linux-vulkan -j$(JOBS) --target whisper-cli
	@mkdir -p $(LINUX_BIN)
	install -m755 $(BUILD)/linux-vulkan/bin/whisper-cli $(LINUX_BIN)/whisper-cli-vulkan
	@echo "installed $(LINUX_BIN)/whisper-cli-vulkan"

# --- windows ---------------------------------------------------------------
#
# CPU and CUDA come from upstream's own MSVC releases rather than being
# cross-built: they are official, tested, and CUDA cannot be cross-compiled
# from Linux at all, since nvcc for Windows requires MSVC.

windows: windows-cpu windows-ffmpeg windows-vulkan release-windows
	@echo
	@echo "note: Windows CUDA is a separate 640 MB download - 'make windows-cuda',"
	@echo "      then re-run 'make release-windows' to fold it into $(REL_WIN)/"

windows-cpu:
	@mkdir -p $(WIN_BIN) $(BUILD)
	curl -fL --retry 3 -o $(BUILD)/whisper-win-cpu.zip $(WHISPER_DL)/whisper-bin-x64.zip
	rm -rf $(BUILD)/win-cpu && mkdir -p $(BUILD)/win-cpu
	unzip -q -o $(BUILD)/whisper-win-cpu.zip -d $(BUILD)/win-cpu
	@set -e; f=$$(find $(BUILD)/win-cpu -name 'whisper-cli.exe' | head -1); \
	  test -n "$$f" || { echo "whisper-cli.exe not in the archive"; exit 1; }; \
	  cp "$$f" $(WIN_BIN)/whisper-cli.exe; \
	  find $$(dirname "$$f") -maxdepth 1 -name '*.dll' -exec cp {} $(WIN_BIN)/ \; ; \
	  echo "installed $(WIN_BIN)/whisper-cli.exe"

windows-cuda:
	@mkdir -p $(WIN_BIN) $(BUILD)
	curl -fL --retry 3 -o $(BUILD)/whisper-win-cuda.zip $(WHISPER_DL)/whisper-cublas-12.4.0-bin-x64.zip
	rm -rf $(BUILD)/win-cuda && mkdir -p $(BUILD)/win-cuda
	unzip -q -o $(BUILD)/whisper-win-cuda.zip -d $(BUILD)/win-cuda
	@set -e; f=$$(find $(BUILD)/win-cuda -name 'whisper-cli.exe' | head -1); \
	  test -n "$$f" || { echo "whisper-cli.exe not in the archive"; exit 1; }; \
	  mkdir -p $(WIN_BIN)/cuda; \
	  cp "$$f" $(WIN_BIN)/cuda/whisper-cli-cuda.exe; \
	  find $$(dirname "$$f") -maxdepth 1 -name '*.dll' -exec cp {} $(WIN_BIN)/cuda/ \; ; \
	  echo "installed $(WIN_BIN)/cuda/whisper-cli-cuda.exe with its CUDA DLLs"

# Upstream publishes no Vulkan build for Windows, so this one is ours. It needs
# a mingw-w64 toolchain plus a vulkan-1 import library; when that is missing the
# supported route is a windows-latest CI job with the LunarG SDK, which is what
# the message below describes.
windows-vulkan: src
	@command -v x86_64-w64-mingw32-g++ >/dev/null || { \
	  echo "mingw-w64 not found - install mingw-w64-gcc"; exit 1; }
	@mkdir -p $(BUILD)
	@printf '%s\n' \
	  'set(CMAKE_SYSTEM_NAME Windows)' \
	  'set(CMAKE_SYSTEM_PROCESSOR x86_64)' \
	  'set(CMAKE_C_COMPILER x86_64-w64-mingw32-gcc)' \
	  'set(CMAKE_CXX_COMPILER x86_64-w64-mingw32-g++)' \
	  'set(CMAKE_RC_COMPILER x86_64-w64-mingw32-windres)' \
	  'set(CMAKE_FIND_ROOT_PATH /usr/x86_64-w64-mingw32)' \
	  'set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)' \
	  'set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)' \
	  'set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)' \
	  > $(BUILD)/mingw-toolchain.cmake
	@echo "cross-building whisper-cli-vulkan.exe with mingw-w64"
	@cmake -B $(BUILD)/win-vulkan -S $(SRC) $(CMAKE_COMMON) $(PORTABLE) \
	      -DCMAKE_TOOLCHAIN_FILE=$(CURDIR)/$(BUILD)/mingw-toolchain.cmake \
	      -DGGML_VULKAN=ON \
	      -DVulkan_INCLUDE_DIR=/usr/include \
	      -DVulkan_GLSLC_EXECUTABLE=$$(command -v glslc) 2>&1 | tail -20; \
	  if [ $${PIPESTATUS[0]} -ne 0 ]; then \
	    echo; \
	    echo "Cross-building Vulkan for Windows failed - this is expected without a"; \
	    echo "mingw vulkan-1 import library. The supported route is a CI job:"; \
	    echo "  runs-on: windows-latest"; \
	    echo "  - uses: jakoch/install-vulkan-sdk-action"; \
	    echo "  - cmake -B build -DGGML_VULKAN=ON -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF"; \
	    echo "  - cmake --build build --config Release --target whisper-cli"; \
	    exit 1; \
	  fi
	cmake --build $(BUILD)/win-vulkan -j$(JOBS) --target whisper-cli
	@mkdir -p $(WIN_BIN)
	install -m755 $(BUILD)/win-vulkan/bin/whisper-cli.exe $(WIN_BIN)/whisper-cli-vulkan.exe
	@echo "installed $(WIN_BIN)/whisper-cli-vulkan.exe"

FFMPEG_BUILD ?= ffmpeg-n9.0.1-27-g9b0578816c-win64-lgpl-9.0
FFMPEG_TAG   ?= autobuild-2026-09-09-14-51

windows-ffmpeg:
	@mkdir -p $(WIN_BIN) $(BUILD)
	curl -fL --retry 3 -o $(BUILD)/ffmpeg-win.zip \
	  https://github.com/BtbN/FFmpeg-Builds/releases/download/$(FFMPEG_TAG)/$(FFMPEG_BUILD).zip
	unzip -j -o $(BUILD)/ffmpeg-win.zip '*/bin/ffmpeg.exe' '*/bin/ffprobe.exe' -d $(WIN_BIN)
	unzip -j -o $(BUILD)/ffmpeg-win.zip '*/LICENSE.txt' -d $(BUILD)
	cp $(BUILD)/LICENSE.txt $(WIN_BIN)/FFMPEG-LICENSE.txt
	@echo "installed $(WIN_BIN)/ffmpeg.exe and ffprobe.exe (LGPL)"

# --- releases --------------------------------------------------------------
#
# A release folder is laid out the way the app expects to find things: the
# engines under binaries/<os>/ beside the executable, and models/ alongside.
# Copy the folder to a user's machine and it runs with no install step.

release-linux: models
	@rm -rf $(REL_LINUX) && mkdir -p $(REL_LINUX)/binaries/linux $(REL_LINUX)/models \
	        $(REL_LINUX)/audios $(REL_LINUX)/videos
	CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -trimpath -ldflags "-s -w" \
	  -o $(REL_LINUX)/transcript ./cmd/transcript
	@for f in whisper-cli whisper-cli-cuda whisper-cli-vulkan; do \
	  [ -x $(LINUX_BIN)/$$f ] && cp $(LINUX_BIN)/$$f $(REL_LINUX)/binaries/linux/ || true; \
	done
	@if [ "$(INCLUDE_MODELS)" = "1" ]; then cp $(MODELS)/*.bin $(REL_LINUX)/models/ 2>/dev/null || true; fi
	@printf '%s\n' '#!/bin/sh' 'cd "$$(dirname "$$0")" && exec ./transcript "$$@"' \
	  > $(REL_LINUX)/run.sh && chmod +x $(REL_LINUX)/run.sh
	@$(MAKE) --no-print-directory release-readme DEST=$(REL_LINUX) OS=Linux RUN=./run.sh
	@echo; echo "$(REL_LINUX)/ ready  ($$(du -sh $(REL_LINUX) | cut -f1))"; ls $(REL_LINUX)

release-windows:
	@rm -rf $(REL_WIN) && mkdir -p $(REL_WIN)/binaries/windows $(REL_WIN)/models \
	        $(REL_WIN)/audios $(REL_WIN)/videos
	CGO_ENABLED=0 GOOS=windows GOARCH=amd64 go build -trimpath -ldflags "-s -w" \
	  -o $(REL_WIN)/transcript.exe ./cmd/transcript
	@cp -r $(WIN_BIN)/. $(REL_WIN)/binaries/windows/ 2>/dev/null || true
	@if [ "$(INCLUDE_MODELS)" = "1" ]; then cp $(MODELS)/*.bin $(REL_WIN)/models/ 2>/dev/null || true; fi
	@printf '%s\r\n' '@echo off' 'cd /d "%~dp0"' 'start "" transcript.exe %*' \
	  > $(REL_WIN)/run.bat
	@$(MAKE) --no-print-directory release-readme DEST=$(REL_WIN) OS=Windows RUN=run.bat
	@echo; echo "$(REL_WIN)/ ready  ($$(du -sh $(REL_WIN) | cut -f1))"; ls $(REL_WIN)

.PHONY: release-readme
release-readme:
	@printf '%s\n' \
	  'Transcript Pipeline ($(OS))' \
	  '' \
	  'Run $(RUN) - it opens the interface in your browser.' \
	  'Everything runs on this machine; nothing is uploaded.' \
	  '' \
	  'Folders:' \
	  '  audios/            put audio here' \
	  '  videos/            put video here' \
	  '  models/            speech models' \
	  '  work/wav/          converted audio (safe to delete)' \
	  '  work/transcripts/  finished transcripts' \
	  '' \
	  'You can also add files from anywhere via Browse or drag and drop.' \
	  '' \
	  'GPU: the Transcribe page has a GPU switch. If your machine has more than' \
	  'one graphics chip, pick the discrete card - the built-in one shares memory' \
	  'with the processor and is often no faster than the processor alone.' \
	  '' \
	  'Licences: ffmpeg is LGPL (see binaries/windows/FFMPEG-LICENSE.txt),' \
	  'whisper.cpp is MIT.' \
	  > $(DEST)/README.txt

# --- models ----------------------------------------------------------------

models: $(MODELS)/$(ASR_MODEL) $(MODELS)/$(VAD_MODEL)

$(MODELS)/$(ASR_MODEL):
	@mkdir -p $(MODELS)
	curl -fL --retry 3 -o $@.part $(MODEL_BASE)/$(ASR_MODEL) && mv $@.part $@

$(MODELS)/$(VAD_MODEL):
	@mkdir -p $(MODELS)
	curl -fL --retry 3 -o $@.part $(VAD_BASE)/$(VAD_MODEL) && mv $@.part $@

# --- the Turkish fine-tune --------------------------------------------------
#
# TurkMedSTT publish whisper large-v3 adapted to Turkish (Apache-2.0), but only
# as transformers weights, so it has to be converted here. On their 1,060-clip
# benchmark it scores 0.078 WER against 0.201 for stock turbo -- at roughly four
# times the runtime, and trained mostly on read speech, so measure it on your
# own recordings before making it the default.

# Any whisper fine-tune published as transformers weights can be brought in:
#   make model-hf HF_REPO=owner/name HF_NAME=short-name
HF_REPO ?= turkmedstt/whisper-large-v3-turkish-general
HF_NAME ?= large-v3-turkish-general
HF_OUT  := $(MODELS)/ggml-$(HF_NAME)-q5_0.bin

# A throwaway virtualenv: torch and transformers are needed only to read the
# safetensors once, and have no place in what gets shipped.
VENV := $(BUILD)/convert-venv
VPY  := $(VENV)/bin/python
# The converter reads mel filters and the tokenizer from the openai/whisper
# source tree, so that has to be present too.
OAI  := $(BUILD)/openai-whisper

$(VPY):
	python3 -m venv $(VENV)
	$(VENV)/bin/pip -q install --upgrade pip
	$(VENV)/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
	$(VENV)/bin/pip install transformers safetensors huggingface_hub numpy

$(OAI)/whisper/assets/mel_filters.npz:
	git clone --depth 1 https://github.com/openai/whisper $(OAI)

model-hf: src $(VPY) $(OAI)/whisper/assets/mel_filters.npz
	@mkdir -p $(BUILD)/hf $(MODELS)
	@echo "fetching $(HF_REPO) (about 3 GB)"
	$(VPY) -c "from huggingface_hub import snapshot_download; snapshot_download('$(HF_REPO)', local_dir='$(BUILD)/hf/$(HF_NAME)')"
	@echo "converting to ggml"
	cd $(BUILD)/hf/$(HF_NAME) && $(CURDIR)/$(VPY) $(CURDIR)/$(SRC)/models/convert-h5-to-ggml.py . $(CURDIR)/$(OAI) .
	@echo "quantising to q5_0"
	@test -x $(BUILD)/linux-cpu/bin/whisper-quantize || cmake --build $(BUILD)/linux-cpu -j$(JOBS) --target whisper-quantize
	$(BUILD)/linux-cpu/bin/whisper-quantize $(BUILD)/hf/$(HF_NAME)/ggml-model.bin $(HF_OUT) q5_0
	@rm -rf $(BUILD)/hf/$(HF_NAME)   # ~6 GB of intermediates, no longer needed
	@echo; echo "installed $(HF_OUT)"

model-turkish:
	@$(MAKE) --no-print-directory model-hf \
	  HF_REPO=turkmedstt/whisper-large-v3-turkish-general HF_NAME=large-v3-turkish-general

model-buzz:
	@$(MAKE) --no-print-directory model-hf \
	  HF_REPO=BuzzASR/turkish HF_NAME=buzzasr-turkish

# --- checks ----------------------------------------------------------------

# Runs each engine over a fragment of silence and prints the devices it found,
# which is the same question the GPU section of the UI asks.
check:
	@model=$$(ls $(MODELS)/ggml-*.bin 2>/dev/null | grep -v -i -E 'silero|vad' | head -1); \
	if [ -z "$$model" ]; then echo "no model in $(MODELS)/ - run 'make models'"; exit 0; fi; \
	probe=internal/asr/probe.wav; \
	for bin in $(LINUX_BIN)/whisper-cli $(LINUX_BIN)/whisper-cli-cuda $(LINUX_BIN)/whisper-cli-vulkan; do \
	  [ -x "$$bin" ] || continue; \
	  printf '%-38s ' "$$bin"; \
	  out=$$("$$bin" -m "$$model" -f "$$probe" -l tr -t 1 -nt 2>&1); \
	  dev=$$(echo "$$out" | grep -oP 'device \d+: \K.*?(?= \(type)' | paste -sd', '); \
	  if echo "$$out" | grep -q 'no GPU found'; then echo "CPU only"; \
	  elif [ -n "$$dev" ]; then echo "$$dev"; \
	  else echo "could not determine"; fi; \
	done

clean:
	rm -rf $(BUILD)

distclean: clean
	rm -f $(LINUX_BIN)/whisper-cli $(LINUX_BIN)/whisper-cli-cuda $(LINUX_BIN)/whisper-cli-vulkan
