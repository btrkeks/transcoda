## musicxml2hum GNU makefile
##
## Programmer:    Craig Stuart Sapp <craig@ccrma.stanford.edu>
## Creation Date: Sat Aug  6 10:57:54 CEST 2016
## Last Modified: Tue Jun 20 11:54:25 CEST 2017
## Filename:      musicxml2hum/Makefile
##
## Description: This Makefile compiles the musicxml2hum program.
##
## To run this makefile, type (without quotes) "make" (or
## "gmake library" on FreeBSD computers).
##

# targets which don't actually refer to files:
.PHONY: all debug release clean install download update test tests update-goldens bindir pgo pgo-generate pgo-use

.SUFFIXES:

# Directories
SRCDIR    = src
INCDIR    = include
BINDIR    = bin
BUILDDIR  = build

# Compiler
CXX       = g++
TARGET    = musicxml2hum

# Source files
SRCS      = $(SRCDIR)/humlib.cpp $(SRCDIR)/musicxml2hum.cpp $(SRCDIR)/pugixml.cpp
OBJS_BASE = $(notdir $(SRCS:.cpp=.o))

# Common flags
CXXFLAGS_COMMON = -Wall -Wextra -std=c++17 -I$(INCDIR)
DEPFLAGS        = -MMD -MP

# Debug configuration
DEBUG_DIR       = $(BUILDDIR)/debug
DEBUG_OBJS      = $(addprefix $(DEBUG_DIR)/,$(OBJS_BASE))
DEBUG_DEPS      = $(DEBUG_OBJS:.o=.d)
DEBUG_TARGET    = $(DEBUG_DIR)/$(TARGET)
DEBUG_CXXFLAGS  = $(CXXFLAGS_COMMON) -O0 -g3 -DDEBUG \
                  -fsanitize=address,undefined -fno-omit-frame-pointer
DEBUG_LDFLAGS   = -fsanitize=address,undefined

# Release configuration
RELEASE_DIR     = $(BUILDDIR)/release
RELEASE_OBJS    = $(addprefix $(RELEASE_DIR)/,$(OBJS_BASE))
RELEASE_DEPS    = $(RELEASE_OBJS:.o=.d)
RELEASE_TARGET  = $(RELEASE_DIR)/$(TARGET)
RELEASE_CXXFLAGS = $(CXXFLAGS_COMMON) -O3 -DNDEBUG -flto \
                   -march=native -fno-rtti -funroll-loops \
                   -fdata-sections -ffunction-sections
RELEASE_LDFLAGS  = -flto -Wl,--gc-sections

# PGO configuration
PGO_DIR         = $(BUILDDIR)/pgo
PGO_PROFILE_DIR = $(PGO_DIR)/profiles
PGO_OBJS        = $(addprefix $(PGO_DIR)/,$(OBJS_BASE))
PGO_DEPS        = $(PGO_OBJS:.o=.d)
PGO_TARGET      = $(PGO_DIR)/$(TARGET)
PGO_GEN_FLAGS   = $(RELEASE_CXXFLAGS) -fprofile-generate=$(PGO_PROFILE_DIR)
PGO_USE_FLAGS   = $(RELEASE_CXXFLAGS) -fprofile-use=$(PGO_PROFILE_DIR) -fprofile-correction

# Default target: build debug
debug: $(DEBUG_TARGET)

# Build both configurations
all: debug release

# Release target
release: $(RELEASE_TARGET) bindir
	@ln -sf ../$(RELEASE_TARGET) $(BINDIR)/$(TARGET)
	@echo Executable created in $(RELEASE_TARGET)
	@echo Symlink created at $(BINDIR)/$(TARGET)
	@echo Type "[32mmake install[0m" to copy to /usr/local/bin.

# Debug build rules
$(DEBUG_DIR):
	@mkdir -p $(DEBUG_DIR)

$(DEBUG_DIR)/%.o: $(SRCDIR)/%.cpp | $(DEBUG_DIR)
	@echo [CC] $<
	@$(CXX) $(DEBUG_CXXFLAGS) $(DEPFLAGS) -c $< -o $@

$(DEBUG_TARGET): $(DEBUG_OBJS)
	@echo [LD] $@
	@$(CXX) $(DEBUG_CXXFLAGS) $(DEBUG_LDFLAGS) $^ -o $@
	@echo Debug executable created in $@

# Release build rules
$(RELEASE_DIR):
	@mkdir -p $(RELEASE_DIR)

$(RELEASE_DIR)/%.o: $(SRCDIR)/%.cpp | $(RELEASE_DIR)
	@echo [CC] $<
	@$(CXX) $(RELEASE_CXXFLAGS) $(DEPFLAGS) -c $< -o $@

$(RELEASE_TARGET): $(RELEASE_OBJS)
	@echo [LD] $@
	@$(CXX) $(RELEASE_CXXFLAGS) $(RELEASE_LDFLAGS) $^ -o $@
	@strip $@
	@echo Release executable created in $@

# PGO build rules
$(PGO_DIR):
	@mkdir -p $(PGO_DIR)
	@mkdir -p $(PGO_PROFILE_DIR)

# Step 1: Build instrumented binary for profiling
pgo-generate: $(PGO_DIR) bindir
	@echo [PGO] Building instrumented binary...
	@mkdir -p $(RELEASE_DIR)
	@$(CXX) $(PGO_GEN_FLAGS) $(DEPFLAGS) $(SRCS) -o $(RELEASE_TARGET) $(RELEASE_LDFLAGS) -fprofile-generate=$(PGO_PROFILE_DIR)
	@ln -sf ../$(RELEASE_TARGET) $(BINDIR)/$(TARGET)
	@echo Instrumented binary created at $(RELEASE_TARGET)
	@echo Run: bin/$(TARGET) on representative input files
	@echo Then run: make pgo-use

# Step 2: Build optimized binary using profile data
pgo-use: $(PGO_DIR)
	@echo [PGO] Building optimized binary with profile data...
	@$(CXX) $(PGO_USE_FLAGS) $(DEPFLAGS) $(SRCS) -o $(RELEASE_TARGET) $(RELEASE_LDFLAGS) -fprofile-use=$(PGO_PROFILE_DIR) -fprofile-correction
	@strip $(RELEASE_TARGET)
	@ln -sf ../$(RELEASE_TARGET) $(BINDIR)/$(TARGET)
	@echo PGO-optimized executable created at $(RELEASE_TARGET)

# Convenience target: full PGO build using test files
pgo: pgo-generate
	@echo [PGO] Running profiling workload...
	@for f in tests/*.xml; do $(RELEASE_TARGET) "$$f" > /dev/null 2>&1 || true; done
	@$(MAKE) pgo-use
	@echo [PGO] Build complete!

# Include dependency files
-include $(DEBUG_DEPS)
-include $(RELEASE_DEPS)
-include $(PGO_DEPS)

# Backward compatibility: bin directory
bindir:
	@mkdir -p $(BINDIR)

# Download/update source files
download: update
update:
	@echo Downloading source code...
	@(cd src; ./.download);
	@(cd include; ./.download);

# Install release binary
install: release
	sudo cp $(RELEASE_TARGET) /usr/local/bin/

# Run tests with debug binary
test: debug
	python3 -m pytest -q

tests: test

update-goldens: debug
	python3 tests/update_goldens.py --binary $(DEBUG_TARGET)

# Clean build artifacts
clean:
	rm -rf $(BUILDDIR)
	rm -f $(BINDIR)/$(TARGET)
