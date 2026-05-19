SHADER  = shaders/octra_vanity.metal
SWIFT   = src/main.swift
BUILD   = build
METALLIB= $(BUILD)/default.metallib
AIR     = $(BUILD)/octra_vanity.air
BIN     = $(BUILD)/octra_vanity_metal

SDK     = macosx
METALFLAGS = -O3 -ffast-math
SWIFTFLAGS = -O -framework Metal -framework QuartzCore -framework Foundation -framework Security

.PHONY: all clean run

all: $(BIN) $(METALLIB)

$(BUILD):
	mkdir -p $(BUILD)

$(AIR): $(SHADER) | $(BUILD)
	xcrun -sdk $(SDK) metal $(METALFLAGS) -c $(SHADER) -o $(AIR)

$(METALLIB): $(AIR)
	xcrun -sdk $(SDK) metallib $(AIR) -o $(METALLIB)

$(BIN): $(SWIFT) | $(BUILD)
	swiftc $(SWIFTFLAGS) -o $(BIN) $(SWIFT)

run: all
	cd $(BUILD) && ./octra_vanity_metal

clean:
	rm -rf $(BUILD)
