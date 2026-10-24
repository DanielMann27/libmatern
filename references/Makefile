# references/Makefile -- the external K_nu supplier objects, built with the flags of
# the published comparison (-march=native -O2, no -std); nothing links libmatern.

# The first rule is the `$(OBJECTS): common/cpu_wm.h` line, not `all`.
.DEFAULT_GOAL := all

# No -std: the GNU dialect defaults to -ffp-contract=fast, while -std=c11 changes
# the arithmetic of these ports and hides M_PI (check-flags counts the FMAs).
CFLAGS   = -march=native -O2 -Wall
CXXFLAGS = -std=c++17 -march=native -O2 -Wall

INCLUDES = -Icommon

# KAUST CuBesselK carries "All rights reserved" and is not distributed; its rows are
# built only when the port exists (top level: make kaust).
# V=yes (also 1, on, true, y) prints the compile commands.
YES_WORDS = yes y 1 on true Yes Y On True YES ON TRUE
Q := $(if $(filter $(YES_WORDS),$(strip $(V))),,@)
KAUST_PORT := $(wildcard build/kaust/kaust_derived.h)
KAUST_SRC  := $(if $(KAUST_PORT),kaust_logbesselk/kaust_port.c)

# GSL: one built by `make externals` (gsl/<version>/), else the system GSL.
GSL_FLAGS   = bash ../benchmark/tools/gsl_flags.sh
GSL_PRESENT := $(shell $(GSL_FLAGS) --exists && echo yes)
ifeq ($(GSL_PRESENT),yes)
GSL_CFLAGS := $(shell $(GSL_FLAGS) --cflags)
endif

# C++ counts as present only if it compiles std::cyl_bessel_k (libc++ has no C++17
# special functions); tools/requirements.sh applies the same test.
CXX_PRESENT := $(shell printf '\043include <cmath>\nint main(){return (int)std::cyl_bessel_k(1.5,2.0);}\n' \
                 | $(CXX) -std=c++17 -x c++ - -o /dev/null >/dev/null 2>&1 && echo yes)

# The CPU Boost row is Boost 1.75.0 or absent.  The compile refuses a
# boost/version.hpp other than BOOST_CPU_VERSION (CPATH can put another Boost ahead of -I).
ifeq ($(CXX_PRESENT),yes)
BOOST_CPU_DIR := $(shell CXX="$(CXX)" BOOST_CPU_ROOT="$(BOOST_CPU_ROOT)" bash ../benchmark/tools/boost_detect.sh --select cpu)
endif
ifneq ($(BOOST_CPU_DIR),)
BOOST_CPU_VERSION := $(shell sed -n 's/^\#define BOOST_VERSION  *\([0-9]*\).*/\1/p' $(BOOST_CPU_DIR)/boost/version.hpp | head -1)
BOOST_CXXFLAGS := -DWMC_WITH_BOOST -DWMC_BOOST_EXPECT=$(BOOST_CPU_VERSION) -I$(BOOST_CPU_DIR)
endif

OBJECTS = build/cpu_bessels_jl.o \
          build/cpu_bessels_old.o \
          build/build_info_c.o
DEFINES =
LINK_LIBS =
ifeq ($(CXX_PRESENT),yes)
OBJECTS   += build/cpu_cxx.o
DEFINES   += -DWMC_WITH_CXX $(if $(BOOST_CPU_DIR),-DWMC_WITH_BOOST)
LINK_LIBS += -lstdc++
endif
ifneq ($(KAUST_SRC),)
OBJECTS += build/kaust_port.o
DEFINES += -DWMC_WITH_KAUST
endif
ifeq ($(GSL_PRESENT),yes)
OBJECTS   += build/cpu_gsl.o
DEFINES   += -DWMC_WITH_GSL
LINK_LIBS += $(shell $(GSL_FLAGS) --libs)
endif

$(OBJECTS): common/cpu_wm.h

# The drivers read build/defines.txt and build/link.txt, so detection happens here
# only; the explicit object list keeps a leftover object of another configuration out.
.PHONY: all clean check-flags FORCE
all: $(OBJECTS)
	@printf '%s\n' "$(strip $(DEFINES))" > build/defines.txt
	@printf '%s\n' "$(strip $(addprefix references/,$(OBJECTS)) $(LINK_LIBS))" > build/link.txt
	@echo "built $(words $(OBJECTS)) supplier objects in build/"
	@[ "$(CXX_PRESENT)" = yes ] || echo "  (cpu_cxx.o skipped: no C++ compiler $(CXX); no std:: and Boost rows)"
	@[ "$(CXX_PRESENT)" != yes ] || [ -n "$(BOOST_CPU_DIR)" ] || echo "  (Boost row skipped: no Boost 1.75.0 selected; std:: row built; make externals adds it)"
	@[ -z "$(BOOST_CPU_DIR)" ] || echo "  Boost row: $(BOOST_CPU_DIR) (BOOST_VERSION $(BOOST_CPU_VERSION))"
	@[ "$(GSL_PRESENT)" = yes ] || echo "  (cpu_gsl.o skipped: no GSL found; make externals builds one)"

build/cpu_bessels_jl.o: bessels_jl/port/cpu_bessels_jl.c
	@mkdir -p build
	$(Q)$(CC) $(CFLAGS) $(INCLUDES) -Ibessels_jl/port -c $< -o $@

build/cpu_bessels_old.o: bessels_jl/port/cpu_bessels_old.c
	@mkdir -p build
	$(Q)$(CC) $(CFLAGS) $(INCLUDES) -Ibessels_jl/port -c $< -o $@

# Compiled with the suppliers' flags, so the build record it reports is theirs.
build/build_info_c.o: common/build_info_c.c common/build_info.h
	@mkdir -p build
	$(Q)$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# The port is never distributed and no rule here makes it.
build/kaust_port.o: kaust_logbesselk/kaust_port.c build/kaust/kaust_derived.h
	@mkdir -p build
	$(Q)$(CC) $(CFLAGS) $(INCLUDES) -Ibuild/kaust -c $< -o $@

# Configuration stamps, rewritten only on change: another GSL or Boost rebuilds the object.
build/gsl.config: FORCE
	@mkdir -p build
	@echo "$(GSL_CFLAGS)" | cmp -s - $@ || echo "$(GSL_CFLAGS)" > $@

build/cpu_gsl.o: gsl/cpu_gsl.c build/gsl.config common/build_info.h
	@mkdir -p build
	$(Q)$(CC) $(CFLAGS) $(INCLUDES) $(GSL_CFLAGS) -c $< -o $@

build/cxx.config: FORCE
	@mkdir -p build
	@echo "$(CXX) $(BOOST_CXXFLAGS)" | cmp -s - $@ || echo "$(CXX) $(BOOST_CXXFLAGS)" > $@

# std::cyl_bessel_k, and Boost (headers, BSL-1.0) when one is selected.
build/cpu_cxx.o: cpp_stdlib/cpu_cxx.cpp build/cxx.config common/build_info.h
	@mkdir -p build
	$(Q)$(CXX) $(CXXFLAGS) $(BOOST_CXXFLAGS) $(INCLUDES) -c $< -o $@

# FMA instruction counts of every C supplier source, GNU dialect against -std=c11.
FLAG_SOURCES := $(KAUST_SRC) \
                bessels_jl/port/cpu_bessels_jl.c \
                bessels_jl/port/cpu_bessels_old.c

check-flags: $(if $(KAUST_SRC),build/kaust/kaust_derived.h)
	@mkdir -p build
	@echo "  fused multiply-adds, GNU dialect (as measured) vs -std=c11:"
	@tg=0; ti=0; \
	 for src in $(FLAG_SOURCES); do \
	     $(CC) $(CFLAGS) $(INCLUDES) -Ibessels_jl/port -Ibuild/kaust -S $$src -o build/flags_gnu.s; \
	     $(CC) $(CFLAGS) -std=c11 -D_GNU_SOURCE $(INCLUDES) -Ibessels_jl/port -Ibuild/kaust \
	         -S $$src -o build/flags_iso.s; \
	     gnu=$$(grep -cE 'vfn?m(add|sub)' build/flags_gnu.s); \
	     iso=$$(grep -cE 'vfn?m(add|sub)' build/flags_iso.s); \
	     printf '    %-34s %4s -> %-4s\n' "$$(basename $$src)" "$$gnu" "$$iso"; \
	     tg=$$((tg+gnu)); ti=$$((ti+iso)); \
	 done; \
	 echo "    $$(printf '%-34s %4s -> %-4s' TOTAL $$tg $$ti)"; \
	 if [ -z "$(KAUST_SRC)" ]; then \
	     echo "  (KAUST not ported; make kaust at the top level)"; \
	 fi; \
	 if [ "$$tg" = "$$ti" ]; then \
	     echo "  the dialect does not change the arithmetic"; \
	 else \
	     echo "  the dialect changes the arithmetic; CFLAGS carries no -std"; \
	 fi

# Checks the KAUST upstream files against the sha256 table of
# kaust_logbesselk/PROVENANCE.md; neither downloads nor ports.
UPSTREAM_KAUST = kaust_logbesselk/upstream/logbesselk.cu \
                 kaust_logbesselk/upstream/logbesselk.h

.PHONY: verify-upstream
verify-upstream:
	@if [ ! -e kaust_logbesselk/upstream/logbesselk.cu ] && [ ! -e kaust_logbesselk/upstream/logbesselk.h ]; then \
	    echo "  KAUST upstream sources not present; nothing to verify (make kaust at the top level)."; \
	    exit 0; \
	 fi; \
	 doc=kaust_logbesselk/PROVENANCE.md; rc=0; \
	 for f in $(UPSTREAM_KAUST); do \
	    base=$$(basename $$f); rel=$${f#kaust_logbesselk/}; \
	    if [ ! -e $$f ]; then echo "  $$base: missing (both files are needed)"; rc=1; continue; fi; \
	    got=$$(sha256sum $$f | cut -d' ' -f1); \
	    want=$$(awk -v b="$$rel" '$$0 ~ "`" b "`" { if (match($$0, /[0-9a-f]{32,}/)) { print substr($$0, RSTART, RLENGTH); exit } }' $$doc); \
	    if [ -z "$$want" ]; then \
	        echo "  $$base: no digest recorded in $$doc"; rc=1; \
	    elif [ "$$got" = "$$want" ]; then \
	        echo "  $$base: ok  $$got"; \
	    else \
	        echo "  $$base: mismatch"; echo "      file: $$got"; echo "      doc : $$want"; rc=1; \
	    fi; \
	done; \
	if [ $$rc -ne 0 ]; then \
	    echo "  the placed upstream files are not the ones PROVENANCE.md records"; exit 1; \
	fi

# Regenerates the Bessels.jl coefficient header from bessels_jl/upstream/.
.PHONY: extract
extract:
	cd bessels_jl/port && ./extract_upstream.sh

clean:
	rm -rf build
