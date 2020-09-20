USE_SDL := 0

# Chemin vers ALE
ALE := Arcade-Learning-Environment/

FLAGS := -I$(ALE)/src -I$(ALE)/src/controllers -I$(ALE)/src/os_dependent -I$(ALE)/src/environment -I$(ALE)/src/external -L$(ALE)
CXX := g++
FILE := agent-pong
LDFLAGS := -lale -lz

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    FLAGS += -Wl,-rpath=$(ALE)
endif
ifeq ($(UNAME_S),Darwin)
    FLAGS += -framework Cocoa
endif

ifeq ($(strip $(USE_SDL)), 1)
  DEFINES += -D__USE_SDL -DSOUND_SUPPORT
  FLAGS += $(shell sdl-config --cflags)
  LDFLAGS += $(shell sdl-config --libs)
endif

all: agent-pong

agent-pong:
	$(CXX) $(DEFINES) $(FLAGS) $(FILE).cpp $(LDFLAGS) -o $(FILE)

clean:
	rm -rf agent-pong *.o
