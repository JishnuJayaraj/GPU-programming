CXXFLAGS =	-O2 -g -Wall -fmessage-length=0

OBJS =		Juliaset.o

LIBS =

TARGET =	Juliaset

$(TARGET):	$(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(LIBS)

all:	$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)
