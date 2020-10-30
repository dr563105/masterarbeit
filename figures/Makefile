SUBDIRS = ppt fig gnuplot xls inkscape png jpg

all:
	@for TARGET in $(SUBDIRS) ; do \
		$(MAKE) -C $$TARGET; \
	done;

clean:
	@for TARGET in $(SUBDIRS) ; do \
		$(MAKE) -C $$TARGET clean; \
	done;
