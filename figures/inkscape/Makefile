############################################################################
#
#  $HeadURL$
#  $Author$
#  $Date$
#  $Revision$
#
###########################################################################
#
#       This confidential and proprietary software may be used only
#      as authorized by a licensing agreement from the Institute for
#     Integrated Signal Processing Systems of the RWTH Aachen (Germany).
#
#                         II     SSSS    SSSS
#                         II    SS      SS
#                         II     SS      SS
#                         II       SS      SS
#                         II    SSSS    SSSS
#
#                            (C) COPYRIGHT
#             INSTITUTE FOR INTEGRATED SIGNAL PROCESSING SYSTEMS
#                             RWTH AACHEN
#                         ALL RIGHTS RESERVED
#
###########################################################################
#
#                     Author:  Ernst Martin Witte
#                    Contact:  witte@iss.rwth-aachen.de
#
###########################################################################

SVG      = $(wildcard *.svg)

SVG_EPS  = $(patsubst %.svg, %.eps, $(SVG))

INKSCAPE = inkscape

all: $(SVG_EPS)
	@# nothing more to do

%.eps: %.svg
	$(INKSCAPE) -z -f$< -E$@ -D

clean:
	rm -f $(SVG_EPS)
