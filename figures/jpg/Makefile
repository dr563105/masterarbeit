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

JPG     = $(wildcard *.jpg)

JPG_EPS = $(patsubst %.jpg, %.eps, $(JPG))

CONVERT = convert

all: $(JPG_EPS)
	@# nothing more to do

%.eps: %.jpg
	$(CONVERT) $< $@

clean:
	rm -f $(JPG_EPS)
