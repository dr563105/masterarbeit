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

PS     = $(wildcard *.ps)

PS_EPS = $(patsubst %.ps, %.eps, $(PS))

PS2EPS = ps2eps -f -l -B -C -H

all: $(PS_EPS)
	@# nothing more to do

%.eps: %.ps
	$(PS2EPS) $<

clean:
	rm -f $(PS_EPS)
