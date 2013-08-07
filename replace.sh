#!/bin/bash
# author:	Felix Baumann
# date:		2013-08-07
# description:	
#		script that replaces the include strings "<opencv/*>" in
#		the source files of OpenTLD for C++ with the the include
#		string <*> where * denotes the original OpenCV library.
#		Please adjust the DIR variable before running.	

LIST=$( mktemp )
DIR=~
if [ "${DIR}" == "$( echo ~ )" ]; then
	echo "Please adjust the DIR variable first, exit."
	exit 1
fi
#TMP=$( mktemp )

grep -ri "<opencv/" ${DIR}/* | grep -v "replace.sh" | awk -F":" '{ print $1 }' > ${LIST}

while read item; do
	mv ${item} ${item}.bak
	cat ${item}.bak | sed -e "s/<opencv\//</g" > ${item}
done < ${LIST}
rm ${LIST}
