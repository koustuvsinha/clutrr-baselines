#!/bin/sh

if [ -z "$1" ] ; then
    echo "Minimum Job Number argument is required.  Run as '$0 jobnum'"
    exit 1
fi

minjobnum="$1"


for j in $(squeue --user="$USER" --noheader --format=%i) ; do
  if [ "$j" -gt "$minjobnum" ] ; then
    scancel "$j"
    echo "cancelled job $j"
  fi
done
