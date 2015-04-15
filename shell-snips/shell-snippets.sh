#!/bin/bash

# Download datasets with:
wget -r -k -R '*index.html*' -nH --cut-dirs=2 --no-parent "$URL"

# Move first column to the last in a csv file:
sed -i -r 's/(\w),(.*)/\2,\1/' "$FILENAME"

# Extract class labels from csv (assuming labels in last column):
cat letter-recognition.tes | awk 'BEGIN{FS=OFS=","} {print $NF}' | sort -u

# Processing the PROG: style logs
cat "$FILENAME" | cut -d' ' -f2- | sed 's/set(\[\(.*\)\])/\1/' | tr -d ' '
