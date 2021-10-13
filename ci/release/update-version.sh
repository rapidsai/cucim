#!/bin/bash
#########################
# cuCIM Version Updater #
#########################

## Usage
# bash update-version.sh <new_version>


# Format is YY.MM.PP - no leading 'v' or trailing 'a'
NEXT_FULL_TAG=$1

# Get current version
CURRENT_TAG=$(git tag --merged HEAD | grep -xE '^v.*' | sort --version-sort | tail -n 1 | tr -d 'v')
CURRENT_MAJOR=$(echo $CURRENT_TAG | awk '{split($0, a, "."); print a[1]}')
CURRENT_MINOR=$(echo $CURRENT_TAG | awk '{split($0, a, "."); print a[2]}')
CURRENT_PATCH=$(echo $CURRENT_TAG | awk '{split($0, a, "."); print a[3]}' | tr -d 'a')
CURRENT_SHORT_TAG=${CURRENT_MAJOR}.${CURRENT_MINOR}
CURRENT_LONG_TAG=${CURRENT_MAJOR}.${CURRENT_MINOR}.${CURRENT_PATCH}

#Get <major>.<minor> for next version
NEXT_MAJOR=$(echo $NEXT_FULL_TAG | awk '{split($0, a, "."); print a[1]}')
NEXT_MINOR=$(echo $NEXT_FULL_TAG | awk '{split($0, a, "."); print a[2]}')
NEXT_SHORT_TAG=${NEXT_MAJOR}.${NEXT_MINOR}

echo "Preparing release $CURRENT_TAG => $NEXT_FULL_TAG"

# Inplace sed replace; workaround for Linux and Mac
function sed_runner() {
    sed -i.bak ''"$1"'' $2 && rm -f ${2}.bak
}

# RTD update
sed_runner 's/version = .*/version = '"'${NEXT_SHORT_TAG}'"'/g' docs/source/conf.py
sed_runner 's/release = .*/release = '"'${NEXT_FULL_TAG}'"'/g' docs/source/conf.py

sed_runner "s/${CURRENT_LONG_TAG}/${NEXT_FULL_TAG}/g" VERSION
sed_runner "s/${CURRENT_LONG_TAG}/${NEXT_FULL_TAG}/g" python/cucim/VERSION
sed_runner "s/${CURRENT_LONG_TAG}/${NEXT_FULL_TAG}/g" cpp/plugins/cucim.kit.cuslide/VERSION
sed_runner "s/${CURRENT_LONG_TAG}/${NEXT_FULL_TAG}/g" cpp/plugins/cucim.kit.cumed/VERSION
sed_runner "s#\[Version ${CURRENT_LONG_TAG}\](release_notes/v${CURRENT_LONG_TAG}.md)#\[Version ${NEXT_FULL_TAG}\](release_notes/v${NEXT_FULL_TAG}.md)#g" python/cucim/docs/index.md
sed_runner "s/v${CURRENT_LONG_TAG}/v${NEXT_FULL_TAG}/g" python/cucim/docs/getting_started/index.md
sed_runner "s#cucim.kit.cuslide@${CURRENT_LONG_TAG}.so#cucim.kit.cuslide@${NEXT_FULL_TAG}.so#g" python/cucim/docs/getting_started/index.md
sed_runner "s#cucim.kit.cuslide@${CURRENT_LONG_TAG}.so#cucim.kit.cuslide@${NEXT_FULL_TAG}.so#g" cucim.code-workspace
