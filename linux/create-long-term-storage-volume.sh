#!/bin/bash
# Tim H 2021
# Script for long-term storage (multi-decade) of large collections of files
# Performs the following tasks:
#   * locates all compressed files and expands them, deletes the original
#   * takes/stores inventory of the system used to generate the archive (OS X system report; pip3/pip/brew package list) in order to make future troubleshooting easier
#   * generates a complete list of all files/directories, their dates, sizes, and permissions- stores in text file
#   * stores the hashsums (SHA 256) of all the files found
#   * digitally signs the hashsum file using GPG key to ensure authenticity/integrity
#   * TBD: compresses the entire thing using plain ZIP as an additional archive option (less reliable in the long term)
#
# Usage:
#   ./create-long-term-storage-volume.sh example-sample-compress-directory
#   ~/source_code/homelab/web-archiving/personal-archiving/create-long-term-storage-volume.sh "$HOME/Downloads/ASE 166M - Space Lab"
# time ~/source_code/homelab/web-archiving/personal-archiving/create-long-term-storage-volume.sh "$HOME/Downloads/college_work"
# time ~/source_code/homelab/web-archiving/personal-archiving/create-long-term-storage-volume.sh "$HOME/Downloads/Projects"

# Install dependencies in OS X
# brew install unar #[sic, yes unar]
# brew install p7zip
# brew install gpg

##############################################################################
#   VARIABLE DEFINITIONS
##############################################################################

PATH_TO_ARCHIVE="$1"                            # relative or absolute path to a directory that you want to prepare for archival
GPG_KEY_NAME="Tim H GPG 2021 - insecure-test5"  # 'Name' of the GPG key used by local user on local system
ARCHIVE_FILE_PREFIX="archive-tdh"               # just a unique filename prefix that only this archiver would use; no trailing - needed

set -e

##############################################################################
#   FUNCTION DEFINITIONS
##############################################################################

function extract_arbitrary_compressed_file {
    LOCAL_COMPRESSED_FILE_PATH="$1"
    ITER_PARENT_DIR=$(dirname "$LOCAL_COMPRESSED_FILE_PATH")
    ITER_FULL_FILENAME=$(basename -- "$LOCAL_COMPRESSED_FILE_PATH")
    ITER_FILENAME="${ITER_FULL_FILENAME%.*}"  # TODO: BUG located here, only goes up to the last period in extension instead of first
    ITER_EXTENSION="${ITER_FULL_FILENAME##*.}"
  
    case $ITER_EXTENSION in
      zip | 7z )
          echo "Extacting ZIP: $LOCAL_COMPRESSED_FILE_PATH"
          # the "${VAR}" produces error code 81 - 81 means it cannot unzip the file since a non-standard method was used.
          # shellcheck disable=SC2086,SC2086
          if 7z e "$LOCAL_COMPRESSED_FILE_PATH" -o${ITER_PARENT_DIR}/${ITER_FILENAME}   > /dev/null 2>&1 ; then
            rm -f "$LOCAL_COMPRESSED_FILE_PATH"
          else
            echo "7z seemed to fail - exit code: $?   - exiting script immediately."
            exit 11
          fi
        ;;

      rar)
          echo "Extacting RAR: $LOCAL_COMPRESSED_FILE_PATH"
          if unar -quiet -force-rename -output-directory "$ITER_PARENT_DIR/$ITER_FILENAME" "$LOCAL_COMPRESSED_FILE_PATH"  > /dev/null 2>&1 ; then
            rm -f "$LOCAL_COMPRESSED_FILE_PATH"
          else
            echo "unar seemed to fail - exit code: $?   - exiting script immediately."
            exit 22
          fi
        ;;

      tar )
          echo "Extacting TAR: $LOCAL_COMPRESSED_FILE_PATH"
          mkdir -p "${ITER_PARENT_DIR}/${ITER_FILENAME}"

          if tar -C "${ITER_PARENT_DIR}/${ITER_FILENAME}" -xvf "${LOCAL_COMPRESSED_FILE_PATH}"  > /dev/null 2>&1 ; then
            rm -f "$LOCAL_COMPRESSED_FILE_PATH"
          else
            echo "tar seemed to fail - exit code: $?   - exiting script immediately."
            exit 33
          fi
        ;;

      tgz | tar.gz | gz)
          echo "Extacting TGZ/GZ: $LOCAL_COMPRESSED_FILE_PATH"
          # TODO: complex filenames with multiple extensions break things like .tar.gz
          # might need to modify how I calculate basename
          #Extacting TGZ/GZ: /Users/tim/Downloads/projects/XBOX Original Backups.tar.gz
          # tar: could not chdir to '/Users/tim/Downloads/projects/XBOX Original Backups.tar'

          mkdir -p "${ITER_PARENT_DIR}/${ITER_FILENAME}"
          if tar -zxvf "${LOCAL_COMPRESSED_FILE_PATH}" -C "${ITER_PARENT_DIR}/${ITER_FILENAME}"  > /dev/null 2>&1 ; then
            rm -f "$LOCAL_COMPRESSED_FILE_PATH"
          else
            echo "TAR's TGZ/GZ TAR.GZ seemed to FAIL - exit code: $?   - exiting script immediately."
            exit 44
          fi

        ;;

      *)
        echo "unknown extension type: $ITER_EXTENSION . EXITING!"
        exit 99
        ;;
	esac    
}


function do_compressed_files_exist_in_dir {
  FILE=$(find "$1"  -type f \( -iname \*.zip -o -iname \*.rar -o -iname \*.tar -o -iname \*.tar.gz -o -iname \*.tgz -o -iname \*.gz -o -iname \*.7z \)  -print -quit 2> /dev/null)

  if [ -n "$FILE" ]; then
      return 0 # true
  else
      return 1 # false
  fi
}


function find_and_extract_compressed_files {

  PATH_TO_ARCHIVE="$1"

  while do_compressed_files_exist_in_dir "$PATH_TO_ARCHIVE"
  do  
    # shellcheck disable=SC2162
    find "$PATH_TO_ARCHIVE" -type f \( -iname \*.zip -o -iname \*.rar -o -iname \*.tar -o -iname \*.tar.gz -o -iname \*.tgz -o -iname \*.gz -o -iname \*.7z \) -print0 | while read -d $'\0' ITER_COMPRESSED_FILE_PATH
    do
      extract_arbitrary_compressed_file "$ITER_COMPRESSED_FILE_PATH"
    done
  done
}



##############################################################################
#   MAIN
##############################################################################

echo "[create-long-term-storage-volume] Script started"


# ensure that the source directory exists
if [ ! -d "$PATH_TO_ARCHIVE" ]; then
    echo "Directory to compress does not exist: $PATH_TO_ARCHIVE"
    exit 1
fi

# TODO: check prereqs like if unar is installed


# define some additional variables 
# shellcheck disable=
HASHSUMS_FILE_FULL_PATH="$PATH_TO_ARCHIVE/$ARCHIVE_FILE_PREFIX-hashsums.sha256"
PARENT_DIR=$(dirname "$PATH_TO_ARCHIVE")
COMPRESSED_FILE_FULL_PATH="$PARENT_DIR/$(basename "$PATH_TO_ARCHIVE").zip"

# Before doing anything - Delete any files from previous runs on the same filenames
# must use the ${} form of variables to work with the * suffix
rm -f "${PATH_TO_ARCHIVE}/${ARCHIVE_FILE_PREFIX}*" "$COMPRESSED_FILE_FULL_PATH"

# locate compressed (ZIP, RAR) files and decompress them, delete the original
# TODO: add support for TAR, TAR.GZ, TGZ, 7Z, GZ volumes too
# can't assume that the file/path won't have a space in it.
echo "Looking for compressed files and expanding them before starting larger archiving task..."
find_and_extract_compressed_files "${PATH_TO_ARCHIVE}"

# output some general data output to a new text file
echo "Archive created: $(date)
Path to archive: $PATH_TO_ARCHIVE
Name of GPG key being used:
$GPG_KEY_NAME

Original size: $(du -sh "$PATH_TO_ARCHIVE")
Hostname: $(hostname)
User: $(whoami)
Kernel: $(uname -a)

Hashing tool version:
$(sha256sum --version)

Digital signature tool version:
$(gpg --version)

GPG Secret Keys on local system:
$(gpg --list-secret-keys)

GPG Settings:
$(gpgconf --list-options gpg-agent)

Compression tool version: 
$(zip --version)
" > "$PATH_TO_ARCHIVE/$ARCHIVE_FILE_PREFIX-info.txt"

echo "Generating OS X system profile file - this takes a while ..."
system_profiler &> "$PATH_TO_ARCHIVE/$ARCHIVE_FILE_PREFIX-system-profile.txt"

echo "Generating list of Brew packages and versions ..."
brew list --versions --verbose --debug &> "$PATH_TO_ARCHIVE/$ARCHIVE_FILE_PREFIX-brew-list.txt"

echo "Generating list of Pip3 packages and versions ..."
pip3 list --verbose &> "$PATH_TO_ARCHIVE/$ARCHIVE_FILE_PREFIX-pip3-list.txt"

echo "Generating list of Pip  packages and versions ..."
pip list --verbose &> "$PATH_TO_ARCHIVE/$ARCHIVE_FILE_PREFIX-pip-list.txt"

# npm seems to be directory dependent, so skipping that.

##############################################################################

cd "$PATH_TO_ARCHIVE"   # make sure to only do this once

# output a tree map of all the files, including timestamps into a new text file inside folder
echo "Generating file index..."
tree -ifpugDs -o "$ARCHIVE_FILE_PREFIX-file-index-tree.txt" --timefmt "%Y-%m-%d %I:%M:%S %p %Z" .

echo "Calculating hashsums..."
# calcuate the hashsums of all the files (recursively) and dump them into a new text file
# on a test set, MD5 took 16.1 sec, SHA256 took 42.5 sec
# had to hardcode the text file name since variable wasn't working for some reason.
find . -type f ! -name '*.sha256' -exec sha256sum {} + > "$HASHSUMS_FILE_FULL_PATH"

echo "Signing and encrypting hashsum file for authenticity and integrity..."
# Add PGP digital signature on just the hashsums file, that's all that's needed to verify
#   integrity across everything. Sign it 3 different ways just to make sure it's
#   easy to test/recover in the distant future
# TODO: maybe include the public key in here or is that  bad idea?
gpg --default-key "$GPG_KEY_NAME" --detach-sig archive-tdh-hashsums.sha256
gpg --default-key "$GPG_KEY_NAME" --sign       archive-tdh-hashsums.sha256
gpg --default-key "$GPG_KEY_NAME" --clearsign  archive-tdh-hashsums.sha256
#gpg --local-user  "$GPG_KEY_NAME" --clearsign archive-tdh-hashsums.sha256

echo "Compressing files without deleting originals..."
#zip -r --compression-method store "$COMPRESSED_FILE_FULL_PATH" "$PATH_TO_ARCHIVE"
# compress with quiet and regular compression ratio
#zip -q -r "$COMPRESSED_FILE_FULL_PATH" "$PATH_TO_ARCHIVE"

echo "[create-long-term-storage-volume] Script finished successfully."

# seems like 7z compression isn't that much better than regular ZIP
#echo "testing compression on directory:"
# good alg for text: PPMd
# original plain ZIP was 7.6 MB
#7z a "$COMPRESSED_FILE_FULL_PATH" "$PATH_TO_ARCHIVE"    # produced 4.5 MB file
#7z a "$COMPRESSED_FILE_FULL_PATH" "$PATH_TO_ARCHIVE" -m0=PPMd   # produced 6.6 MB file in test set
#7z a "$COMPRESSED_FILE_FULL_PATH" "$PATH_TO_ARCHIVE" -mx9   # 
#7z a -t7z -m0=zip -mx=9 "$COMPRESSED_FILE_FULL_PATH" "$PATH_TO_ARCHIVE"
#7z a -t7z -m0=lzma -mx=9 -mfb=64 -md=32m -ms=on archive.7z dir1

#du -sh "$PATH_TO_ARCHIVE"
#ls -lah "$COMPRESSED_FILE_FULL_PATH"


#gpg --delete-keys  28243B80076EC786E47AD2C1D5E4BA3013B1357C B91FC31DDD1BA386F099490ED70BD52F8C39FFC9 3B1AC414AF31EBBC215C5F36AC8592F02C16C20B 19851A15AD16FA3685BA64B25D6079695FC6E9ED
