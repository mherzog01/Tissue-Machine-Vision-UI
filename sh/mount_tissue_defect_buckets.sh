# TODO Make xref of buckets and mount points
if [ $1 = '-u' ]
then
    echo "Unmounting buckets"
    fusermount -u ./gcp/tmp
    fusermount -u ./gcp/static
    fusermount -u ./gcp/ov
    exit
fi

# per https://stackoverflow.com/questions/42602356/how-to-unmount-google-bucket-in-linux-created-with-gcsfuse
#
# Don't use --implicit-dirs per https://github.com/GoogleCloudPlatform/gcsfuse/issues/234

# TODO Support different environments, not just "dev"
mkdir -p gcp

mkdir -p gcp/tmp
gcsfuse tissue-defect-tmp ./gcp/tmp

mkdir -p gcp/static
gcsfuse tissue-defect-dev ./gcp/static

# OV - Object Versioning - for code and other objects where we want to track versions
mkdir -p gcp/ov
gcsfuse tissue-defect-dev-ov ./gcp/ov
