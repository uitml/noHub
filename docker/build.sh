while getopts u:n: flag
do
    case "${flag}" in
        u) userName=${OPTARG};;
        n) imageName=${OPTARG};;
    esac
done

if [ -z $userName ]; then
  echo "Usage: build.sh -u docker-username [-n image-name']"
  exit
fi

if [ -z $imageName ]; then
  imageName="hubs"
  echo "Image name not specified. Using default name: ${imageName}"
fi

echo "Building image ${userName}/${imageName}..."

docker build \
  --tag $userName/$imageName \
  --rm \
  --no-cache \
  --pull \
  -f docker/Dockerfile \
  .
