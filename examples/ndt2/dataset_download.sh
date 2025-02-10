if [ ! -d "data" ]; then
  echo "data/ does not exist, please create it e.g. a symlink to your data directory"
  exit 1
fi

cd data
# RTT
mkdir odoherty_rtt
cd odoherty_rtt
zenodo_get 3854034
cd ..