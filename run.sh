#!/bin/bash
<<<<<<< HEAD
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

python server.py --toy  &
sleep 10  # Sleep for 10s to give the server enough time to start and dowload the dataset

for i in `seq 0 9`; do
    echo "Starting client $i"
    python client.py --client-id=${i} --toy &
done

# Enable CTRL+C to stop all background processes
=======

echo "Starting server"
python server.py &
sleep 3  # Sleep for 3s to give the server enough time to start

for i in $(seq 0 1); do
    echo "Starting client $i"
    python client.py --partition-id $i &
done

# This will allow you to use CTRL+C to stop all background processes
>>>>>>> f9786dc (update)
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
