echo ping
shuffler autoplay.bin.bin autoplay_shuffled.bin --min 1000 --max 2000
echo pong
shuffler autoplay_shuffled.bin autoplay.bin --min 50000 --max 100000
