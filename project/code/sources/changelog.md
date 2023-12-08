v1:

* initial version

v2:

* changed pt_idx structure
* Does not use pt_idx pointer vectors anymore (free from deep copy!!)

v3:

* Apply threshold to apply CUDA kernel
* Best perf at 32 * BLOCKSIZE

upgrade room:

* unified memory
* change max distance finding algo to min copy
* change ccw sorting algo to min copy
* when there is not enough points (i.e. <= BLOCK_SIZE) use CPU for ccw and sorting
* what else?r
