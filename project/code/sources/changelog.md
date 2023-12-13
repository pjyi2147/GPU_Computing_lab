v1:

* initial version

v2:

* changed pt_idx structure
* Does not use pt_idx pointer vectors anymore (free from deep copy!!)

v3:

* Apply threshold to apply CUDA kernel
* Best perf at 32 * BLOCKSIZE

v4:

* Apply parallel reduction on the max dist calculation
* Best perf at 32 * BLOCKSIZE (85ms on 2M points)

v5:

* Apply Unified memory
* Does not work well

v6:

* Sort using thrust
* Works really well

upgrade room:

* unified memory
* change max distance finding algo to min copy
* change ccw sorting algo to min copy
* when there is not enough points (i.e. <= BLOCK_SIZE) use CPU for ccw and sorting
* what else?
* GPU sorting??
