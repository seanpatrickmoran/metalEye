


# rs = np.random.RandomState(123)
# subset = rs.choice(ds.nb, 50, replace=False).astype("int64")
# bitmap = np.zeros(ds.nb, dtype=bool)
# bitmap[subset] = True
# bitmap = np.packbits(bitmap, bitorder='little')


import mlx.core as mx
# mx.random.uniform(0,99130,dtype=mx.uint64)
# mx.random.randint(0,99130)

#done for each region
entryVal = 99130
bitmap = mx.zeros(entryVal + entryVal%8,dtype=mx.uint8) #now we always have a number perfectly divisible by 8. 
subset = mx.arange(0,1111) #take from partition hashmap
bitmap[subset] = 1 
#set to bloomfilter. we need to pad this to be 8bit unsigned.


#little endian bitmap for bloom filter
mxBitMap = mx.zeros((entryVal + entryVal%8)//8,dtype=mx.int64)
bw = mx.zeros(8, dtype=mx.uint8)
for i in range(len(bitmap)):
    bw[i%8]=bitmap[i] 
    if not (i+1)%8:
        accumulator = 0
        for j in range(8):
            accumulator += (2**j) * bw[j].item()
            bw[j] = 0 
        mxBitMap[((i+1)//8)-1]=accumulator
        # print(i, mxBitMap[((i+1)//8)-1].item())

