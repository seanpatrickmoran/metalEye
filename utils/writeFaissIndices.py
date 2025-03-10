import faiss
import sqlite3
# import sqlite_vec
from typing import List
import struct
import numpy as np

import time
import sys, os

import json

# rootpath = "/Users/sean/Documents/Master/2025/March2025/SqueakToy/"
# dbSQLITE3VEC = rootpath + "virtualTables/EB_databaseVEC_18_fts5vec.db"

def call(PATH,TIMEOUT):

    connection = sqlite3.connect(PATH, timeout=TIMEOUT)  # Set timeout to 10 seconds
    cursor = connection.cursor()
    return connection,cursor


def deserialize_f32(vector,size=512):
    """serializes a list of floats into a compact "raw bytes" format"""
    return struct.unpack(f"{size}f", vector)


def getEverything(dbPATH, dimensions, timeout=10):
    connection,cursor=call(dbPATH,timeout)
    try:
        cursor.row_factory = sqlite3.Row
        cursor.execute("SELECT rowid,embedding from vector_table")
        print(f"success")
        reply = [(a,deserialize_f32(b, dimensions)) for a,b in cursor.fetchall()]
    except Exception as e:
        print(e)
        reply = []
    finally:
        cursor.close()
        connection.close()

    return reply



def runMain(dbSQLITE3VEC, dimensions):
    _nameDecorator = dbSQLITE3VEC.split("/")[-1].split(".db")[0]
    if not os.path.isdir('/'.join(dbSQLITE3VEC.split("/")[:-2])+f"/FAISSIndex"):
        os.mkdir('/'.join(dbSQLITE3VEC.split("/")[:-2])+f"/FAISSIndex")

    if not os.path.isdir('/'.join(dbSQLITE3VEC.split("/")[:-2])+f"/FAISSIndex"+f"/{_nameDecorator}"):
        os.mkdir('/'.join(dbSQLITE3VEC.split("/")[:-2])+f"/FAISSIndex"+f"/{_nameDecorator}")
        print('/'.join(dbSQLITE3VEC.split("/")[:-2])+f"/FAISSIndex"+f"/{_nameDecorator}")

    rootpath = '/'.join(dbSQLITE3VEC.split("/")[:-2])+f"/FAISSIndex"+f"/{_nameDecorator}"

    rows = getEverything(dbSQLITE3VEC, dimensions);
    # print([(row[0],len(row[1])) for row in rows])
    print('rad')

    ### faiss here.
    d = dimensions                           # dimension
    nb = len(rows)                      # database size
    nq = nb//10                       # nb of queries

    jsonKV_indices = {}
    for i in range(len(rows)):
        #use ID to get the range used in FAISS.
        jsonKV_indices[rows[i][0]]=i


    with open(rootpath+"/faiss.keyValue.map.json", "w") as zug:
        zug.write(json.dumps(jsonKV_indices))

    print(dimensions, nb)
    xb=np.array([np.array(xi[1]) for xi in rows]).astype('float32')   #EB_14_bin has gaps in key_ids. we can just ignore these as source_17 has no gaps.
    # xb = np.random.random((nb, d)).astype('float32')
    print(xb.shape)
    xb[:, 0] += np.arange(nb) / 1000.
    xq = np.random.random((nq, dimensions)).astype('float32')
    xq[:, 0] += np.arange(nq) / 1000.


    queries_raw = [];
    nlist = 100
    m = 8                             # number of subquantizers
    k = 4

    quantizer = faiss.IndexFlatL2(d)  # this remains the same
    print("init quantizer")
    index = faiss.IndexIVFPQ(quantizer, dimensions, nlist, m, 8)
    print("init index")
                                        # 8 specifies that each sub-vector is encoded as 8 bits
    index.train(xb)
    print("trained index")
    index.add(xb)
    faiss.write_index(index, rootpath+"/faiss.IndexIVFPQ.test.index")





# if 'hnsw_sq' in todo:

#     print("Testing HNSW with a scalar quantizer")
#     # also set M so that the vectors and links both use 128 bytes per
#     # entry (total 256 bytes)
#     index = faiss.IndexHNSWSQ(d, faiss.ScalarQuantizer.QT_8bit, 16)

#     print("training")
#     # training for the scalar quantizer
#     index.train(xt)

#     # this is the default, higher is more accurate and slower to
#     # construct
#     index.hnsw.efConstruction = 40

#     print("add")
#     # to see progress
#     index.verbose = True
#     index.add(xb)

#     print("search")
#     for efSearch in 16, 32, 64, 128, 256:
#         print("efSearch", efSearch, end=' ')
#         index.hnsw.efSearch = efSearch
#         evaluate(index)


    # print("added index")
    # print([x[0] for x in rows[:5]])
    # D, I = index.search(xb[:5], k) # sanity check
    # print(I)
    # print(D)
    # index.nprobe = 10              # make comparable with experiment above
    # D, I = index.search(xq, k)     # search
    # print(I[-5:])



    # # del index
    # M = 128
    # index = faiss.IndexHNSWSQ(d, faiss.ScalarQuantizer.QT_8bit, M)
    # # index_ivf_hnsw = faiss.IndexIVFFlat(quantizer, d, nlist)
    # index.train(xb)    
    # index.hnsw.efConstruction = 64    
    # index.verbose = True    
    # index.add(xb)
    # faiss.write_index(index, "faiss.IndexHNSWSQ_QT8_m128eConst64.index")

    # start = time.time()

    # search_time_IndexHNSWSQ_QT8 = time.time() - start
    # D, I = index.search(xq, 5)
    # print(f"IndexHNSWSQ_QT8 search time: {search_time_IndexHNSWSQ_QT8}")
    # print(I,D)

    # #  HNSW with Scalar Quantization (SQ)
    # quantizer_sq = faiss.IndexScalarQuantizer(d, faiss.ScalarQuantizer.QT_8bit)
    # index_hnswsq = faiss.IndexHNSWFlat(d, 32)

    # # Add the data
    # start = time.time()
    # index_hnswsq.add(xb)
    # indexing_time_hnswsq = time.time() - start
    # print(f"HNSWQ indexing time: {indexing_time_hnswsq}")
    # faiss.write_index(index_hnswsq, "faiss.IndexHNSWSQ_FLAT_QT8.index")
    # # Search with HNSWSQ
    # # start = time.time()
    # # D, I = index_hnswsq.search(xq, 5)
    # # search_time_hnswsq = time.time() - start

    # # Search with HNSWSQ
    # start = time.time()
    # D, I = index_hnswsq.search(xq, 5)
    # search_time_hnswsq = time.time() - start
    # print(f"HNSWQ search time: {search_time_hnswsq}")
    # print(I,D)


    del index
    quantizer = faiss.IndexFlatL2(dimensions)  # the other index
    index = faiss.IndexIVFFlat(quantizer, dimensions, nlist)
    index.train(xb)
    index.add(xb)
    # index.make_direct_map()
    index.set_direct_map_type(faiss.DirectMap.Array)
    faiss.write_index(index, rootpath+"/faiss.IndexIVFFlat.index")

    start = time.time()
    D, I = index.search(xq, 5)
    search_time_index = time.time() - start
    print(f"IndexFlatL2_dirMap search time: {search_time_index}")
    print(I,D)



    index2 = faiss.IndexFlatL2(dimensions)
    index2.add(xb)
    faiss.write_index(index, rootpath+"/faiss.IndexFlatL2_dirMap.index")


if __name__ == "__main__":
    dbSQLITE3VEC = sys.argv[1]
    # nameDecorator = sys.argv[2]
    dimensions = int(sys.argv[2])
    runMain(dbSQLITE3VEC, dimensions)
