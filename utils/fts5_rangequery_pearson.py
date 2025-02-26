# import sqlite_vec
# from termcolor import colored, cprint

from typing import List
import faiss
import struct
import datetime
import sqlite3
import json

import mlx.core as mx
import numpy as np
import sys

####################################
#
# INIT IN-MEM dbPATH and JSON stores
#
####################################


dbSOURCE = "/Users/sean/Documents/Master/2025/Feb2025/sourceTables/database_17_bin.db"
# dbSOURCE = "/Users/seanmoran/Documents/Master/2025/Feb2025/database_TEST/database_14_bin.db"
# dbVECTOR_FTS5 = "/Users/seanmoran/Documents/Master/2025/Feb2025/EB_databaseVEC_14.db"
dbVECTOR_FTS5 = "/Users/sean/Documents/Master/2025/Feb2025/virtualTables/EB_14_fts5vec.db"

sourcePartitions = "/Users/sean/git/MLX/metalEye/utils/database_17_2025-02-24T16.json"

#99129

colorMap = {
            5: '\x1b[95m',
            4: '\x1b[96m',
            3: '\x1b[92m',
            2:  '\x1b[39m',
            1:  '\x1b[93m',
            0:  '\x1b[91m\x1b[5m'
            }


store_answer = {
                # "imageScore":np.zeros(99129),
                # "epiScore":np.zeros(99129),
                # # "histogramScore":np.zeros(99129),
                # "p@k":np.zeros(99129),
                "imageScore": mx.array([0]*99129, dtype=mx.float32),
                "epiScore": mx.array([0]*99129, dtype=mx.float32),
                # "histogramScore":np.zeros(99129),
                "p@k": mx.array([0]*99129, dtype=mx.float32),
                }



import mlx.core as mx

def createBitMask(arraySize, subset):



    # # subset must be a mx.array...
    # # i.e. createBitMask(99130,mx.arange(0,1111))

    # # entryVal = 99130
    # bitmap = mx.zeros(arraySize + arraySize%8,dtype=mx.uint8) #now we always have a number perfectly divisible by 8. 
    # # subset = mx.arange(0,1111) #take from partition hashmap
    # bitmap[subset] = 1 
    # #set to bloomfilter. we need to pad this to be 8bit unsigned.

    # #little endian bitmap for bloom filter
    # mxBitMap = mx.zeros((arraySize + arraySize%8)//8,dtype=mx.int64)

    # bw = mx.zeros(8, dtype=mx.uint8)
    # for i in range(len(bitmap)):
    #     bw[i%8]=bitmap[i] 
    #     if not (i+1)%8:
    #         accumulator = 0
    #         for j in range(8):
    #             accumulator += (2**j) * bw[j].item()
    #             bw[j] = 0 
    #         mxBitMap[((i+1)//8)-1]=accumulator
    #         print(i, mxBitMap[((i+1)//8)-1].item())
    # return np.array(mxBitMap)

    bitmap = np.zeros(arraySize, dtype=bool)
    bitmap[np.array(subset)] = True
    bitmap = np.packbits(bitmap, bitorder='little')
    return bitmap








# def pairwise_correlation(A, B):
#     am = A - np.mean(A, axis=0, keepdims=True)
#     bm = B - np.mean(B, axis=0, keepdims=True)
#     return am.T @ bm /  (np.sqrt(
#         np.sum(am**2, axis=0,
#                keepdims=True)).T * np.sqrt(
#         np.sum(bm**2, axis=0, keepdims=True)))

def mlx_pairwise_correlation(A, B):
    am = A - A.mean(axis=0)
    bm = B - B.mean(axis=0)
    # am = A - mx.mean(A, axis=0, keepdims=True)
    # bm = B - mx.mean(B, axis=0, keepdims=True)
    _numer = mx.matmul(am.T, bm, stream=mx.gpu)
    return _numer /  (mx.sqrt(
        mx.sum(am**2, axis=0,
               keepdims=True), stream=mx.gpu).T * mx.sqrt(
        mx.sum(bm**2, axis=0, keepdims=True), stream=mx.gpu))

mx_comp_pairwise = mx.compile(mlx_pairwise_correlation)



class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)



class MLXEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        # if isinstance(obj, mx.int8):
        #     return int(obj)
        # elif isinstance(obj, mx.float32):
        #     return float(obj)
        if isinstance(obj, mx.array):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)




def serialize_f32(vector: List[float]) -> bytes:
    return struct.pack("%sf" % len(vector), *vector)


def deserialize_f32(vector,size=5120):
    return struct.unpack(f"{size}f", vector)



def call(PATH,TIMEOUT):
    connection = sqlite3.connect(PATH, timeout=TIMEOUT)  # Set timeout to 10 seconds
    cursor = connection.cursor()
    return connection,cursor



def _readEmbeddingByKeyId(dbPATH, timeout, key_id=0):
    connection,cursor=call(dbPATH,timeout)
    reply = [-1]
    try:
        cursor.row_factory = sqlite3.Row
        cursor.execute("SELECT * FROM vector_table WHERE rowid = ? LIMIT 1", [key_id,])
        print(key_id, end=": ")
        # print([b for b in cursor.fetchall()])
        # print([deserialize_f32(b, 5120) for b in cursor.fetchall()])
        reply = [deserialize_f32(b[0], 5120) for b in cursor.fetchall()]
        if reply == []:
            return -404
    except Exception as e:
        print(e, end=" ")
        print("failure")
        reply = [-1]
    finally:
        cursor.close()
        connection.close()

    return reply.pop()


# WIP?
# def batchedFaissHNSW(id, eValue):

#     flagSourceAvail = True
#     print("@@@", end=" ")
#     # print("@@@", end=" ", file=sys.stderr, flush=True)
#     #call source
#     connection,cursor=call(dbSOURCE,10)

#     try:
#         cursor.row_factory = sqlite3.Row
#         # cursor.execute("SELECT rowid,embedding from vector_table")
#         query_Rows = [row for row in keyIdToRow(dbSOURCE,id, 10)]
#         q_vmaxes  = query_Rows[6]
#         q_images = mx.array(query_Row[5], dtype=mx.float32) / q_vmax * 255
#         q_images = q_image.astype(dtype=mx.int8)
#         q_epigenomics = mx.array(query_Row[17],dtype=mx.float32)
#         #if source works, keep going else

#     except Exception as e:
#         print(e)
#         flagSourceAvail = False
#     finally:
#         cursor.close()
#         connection.close()       
#         if flagSourceAvail==False:
#             return 
#     xq = np.reshape(xb[id,:], (1,-1))
#     print(xq.shape)
#     k = 7
#     D, I = index.search(xq, k) # search @id KNN

#     print(I)
#     print(D)






####################################
#
# Broad Search Methods
#
####################################
        

def faissHNSW(id, eValue, positions):

    flagSourceAvail = True
    print("@@@", end=" ")
    # print("@@@", end=" ", file=sys.stderr, flush=True)
    #call source
    connection,cursor=call(dbSOURCE,10)

    try:
        cursor.row_factory = sqlite3.Row
        # cursor.execute("SELECT rowid,embedding from vector_table")
        # print(id)
        query_Row = keyIdToRow(dbSOURCE,id, 1000)
        q_vmax  = query_Row[6]
        q_image = mx.array(query_Row[5], dtype=mx.float32) / q_vmax * 255
        q_image = q_image.astype(dtype=mx.int8)
        q_epigenomic = mx.array(query_Row[17],dtype=mx.float32)
        #if source works, keep going else

    except Exception as e:
        print(e)
        flagSourceAvail = False
    finally:
        cursor.close()
        connection.close()       
        if flagSourceAvail==False:
            return 


    # positions = json.loads(sourcePartitions).decode('utf-8')
    keyword = str(query_Row[11])+str(query_Row[13])+str(query_Row[15])

    # db = sqlite3.connect(dbVECTOR_FTS5)
    # db.enable_load_extension(True)
    # sqlite_vec.load(db)
    # db.enable_load_extension(False)


    # rows = db.execute(
    #   """
    #     SELECT
    #       key_id,
    #       distance
    #     FROM vec_items
    #     WHERE embedding MATCH ?
    #     AND k = 8
    #   """,
    # eValue,
    # ).fetchall()
    # print([x[0] for x in rows])

    #we query from FAISS not from a DB anymore. 

    # name = request.args.get('name', default = "", type = str)
    # nArr = nameToNArr(dbSOURCE, name, 10);
    # query = ollama.embed(model='llama3.2', input=str(nArr),)
    start,end = positions[keyword]
    # print(query.embeddings[0])

    xq = np.reshape(xb[id,:], (1,-1))
    k = 200 #this doesn't cost much.
    # D, I = index.search(xq, k) # search @id KNN


    # now = datetime.datetime.now()
    mitBap = createBitMask(nb, mx.arange(start,end))
    # _D, _I = index.search(xq, k, params=faiss.SearchParametersIVF(sel=faiss.IDSelectorBitmap(mitBap),nprobe=200))

    _D, _I = index.search(xq, k, params=faiss.SearchParametersIVF(sel=faiss.IDSelectorRange(start, end, True),nprobe=200))
    # print("Search ", datetime.datetime.now() - now)

    # print(_I)
    if len(_D[0]) == 0 or len(_I[0]) == 0:
        return

    D, I = zip(*sorted(zip(_D[0], _I[0])))   
    #necessary to sort these. 


    # print(I)

    # print(I)
    # print(D)

    # I[[]] -> our indices.
    # D[[]] -> 1-D[0][i] is our recall

    # message = {"message":[]}
    # for node in I[0]:
        # print(node)
        # node = rows.pop()
        # val = rowIdToName(dbSOURCE, str(node+1), 10)
        # if val != [0]:
            # message["message"]+=[val]


    store_answer["p@k"][id] = sum(1-x for x in D[:7])/len(D[:7])


    store_epiP = mx.array([0]*7, dtype=mx.float32)
    store_imgP = mx.array([0]*7, dtype=mx.float32)
    store_histP = mx.array([0]*7, dtype=mx.float32)

    mxIndex = 0
    for imx in range(7):
    # for imx in range(len(I)):
        # print(imx)
        # print(I[0][imx],type(I[0][imx]))

        # query_Row = keyIdToRow(dbSOURCE,id, 10)
        val = keyIdToRow(dbSOURCE, int(I[imx]), 1000)
    #     node = rows.pop()
    #     val = keyIdToRow(dbSOURCE, node[0], 10)

        if val is None or val == -2:
            print('\x1b[91m\x1b[5mMissed Result!!!\x1b[0m')
            pass
        elif val[0]==id:
            pass
        else:
            a_vmax  = val[6]
            a_image = mx.array(val[5], dtype=mx.float32) / a_vmax * 255
            a_image = a_image.astype(dtype=mx.int8)
            a_epigenomic = mx.array(val[17],dtype=mx.float32)
            try:
                epiP = mlx_pairwise_correlation(q_epigenomic, a_epigenomic)
            except ValueError as e:
                print(f"ValueError: {e}")
                epi = mx.array(0,dtype=mx.float32)
            try:
                imgP = mlx_pairwise_correlation(q_image, a_image)
            except ValueError as e:
                print(f"ValueError: {e}")
                imgP = mx.array(0,dtype=mx.float32)

            store_epiP[mxIndex] = epiP.item()
            store_imgP[mxIndex] = imgP.item()
            mxIndex += 1

    store_answer["epiScore"][id] = store_epiP.sum()/mxIndex
    store_answer["imageScore"][id] = store_imgP.sum()/mxIndex

    logging=""
    for pname in ["epiScore", "imageScore", "p@k"]:
        logging +=  pname+": "

        _store = (store_answer[pname][id].item() * 100 // 20)
        _store = 5 if store_answer[pname][id] > 0.94 else _store
        _store = _store if _store > 0 else 0

        logging += f"{colorMap[_store]}{store_answer[pname][id]}\x1b[0m" + " "
    logging += "\n"

    print(logging)


####################################
#
# SQL touch methods
#
####################################


def keyIdToRow(dbPATH=dbSOURCE, key_id=1, timeout=10):
    if key_id==-1:
        return

    #Full DB, not VEC
    # print(f"keyid2 row @ {key_id}", end=": ")
    connection_s,cursor_s=call(dbPATH,timeout)

    try:
        cursor_s.row_factory = sqlite3.Row
        # print(key_id)
        cursor_s.execute("SELECT * FROM imag WHERE key_id = (?)", (key_id,))
        row = cursor_s.fetchone()
        cursor_s.close()
        connection_s.close()
        # print(row[0])
        return row

    except Exception as e:
        cursor_s.close()
        connection_s.close()
        print(e)
        return -2


def batchedKeyIdToRow(dbPATH=dbSOURCE, key_id=1, timeout=10):
    if key_id==-1:
        return

    #Full DB, not VEC
    connection_s,cursor_s=call(dbPATH,timeout)

    try:
        cursor_s.row_factory = sqlite3.Row
        # print(key_id)
        cursor_s.execute("SELECT * FROM imag WHERE OFFSET (?) LIMIT 200", (key_id,))
        # row = cursor_s.fetchone()
        rows = cursor_s.fetchall()
        cursor_s.close()
        connection_s.close()
        # print(row)
        return rows

    except Exception as e:
        cursor_s.close()
        connection_s.close()
        print(e)
        return [-2]





def nameToKeyID(dbPATH=dbSOURCE, name="", timeout=10):
    print(name)
    if name=="":
        return

    #Full DB, not VEC
    connection_s,cursor_s=call(dbPATH,timeout)

    try:
        print(name)
        cursor_s.execute("SELECT key_id FROM imag WHERE name = ?", (name,))
        row = cursor_s.fetchone()
        cursor_s.close()
        connection_s.close()
        return row[0]

    except Exception as e:
        cursor_s.close()
        connection_s.close()
        print(e)
        return -2




def getEverything(dbPATH, timeout=10):
    connection,cursor=call(dbPATH,timeout)
    try:
        cursor.row_factory = sqlite3.Row
        cursor.execute("SELECT rowid,embedding from vector_table")
        print(f"success")
        reply = [(a,deserialize_f32(b, 5120)) for a,b in cursor.fetchall()]
    except Exception as e:
        print(e)
        reply = []
    finally:
        cursor.close()
        connection.close()

    return reply



############################
#
# INIT IN-MEM FAISS INDEX
#
############################

# index = faiss.read_index("/Users/seanmoran/Documents/Master/2025/Feb2025/022425_faissPrep/faiss.IndexFlatL2_dirMap.index")
# index = faiss.read_index("/Users/seanmoran/Documents/Master/2025/Feb2025/022425_faissPrep/faiss.IndexHNSWSQ_QT8_m128eConst64.index")
# index = faiss.read_index("/Users/seanmoran/Documents/Master/2025/Feb2025/022425_faissPrep/faiss.IndexHNSWSQ_QT8.index")
# indexMap = faiss.IndexIDMap2(index)
index = faiss.read_index("/Users/sean/git/MLX/metalEye/utils/faiss.IndexIVFPQ.test.index")
assert index.is_trained

rows = getEverything(dbVECTOR_FTS5);
# k = 50
d = 5120                           # dimension
nb = len(rows)                      # database size
nq = nb//10                       # nb of queries
xb=np.array([np.array(xi[1]) for xi in rows]).astype('float32') #float32? this may be wrong
############











####################################
#
# Main Program
#
####################################

def mainProg():
    with open(sourcePartitions, 'r') as file:
        positions = json.load(file)

    print(xb.shape)
    for i in range(0,99130):
        embedded = _readEmbeddingByKeyId(dbVECTOR_FTS5, 1000, i)

        #batches of 200?

        if embedded == -404:
            pass

        if embedded!=-1:
            faissHNSW(i, embedded, positions)

        if i%999==0:
            with open("022425_faissIVFPQ_vector_pearson_analytics.json_2", "w") as zug:
                zug.write(json.dumps(store_answer,cls=MLXEncoder))

    with open("022425_faissIVFPQ_vector_pearson_analytics.json_2", "w") as zug:
        zug.write(json.dumps(store_answer,cls=MLXEncoder))


if __name__ == "__main__":
    mainProg()

