
# import sqlite_vec
# import ollama


from typing import List
import struct
import datetime
import faiss

import os, sys
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)


 
from flask import Flask, jsonify
from flask import request

from types import SimpleNamespace
import numpy as np
import sqlite3
import json

cache = SimpleNamespace()
app = Flask(__name__)

import mlx.core as mx








def serialize_f32(vector: List[float]) -> bytes:
    """serializes a list of floats into a compact "raw bytes" format"""
    return struct.pack("%sf" % len(vector), *vector)

def deserialize_f32(vector,size=5120):
    return struct.unpack(f"{size}f", vector)


 
@app.route("/test", methods=['GET'])
def hello_microservice():
    message = {"message": "meow! @ flask"}
    return jsonify(message)
 
# @app.route("/apiv2/faiss_query", methods=['GET'])
# def hello_microservice():
#     message = {"message": "meow! @ flask"}
#     return jsonify(message)
 
# @app.route("/apiv1/sqlitevec_knn", methods=['GET'])
# def hello_microservice():
#     message = {"message": "meow! @ flask"}
#     return jsonify(message)
 
@app.route("/searchtotal", methods=['GET'])
def checkDB():
    name = request.args.get('name', default = "", type = str)
    print(name)
    print("@@@")

    key_id = nameToKeyID(cache.dbSOURCE, name, 10);

    # nArr = nameToNArr(dbSOURCE, name, 10);
    # rowId = nameToRowId(dbSOURCE, "GM12878_2000_mustache_#5", 10);
    # nArr = getRowId(dbSOURCE, "1", 10);
    # query = ollama.embed(model='llama3.2', input=str(nArr),)

    flagSourceAvail = True
    print("@@@", end=" ")
    # print("@@@", end=" ", file=sys.stderr, flush=True)
    #call source
    connection,cursor=call(cache.dbSOURCE,10)

    try:
        cursor.row_factory = sqlite3.Row
        # cursor.execute("SELECT rowid,embedding from vector_table")
        # print(id)
        # query_Row = keyIdToRow(dbSOURCE,id, 10)

        #change this.

        query_Row = keyIdToRow(cache.dbSOURCE,key_id, 10)
        q_vmax  = query_Row[6]
        q_image = mx.array(query_Row[5], dtype=mx.float32) / q_vmax * 255
        q_image = q_image.astype(dtype=mx.int8)
        q_epigenomic = mx.array(query_Row[17],dtype=mx.float32)
        #if source works, keep going else

    except Exception as e:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(e).__name__, e.args)
        print(message)
        print(e)
        flagSourceAvail = False
    finally:
        cursor.close()
        connection.close()       
        if flagSourceAvail==False:
            return

    #k,v here.
    print(cache.xb.shape)
    print(cache.sourceIndexKVMap[str(key_id)])

    xq = np.reshape(cache.xb[cache.sourceIndexKVMap[str(key_id)]], (1,-1))

    k = 8 #this doesn't cost much.
    D, I = cache.index.search(xq, k) # search @id KNN

    # store_epiP = mx.array([0]*7, dtype=mx.float32)
    # store_imgP = mx.array([0]*7, dtype=mx.float32)
    # store_pAtK = mx.array([0]*7, dtype=mx.float32)


    message = {"message":[]}

    mxIndex = 0
    for imx in range(len(I[0])):
        print(cache.IVsourceIndexKVMap[I[0][imx]])
        val = rowIdToName(cache.dbSOURCE, cache.IVsourceIndexKVMap[I[0][imx]], 10)
        # val = keyIdToRow(cache.dbSOURCE, cache.IVsourceIndexKVMap[I[0][imx]], 10)
        print(val)
        if val is None or val == -2:
            print('\x1b[91m\x1b[5mMissed Result!!!\x1b[0m')
            pass
        elif val[0]==id:
            pass
        # else:
        #     a_vmax  = val[6]
        #     a_image = mx.array(val[5], dtype=mx.float32) / a_vmax * 255
        #     a_image = a_image.astype(dtype=mx.int8)
        #     a_epigenomic = mx.array(val[17],dtype=mx.float32)
            # try:
            #     epiP = mlx_pairwise_correlation(q_epigenomic, a_epigenomic)
            # except ValueError as e:
            #     print(f"ValueError: {e}")
            #     epi = mx.array(0,dtype=mx.float32)
            # try:
            #     imgP = mlx_pairwise_correlation(q_image, a_image)
            # except ValueError as e:
            #     print(f"ValueError: {e}")
            #     imgP = mx.array(0,dtype=mx.float32)

            # store_epiP[mxIndex] = epiP.item()
            # store_imgP[mxIndex] = imgP.item()
            # xq_choose = np.reshape(cache.xb[int(I[0][imx]),:], (1,-1))
            # store_pAtK[mxIndex] = cosine_similarity_loss(mx.array(xq),mx.array(xq_choose)).item()
            # mxIndex += 1

    # store_answer["epiScore"][id] = store_epiP.sum()/mxIndex
    # store_answer["imageScore"][id] = store_imgP.sum()/mxIndex
    # store_answer["p@k"][id] = store_pAtK.sum()/mxIndex

    # logging=""
    # for pname in ["epiScore", "imageScore", "p@k"]:
    #     logging +=  pname+": "

    #     _store = (store_answer[pname][id].item() * 100 // 20)
    #     _store = 5 if store_answer[pname][id] > 0.94 else _store
    #     _store = _store if _store > 0 else 0

    #     logging += f"{colorMap[_store]}{store_answer[pname][id]}\x1b[0m" + " "
    # logging += "\n"

    # print(logging)



    # while rows:
        # node = rows.pop()
        # val = rowIdToName(cache.dbSOURCE, str(node[0]), 10)
        if val != [0]:
            message["message"]+=[val]

    return jsonify(message)





def call(PATH,TIMEOUT):
    #only for SOURCE db
    connection = sqlite3.connect(PATH, timeout=TIMEOUT)  # Set timeout to 10 seconds
    cursor = connection.cursor()
    return connection,cursor



def nameToKeyID(dbPATH, name="", timeout=10):
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


# def nameToNArr(dbPATH=dbSOURCE, name="", timeout=10):
# # def nameToRowId(dbPATH=dbSOURCE, name="", timeout=10):
#     print(name)
#     if name=="":
#         return

#     #Full DB, not VEC
#     connection_s,cursor_s=call(dbPATH,timeout)

#     try:
#         print(name)
#         cursor_s.execute("SELECT * FROM imag WHERE name = ?", (name,))
#         # cursor_s.execute("SELECT * FROM imag WHERE rowid = ? LIMIT 1", rowID) 
#         row = cursor_s.fetchone()
#         cursor_s.close()
#         connection_s.close()
#         # print(row[4])
#         # return row
#         return row[4]

#     except Exception as e:
#         cursor_s.close()
#         connection_s.close()
#         print(e)
#         return [0]



def rowIdToName(dbPATH, rowID="", timeout=10):
    if rowID=="":
        return

    #Full DB, not VEC
    connection_s,cursor_s=call(dbPATH,timeout)

    try:
        cursor_s.execute("SELECT * FROM imag WHERE rowid = ?", (rowID,))
        # cursor_s.execute("SELECT * FROM imag WHERE rowid = ? LIMIT 1", rowID) 
        row = cursor_s.fetchone()
        # print(row)
        cursor_s.close()
        connection_s.close()
        # print(row[0],row[1],row[2])
        return row[1]

    except Exception as e:
        cursor_s.close()
        connection_s.close()
        print(e)
        return [0]


def keyIdToRow(dbPATH, key_id=1, timeout=10):
    if key_id==-1:
        return

    connection_s,cursor_s=call(dbPATH,timeout)
    try:
        cursor_s.row_factory = sqlite3.Row
        cursor_s.execute("SELECT * FROM imag WHERE key_id = (?)", (key_id,))
        row = cursor_s.fetchone()
        cursor_s.close()
        connection_s.close()
        return row

    except Exception as e:
        cursor_s.close()
        connection_s.close()
        print(e)
        return -2


def _readEmbeddingByKeyId(dbPATH, timeout, key_id=0, dimensions=5120):
    connection,cursor=call(dbPATH,timeout)
    reply = [-1]
    try:
        cursor.row_factory = sqlite3.Row
        cursor.execute("SELECT * FROM vector_table WHERE rowid = ? LIMIT 1", [key_id,])
        print(key_id, end=": ")
        # print([b for b in cursor.fetchall()])
        # print([deserialize_f32(b, 512) for b in cursor.fetchall()])
        reply = [deserialize_f32(b[0], dimensions) for b in cursor.fetchall()]
        if reply == []:
            return -404
    except Exception as e:
        print(dbPATH)
        print(e, end=" ")
        print("failure")
        reply = [-1]
    finally:
        cursor.close()
        connection.close()
    return reply.pop()




def getEverything(dbPATH, dimensions=5120, timeout=10):
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




if __name__ == "__main__":
    ###move to main
    # dbSOURCE = "/Users/seanmoran/Documents/Master/2025/Feb2025/LariatTables/sourceTables/database_24_1bin.db";
    ## use FTS5 table
    # dbVECTOR_FTS5 = "/Users/seanmoran/Documents/Master/2025/Feb2025/vectorPilot/EB_databaseVEC.db.checkpoint_Feb10_2025.db"

    FAISS_Index = "/Users/seanmoran/Documents/Master/2025/Mar2025/030925_live_demo/SqueakToy/FAISSIndex/D3_llama3.2-3B-fts5_khImage+hist+epi-11bin/faiss.IndexIVFPQ.test.index"
    cache.dbSOURCE = "/Users/seanmoran/Documents/Master/2025/Mar2025/030925_live_demo/database_26_11bin.db";
    cache.dbVECTOR_FTS5 = "/Users/seanmoran/Documents/Master/2025/Mar2025/030925_live_demo/SqueakToy/vTables/D3_llama3.2-3B-fts5_khImage+hist+epi-11bin.db"

    k = 50
    d = 3072           # dimension
    rows = getEverything(cache.dbVECTOR_FTS5, d);
    print(len(rows))
    nb = len(rows)     # database size
    nq = nb//10        # nb of queries
    # xb=np.array([np.array(xi[1]) for xi in rows]).astype('float32')
    cache.xb = np.array([np.array(xi[1]) for xi in rows]).astype('float32')


    # index = faiss.read_index(FAISS_Index)
    cache.index = faiss.read_index(FAISS_Index)
    # assert index.is_trained
    assert cache.index.is_trained

    #only for subsampled, change to full in production. 
    KV_indices = "/".join(FAISS_Index.split("/")[:-1])+"/faiss.keyValue.map.json" 


    #from write FAISS. change the sampling rate upstream to fix KV map if using more than 1:8

    with open(KV_indices, "r") as zub:
        sourceIndexKVMap = json.load(zub)
    cache.sourceIndexKVMap = sourceIndexKVMap

    # IVsourceIndexKVMap = {v: int(k) for k, v in sourceIndexKVMap.items()}

    # cache.xb = xb
    # cache.index = index
    cache.IVsourceIndexKVMap = {v: int(k) for k, v in sourceIndexKVMap.items()}

    # cache.dbSOURCE = dbSOURCE
    # cachce.dbVECTOR_FTS5 = dbVECTOR_FTS5

    app.run(port=9999)











