import requests
import json
# x = requests.post("http://127.0.0.1:8080/v1/chat/completions", headers = {"Content-Type": "application/json"},json={"messages": [{"role":"user", "content":"Katarina liked your song"}]} )

from llama_cpp import Llama
import numpy as np

import sqlite3
import sqlite_vec
# from ollama import embed
# import ollama
from typing import List
import struct
import array
import datetime


def serialize_f32(vector: List[float]) -> bytes:
    """serializes a list of floats into a compact "raw bytes" format"""
    return struct.pack("%sf" % len(vector), *vector)




def call(PATH,TIMEOUT):

    connection = sqlite3.connect(PATH, timeout=TIMEOUT)  # Set timeout to 10 seconds
    cursor = connection.cursor()
    return connection,cursor


def untouch(dbPATH,timeout):
    db = sqlite3.connect(dbPATH)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    try:
        db.execute("DROP TABLE vec_items")#, (kwargs['tablename'],))
#         print(cursor.fetchall())
        print(f"success")
    except Exception as e:
        print("connection failure")
        print(e)
    finally:
        db.commit()
        db.close()






def _createTable(dbPATH, timeout,**kwargs):

    # connection,cursor=call(dbPATH,timeout)
    try:
        db = sqlite3.connect(dbPATH)
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)
        # cursor = connection.execute("CREATE TABLE vec_items USING vec0(embedding float[256])")
        db.execute("CREATE VIRTUAL TABLE vec_items USING vec0(key_id integer primary key, embedding float[3072])")
        # cursor = connection.execute("CREATE TABLE imag(name, dataset, condition, coordinates, numpyarr, viewing_vmax, dimensions, hic_path, PUB_ID, resolution, norm, meta)")
        print("table make; success")
        # db.close()
    finally:
        db.close()
        # cursor.close()
        # connection.close()


    # sqlite_vec.load(db)
def _readSOURCE_writeVECTOR(dbPATH1, dbPATH2,timeout,**kwargs):
# def _readMatchAllTable(dbPATH,timeout,**kwargs):
    def _readDB(offset, limit):
        connection_s,cursor_s=call(dbPATH1,timeout)
        # connection_t,cursor_t=call(dbPATH2,timeout)
        # dbvec = sqlite3.connect(dbPATH2)




        try:
            db = sqlite3.connect(dbPATH2)
            db.enable_load_extension(True)
            sqlite_vec.load(db)
            db.enable_load_extension(False)
            cursor_s.row_factory = sqlite3.Row
            # cursor_s.execute("SELECT key_id, hist_rel, numpyarr FROM imag LIMIT ? OFFSET ?", (limit,offset))
            ### for bin16
            
            cursor_s.execute("SELECT key_id, hist_rel, numpyarr, epigenomicFactors, motifDirection FROM imag LIMIT ? OFFSET ?", (limit,offset))
            row_ids = []
            reply = []
            llm = kwargs["model"]
            response = lambda: None
            response.embeddings = []
            incrementor = 0
            for en in cursor_s.fetchall():
                if incrementor%8!=0:
                    incrementor+=1
                    continue
                incrementor+=1

                try:
                    assert len(en[2])==4*65*65
                    row_ids += [en[0]]
                    rarr = b''

                    # harr = array.array('I', en[1])
                    barr = array.array('f', en[2])

                    ### for epigenomics
                    earr = array.array('f', en[3])

                    # for el in harr:
                    #     rarr += struct.pack('l', el)

                    for i in range(65):
                        for j in range(65):
                            rarr += struct.pack('f',barr[i*65+j])
                            # if 28<i<38 and 28<j<38:
                            #     rarr += struct.pack('f',barr[i*65+j])
                    # ### for epigenomics
                    for el in earr:
                        rarr += struct.pack('f', el)


                    # ### for motifs
                    # if en[4]!=":":
                    #     byteDir = bytes(en[4],'utf8')
                    #     for zbyte in byteDir:
                    #         rarr += struct.pack('I', zbyte)


                    embeddings = llm.create_embedding(str(rarr))
                    _vec = np.zeros(len(embeddings['data'][0]['embedding'][0]))
                    for em in embeddings['data'][0]['embedding']:
                        _vec[:] += np.array(em)[:]
                    _vec /= len(embeddings['data'][0]['embedding'])
                    response.embeddings +=[_vec]

                except Exception as e:
                    template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                    message = template.format(type(e).__name__, e.args)
                    print(message)
                    print(e)

            print("dataloaded@@",end="")
            # print(len(response.embeddings))
            for idx,embd in enumerate(response.embeddings):
                # print(idx, type(idx),row_ids[idx], len(embd))
                db.execute("INSERT INTO vec_items(key_id, embedding) VALUES (?, ?)", [row_ids[idx], serialize_f32(embd[:])],)
            # print(f"success")



        except sqlite3.OperationalError as e:
            # the limiter overflows whatever is left.
            # just read everything left.
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            # print([len(x) for x in reply])
            for x in range(len(reply)):
                if len(reply[x])!=16900:
                    print(row_ids[x])
                    continue    


            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(e).__name__, e.args)
            print(message)
            print(e)



        except Exception as e:
            print(e)

        finally:
            cursor_s.close()
            connection_s.close()
            db.commit()
            db.close()

    if not all(i in kwargs for i in ["limit","offset"]):
        raise Exception("need  \"limit\",\"offset\" in kwargs")
    return _readDB(kwargs['offset'],kwargs["limit"])


def mainProg():
    # dbSOURCE = "/Users/sean/Documents/Master/2025/Feb2025/sourceTables/database_14_bin.db"
    # dbSOURCE = "/Users/seanmoran/Documents/Master/2025/Feb2025/database_TEST/database_14_bin.db"
    dbSOURCE = "/Users/sean/Documents/Master/2025/Feb2025/sourceTables/database_25_5bin.db"

    # dbVECTOR = "/Users/sean/Documents/Master/2025/Feb2025/embeddedLoops/EB_databaseVEC_14.db"
    # dbVECTOR = "/Users/sean/Documents/Master/2025/Feb2025/testTables/metalLlamacppSPEEDTEST.db"
    # dbVECTOR = "debug.db"
    dbVECTOR = "/Users/sean/Documents/Master/2025/March2025/SqueakToy/ebTable/F2_llama3.2-3B-Image+epi-5bin.db"

    try:
        _createTable(dbVECTOR, 10)
    except sqlite3.OperationalError:
        untouch(dbVECTOR,100)
        _createTable(dbVECTOR, 10)

    hardLimiter = 99130;

    insert_kwargs = {
        "limit": 256,
        "offset": 0,
        "entrypoint": "llamacpp"
        }

    if insert_kwargs["entrypoint"]=="llamacpp":
        insert_kwargs["model"] = Llama(
        model_path="/Users/sean/Downloads/llama-3.2-3b-instruct-q4_k_m.gguf",
        n_gpu_layers=-1, # Uncomment to use GPU acceleration
        # seed=1337, # Uncomment to set a specific seed
        n_ctx=131072, # Uncomment to increase the context window
        embedding=True,
        verbose=False,
        )

    while insert_kwargs["offset"] < hardLimiter:
        tnow = datetime.datetime.now()
        _readSOURCE_writeVECTOR(dbSOURCE, dbVECTOR, 10, **insert_kwargs)
        insert_kwargs["offset"] = insert_kwargs.get("offset", 0) + insert_kwargs.get("limit", 10)
        print(datetime.datetime.now() - tnow, datetime.datetime.now())
        print(insert_kwargs)


if __name__ == "__main__":
    now = datetime.datetime.now()
    mainProg();
    print("MLX: ", end="")
    print(datetime.datetime.now() - now)






