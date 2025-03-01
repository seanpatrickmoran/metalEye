# import faiss
import sqlite3
import sqlite_vec
from typing import List
import struct

import os, sys
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

# dbVECTOR = "/Users/sean/Documents/Master/2025/Feb2025/embeddedLoops/EB_databaseVEC_18.db"
# # dbVECTOR = "/Users/sean/Documents/Master/2025/Feb2025/embeddedLoops/EB_databaseVEC_14.db"
# # dbVECTOR = "/Users/seanmoran/Documents/Master/2025/Feb2025/vectorPilot/EB_databaseVEC.db"

# dbSQLITE3VEC = "/Users/sean/Documents/Master/2025/Feb2025/virtualTables/EB_databaseVEC_18_fts5vec.db"



def call(PATH,TIMEOUT):
    connection = sqlite3.connect(PATH, timeout=TIMEOUT)  # Set timeout to 10 seconds
    cursor = connection.cursor()
    return connection,cursor


def _createTable(dbPATH, timeout,**kwargs):

    connection,cursor=call(dbPATH,timeout)
    try:
        cursor = connection.execute("CREATE VIRTUAL TABLE vector_table USING fts5(embedding)")
        print("table make; success")
    except Exception as e:
        print(e)
    finally:
        cursor.close()
        connection.close()



def _writeManyToTable(dbPATH,timeout,**kwargs):
    def _write_db(index,data):
        connection,cursor=call(dbPATH,timeout)
        try:
            submission = [[x[1]] for x in data]
            cursor.executemany("INSERT INTO vector_table(embedding) VALUES(?)", submission)
            print(f"success")

        except Exception as e:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(e).__name__, e.args)
            print(message)
            print(e)

        finally:
            connection.commit()
            cursor.close()
            connection.close()
    
    payload = [x for x in kwargs["data"]]
    d_i = 0
    while d_i + 999 < len(payload):
        _write_db(d_i,payload[d_i:d_i+999])
        d_i += 999
    _write_db(d_i, payload[d_i:])




# def deserialize_f32(vector):
    """serializes a list of floats into a compact "raw bytes" format"""
    # return struct.unpack("256f", vector[1])



def getEverything(dbPATH, timeout=10):
    try:
        db = sqlite3.connect(dbPATH)
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)
        row = db.execute("SELECT key_id, embedding from vec_items").fetchall()

        db.close()
        return row

    except Exception as e:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(e).__name__, e.args)
        print(message)
        db.close()
        print(e)
        return [0]



def runMain(dbVECTOR,dbSQLITE3VEC):
    rows = getEverything(dbVECTOR);
    insertion_kwargs = {
    "data":rows
    }
    try:
        _createTable(dbSQLITE3VEC,10)
    except Exception as e:
        print("table already exists")
        print(e)

    _writeManyToTable(dbSQLITE3VEC,10,**insertion_kwargs)





if __name__ == "__main__":
    dbVECTOR = sys.argv[1]
    if not os.path.isdir('/'.join(dbVECTOR.split("/")[:-1])):
        os.mkdir('/'.join(dbVECTOR.split("/")[:-1]))

    dbSQLITE3VEC = sys.argv[2]
    if not os.path.isdir('/'.join(dbSQLITE3VEC.split("/")[:-1])):
        os.mkdir('/'.join(dbSQLITE3VEC.split("/")[:-1]))

    # dbVECTOR = "/Users/sean/Documents/Master/2025/Feb2025/embeddedLoops/EB_databaseVEC_18.db"
    # dbSQLITE3VEC = "/Users/sean/Documents/Master/2025/Feb2025/virtualTables/EB_databaseVEC_18_fts5vec.db"
    runMain(dbVECTOR,dbSQLITE3VEC)
