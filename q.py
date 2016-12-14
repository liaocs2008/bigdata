# spark-submit --py-files geohash.py q.py

from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, Row
from pyspark.sql.types import *
from pyspark.sql.functions import *


import sys


def q(taxi):
    import geohash

    conf = SparkConf()
    conf.set("spark.sql.crossJoin.enabled","true")
    sc = SparkContext(conf=conf)
    sqlCtx = SQLContext(sc)


    df = sqlCtx.read.format('com.databricks.spark.csv').option('header', 'true').option('inferSchema', 'true').load('/data/share/tlc/TripRecord/%s' % taxi)
    df = df.select(df['lpep_pickup_datetime'], df['Pickup_latitude'], df['Pickup_longitude'])
    df = df.filter(df['lpep_pickup_datetime'].isNotNull() &
                   df['Pickup_latitude'].isNotNull() &
                   df['Pickup_longitude'].isNotNull()
                  )

    timeframe_id = floor( (col("lpep_pickup_datetime").cast("long") / 3600)  )
    df = df.withColumn("tf", timeframe_id)

    # encode
    enc = udf(lambda lat, lon: geohash.encode(lat, lon, 7), StringType())
    df = df.withColumn("hash", enc(col("Pickup_latitude"), col("Pickup_longitude")))

    # group
    df.groupBy(['tf', 'hash']).agg({"*": "count"}).orderBy(desc("tf")).show(30)

    # decode
    schema = StructType([
        StructField("lat", FloatType(), False),
        StructField("lon", FloatType(), False)
    ])
    dec = udf(lambda h: geohash.decode(h), schema)

    df = df.select(dec(df['hash']).alias('latlon'))
    df.select('latlon.lat', 'latlon.lon').show(30)

    return



if __name__ == "__main__":
    #q(sys.argv[1])
    q('green_tripdata_2013-08.csv')

