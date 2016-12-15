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

    schema = StructType([
        StructField("lat", FloatType(), False),
        StructField("lon", FloatType(), False)
    ])

    enc = udf(lambda lat, lon: geohash.encode(lat, lon, 6), StringType())
    dec = udf(lambda h: geohash.decode(h), schema)

    p_tf = floor( (col("lpep_pickup_datetime").cast("long") / 3600) % (365 * 24) )
    d_tf = floor( (col("Lpep_dropoff_datetime").cast("long") / 3600) % (365 * 24) )


    res = df.select(df['lpep_pickup_datetime'], df['Pickup_latitude'], df['Pickup_longitude'])
    res = res.filter(res['lpep_pickup_datetime'].isNotNull() &
                     res['Pickup_latitude'].isNotNull() &
                     res['Pickup_longitude'].isNotNull()
                    )
    res = res.filter((res['Pickup_latitude']  < 50) &
                     (res['Pickup_latitude']  > 35) &
                     (res['Pickup_longitude'] < -50) &
                     (res['Pickup_longitude'] > -80)
                    )

    res = res.withColumn("p_tf", p_tf)
    res = res.withColumn("hash", enc(col("Pickup_latitude"), col("Pickup_longitude")))
    res = res.groupBy(['p_tf', 'hash']).agg({"*": "count"}).orderBy(desc("p_tf"))
    res = res.select(res['p_tf'], res['hash'], res['count(1)'].alias('p_cnt'))
    #res.show(10)


    res1 = df.select(df['Lpep_dropoff_datetime'], df['Dropoff_latitude'], df['Dropoff_longitude'])
    res1 = res1.filter(res1['Lpep_dropoff_datetime'].isNotNull() &
                       res1['Dropoff_latitude'].isNotNull() &
                       res1['Dropoff_longitude'].isNotNull()
                      )
    res1 = res1.filter((res1['Dropoff_latitude']  < 50) &
                       (res1['Dropoff_latitude']  > 35) &
                       (res1['Dropoff_longitude'] < -50) &
                       (res1['Dropoff_longitude'] > -80)
                      )

    res1 = res1.withColumn("d_tf", d_tf)
    res1 = res1.withColumn("hash", enc(col("Dropoff_latitude"), col("Dropoff_longitude")))
    res1 = res1.groupBy(['d_tf', 'hash']).agg({"*": "count"}).orderBy(desc("d_tf"))
    res1 = res1.select(res1['d_tf'], res1['hash'], res1['count(1)'].alias('d_cnt'))
    #res1.show(10)


    # JOIN RES & RES1
    res2 = res.join(res1, 'hash').filter(res['p_tf'] + lit(3) > res1['d_tf'])
    res2 = res2.groupBy(['p_tf', 'hash']).agg({"d_cnt": "count", "p_cnt":"mean"}).orderBy(desc("p_tf"))
    #res2.show(30)

    res2 = res2.select(res2['p_tf'], dec(res2['hash']).alias('latlon'), \
                       res2['count(d_cnt)'].alias('d_cnt'), res2['avg(p_cnt)'].cast(IntegerType()).alias('p_cnt'))
    res2 = res2.select(res2['p_tf'].alias('tf'), \
                       res2['latlon.lat'].alias('lat'), \
                       res2['latlon.lon'].alias('lon'),\
                       res2['d_cnt'], res2['p_cnt'])
    #res2.show(30)
    res2.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").save('proc_%s' % taxi)

    return



if __name__ == "__main__":
    q(sys.argv[1])
    #q('green_tripdata_2013-08.csv')


