from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, Row
from pyspark.sql.types import *
from pyspark.sql.functions import *

import sys

def q(ym):
    import geohash

    conf = SparkConf()
    conf.set("spark.sql.crossJoin.enabled","true")
    sc = SparkContext(conf=conf)
    sqlCtx = SQLContext(sc)


    yellow = sqlCtx.read.format('com.databricks.spark.csv').option('header', 'true').option('inferSchema', 'true').load('/data/share/tlc/TripRecord/yellow_tripdata_%s' % ym)
    yellow.printSchema()
    yellow = yellow.select(yellow['tpep_pickup_datetime'].alias("lpep_pickup_datetime"),
                           yellow['tpep_dropoff_datetime'].alias("Lpep_dropoff_datetime"),
                           yellow['pickup_latitude'].alias( "Pickup_latitude"),
                           yellow['pickup_longitude'].alias("Pickup_longitude"),
                           yellow['dropoff_latitude'].alias( "Dropoff_latitude"),
                           yellow['dropoff_longitude'].alias("Dropoff_longitude")
                          )
    yellow = yellow.filter(yellow['lpep_pickup_datetime'].isNotNull() &
                           yellow['Lpep_dropoff_datetime'].isNotNull() &
                           yellow['Pickup_latitude'].isNotNull() &
                           yellow['Pickup_longitude'].isNotNull() &
                           yellow['Dropoff_latitude'].isNotNull() &
                           yellow['Dropoff_longitude'].isNotNull()
                          )
    yellow = yellow.filter((yellow['Pickup_latitude']  <  50) & (yellow['Dropoff_latitude']  <  50) &
                           (yellow['Pickup_latitude']  >  35) & (yellow['Dropoff_latitude']  >  35) &
                           (yellow['Pickup_longitude'] < -50) & (yellow['Dropoff_longitude'] < -50) &
                           (yellow['Pickup_longitude'] > -80) & (yellow['Dropoff_longitude'] > -80)
                          )
    yellow.show(10)

    green = sqlCtx.read.format('com.databricks.spark.csv').option('header', 'true').option('inferSchema', 'true').load('/data/share/tlc/TripRecord/green_tripdata_%s' % ym)
    green = green.select(green["lpep_pickup_datetime"],
                         green["Lpep_dropoff_datetime"],
                         green["Pickup_latitude"],
                         green["Pickup_longitude"],
                         green["Dropoff_latitude"],
                         green["Dropoff_longitude"]
                        )
    green = green.filter(green['lpep_pickup_datetime'].isNotNull() &
                         green['Lpep_dropoff_datetime'].isNotNull() &
                         green['Pickup_latitude'].isNotNull() &
                         green['Pickup_longitude'].isNotNull() &
                         green['Dropoff_latitude'].isNotNull() &
                         green['Dropoff_longitude'].isNotNull()
                        )
    green = green.filter((green['Pickup_latitude']  <  50) & (green['Dropoff_latitude']  <  50) &
                         (green['Pickup_latitude']  >  35) & (green['Dropoff_latitude']  >  35) &
                         (green['Pickup_longitude'] < -50) & (green['Dropoff_longitude'] < -50) &
                         (green['Pickup_longitude'] > -80) & (green['Dropoff_longitude'] > -80)
                        )

    df = yellow.union(green)

    schema = StructType([
        StructField("lat", FloatType(), False),
        StructField("lon", FloatType(), False)
    ])

    enc = udf(lambda lat, lon: geohash.encode(lat, lon, 6), StringType())
    dec = udf(lambda h: geohash.decode(h), schema)

    p_tf = floor( (col("lpep_pickup_datetime").cast("long") / 3600) % (365 * 24) )
    d_tf = floor( (col("Lpep_dropoff_datetime").cast("long") / 3600) % (365 * 24) )


    res = df.select(df['lpep_pickup_datetime'], df['Pickup_latitude'], df['Pickup_longitude'])
    res = res.withColumn("p_tf", p_tf)
    res = res.withColumn("hash", enc(col("Pickup_latitude"), col("Pickup_longitude")))
    res = res.groupBy(['p_tf', 'hash']).agg({"*": "count"}).orderBy(desc("p_tf"))
    res = res.select(res['p_tf'], res['hash'], res['count(1)'].alias('p_cnt'))

    print "LEO: Finish PROCESSING PICKUP", res.count()
    res.show(10)

    res1 = df.select(df['Lpep_dropoff_datetime'], df['Dropoff_latitude'], df['Dropoff_longitude'])
    res1 = res1.withColumn("d_tf", d_tf)
    res1 = res1.withColumn("hash", enc(col("Dropoff_latitude"), col("Dropoff_longitude")))
    res1 = res1.groupBy(['d_tf', 'hash']).agg({"*": "count"}).orderBy(desc("d_tf"))
    res1 = res1.select(res1['d_tf'], res1['hash'], res1['count(1)'].alias('d_cnt'))

    print "LEO: Finish PROCESSING DROPOFF", res1.count()
    res1.show(10)


    # JOIN RES & RES1
    res2 = res.join(res1, 'hash').filter(res['p_tf'] + lit(3) > res1['d_tf'])
    res2 = res2.filter(res2['hash'] != "")
    print "LEO: FINISH JOIN step 1", res2.count()
    res2 = res2.groupBy(['p_tf', 'hash']).agg({"d_cnt": "count", "p_cnt":"mean"}).orderBy(desc("p_tf"))
    print "LEO: FINISH JOIN step 2", res2.count()
    res.unpersist()
    res1.unpersist()


    res2 = res2.select(res2['p_tf'], dec(res2['hash']).alias('latlon'), \
                       res2['count(d_cnt)'].alias('d_cnt'), res2['avg(p_cnt)'].cast(IntegerType()).alias('p_cnt'))
    res2 = res2.select(res2['p_tf'].alias('tf'), \
                       res2['latlon.lat'].alias('lat'), \
                       res2['latlon.lon'].alias('lon'),\
                       res2['d_cnt'], res2['p_cnt'])
    res2.show(30)
    res2.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").save('proc_yellow_green_%s' % ym)

    return



if __name__ == "__main__":
    q(sys.argv[1])

