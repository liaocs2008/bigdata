from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, Row
from pyspark.sql.types import *
from pyspark.sql.functions import *

import sys


# https://vanishingcodes.wordpress.com/2016/06/09/pyspark-tutorial-building-a-random-forest-binary-classifier-on-unbalanced-dataset/
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier as RF
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler, SQLTransformer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


def q():
    conf = (SparkConf().set("spark.driver.maxResultSize", "20g"))
    conf.set("spark.sql.crossJoin.enabled","true")
    sc = SparkContext(conf=conf)
    sqlCtx = SQLContext(sc)

    df = sqlCtx.read.format('com.databricks.spark.csv').option('header', 'true').option('inferSchema', 'true').load('yg2014.csv')
    df = df.withColumn("day", floor(col("tf") / 24) % 7 )
    df = df.withColumn("hour", col("tf") % 24 )

    l = [1, 20, 48, 106, 131, 146, 167, 185, 244, 286, 315, 331, 332, 359, 360]
    is_holiday = udf(lambda x: 1 if x in l else 0, IntegerType())
    df = df.withColumn("is_holiday", is_holiday(col('tf') % 365))

    df.show(30)


    df2 = sqlCtx.read.format('com.databricks.spark.csv').option('header', 'true').option('inferSchema', 'true').load('yg2015.csv')
    df2 = df2.withColumn("day", floor(col("tf") / 24) % 7 )
    df2 = df2.withColumn("hour", col("tf") % 24 )

    l = [1, 19, 47, 106, 130, 145, 172, 184, 250, 285, 315, 330, 331, 359]
    is_holiday = udf(lambda x: 1 if x in l else 0, IntegerType())
    df2 = df2.withColumn("is_holiday", is_holiday(col('tf') % 365))


    # apply random forest
    #

    cols_now = ['day',
                'hour',
                'is_holiday',
                'lat',
                'lon',
                'd_cnt']

    from pyspark.ml import Pipeline                                                    
    from pyspark.ml.regression import RandomForestRegressor                            
    from pyspark.ml.feature import VectorIndexer                                       
    from pyspark.ml.evaluation import RegressionEvaluator                              


                                                                                       
    assembler_features = VectorAssembler(inputCols=cols_now, outputCol='features')  
    labelIndexer = StringIndexer(inputCol='p_cnt', outputCol="label")                  
    tmp = [assembler_features, labelIndexer]                                           
    pipeline = Pipeline(stages=tmp)                                                    
                                                                                       
    trainingData = pipeline.fit(df).transform(df)                                      
    testData = pipeline.fit(df2).transform(df2)                                        
                                                                                       
                                                                                       
    # Set maxCategories so features with > 4 distinct values are treated as continuous.
    featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(trainingData)
                                                                                       
    # Train a RandomForest model.                                                      
    rf = RandomForestRegressor(featuresCol="indexedFeatures", numTrees=200)                          
                                                                                       
    # Chain indexer and forest in a Pipeline                                           
    pipeline = Pipeline(stages=[featureIndexer, rf])                                   
                                                                                       
    # Train model.  This also runs the indexer.                                        
    model = pipeline.fit(trainingData)                                                 
                                                                                       
    # Make predictions.                                                                
    predictions = model.transform(testData)                                            
                                                                                       
    # Select example rows to display.                                                  
    predictions.select("prediction", "label", "features").show(5)                      
                                                                                       
    # Select (prediction, true label) and compute test error                           
    evaluator = RegressionEvaluator(                                                   
                labelCol="label", predictionCol="prediction", metricName="rmse")       
    rmse = evaluator.evaluate(predictions)                                             
    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)                   


    evaluator = RegressionEvaluator(                                                   
                labelCol="label", predictionCol="prediction", metricName="r2")       
    r2 = evaluator.evaluate(predictions)                                             
    print("R2 on test data = %g" % r2)                   

                                                                                       
    rfModel = model.stages[1]                                                          
    print(rfModel)  # summary only                     


    return



if __name__ == "__main__":
    q()

