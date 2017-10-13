##################### Library Imports ########################################

##### Import for Vagrant run ########
# import findspark
# findspark.init("/usr/local/bin/spark-1.3.1-bin-hadoop2.6")


import sys
from operator import add
from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating


###################### FILE TO RDD FUNCTIONS ######################

def getArtistMapping(line):
    tokens = line.split('\t')
    
    if (len(tokens[0]) == 0):
        return None

    else:
        try:
            
            artist_id = int(tokens[0])
            artist_name = tokens[1]
            return (artist_id, artist_name)

        except:
            return None


def getArtistAlias(line):
    tokens = line.split('\t')

    if (len(tokens[0]) == 0):
        return None

    else:
        try:
            id1 = int(tokens[0])
            id2 = int(tokens[1])
            return (id1 , id2)

        except:
            return None



def getUserArtist(line):
    tokens = line.split(' ')

    artistId = int(tokens[1])

    # Check if alias exists
    if (artistId in bArtistAlias.value):
        artistId = bArtistAlias.value[artistId]
    
    userid = int(tokens[0])
    playcount = int(tokens[2])
    
    return Rating(userid, artistId, playcount)


################################################################


#for SparkConf() check out http://spark.apache.org/docs/latest/configuration.html
conf = (SparkConf()
         .setMaster("local")
         .setAppName("AudioRecommender")
         .set("spark.executor.memory", "1g"))
sc = SparkContext(conf = conf)

print("Launch App..")
if __name__ == "__main__":
    print("Initiating main..")
    
    # lines_nonempty = lines.filter( lambda x: len(x) > 0 )
    # counts = lines_nonempty.flatMap(lambda x: x.split(' ')) \
    #               .map(lambda x: (x, 1)) \
    #               .reduceByKey(add)
    # output = counts.collect()
    # for (word, count) in output:
    #     print("%s: %i" % (word, count))

    ###################### LOADING FILES ######################

    # Change the path to local path (vagrant data directory) while running in vagrant
    rawArtistData = sc.textFile("s3://csds-emr-nitesh/csds-spark-emr/artist_data.txt")
    rawUserArtistData = sc.textFile("s3://csds-emr-nitesh/csds-spark-emr/user_artist_data.txt")
    rawArtistAlias = sc.textFile("s3://csds-emr-nitesh/csds-spark-emr/artist_alias.txt")


    artistAlias = rawArtistAlias.map(lambda line: getArtistAlias(line)).filter(lambda line: line != None).collectAsMap()
    artistByID = rawArtistData.map(lambda line: getArtistMapping(line)).filter(lambda line: line != None).collectAsMap()
    bArtistAlias = sc.broadcast(artistAlias)

    ################################################################



    ###################### TRAINING THE MODEL ######################
    training_data = rawUserArtistData.map(lambda line: getUserArtist(line)).cache()

    model = ALS.trainImplicit(training_data, 50, 5, lambda_ = 1.0, alpha = 40.0, seed = 42)

    ################################################################



    ###################### user_recommendations ######################

    userId = 2093760
    num_recommendations = 10

    user_recommendations = map(lambda x: artistByID.get(x.product), model.call("recommendProducts", userId, num_recommendations))

    print("\nRecommended artists:\n")
    for recommendation in user_recommendations:
        print(recommendation)

    ################################################################


    sc.stop()
