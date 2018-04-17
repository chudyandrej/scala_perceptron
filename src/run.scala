
import org.apache.Perceptron
import org.apache.log4j._
import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.spark.sql._
import breeze.linalg.DenseVector



object run {






  def parseLine(line:String):DenseVector[Double] = {
    val fields = line.split(",")

    DenseVector(fields(0).toDouble, fields(1).toDouble, fields(2).toDouble,fields(3).toDouble)
  }

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.ERROR)

    // Create a SparkContext using every core of the local machine, named RatingsCounter
    val sc = new SparkContext("local[*]", "RatingsCounter")

//    // Load up each line of the ratings data into an RDD
    val lines = sc.textFile("./data/rawstockdata_binary_test.csv")
    val features = lines.map(parseLine)
    val labels = lines.map(x => x.split(",")(4).toInt)

    labels.collect()
    features.collect()


    val removePunctuation: String => String = (text: String) => {
      val punctPattern = "[^a-zA-Z0-9\\s]".r
      punctPattern.replaceAllIn(text, "").toLowerCase
    }
    sc.textFile("/home/ubuntu/data.txt",4).map(removePunctuation)


    val a : Perceptron = new Perceptron(0.005f,100000)
    a.fit(features, labels, sc)




//    sqlContext.csvFile("cars.csv")

  }

}
